// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Seven-stage stable-backport reviewer.
//!
//! Adapts `~/stable-ai/reviewer/prompt.txt`'s 7-phase forensic pipeline to
//! sashiko's per-stage tool-calling JSON contract. Each stage is one
//! tool-augmented LLM conversation that emits a uniform
//! `{ "concerns": [...], "stage_summary": "..." }` payload (stage 7
//! emits the synthesis verdict instead).
//!
//! The runner is intentionally simpler than `Worker::run` in `prompts.rs`:
//! no Phase 0 prescreening, no inline-template format validators, no
//! token-budget enforcement (the AI provider already enforces its own
//! per-call budget). What sashiko's reviewer needs from this module is:
//! drive a deterministic, traceable conversation per stage, accumulate
//! structured concerns, and synthesize a verdict.

use crate::ai::{AiMessage, AiProvider, AiRequest, AiRole, AiUsage};
use crate::worker::tools::ToolBox;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::warn;

/// Maximum tool-call interactions allowed per stage. Hit this and the stage
/// errors out with `LimitExceeded`. Generous: a thorough Phase 4 might
/// involve 6-10 lei queries, a few b4 digs, plus surrounding git_log.
const DEFAULT_STAGE_TURN_BUDGET: usize = 30;

/// Sampling temperature for the stages. Mirrors stable-ai's effective
/// behavior — low temperature to keep the model's chain-of-thought
/// disciplined, but not zero (we still want some judgment headroom).
const DEFAULT_STAGE_TEMPERATURE: f32 = 0.2;

/// Concrete metadata for a single (queue commit, target version) review.
/// This is the input to [`BackportStageRunner::run_all`].
#[derive(Debug, Clone)]
pub struct StageInput {
    pub queue_sha: String,
    pub queue_branch: String, // e.g. "stable-rc/queue/6.12"
    pub queue_subject: String,
    pub queue_body: String,
    pub queue_diff: String,
    pub upstream_sha: Option<String>,
    pub upstream_body: Option<String>,
    pub upstream_diff: Option<String>,
    pub target_version: String, // e.g. "6.12"
    pub target_branch: String,  // e.g. "linux-6.12.y"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concern {
    /// Filled in by the runner after parsing the model's payload — the
    /// model itself doesn't emit `stage`, so it defaults to 0 here.
    #[serde(default)]
    pub stage: u8,
    pub kind: String,
    pub severity: String, // "low" | "medium" | "high" | "critical"
    pub problem: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageOutcome {
    pub verdict: String, // "yes" | "no" | "needs_review"
    pub confidence: Option<f64>,
    pub summary: String,
    pub concerns: Vec<Concern>,
    pub usage: AiUsage,
    pub per_stage: Vec<StageRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: u8,
    pub stage_summary: String,
    pub concerns: Vec<Concern>,
    pub turns: usize,
}

pub struct BackportStageRunner {
    provider: Arc<dyn AiProvider>,
    tools: ToolBox,
    stages: Option<Vec<u8>>,
    turn_budget: usize,
    temperature: f32,
    context_tag: String,
}

impl BackportStageRunner {
    pub fn new(
        provider: Arc<dyn AiProvider>,
        tools: ToolBox,
        stages: Option<Vec<u8>>,
        context_tag: String,
    ) -> Self {
        Self {
            provider,
            tools,
            stages,
            turn_budget: DEFAULT_STAGE_TURN_BUDGET,
            temperature: DEFAULT_STAGE_TEMPERATURE,
            context_tag,
        }
    }

    pub fn with_turn_budget(mut self, n: usize) -> Self {
        self.turn_budget = n.max(1);
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Run the 7 stages (or the configured subset). Returns a final
    /// verdict and per-stage records. A stage that errors causes the
    /// run to abort with the partial state visible in the error chain.
    pub async fn run_all(&self, input: StageInput) -> Result<StageOutcome> {
        let active: Vec<u8> = match &self.stages {
            Some(list) => list
                .iter()
                .copied()
                .filter(|s| (1..=7).contains(s))
                .collect(),
            None => (1u8..=7).collect(),
        };

        let mut per_stage: Vec<StageRecord> = Vec::with_capacity(active.len());
        let mut all_concerns: Vec<Concern> = Vec::new();
        let mut total_usage = AiUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: Some(0),
        };

        // Stages 1-6 emit the uniform concerns/stage_summary payload.
        for stage in active.iter().copied().filter(|s| *s != 7) {
            let out = self
                .run_one_stage(stage, &input, &per_stage)
                .await
                .with_context(|| format!("stage {stage}"))?;
            for u in &out.usages {
                accumulate(&mut total_usage, u);
            }
            for c in &out.concerns {
                all_concerns.push(c.clone());
            }
            per_stage.push(StageRecord {
                stage,
                stage_summary: out.stage_summary,
                concerns: out.concerns,
                turns: out.turns,
            });
        }

        // Stage 7 (synthesis): only run if it's part of the active set.
        let synth = if active.contains(&7) {
            self.run_synthesis(&input, &per_stage)
                .await
                .with_context(|| "stage 7 synthesis")?
        } else {
            // No synthesis requested — collapse to a needs_review verdict.
            SynthesisOutput {
                verdict: "needs_review".into(),
                confidence: None,
                summary: "synthesis stage skipped".into(),
                concerns: all_concerns.clone(),
                usages: vec![],
                turns: 0,
            }
        };
        for u in &synth.usages {
            accumulate(&mut total_usage, u);
        }
        per_stage.push(StageRecord {
            stage: 7,
            stage_summary: synth.summary.clone(),
            concerns: synth.concerns.clone(),
            turns: synth.turns,
        });

        Ok(StageOutcome {
            verdict: synth.verdict,
            confidence: synth.confidence,
            summary: synth.summary,
            concerns: synth.concerns,
            usage: total_usage,
            per_stage,
        })
    }

    async fn run_one_stage(
        &self,
        stage: u8,
        input: &StageInput,
        prior: &[StageRecord],
    ) -> Result<SimpleStageOutput> {
        let system = build_system_prompt(stage, input);
        let user = build_stage_user_prompt(stage, input, prior);

        let initial_msg = AiMessage {
            role: AiRole::User,
            content: Some(user),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let mut history: Vec<AiMessage> = vec![initial_msg];

        let mut usages: Vec<AiUsage> = Vec::new();
        let mut turns = 0usize;

        let final_text = loop {
            turns += 1;
            if turns > self.turn_budget {
                bail!(
                    "stage {} exceeded turn budget of {}",
                    stage,
                    self.turn_budget
                );
            }

            let req = AiRequest {
                system: Some(system.clone()),
                messages: history.clone(),
                tools: Some(self.tools.get_declarations_generic()),
                temperature: Some(self.temperature),
                response_format: None,
                context_tag: Some(format!(
                    "{} s:{}] ",
                    self.context_tag.trim_end_matches(']'),
                    stage
                )),
            };

            let resp = self
                .provider
                .generate_content(req)
                .await
                .with_context(|| format!("stage {stage} provider call (turn {turns})"))?;

            if let Some(u) = &resp.usage {
                usages.push(u.clone());
            }

            let assistant = AiMessage {
                role: AiRole::Assistant,
                content: resp.content.clone(),
                thought: resp.thought.clone(),
                thought_signature: resp.thought_signature.clone(),
                tool_calls: resp.tool_calls.clone(),
                tool_call_id: None,
            };
            history.push(assistant);

            if let Some(calls) = resp.tool_calls
                && !calls.is_empty()
            {
                let mut tool_msgs = Vec::with_capacity(calls.len());
                for call in calls {
                    let result = match self
                        .tools
                        .call(&call.function_name, call.arguments.clone())
                        .await
                    {
                        Ok(v) => v.to_string(),
                        Err(e) => json!({ "error": e.to_string() }).to_string(),
                    };
                    tool_msgs.push(AiMessage {
                        role: AiRole::Tool,
                        content: Some(result),
                        thought: None,
                        thought_signature: None,
                        tool_calls: None,
                        tool_call_id: Some(call.id),
                    });
                }
                history.extend(tool_msgs);
                continue;
            }

            // No tool calls — should be the final structured payload.
            match resp.content {
                Some(c) if !c.trim().is_empty() => break c,
                _ => bail!("stage {stage} produced empty response with no tool calls"),
            }
        };

        let parsed = parse_stage_payload(&final_text)
            .with_context(|| format!("stage {stage} payload parse"))?;
        Ok(SimpleStageOutput {
            stage_summary: parsed.stage_summary,
            concerns: parsed
                .concerns
                .into_iter()
                .map(|c| Concern { stage, ..c })
                .collect(),
            usages,
            turns,
        })
    }

    async fn run_synthesis(
        &self,
        input: &StageInput,
        prior: &[StageRecord],
    ) -> Result<SynthesisOutput> {
        let system = build_synthesis_system_prompt(input);
        let user = build_synthesis_user_prompt(input, prior);

        let req = AiRequest {
            system: Some(system),
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some(user),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: Some(self.temperature),
            response_format: None,
            context_tag: Some(format!("{} s:7] ", self.context_tag.trim_end_matches(']'))),
        };

        let resp = self
            .provider
            .generate_content(req)
            .await
            .context("synthesis provider call")?;
        let content = resp
            .content
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("synthesis stage produced no content"))?;

        let synth: SynthesisPayload = parse_synthesis_payload(content)?;
        let mut usages = Vec::new();
        if let Some(u) = resp.usage {
            usages.push(u);
        }
        Ok(SynthesisOutput {
            verdict: synth.verdict,
            confidence: synth.confidence,
            summary: synth.summary,
            concerns: synth
                .concerns
                .into_iter()
                .map(|c| Concern { stage: 7, ..c })
                .collect(),
            usages,
            turns: 1,
        })
    }
}

fn accumulate(total: &mut AiUsage, u: &AiUsage) {
    total.prompt_tokens += u.prompt_tokens;
    total.completion_tokens += u.completion_tokens;
    total.total_tokens += u.total_tokens;
    if let Some(c) = u.cached_tokens {
        let prev = total.cached_tokens.unwrap_or(0);
        total.cached_tokens = Some(prev + c);
    }
}

struct SimpleStageOutput {
    stage_summary: String,
    concerns: Vec<Concern>,
    usages: Vec<AiUsage>,
    turns: usize,
}

struct SynthesisOutput {
    verdict: String,
    confidence: Option<f64>,
    summary: String,
    concerns: Vec<Concern>,
    usages: Vec<AiUsage>,
    turns: usize,
}

#[derive(Debug, Deserialize)]
struct StagePayload {
    #[serde(default)]
    concerns: Vec<Concern>,
    #[serde(default)]
    stage_summary: String,
}

#[derive(Debug, Deserialize)]
struct SynthesisPayload {
    verdict: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    concerns: Vec<Concern>,
}

fn parse_stage_payload(text: &str) -> Result<StagePayload> {
    let json = strip_fences(text);
    let v: Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("not valid JSON: {e}; got: {}", trim_for_log(&json)))?;
    Ok(serde_json::from_value(v)?)
}

fn parse_synthesis_payload(text: &str) -> Result<SynthesisPayload> {
    let json = strip_fences(text);
    let v: Value = serde_json::from_str(&json).map_err(|e| {
        anyhow::anyhow!(
            "synthesis not valid JSON: {e}; got: {}",
            trim_for_log(&json)
        )
    })?;
    let mut p: SynthesisPayload = serde_json::from_value(v)?;
    let v = p.verdict.trim().to_ascii_lowercase();
    if !matches!(v.as_str(), "yes" | "no" | "needs_review") {
        warn!("synthesis verdict {v:?} not in {{yes,no,needs_review}} — coercing to needs_review");
        p.verdict = "needs_review".into();
    } else {
        p.verdict = v;
    }
    Ok(p)
}

fn strip_fences(text: &str) -> String {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")) {
        let rest = rest.trim_start_matches('\n');
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim().to_string();
        }
    }
    t.to_string()
}

fn trim_for_log(s: &str) -> String {
    s.chars().take(400).collect()
}

// ---- Prompt construction ---------------------------------------------------

fn build_system_prompt(stage: u8, input: &StageInput) -> String {
    let stage_intro = stage_instruction(stage, input);
    format!(
        "{shared}\n\n{stage_intro}\n\n{output_contract}",
        shared = shared_system_prompt(input),
        stage_intro = stage_intro,
        output_contract = STAGE_OUTPUT_CONTRACT,
    )
}

fn build_synthesis_system_prompt(input: &StageInput) -> String {
    format!(
        "{shared}\n\n{stage_intro}\n\n{output_contract}",
        shared = shared_system_prompt(input),
        stage_intro = stage_instruction(7, input),
        output_contract = synthesis_output_contract(input),
    )
}

fn shared_system_prompt(input: &StageInput) -> String {
    let date = chrono::Utc::now().format("%A, %B %d, %Y");
    format!(
        "You are a Linux kernel stable-tree reviewer performing deep verification of\n\
         a commit that has ALREADY BEEN SELECTED for backport to {target_branch}.\n\
         Today's date: {date}.\n\n\
         The selection decision is already made. Your job is NOT to decide whether\n\
         the commit *should* be backported per stable-process rules. Your job is to\n\
         VERIFY that the selection is sound — that the commit, as it sits on\n\
         `{queue_branch}`, will be CORRECT IN THE CONTEXT OF the target tree.\n\n\
         Specifically check:\n\
         1. ALL DEPENDENCIES ARE PRESENT — every commit this patch depends on is\n\
            either already in {target_branch} (released) or also queued ahead of\n\
            this commit on {queue_branch}. Missing dependencies cause build\n\
            failures, incorrect behavior, or silent corruption on this tree.\n\
         2. NO IMPORTANT FOLLOW-UP FIXES ARE MISSING — if mainline (origin/master)\n\
            has commits with `Fixes: {upstream}` or that revert this commit, those\n\
            follow-ups must also be queued. A backport that itself has known bugs\n\
            is worse than no backport.\n\
         3. THE CODE IT TOUCHES ACTUALLY EXISTS IN THIS TREE — functions, struct\n\
            members, macros, config options, files, and surrounding context lines\n\
            must match in the target tree. Even if the patch applies cleanly,\n\
            semantic divergence (reordered ops, missing prerequisites, refactored\n\
            error paths) can silently break behavior on this version.\n\
         4. NO INAPPROPRIATE FEATURES OR REGRESSIONS — verify the commit doesn't\n\
            sneak in new APIs, new config options, or changed behavior that breaks\n\
            existing callers in this tree.\n\
         5. THE BUG ACTUALLY EXISTS IN THIS VERSION — fixes for bugs introduced\n\
            after the {version} branch point are unnecessary and add risk for no\n\
            benefit.\n\n\
         IMPORTANT: \"this is a cleanup, not a user-visible bug fix\" or \"there's\n\
         no Cc: stable\" are NOT problems. Cleanups, code-quality changes, and\n\
         dead-code removals routinely ship in stable. What matters is whether the\n\
         commit applies CORRECTLY in this tree's context. Eligibility per\n\
         stable-kernel-rules.rst is upstream-process; by the time this patch is on\n\
         {queue_branch} someone has already decided.\n\n\
         Every factual claim must be verified against actual code, git history,\n\
         or mailing-list discussions via the available tools (git_show, git_log,\n\
         search_file_content, lei_search, b4_dig, lore_thread, read_files). If\n\
         you cannot verify a claim, mark it UNVERIFIED — do NOT let unverified\n\
         claims drive the verdict.\n\n\
         TARGET VERSION: {version}\n\
         TARGET BRANCH:  {target_branch}   (last released)\n\
         QUEUE BRANCH:   {queue_branch}    (this commit + its peers, queued for next release)\n\
         QUEUE SHA:      {queue_sha}\n\
         UPSTREAM SHA:   {upstream}        (original commit on origin/master)\n\n\
         CRITICAL git rules:\n\
         - NEVER use `git log --all` or any --all flag — it scans hundreds of\n\
           branches and takes hours. Use specific refs (`origin/master`,\n\
           `{target_branch}`, `{queue_branch}`).\n\
         - NEVER run `git log` without a path limiter (`-- <file>`), an explicit\n\
           ref range, a `--grep`, or `-n <count>`. The kernel has millions of\n\
           commits.\n\n\
         GUIDANCE: batch independent tool calls in a single response (multiple\n\
         git_log queries, several lei_search queries) so this conversation does\n\
         not burn turns on serial round-trips. lei_search ALWAYS runs with\n\
         --no-save, so speculative queries are free.",
        date = date,
        version = input.target_version,
        target_branch = input.target_branch,
        queue_branch = input.queue_branch,
        queue_sha = input.queue_sha,
        upstream = input
            .upstream_sha
            .as_deref()
            .unwrap_or("(none — recover via Link: trailer + lei_search/b4_dig if needed)"),
    )
}

fn stage_instruction(stage: u8, input: &StageInput) -> String {
    match stage {
        1 => STAGE_1.to_string(),
        2 => STAGE_2.to_string(),
        3 => stage_3(input),
        4 => STAGE_4.to_string(),
        5 => stage_5(input),
        6 => stage_6(input),
        7 => stage_7(input),
        _ => String::new(),
    }
}

fn stage_3(input: &StageInput) -> String {
    let tb = &input.target_branch;
    let qsha = &input.queue_sha;
    let ver = &input.target_version;
    format!(
        "STAGE 3 — Dependency verification.\n\n\
This is the most important load-bearing check. The reference for\n\
\"present in the tree when this commit applies\" is {qsha}^ — the\n\
parent of the queue commit, which represents the tree state\n\
immediately before this cherry-pick lands. It includes everything in\n\
{tb} plus every queued commit ahead of this one. It does NOT include\n\
this commit itself or any later queue commits.\n\n\
HOW TO PROBE PRESENCE — USE STATE READS, NOT COMMIT-PRESENCE\n\
HEURISTICS. The available tools (`git_log --grep` and `git_show`)\n\
do NOT give clean answers to \"is commit X in ref Y's history\".\n\
Specifically:\n\
\n\
  - `git log <ref> --grep='<sha-prefix>'` searches COMMIT MESSAGES.\n\
    On a stable tree it finds backport-of-X entries (because the\n\
    cherry-pick trailer records the upstream SHA), but on mainline\n\
    it finds only commits that REFERENCE X (e.g. `Fixes: <X>`\n\
    trailers) — NOT commit X itself. A negative result on mainline\n\
    does not mean X is absent.\n\
  - `git log <ref> --grep='<subject-keyword>'` is a heuristic.\n\
    Similar subjects in the same subsystem produce false positives;\n\
    Greg sometimes rewrites subjects (e.g. adds a subsystem prefix)\n\
    so a stable backport's subject may differ from upstream.\n\
  - `git log A..B` is NOT an ancestor test. `A..B` returns commits\n\
    reachable from B not from A; the output is non-empty in both\n\
    the ancestor and non-ancestor cases. Do not use it for presence\n\
    checks.\n\
\n\
THE RELIABLE TEST is direct state comparison: read the modified\n\
file at {qsha}^ and compare to the upstream pre-image (the\n\
\"before\" state visible in the upstream commit's diff). The bug\n\
exists in this tree iff the buggy pattern is present in\n\
{qsha}^:<file>; the fix is appropriate iff the patch transforms\n\
that buggy state into the upstream post-image (or close to it,\n\
allowing for stable-specific adjustments).\n\
\n\
For commit-level presence checks, use them as CORROBORATING\n\
signals only, with awareness of their failure modes:\n\
  - On STABLE TREES (e.g. {tb} or {qsha}^):\n\
      `git_log <ref> --grep='cherry picked from commit <X-sha>'`\n\
    finds the backport entry for upstream X if X has been picked\n\
    onto this tree.\n\
  - On MAINLINE (origin/master): use `git_show <sha>` to confirm a\n\
    commit exists; do NOT use `--grep='<sha>'` to test whether it\n\
    has landed (that searches messages, not the commit itself).\n\n\
3.1 FIXES: TARGETS. For every Fixes: trailer captured in stage 1:\n\
    - `git_show <buggy_sha>` with `suppress_diff:true` to confirm it\n\
      exists in your local repo and to read its date / subject.\n\
    - PRIMARY TEST — read the affected file at {qsha}^ and check for\n\
      the buggy code pattern:\n\
        `git_show {qsha}^:<modified_file>`\n\
      Compare to the \"before\" hunks in the upstream commit's diff.\n\
      If the buggy pattern is present at {qsha}^, the bug exists and\n\
      the fix is meaningful. If the file at {qsha}^ already shows\n\
      the upstream POST-image, the bug has already been resolved\n\
      (perhaps via a different backport path) — flag this for\n\
      stage 7 (likely needs_review unless the patch is a no-op).\n\
    - CORROBORATING (not load-bearing) checks:\n\
        `git_log --oneline {tb} --grep='cherry picked from commit <buggy-sha>' -n 5`\n\
          (finds a backport of the buggy commit into the released tree)\n\
        `git_log --oneline {qsha}^ --grep='cherry picked from commit <buggy-sha>' -n 5`\n\
          (finds a backport queued ahead of this commit)\n\
      A positive hit confirms the bug commit is in the effective\n\
      tree. A negative hit DOES NOT prove absence (the file-read\n\
      test is what decides).\n\
3.2 CODE CONTEXT MATCHES IN THE TARGET TREE. For every file modified\n\
    in the diff, read it as it exists at {qsha}^ via\n\
    `git_show {qsha}^:<file_path>`. Compare the context lines around\n\
    each hunk to what's actually in {qsha}^. Even if the patch applies\n\
    cleanly, semantics may differ (reordered ops, refactored error\n\
    paths, double-unlock from different cleanup structure). Real\n\
    failure modes from past RC reviews:\n\
      * disable_irq() reordered relative to netif_napi_del()\n\
      * unlock-on-error path that double-unlocks because the target\n\
        tree's error path already unlocks\n\
      * put_device() called when device_initialize() prerequisite\n\
        is absent from this version, causing a crash\n\
3.3 NEW SYMBOLS USED BY THE PATCH MUST EXIST AT {qsha}^. For each new\n\
    function call / macro / struct field / type the patch introduces,\n\
    confirm it exists at {qsha}^ by READING the relevant file or\n\
    grepping the tree:\n\
      `git_show {qsha}^:<header>` (verify struct field is declared)\n\
      `search_file_content '<symbol>'` (verify it's defined somewhere)\n\
    Real examples of missing-dependency failures: `.remove_new`\n\
    callback on platform_driver (added in v6.3, absent in 5.10/5.15);\n\
    `struct devlink_fmsg.err` field absent in older kernels.\n\
3.4 OTHER QUEUED PATCHES IN THE SAME AREA. Find related queued or\n\
    recently released patches via `git_log {qsha}^ -- <file>` and\n\
    `lei_search`. If this commit depends on another commit that is\n\
    queued AFTER this one (i.e. its subject DOESN'T appear in any\n\
    `git_log {qsha}^ --grep` result but DOES appear in the queue\n\
    branch's overall log), that's an ordering bug — flag it.\n\
3.5 PREREQUISITE COMMITS IN MAINLINE. If 3.2 or 3.3 found code that\n\
    the patch needs but isn't present at {qsha}^, identify the\n\
    upstream commit that introduced it. Then check whether THAT\n\
    commit is queued anywhere in the {ver} queue (via lei_search\n\
    'Fixes:<prereq>' or by subject). If not, this is a MISSING\n\
    DEPENDENCY and a NO.\n\n\
Tools: git_log, git_show, git_show <ref>:<path>, search_file_content,\n\
lei_search."
    )
}

const STAGE_OUTPUT_CONTRACT: &str = "OUTPUT FORMAT: when you are done with this stage, respond with a SINGLE \
JSON object only, no markdown fences, no surrounding prose:\n\
{\n  \"concerns\": [\n    { \"kind\": \"<short snake_case category>\",\n      \"severity\": \"low|medium|high|critical\",\n      \"problem\": \"<one-line description>\",\n      \"evidence\": \"<tool output excerpt or git ref proving the concern>\" },\n    ...\n  ],\n  \"stage_summary\": \"<one paragraph: what you checked, what you found, what you couldn't verify>\"\n}\n\
If there are no concerns, emit `\"concerns\": []` and still write a stage_summary.";

fn synthesis_output_contract(input: &StageInput) -> String {
    let tb = &input.target_branch;
    let qb = &input.queue_branch;
    let ver = &input.target_version;
    format!(
        "OUTPUT FORMAT: respond with a SINGLE JSON object only, no markdown fences:\n\
{{\n  \"verdict\": \"yes\" | \"no\" | \"needs_review\",\n  \"confidence\": <float 0.0-1.0>,\n  \"summary\": \"<one paragraph rationale, naming specific evidence>\",\n  \"concerns\": [ {{ \"kind\": \"...\", \"severity\": \"...\", \"problem\": \"...\", \"evidence\": \"...\" }}, ... ]\n}}\n\n\
VERDICT POLICY (the question is correctness IN THIS TREE, not stable-process eligibility):\n\
- \"yes\": ALL of the following hold:\n\
  * The bug exists in the target tree (or this is a device-ID / quirk / build-only fix).\n\
  * All dependencies are present (already released in {tb} or queued ahead on {qb}).\n\
  * No critical follow-up fixes for the upstream commit are missing.\n\
  * The code being patched exists and surrounding context matches in {tb}.\n\
  * No inappropriate features or behavioral changes for a stable tree.\n\
  * Regression risk on this tree's callers is acceptable.\n\
- \"no\": there is a SPECIFIC defect. Be concrete — name the defect:\n\
  * Missing prerequisite commit X (give SHA + subject) absent from {tb} and {qb},\n\
    blocking apply or causing semantic mismatch.\n\
  * The buggy commit X (Fixes: target) was introduced AFTER the {ver} branch point,\n\
    so the bug doesn't exist here and the fix is unnecessary.\n\
  * Important follow-up fix Y (give SHA + subject) is in mainline but not queued.\n\
  * The upstream commit was reverted in mainline.\n\
  * File or function the patch touches doesn't exist in {tb}.\n\
  * Concrete regression risk on a named caller / file in this tree.\n\
  * Adds a new feature / API / config option inappropriate for stable.\n\
- \"needs_review\": load-bearing evidence couldn't be obtained (e.g. lei is unavailable,\n\
  a required ref isn't fetched, an ambiguous symbol couldn't be located). Use this\n\
  when a check failed silently rather than confirming presence.\n\n\
NOT grounds for \"no\" (these are upstream-process concerns, not our job):\n\
- \"This is a cleanup / not a user-visible bug fix\" — cleanups ship in stable.\n\
- \"No Cc: stable@vger.kernel.org\" — somebody already queued it; nomination path\n\
  is irrelevant here.\n\
- \"The author said this is mainline-only\" — selection has been made by a maintainer.\n\
- \"There's no Fixes: trailer\" — this is informational, not disqualifying.\n\n\
Be conservative — when in doubt about a SPECIFIC defect's existence, prefer\n\
needs_review over no. But do not return needs_review when the evidence is clear."
    )
}

const STAGE_1: &str = "STAGE 1 — Commit message and trailer inventory.\n\n\
Goal: extract the evidence later stages need, NOT to judge eligibility.\n\n\
1.1 SUBJECT: extract subsystem prefix (e.g. \"net: tcp:\", \"drm/i915:\",\n\
    \"mm/slub:\") and one-line summary of what the commit claims to do.\n\
1.2 TRAILERS: record every trailer with its value. Critically:\n\
    a) Fixes: <sha> — the upstream commit that introduced the bug. This is\n\
       LOAD-BEARING for stage 3 dependency verification.\n\
    b) Link: — pointers to lore discussion, syzbot reports, bug trackers.\n\
       Used in stages 1 and 4 if you need to recover upstream SHA or look\n\
       up follow-up discussion.\n\
    c) Reported-by:, Tested-by:, Reviewed-by:, Acked-by: — quality signals.\n\
    d) `(cherry picked from commit <sha>)` — if present, this is the\n\
       upstream SHA. If absent (common on commits queued as raw patches),\n\
       use `b4_dig -c` on Link: trailers or `lei_search 's:\"<subject>\"'`\n\
       to recover the upstream commit reference.\n\
1.3 BODY: identify the bug described, its symptom, root cause as the author\n\
    states it. Note any references to other commits or kernel versions.\n\
1.4 HIDDEN BUG FIXES: a commit framed as \"clean up\" or \"refactor\" can be a\n\
    real bug fix. Look for telltales: \"Handle X properly\", \"Initialize X\",\n\
    \"Balance refcount\", \"Improve locking\". Note the actual mechanism.\n\n\
DO NOT flag concerns based on missing Cc: stable, missing Fixes:, or\n\
\"this seems like a cleanup, not stable-eligible\". The selection has been\n\
made; eligibility is not your concern. Stage 1 concerns are limited to:\n\
- the commit message materially contradicts the diff (e.g. claims to fix X\n\
  but the diff doesn't touch X)\n\
- the upstream SHA cannot be recovered after exhausting the available tools\n\
  (this becomes a blocker for stage 3 deps and stage 4 follow-ups; flag it\n\
  here so the synthesis stage can mark needs_review).\n\n\
Tools: git_show, git_log, b4_dig, lei_search, lore_thread.";

const STAGE_2: &str = "STAGE 2 — Diff analysis (what does this commit do, in detail).\n\n\
Goal: understand the change well enough that stages 3, 5, and 6 can verify\n\
it against the target tree. NOT bug-class taxonomy — concrete mechanism.\n\n\
2.1 INVENTORY: list every file modified with +/- counts; identify each\n\
    function modified (read @@ hunk headers). Classify scope: single-file\n\
    surgical / multi-file / cross-subsystem.\n\
2.2 PER-HUNK CODE FLOW: for each hunk, what was the code doing BEFORE,\n\
    what does it do AFTER, and is this the normal path / error path /\n\
    init path? Note any new function calls, new struct field accesses,\n\
    new macros — these are the candidates stages 3 and 5 must verify\n\
    exist in the target tree.\n\
2.3 BUG MECHANISM: classify what the fix actually does (error-path leak,\n\
    race, refcount, NULL/bounds check, initialization, off-by-one, hardware\n\
    quirk). Stages 3-6 use this to know what to look for.\n\
2.4 NEW SYMBOLS: list every function call, macro, struct field, or type\n\
    introduced by the patch that the surrounding (unchanged) target-tree\n\
    code wouldn't already have. Stage 3 will check whether each exists.\n\n\
Stage 2 concerns should be specific defects in the patch ITSELF (a hunk\n\
that obviously doesn't compile, a function call to a symbol the diff also\n\
removes, an unrelated change snuck into a fix). Bug-class taxonomy is NOT\n\
a concern — it's information for downstream stages.\n\n\
Tools: git_show -p (queue diff), git_show on the upstream SHA for the\n\
upstream diff comparison.";

// (Stage 3 uses ref-substituted text via stage_3() below.)

const STAGE_4: &str = "STAGE 4 — Follow-up fix and revert search.\n\n\
For the upstream SHA, find every commit in `origin/master` that points\n\
back at it via `Fixes: <upstream_sha>` — `git_log --oneline\n\
--grep='Fixes: <prefix>' origin/master`. Each follow-up that fixes a\n\
real bug introduced by the upstream commit must ALSO be present in the\n\
target tree or be queued — otherwise this backport is shipping a known\n\
broken commit.\n\n\
Search for reverts: `git_log --oneline --grep='Revert.*<short-subject>'`\n\
on origin/master and on the queue branch. A revert upstream is usually\n\
fatal for the backport.\n\n\
Cross-check with lore: `lei_search 'Fixes:<upstream_sha>'` and\n\
`b4_dig -c <upstream_sha>` will find mailing-list discussion of\n\
regressions or follow-up postings that aren't yet committed. Pull the\n\
thread with `lore_thread <mid>` if a hit looks load-bearing.\n\n\
Tools: git_log --grep, lei_search, b4_dig, lore_thread.";

fn stage_5(input: &StageInput) -> String {
    let tb = &input.target_branch;
    let qb = &input.queue_branch;
    let qsha = &input.queue_sha;
    format!(
        "STAGE 5 — Version applicability (does the bug exist in THIS tree).\n\n\
Goal: confirm the conditions for this backport to be meaningful actually\n\
hold in the target tree. If the bug doesn't exist here, the patch is\n\
unnecessary and adds risk for no benefit.\n\n\
THE EFFECTIVE TREE FOR THIS COMMIT IS {qsha}^.\n\
That ref is the state of the queue branch JUST BEFORE this commit\n\
applies — it includes everything released in {tb} plus every commit\n\
queued AHEAD of this one. It does NOT include this commit itself, and\n\
it does NOT include later queue commits whose presence in {qb}\n\
post-dates the moment this commit lands. Use {qsha}^ for every\n\
existence check below — NEVER use {qb} as the existence ref:\n\
\n\
  * `git_show {qb}:<path>` would return success even for a NEW file\n\
    that THIS commit creates, because the queue tip includes this\n\
    commit. That's circular evidence and would falsely confirm\n\
    'file exists' when its existence depends on the very patch we\n\
    are validating.\n\
  * `git_show {qsha}^:<path>` is the patch's pre-image — exactly what\n\
    git apply will see when the cherry-pick runs.\n\n\
5.1 DOES THE BUG EXIST in the effective tree?\n\
    Use the SAME presence-check discipline as stage 3 (read the file\n\
    at {qsha}^; do NOT lean on `git log A..B` for ancestor tests; do\n\
    NOT treat `--grep='<sha>'` as a presence test on mainline since\n\
    that searches messages and only finds REFERENCES to the SHA, not\n\
    the commit itself).\n\
    - If a Fixes: trailer is present:\n\
        `git_show <buggy_sha>` with `suppress_diff:true` to read its\n\
          date / subject and confirm it resolves locally.\n\
      PRIMARY TEST: read the modified file at {qsha}^ and compare to\n\
      the upstream commit's BEFORE hunks:\n\
        `git_show {qsha}^:<modified_file>`\n\
      The bug exists in {qsha}^ iff the buggy pattern is present in\n\
      that file. If the file at {qsha}^ already shows the upstream\n\
      AFTER state (or a close variant), the bug has been resolved\n\
      somehow (different backport path, or independent fix) — likely\n\
      needs_review unless the patch is a clean no-op.\n\
      Corroborate via `git_log --oneline <ref> --grep='cherry picked\n\
      from commit <buggy-sha>' -n 5` against {tb} and {qsha}^ to spot\n\
      a backport of the buggy commit. A hit corroborates the bug's\n\
      presence; a miss is NOT proof of absence.\n\
    - If no Fixes:: identify the buggy pre-image purely from the\n\
      upstream commit's BEFORE hunks and check whether that pattern\n\
      shows up in `git_show {qsha}^:<file>`. Stage 3.2 already did\n\
      much of this; cross-reference its findings.\n\
5.2 FILES AND FUNCTIONS EXIST in the effective tree:\n\
    - For every file in the diff: `git_show {qsha}^:<path>`.\n\
      Exception: if the diff itself CREATES the file (look at the diff\n\
      header — `new file mode`), absence from {qsha}^ is expected and\n\
      not a defect. For every other file, absence from {qsha}^ is a\n\
      hard NO — the patch can't apply.\n\
    - For every modified function: confirm it exists with the expected\n\
      signature via `git_show {qsha}^:<path>` and `search_file_content`.\n\
5.3 DIVERGENCE ASSESSMENT:\n\
    - `git_diff {qsha}^ origin/master -- <path>` (use head/wc to gauge\n\
      magnitude without dumping huge diffs).\n\
    - Significant divergence around the modified hunks raises the chance\n\
      of silent semantic mismatch even when the patch applies textually.\n\
      Note the level (minimal / moderate / significant) and call out any\n\
      hunks where divergence overlaps the patch's change region.\n\n\
Stage 5 concerns are: bug verifiably doesn't exist in {qsha}^,\n\
non-created file/function missing from {qsha}^, or significant\n\
divergence around the patched region. Do NOT use {qb} (the queue tip)\n\
as the existence ref — it includes this commit and its successors.\n\n\
Tools: git_show <ref>:<file>, git_diff <ref> <ref> -- <file>,\n\
git_log <ref> --grep, search_file_content."
    )
}

fn stage_6(input: &StageInput) -> String {
    let tb = &input.target_branch;
    format!(
        "STAGE 6 — Feature creep and regression risk on THIS tree.\n\n\
Goal: even if the patch applies cleanly and the bug is present, does it\n\
introduce things stable shouldn't carry, or change behavior in a way that\n\
would break callers in {tb} specifically?\n\n\
6.1 NEW FEATURES introduced by this commit:\n\
    - New userspace APIs, syscalls, ioctls, sysfs/procfs entries, module\n\
      params, CONFIG_* options, new hardware support beyond device-IDs.\n\
    - These are stable-inappropriate and become a NO if found.\n\
6.2 BEHAVIORAL CHANGES beyond the bug fix proper:\n\
    - Changed return values in non-error cases.\n\
    - Modified semantics of existing interfaces.\n\
    - Changed default behavior visible to callers.\n\
6.3 REGRESSION RISK ON {tb} CALLERS:\n\
    - Caller fanout: `search_file_content '<changed-fn>(' {tb}`\n\
      to enumerate every caller in the target tree, then verify the\n\
      caller's expectations still match the patched callee. This is\n\
      where most subtle stable regressions hide.\n\
    - Locking divergence: does a new lock acquisition in the patch\n\
      interact with locks already held elsewhere in {tb} that were\n\
      refactored away in mainline?\n\
    - Cleanup-path symmetry: does an alloc/free pair the patch adds\n\
      survive {tb}'s error-path structure (which may differ from\n\
      mainline's)?\n\
    - Timing changes that could expose latent races present in {tb}\n\
      but not mainline.\n\
6.4 SCOPE PROPORTIONALITY: a 5-line fix for a crash is proportional; a\n\
    200-line refactor for a cosmetic issue is suspicious.\n\n\
Stage 6 concerns must NAME the specific caller / file / line / lock at\n\
risk. Generic \"this might cause issues\" is not actionable. NEW FEATURE\n\
INTRODUCTION is the one exception where the concern itself is the\n\
defect — name what feature was added.\n\n\
Tools: read_files, git_blame, search_file_content, git_log."
    )
}

fn stage_7(input: &StageInput) -> String {
    let tb = &input.target_branch;
    format!(
        "STAGE 7 — Synthesis: is the selection sound?\n\n\
You have findings from stages 1-6. The question is NOT whether this\n\
commit should have been selected for backport. The question is whether\n\
the selection — already made — is sound for {tb}.\n\n\
7.1 COMPILE EVIDENCE\n\
    - Evidence the selection IS sound (positive findings from 1-6).\n\
    - Evidence of SPECIFIC defects (missing prereq SHA, missing follow-up\n\
      SHA, missing file/function, named caller regression, feature creep,\n\
      bug not in target version, upstream revert).\n\
    - Unverified claims (lookups that failed, ambiguous evidence).\n\n\
7.2 DEDUPLICATE concerns from 1-6 — multiple stages may have reported the\n\
    same defect. Discard concerns that later-stage evidence refuted.\n\n\
7.3 EMIT VERDICT per the verdict policy. Be specific: a 'no' must name\n\
    the exact defect (which SHA is missing, which function doesn't exist,\n\
    which caller breaks). Vague 'might have problems' is a 'needs_review'."
    )
}

fn build_stage_user_prompt(stage: u8, input: &StageInput, prior: &[StageRecord]) -> String {
    let prior_summary = if prior.is_empty() {
        String::new()
    } else {
        let mut s = String::from("\n\n<prior_stage_findings>\n");
        for rec in prior {
            s.push_str(&format!(
                "Stage {}: {}\n",
                rec.stage,
                rec.stage_summary.trim()
            ));
            for c in &rec.concerns {
                s.push_str(&format!("  - [{}] {}: {}\n", c.severity, c.kind, c.problem));
            }
        }
        s.push_str("</prior_stage_findings>\n");
        s
    };

    let upstream_block = match (
        &input.upstream_sha,
        &input.upstream_body,
        &input.upstream_diff,
    ) {
        (Some(sha), Some(body), Some(diff)) => format!(
            "\n\n<upstream_commit sha=\"{sha}\">\n<message>\n{body}\n</message>\n<diff>\n{diff}\n</diff>\n</upstream_commit>",
            sha = sha,
            body = trim_block(body, 4000),
            diff = trim_block(diff, 12000),
        ),
        _ => String::from(
            "\n\n<upstream_commit>\n  (not available — recover via stage 1 if not yet known)\n</upstream_commit>",
        ),
    };

    format!(
        "Run stage {stage}.\n\n\
         <queue_commit sha=\"{queue_sha}\" branch=\"{queue_branch}\">\n\
         <subject>{subject}</subject>\n\
         <message>\n{body}\n</message>\n\
         <diff>\n{diff}\n</diff>\n\
         </queue_commit>{upstream_block}{prior_summary}",
        stage = stage,
        queue_sha = input.queue_sha,
        queue_branch = input.queue_branch,
        subject = input.queue_subject,
        body = trim_block(&input.queue_body, 4000),
        diff = trim_block(&input.queue_diff, 12000),
        upstream_block = upstream_block,
        prior_summary = prior_summary,
    )
}

fn build_synthesis_user_prompt(input: &StageInput, prior: &[StageRecord]) -> String {
    let mut s = format!(
        "Synthesize a verdict for the queued backport of {} on {} (target {}).\n\n\
         <queue_commit sha=\"{}\">\n\
         <subject>{}</subject>\n\
         </queue_commit>\n\n\
         <prior_stages>\n",
        input
            .upstream_sha
            .as_deref()
            .unwrap_or("(unknown upstream)"),
        input.queue_branch,
        input.target_version,
        input.queue_sha,
        input.queue_subject,
    );
    for rec in prior {
        s.push_str(&format!(
            "Stage {} — {}\n",
            rec.stage,
            rec.stage_summary.trim()
        ));
        for c in &rec.concerns {
            let evidence = c.evidence.as_deref().unwrap_or("");
            s.push_str(&format!(
                "  [{}] {}: {}{}\n",
                c.severity,
                c.kind,
                c.problem,
                if evidence.is_empty() {
                    String::new()
                } else {
                    format!(" — evidence: {}", trim_block(evidence, 400))
                }
            ));
        }
    }
    s.push_str("</prior_stages>");
    s
}

fn trim_block(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push_str("\n…[truncated]…");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_fences_handles_json_block() {
        let t = "```json\n{\"x\": 1}\n```";
        assert_eq!(strip_fences(t), "{\"x\": 1}");
    }

    #[test]
    fn strip_fences_handles_bare_block() {
        let t = "```\n{\"x\": 1}\n```";
        assert_eq!(strip_fences(t), "{\"x\": 1}");
    }

    #[test]
    fn strip_fences_passes_plain_through() {
        let t = "{\"x\": 1}";
        assert_eq!(strip_fences(t), "{\"x\": 1}");
    }

    #[test]
    fn parse_stage_payload_minimal() {
        let p = parse_stage_payload("{\"concerns\":[],\"stage_summary\":\"ok\"}").unwrap();
        assert_eq!(p.stage_summary, "ok");
        assert!(p.concerns.is_empty());
    }

    #[test]
    fn parse_stage_payload_with_concern() {
        let raw = r#"{"concerns":[{"kind":"missing_dep","severity":"high","problem":"x","evidence":"y"}],"stage_summary":"foo"}"#;
        let p = parse_stage_payload(raw).unwrap();
        assert_eq!(p.concerns.len(), 1);
        assert_eq!(p.concerns[0].kind, "missing_dep");
        assert_eq!(p.concerns[0].severity, "high");
    }

    #[test]
    fn parse_synthesis_normalises_verdict_case() {
        let raw = r#"{"verdict":"YES","summary":"s","concerns":[],"confidence":0.9}"#;
        let p = parse_synthesis_payload(raw).unwrap();
        assert_eq!(p.verdict, "yes");
        assert_eq!(p.confidence, Some(0.9));
    }

    #[test]
    fn parse_synthesis_coerces_unknown_verdict() {
        let raw = r#"{"verdict":"maybe","summary":"s","concerns":[]}"#;
        let p = parse_synthesis_payload(raw).unwrap();
        assert_eq!(p.verdict, "needs_review");
    }

    #[test]
    fn build_stage_user_prompt_truncates() {
        let big = "x".repeat(20_000);
        let input = StageInput {
            queue_sha: "a".repeat(40),
            queue_branch: "stable-rc/queue/6.12".into(),
            queue_subject: "subj".into(),
            queue_body: big.clone(),
            queue_diff: big.clone(),
            upstream_sha: None,
            upstream_body: None,
            upstream_diff: None,
            target_version: "6.12".into(),
            target_branch: "linux-6.12.y".into(),
        };
        let p = build_stage_user_prompt(1, &input, &[]);
        assert!(p.contains("[truncated]"));
        assert!(p.len() < big.len() * 2 + 4096);
    }

    #[test]
    fn shared_system_prompt_mentions_no_save() {
        let input = sample_input();
        let s = shared_system_prompt(&input);
        assert!(
            s.contains("--no-save"),
            "system prompt must mention --no-save"
        );
        assert!(s.contains("6.12"));
    }

    /// All stage prompts and the synthesis output contract must finish
    /// substitution before reaching the model. A leftover `{x}` would
    /// drive the model to copy the placeholder verbatim into tool calls
    /// and produce nonsense reviews — Codex caught this once already.
    #[test]
    fn stage_prompts_have_no_unsubstituted_placeholders() {
        let input = sample_input();
        for stage in 1u8..=7 {
            let sys = build_system_prompt(stage, &input);
            assert!(
                !contains_unsubstituted_template(&sys),
                "stage {stage} system prompt has an unsubstituted {{...}} placeholder:\n{sys}"
            );
        }
        let synth = build_synthesis_system_prompt(&input);
        assert!(
            !contains_unsubstituted_template(&synth),
            "synthesis system prompt has an unsubstituted {{...}} placeholder:\n{synth}"
        );
    }

    /// True if `s` contains a `{ident}` placeholder that looks like a
    /// Rust format spec rather than the prose use of literal braces (we
    /// don't write braces in stage prose otherwise).
    fn contains_unsubstituted_template(s: &str) -> bool {
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '{' {
                continue;
            }
            // collect run of letters/underscores
            let mut name = String::new();
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphanumeric() || n == '_' {
                    name.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if !name.is_empty() && chars.peek() == Some(&'}') {
                return true;
            }
        }
        false
    }

    fn sample_input() -> StageInput {
        StageInput {
            queue_sha: "a".repeat(40),
            queue_branch: "stable-rc/queue/6.12".into(),
            queue_subject: "subj".into(),
            queue_body: "body".into(),
            queue_diff: "diff".into(),
            upstream_sha: Some("b".repeat(40)),
            upstream_body: None,
            upstream_diff: None,
            target_version: "6.12".into(),
            target_branch: "linux-6.12.y".into(),
        }
    }
}
