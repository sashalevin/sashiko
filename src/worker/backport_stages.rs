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
    let stage_intro = stage_instruction(stage);
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
        stage_intro = stage_instruction(7),
        output_contract = SYNTHESIS_OUTPUT_CONTRACT,
    )
}

fn shared_system_prompt(input: &StageInput) -> String {
    let date = chrono::Utc::now().format("%A, %B %d, %Y");
    format!(
        "You are a Linux kernel stable-tree maintainer reviewing a queued backport.\n\
         Today's date: {date}.\n\n\
         Your job: decide whether the queued commit on `{queue_branch}` is a correct,\n\
         safe, and shippable backport for the `{target_branch}` release. The original\n\
         commit lives in mainline (`origin/master`) and was applied to this stable\n\
         branch's queue. Your evaluation must be evidence-driven: every claim you\n\
         make MUST be backed by a tool-derived observation (git_show, git_log,\n\
         git_grep via search_file_content, lei_search, b4_dig, lore_thread, read_files).\n\
         If you cannot verify a claim, raise a concern with low confidence rather\n\
         than asserting it.\n\n\
         TARGET VERSION: {version}\n\
         TARGET BRANCH:  {target_branch}\n\
         QUEUE BRANCH:   {queue_branch}\n\
         QUEUE SHA:      {queue_sha}\n\
         UPSTREAM SHA:   {upstream}\n\n\
         GUIDANCE: batch independent tool calls in a single response (e.g. multiple\n\
         git_log queries, or several lei_search queries) so this conversation does\n\
         not blow turns on serial round-trips. lei_search ALWAYS runs with --no-save,\n\
         so feel free to issue speculative queries.",
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

fn stage_instruction(stage: u8) -> &'static str {
    match stage {
        1 => STAGE_1,
        2 => STAGE_2,
        3 => STAGE_3,
        4 => STAGE_4,
        5 => STAGE_5,
        6 => STAGE_6,
        7 => STAGE_7,
        _ => "",
    }
}

const STAGE_OUTPUT_CONTRACT: &str = "OUTPUT FORMAT: when you are done with this stage, respond with a SINGLE \
JSON object only, no markdown fences, no surrounding prose:\n\
{\n  \"concerns\": [\n    { \"kind\": \"<short snake_case category>\",\n      \"severity\": \"low|medium|high|critical\",\n      \"problem\": \"<one-line description>\",\n      \"evidence\": \"<tool output excerpt or git ref proving the concern>\" },\n    ...\n  ],\n  \"stage_summary\": \"<one paragraph: what you checked, what you found, what you couldn't verify>\"\n}\n\
If there are no concerns, emit `\"concerns\": []` and still write a stage_summary.";

const SYNTHESIS_OUTPUT_CONTRACT: &str = "OUTPUT FORMAT: respond with a SINGLE JSON object only, no markdown fences:\n\
{\n  \"verdict\": \"yes\" | \"no\" | \"needs_review\",\n  \"confidence\": <float 0.0-1.0>,\n  \"summary\": \"<one paragraph rationale>\",\n  \"concerns\": [ { \"kind\": \"...\", \"severity\": \"...\", \"problem\": \"...\", \"evidence\": \"...\" }, ... ]\n}\n\n\
VERDICT POLICY (be conservative — when in doubt, prefer needs_review):\n\
- \"yes\": evidence is strong that the backport is correct, applicable, and ships safely.\n\
  All Fixes: prerequisites are satisfied (in target tree or queued). No follow-up\n\
  fixes for this commit exist in mainline that aren't also queued. No reverts.\n\
  The bug exists in the target version and the fix matches the upstream code path\n\
  closely enough that no semantic divergence is suspected.\n\
- \"no\": there is concrete evidence the backport is wrong, dangerous, or not applicable.\n\
  Examples: the upstream commit was reverted; an indispensable Fixes: prerequisite\n\
  is missing from the target tree; the bug doesn't exist in this version; the\n\
  cherry-pick has known follow-up fixes that aren't queued.\n\
- \"needs_review\": the evidence is mixed, missing, or uncertain. Use this whenever\n\
  you couldn't reliably establish applicability or follow-up coverage.";

const STAGE_1: &str = "STAGE 1 — Commit message + trailers.\n\n\
Read the queue commit's message. Extract every trailer: Fixes:, Reported-by:,\n\
Tested-by:, Reviewed-by:, Acked-by:, Link:, Cc: stable@vger.kernel.org,\n\
Signed-off-by:. If the canonical `(cherry picked from commit <sha>)` trailer\n\
is present, note the upstream SHA. If it is ABSENT (common on raw queue\n\
patches that haven't been picked yet), use the Link: trailer's message-id\n\
with `b4_dig` and/or `lei_search` to recover the upstream commit reference.\n\
This stage's concerns should flag missing/inconsistent trailers, missing\n\
Cc: stable when the patch is fix-shaped, unverifiable Link:, or claims in\n\
the commit message that contradict the diff (do a quick sanity read).\n\n\
Tools you'll use most: git_show, git_log, b4_dig, lei_search.";

const STAGE_2: &str = "STAGE 2 — Diff inventory and bug classification.\n\n\
Read the queue diff. Inventory the files and functions modified. Classify\n\
the change into one or more bug classes — race, leak, UAF, init-order,\n\
refcount, locking, lock-ordering, hardware/dma, build, sparse, security,\n\
revert, cleanup, feature/non-fix. Note the scope: surgical hunk vs. broad\n\
refactor. Concerns this stage produces: scope mismatch (e.g. a 'fix' that\n\
also adds new APIs), bug class incompatible with stable-tree rules\n\
(stable doesn't want feature work), or files modified that don't exist\n\
in the target version.\n\n\
Tools: git_show -p (queue diff), git_show on the upstream SHA for the\n\
upstream diff, list_dir / find_files for target-tree existence checks.";

const STAGE_3: &str = "STAGE 3 — Dependency verification.\n\n\
This is the most important load-bearing check. For every Fixes: trailer\n\
target identified in stage 1, verify whether that target commit is\n\
reachable from the target stable branch. Use `git_log --oneline\n\
linux-<ver>.y --grep=<short-subject>` and `git_show` to confirm.\n\n\
Beyond explicit Fixes:, identify implicit prerequisites: helper\n\
functions, struct fields, macros, callbacks, or refactored APIs the\n\
upstream commit relies on. For each, verify the prerequisite is present\n\
in the target tree (use `search_file_content` and `git_show <ref>:<path>`).\n\
If a prerequisite is missing in the target tree but present elsewhere in\n\
the queue, that's still a soft pass. If it's missing both places, that's\n\
a hard concern.\n\n\
Use `lei_search` to check whether prerequisite commits have been queued\n\
separately (e.g. \"<prereq-subject>\" or \"Fixes:<prereq-sha>\" queries).\n\n\
Tools: git_log, git_show, git_show <ref>:<path>, search_file_content, lei_search.";

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

const STAGE_5: &str = "STAGE 5 — Version applicability.\n\n\
Confirm the bug being fixed actually exists in the target stable\n\
branch. Read the upstream commit's diff to identify the buggy\n\
pre-image — then `git_show stable-rc/linux-<ver>.y:<path>` to read\n\
the target file at the same line range and verify the bug is present.\n\
If the buggy code doesn't exist in the target version (e.g. the function\n\
was rewritten between target and mainline), the backport is not\n\
applicable.\n\n\
Assess code divergence: `git_diff stable-rc/linux-<ver>.y origin/master\n\
-- <path>` to see how much the file has drifted. Heavy divergence around\n\
the modified hunks raises the chance of a silent semantic mismatch when\n\
the queue patch is applied — even when it applies textually.\n\n\
Tools: git_show <ref>:<file>, git_diff <ref> <ref> -- <file>, git_log <ref> -- <file>.";

const STAGE_6: &str = "STAGE 6 — Regression risk on the target branch.\n\n\
With everything above in mind, assess what could break if this backport\n\
ships on the target stable branch. Look for:\n\n\
- Locking / concurrency divergence: does the lock acquired in the\n\
  upstream patch exist in the target tree? Same lock type? Same order?\n\
- Resource-management regressions: the patch adds an alloc/free pair\n\
  that pairs differently in the target tree (e.g. cleanup paths refactored).\n\
- Security regressions: the patch fixes a security issue but the bounds\n\
  check it adds is wrong for the older API.\n\
- Caller fanout: search the target tree for callers of any function\n\
  signature/contract changed by this patch (`search_file_content` for\n\
  the function name) and verify all callers are consistent.\n\n\
Concerns from this stage are 'this might break <X> on <ver>' style.\n\
Cite which caller / which file / which line.\n\n\
Tools: read_files, git_blame, search_file_content, git_log.";

const STAGE_7: &str = "STAGE 7 — Synthesis and verdict.\n\n\
You are given the union of concerns from stages 1-6. Deduplicate\n\
overlapping concerns. Discard any concern that turned out to be a false\n\
positive based on later-stage evidence. Then emit a single verdict:\n\n\
- yes: ship this backport as-is.\n\
- no: do NOT ship; provide the load-bearing reason in summary.\n\
- needs_review: human attention required (mixed evidence, lookups\n\
  failed, or the model is uncertain).\n\n\
Be conservative. The cost of a wrong 'yes' (shipping a broken backport\n\
to millions of users) is much higher than the cost of a wrong\n\
'needs_review' (a maintainer spends 10 minutes double-checking).";

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
        let input = StageInput {
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
        };
        let s = shared_system_prompt(&input);
        assert!(
            s.contains("--no-save"),
            "system prompt must mention --no-save"
        );
        assert!(s.contains("6.12"));
    }
}
