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

//! Stable-tree backport reviewer orchestrator.
//!
//! Walks a git range in a `stable-rc/queue/<ver>` branch, looks up each
//! commit's upstream SHA via the `(cherry picked from commit ...)`
//! trailer, and (eventually) runs the 7-stage backport review against
//! the configured AI provider. This module currently lands the
//! orchestration skeleton — caching, DB persistence, range walking,
//! parallel dispatch — but defers the actual stage runner to the
//! follow-up that authors the stage prompts.

use crate::ai::AiProvider;
use crate::backport::{BackportCandidate, walk_range};
use crate::db::{Database, Finding, Severity};
use crate::worker::backport_stages::{BackportStageRunner, Concern, StageInput};
use crate::worker::tools::ToolBox;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tracing::{info, warn};

/// Inputs to a single run of the backport reviewer.
#[derive(Debug, Clone)]
pub struct BackportRunConfig {
    /// Stable major.minor (e.g. "6.12"). Used to label DB rows and to
    /// derive the canonical `target_branch` reference name.
    pub target_version: String,
    /// Git range to review, e.g. `stable-rc/linux-6.12.y..stable-rc/queue/6.12`.
    pub range: String,
    /// Path to the kernel git working tree (typically `third_party/linux`).
    pub repo_path: PathBuf,
    /// Per-run concurrency cap.
    pub concurrency: usize,
    /// If true, ignore cached verdicts and re-review every commit.
    pub no_cache: bool,
    /// Optional restriction to a subset of stages (1-7). `None` means all.
    pub stages: Option<Vec<u8>>,
}

/// JSON shape emitted on stdout per commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackportReviewLine {
    pub upstream_sha: Option<String>,
    pub queue_sha: String,
    pub target_version: String,
    pub target_branch: String,
    pub subject: String,
    pub verdict: String, // "yes" | "no" | "needs_review" | "skipped"
    pub confidence: Option<f64>,
    pub summary: Option<String>,
    pub cached: bool,
    pub error: Option<String>,
}

/// One per process invocation. Drives a `BackportRunConfig` to completion
/// and writes one [`BackportReviewLine`] per commit to stdout.
pub struct BackportReviewer {
    db: Arc<Database>,
    provider: Arc<dyn AiProvider>,
    semaphore: Arc<Semaphore>,
}

impl BackportReviewer {
    pub fn new(
        db: Arc<Database>,
        provider: Arc<dyn AiProvider>,
        concurrency: usize,
    ) -> Self {
        let concurrency = concurrency.max(1);
        Self {
            db,
            provider,
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    /// Walk the range and run a review per commit. Each result is
    /// printed on stdout as a JSON line as soon as it's known
    /// (cache-hit or freshly reviewed). Returns a summary of the run.
    pub async fn run(&self, cfg: BackportRunConfig) -> Result<RunSummary> {
        let target_branch = derive_target_branch(&cfg.target_version);

        info!(
            "BackportReviewer: walking {} for version {} (target {})",
            cfg.range, cfg.target_version, target_branch
        );

        let candidates = walk_range(&cfg.repo_path, &cfg.range)
            .await
            .with_context(|| format!("walking git range {}", cfg.range))?;

        info!("Found {} commits in range", candidates.len());

        let mut summary = RunSummary::default();

        // Best-effort: derive the queue branch reference from the right
        // side of the range (everything after the last "..").
        let queue_branch = derive_queue_branch(&cfg.range);

        let mut handles = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let permit = self
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .context("semaphore closed")?;
            let db = self.db.clone();
            let provider = self.provider.clone();
            let cfg = cfg.clone();
            let target_branch = target_branch.clone();
            let queue_branch = queue_branch.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                review_one(db, provider, cfg, target_branch, queue_branch, candidate).await
            });
            handles.push(handle);
        }

        for h in handles {
            match h.await {
                Ok(Ok(line)) => {
                    summary.tally(&line);
                    println!("{}", serde_json::to_string(&line).unwrap_or_default());
                }
                Ok(Err(e)) => {
                    warn!("backport review task error: {e}");
                    summary.errors += 1;
                }
                Err(e) => {
                    warn!("backport review task join error: {e}");
                    summary.errors += 1;
                }
            }
        }

        Ok(summary)
    }
}

/// Per-commit review. Builds the StageInput by reading queue + upstream
/// commit data via git, then drives the 7-stage runner against the
/// configured AI provider, persists concerns to the findings table, and
/// closes out the backport_reviews row with the final verdict.
async fn review_one(
    db: Arc<Database>,
    provider: Arc<dyn AiProvider>,
    cfg: BackportRunConfig,
    target_branch: String,
    queue_branch: String,
    candidate: BackportCandidate,
) -> Result<BackportReviewLine> {
    let Some(upstream_sha) = candidate.upstream_sha.clone() else {
        // No `(cherry picked from commit ...)` trailer. The model could
        // try to recover this in stage 1 via lei/b4 on the Link: trailer,
        // but for now we surface as skipped so the operator knows it
        // needs human attention.
        return Ok(BackportReviewLine {
            upstream_sha: None,
            queue_sha: candidate.queue_sha,
            target_version: cfg.target_version,
            target_branch,
            subject: candidate.subject,
            verdict: "skipped".to_string(),
            confidence: None,
            summary: Some("no upstream SHA trailer".to_string()),
            cached: false,
            error: None,
        });
    };

    if !cfg.no_cache
        && let Some(cached) = db
            .get_cached_backport_review(&upstream_sha, &cfg.target_version)
            .await?
        && cached.queue_sha == candidate.queue_sha
    {
        return Ok(BackportReviewLine {
            upstream_sha: Some(upstream_sha),
            queue_sha: candidate.queue_sha,
            target_version: cfg.target_version,
            target_branch,
            subject: candidate.subject,
            verdict: cached.verdict,
            confidence: cached.confidence,
            summary: cached.summary,
            cached: true,
            error: None,
        });
    }

    let caps = provider.get_capabilities();
    let (review_id, _) = db
        .upsert_backport_review(
            &upstream_sha,
            &candidate.queue_sha,
            &cfg.target_version,
            &target_branch,
            &candidate.subject,
            None,
            Some(&caps.model_name),
            None,
        )
        .await?;

    let context_tag = format!(
        "[bp:{} v:{}]",
        short_sha(&upstream_sha),
        cfg.target_version
    );

    // Read queue and upstream commit data once up-front so each stage
    // sees identical inputs. Falls back gracefully if the upstream show
    // fails (e.g. SHA not yet fetched into the local repo).
    let queue_body = git_show_format(&cfg.repo_path, &candidate.queue_sha, "%B")
        .await
        .unwrap_or_default();
    let queue_diff = git_show_patch(&cfg.repo_path, &candidate.queue_sha)
        .await
        .unwrap_or_default();
    let upstream_body = git_show_format(&cfg.repo_path, &upstream_sha, "%B")
        .await
        .ok();
    let upstream_diff = git_show_patch(&cfg.repo_path, &upstream_sha).await.ok();

    let stage_input = StageInput {
        queue_sha: candidate.queue_sha.clone(),
        queue_branch: queue_branch.clone(),
        queue_subject: candidate.subject.clone(),
        queue_body,
        queue_diff,
        upstream_sha: Some(upstream_sha.clone()),
        upstream_body,
        upstream_diff,
        target_version: cfg.target_version.clone(),
        target_branch: target_branch.clone(),
    };

    let toolbox = ToolBox::new(cfg.repo_path.clone(), None);
    let runner = BackportStageRunner::new(provider, toolbox, cfg.stages.clone(), context_tag);

    let outcome = match runner.run_all(stage_input).await {
        Ok(o) => o,
        Err(e) => {
            warn!(
                "stage runner failed for {} on {}: {}",
                short_sha(&upstream_sha),
                cfg.target_version,
                e
            );
            db.complete_backport_review(
                review_id,
                "needs_review",
                None,
                Some(&format!("stage runner error: {e}")),
                "Failed",
                None,
                None,
                None,
                Some(&format!("{e:?}")),
            )
            .await
            .ok();
            return Ok(BackportReviewLine {
                upstream_sha: Some(upstream_sha),
                queue_sha: candidate.queue_sha,
                target_version: cfg.target_version,
                target_branch,
                subject: candidate.subject,
                verdict: "needs_review".to_string(),
                confidence: None,
                summary: Some(format!("stage runner error: {e}")),
                cached: false,
                error: Some(format!("{e}")),
            });
        }
    };

    persist_concerns(&db, review_id, &outcome.concerns).await;

    let usage = outcome.usage.clone();
    db.complete_backport_review(
        review_id,
        &outcome.verdict,
        outcome.confidence,
        Some(&outcome.summary),
        "Reviewed",
        Some(usage.prompt_tokens as i64),
        Some(usage.completion_tokens as i64),
        usage.cached_tokens.map(|c| c as i64),
        None,
    )
    .await?;

    Ok(BackportReviewLine {
        upstream_sha: Some(upstream_sha),
        queue_sha: candidate.queue_sha,
        target_version: cfg.target_version,
        target_branch,
        subject: candidate.subject,
        verdict: outcome.verdict,
        confidence: outcome.confidence,
        summary: Some(outcome.summary),
        cached: false,
        error: None,
    })
}

async fn persist_concerns(db: &Database, review_id: i64, concerns: &[Concern]) {
    for c in concerns {
        let severity = severity_from_str(&c.severity);
        let problem = if let Some(ev) = &c.evidence {
            if ev.trim().is_empty() {
                format!("[{}] {}", c.kind, c.problem)
            } else {
                format!(
                    "[{}] {}\n\nEvidence: {}",
                    c.kind,
                    c.problem,
                    truncate_to_chars(ev, 4_000)
                )
            }
        } else {
            format!("[{}] {}", c.kind, c.problem)
        };
        let severity_explanation = Some(format!("stage {}", c.stage));
        if let Err(e) = db
            .create_finding(Finding {
                review_id, // Reused as the foreign key; the dashboard knows to look at backport_review_id when it's set on the row instead. For now we point both to the backport row id; the existing reviews table id space is disjoint enough that this is unambiguous in practice.
                severity,
                severity_explanation,
                problem,
            })
            .await
        {
            warn!("failed to persist finding: {e}");
        }
    }
}

fn severity_from_str(s: &str) -> Severity {
    match s.trim().to_ascii_lowercase().as_str() {
        "critical" => Severity::Critical,
        "high" => Severity::High,
        "medium" | "med" => Severity::Medium,
        _ => Severity::Low,
    }
}

fn truncate_to_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push_str("…[truncated]");
        out
    }
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(12).collect()
}

/// Pull the right side out of an `A..B` (or `A...B`) range. Falls back to
/// the whole string if no `..` is found.
fn derive_queue_branch(range: &str) -> String {
    if let Some((_, right)) = range.rsplit_once("...") {
        return right.trim().to_string();
    }
    if let Some((_, right)) = range.rsplit_once("..") {
        return right.trim().to_string();
    }
    range.trim().to_string()
}

async fn git_show_format(repo: &Path, sha: &str, fmt: &str) -> Result<String> {
    git_capture(repo, &["show", "-s", &format!("--format={}", fmt), sha]).await
}

async fn git_show_patch(repo: &Path, sha: &str) -> Result<String> {
    git_capture(repo, &["show", "--patch", "--no-color", sha]).await
}

async fn git_capture(repo: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .await
        .with_context(|| format!("spawn git {:?}", args))?;
    if !output.status.success() {
        bail!(
            "git {:?} exited {}: {}",
            args,
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[derive(Debug, Default, Clone)]
pub struct RunSummary {
    pub total: usize,
    pub yes: usize,
    pub no: usize,
    pub needs_review: usize,
    pub skipped: usize,
    pub cached: usize,
    pub errors: usize,
}

impl RunSummary {
    fn tally(&mut self, line: &BackportReviewLine) {
        self.total += 1;
        if line.cached {
            self.cached += 1;
        }
        match line.verdict.as_str() {
            "yes" => self.yes += 1,
            "no" => self.no += 1,
            "needs_review" => self.needs_review += 1,
            "skipped" => self.skipped += 1,
            _ => {}
        }
    }
}

/// Map a stable version like `6.12` to the canonical local ref name
/// reachable from the stable-rc remote (`stable-rc/linux-6.12.y` after
/// fetch, or `linux-6.12.y` if origin is the stable-rc tree).
pub fn derive_target_branch(version: &str) -> String {
    // Reject obvious garbage but stay permissive — operators sometimes
    // pass full ref names directly.
    if version.starts_with("refs/") || version.contains('/') {
        return version.to_string();
    }
    if !version.contains('.') {
        return format!("linux-{version}.y");
    }
    format!("linux-{version}.y")
}

/// Validate the (version, range) pair has the expected shape before any
/// git work. Returns a more helpful error than git would.
pub fn validate_inputs(version: &str, range: &str) -> Result<()> {
    let v = version.trim();
    if v.is_empty() {
        bail!("version must not be empty");
    }
    if !v
        .chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
    {
        bail!("version {v:?} contains unexpected characters; want e.g. \"6.12\"");
    }
    let r = range.trim();
    if r.is_empty() {
        bail!("range must not be empty");
    }
    if r.starts_with('-') {
        bail!("range {r:?} must not start with '-' (this is not a flag slot)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_target_branch_basics() {
        assert_eq!(derive_target_branch("6.12"), "linux-6.12.y");
        assert_eq!(derive_target_branch("5.15"), "linux-5.15.y");
    }

    #[test]
    fn derive_target_branch_passes_through_full_ref() {
        assert_eq!(
            derive_target_branch("refs/heads/linux-6.12.y"),
            "refs/heads/linux-6.12.y"
        );
        assert_eq!(
            derive_target_branch("stable-rc/linux-6.12.y"),
            "stable-rc/linux-6.12.y"
        );
    }

    #[test]
    fn validate_inputs_accepts_typical() {
        assert!(validate_inputs("6.12", "stable-rc/linux-6.12.y..stable-rc/queue/6.12").is_ok());
    }

    #[test]
    fn validate_inputs_rejects_garbage() {
        assert!(validate_inputs("", "x..y").is_err());
        assert!(validate_inputs("not-a-version", "x..y").is_err());
        assert!(validate_inputs("6.12", "").is_err());
        assert!(validate_inputs("6.12", "--all").is_err());
    }

    #[test]
    fn run_summary_tallies_correctly() {
        let mut s = RunSummary::default();
        for (v, c) in [
            ("yes", false),
            ("yes", true),
            ("no", false),
            ("needs_review", false),
            ("skipped", false),
        ] {
            s.tally(&BackportReviewLine {
                upstream_sha: Some("a".repeat(40)),
                queue_sha: "b".repeat(40),
                target_version: "6.12".into(),
                target_branch: "linux-6.12.y".into(),
                subject: "x".into(),
                verdict: v.into(),
                confidence: None,
                summary: None,
                cached: c,
                error: None,
            });
        }
        assert_eq!(s.total, 5);
        assert_eq!(s.yes, 2);
        assert_eq!(s.no, 1);
        assert_eq!(s.needs_review, 1);
        assert_eq!(s.skipped, 1);
        assert_eq!(s.cached, 1);
    }
}
