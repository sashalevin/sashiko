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

use crate::backport::{BackportCandidate, walk_range};
use crate::db::Database;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
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
    semaphore: Arc<Semaphore>,
}

impl BackportReviewer {
    pub fn new(db: Arc<Database>, concurrency: usize) -> Self {
        let concurrency = concurrency.max(1);
        Self {
            db,
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

        let mut handles = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let permit = self
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .context("semaphore closed")?;
            let db = self.db.clone();
            let cfg = cfg.clone();
            let target_branch = target_branch.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                review_one(db, cfg, target_branch, candidate).await
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

/// Per-commit review. Currently emits a stub `needs_review` verdict so the
/// pipeline is end-to-end runnable; the LLM-driven stage runner will
/// replace the body of this function once the prompts land.
async fn review_one(
    db: Arc<Database>,
    cfg: BackportRunConfig,
    target_branch: String,
    candidate: BackportCandidate,
) -> Result<BackportReviewLine> {
    let Some(upstream_sha) = candidate.upstream_sha.clone() else {
        // No `(cherry picked from commit ...)` trailer. We can't yet
        // do version-applicability reasoning meaningfully; surface as
        // skipped so the operator knows it needs human attention.
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

    let (review_id, _) = db
        .upsert_backport_review(
            &upstream_sha,
            &candidate.queue_sha,
            &cfg.target_version,
            &target_branch,
            &candidate.subject,
            None,
            None,
            None,
        )
        .await?;

    // TODO(stage-runner): replace with the 7-stage LLM-driven review.
    // For now persist a placeholder so the row is always closed and the
    // CLI is end-to-end runnable.
    let verdict = "needs_review";
    let summary = Some("stage runner not yet wired — placeholder verdict".to_string());
    db.complete_backport_review(
        review_id,
        verdict,
        None,
        summary.as_deref(),
        "Reviewed",
        None,
        None,
        None,
        None,
    )
    .await?;

    Ok(BackportReviewLine {
        upstream_sha: Some(upstream_sha),
        queue_sha: candidate.queue_sha,
        target_version: cfg.target_version,
        target_branch,
        subject: candidate.subject,
        verdict: verdict.to_string(),
        confidence: None,
        summary,
        cached: false,
        error: None,
    })
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
