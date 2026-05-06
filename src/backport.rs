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

//! Stable-tree backport review primitives.
//!
//! Helpers for walking a git range in the `stable-rc/queue/<ver>` branches
//! and matching each queued commit to its upstream SHA via the
//! `(cherry picked from commit <sha>)` trailer the stable maintainers add
//! when applying a backport.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::process::Command;

/// One commit observed on a queue branch, with its upstream SHA when the
/// trailer is present. Subject is the first line of the commit message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackportCandidate {
    pub queue_sha: String,
    pub upstream_sha: Option<String>,
    pub subject: String,
}

/// Parse a commit body for an upstream-SHA marker. Recognised forms (in
/// priority order — first match wins, except form 3 takes the LAST
/// occurrence when multiple are present):
///
/// 1. `commit <40-hex> upstream[.]` — Greg KH's marker on
///    `stable-rc/queue/<ver>` and many released stable commits. The
///    canonical "header line" placed right after the subject.
/// 2. `[ Upstream commit <40-hex> ]` (case-insensitive on "upstream",
///    spaces inside brackets optional) — Sasha Levin's AUTOSEL marker.
/// 3. `(cherry picked from commit <40-hex>)` (with optional trailing
///    `.`) — the canonical trailer at the end of the message. Multiple
///    occurrences possible when a commit was picked through several
///    trees; the LAST one is the most recent provenance.
///
/// Returns the 40-char lowercase hex SHA, or `None` when no marker is
/// found. Mixed-case SHAs are normalised to lowercase.
pub fn parse_upstream_sha(commit_body: &str) -> Option<String> {
    // First pass: form 1 ("commit <sha> upstream") and form 2
    // ("[Upstream commit <sha>]") are first-match-wins headers.
    for line in commit_body.lines() {
        let trimmed = line.trim();

        // Form 1: "commit <sha> upstream"
        if let Some(after) = trimmed.strip_prefix("commit ")
            && after.len() >= 40
        {
            let (sha, rest) = after.split_at(40);
            if sha.chars().all(|c| c.is_ascii_hexdigit()) {
                let tail = rest.trim_start();
                if let Some(after_upstream) = tail.strip_prefix("upstream")
                    && after_upstream
                        .chars()
                        .next()
                        .is_none_or(|c| c == '.' || c.is_whitespace())
                {
                    return Some(sha.to_ascii_lowercase());
                }
            }
        }

        // Form 2: "[ Upstream commit <sha> ]" — strip outer brackets,
        // then look for "upstream commit <sha>" (case-insensitive on
        // the keyword).
        if let Some(inside) = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
        {
            let inside = inside.trim();
            // Lowercase only the keyword prefix to test it; keep the
            // SHA region intact for hex parsing.
            let lower_lead: String = inside.chars().take(16).collect::<String>().to_ascii_lowercase();
            if let Some(prefix_len) = lower_lead.strip_prefix("upstream commit ").map(|_| 16)
                && inside.len() >= prefix_len + 40
            {
                let after = &inside[prefix_len..];
                let (sha, _rest) = after.split_at(40);
                if sha.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Some(sha.to_ascii_lowercase());
                }
            }
        }
    }

    // Second pass: form 3, cherry-pick trailer (last occurrence wins).
    let mut last: Option<String> = None;
    for line in commit_body.lines() {
        let trimmed = line.trim();
        let prefix = "(cherry picked from commit ";
        let Some(after) = trimmed.strip_prefix(prefix) else {
            continue;
        };
        if after.len() < 40 {
            continue;
        }
        let (sha, rest) = after.split_at(40);
        if !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        if !rest.starts_with(')') {
            continue;
        }
        last = Some(sha.to_ascii_lowercase());
    }
    last
}

/// Enumerate the commits in `range` (oldest-first), shelling out to
/// `git rev-list --reverse <range>` against `repo_path`.
///
/// For each commit, also fetch the full body to extract the upstream SHA via
/// [`parse_upstream_sha`]. Commits without an upstream trailer are still
/// returned (with `upstream_sha: None`) so the caller can decide whether to
/// review them under a reduced rule set.
pub async fn walk_range(repo_path: &Path, range: &str) -> Result<Vec<BackportCandidate>> {
    if range.trim().is_empty() {
        bail!("walk_range: empty range");
    }
    if range.starts_with('-') {
        // Defensive: rev-list takes flags too; reject anything that looks
        // like an option to keep this purely a range walker.
        bail!("walk_range: range argument must not start with '-' (got {range:?})");
    }

    let revs = run_git_capture(repo_path, &["rev-list", "--reverse", range]).await?;
    let queue_shas: Vec<String> = revs
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if queue_shas.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(queue_shas.len());
    for sha in queue_shas {
        let body = run_git_capture(repo_path, &["show", "-s", "--format=%B", &sha])
            .await
            .with_context(|| format!("git show -s --format=%B {sha}"))?;
        let subject = body
            .lines()
            .next()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let upstream_sha = parse_upstream_sha(&body);
        out.push(BackportCandidate {
            queue_sha: sha,
            upstream_sha,
            subject,
        });
    }
    Ok(out)
}

async fn run_git_capture(repo_path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .args(args)
        .output()
        .await
        .with_context(|| format!("failed to spawn git {:?}", args))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "git {:?} exited with {}: {}",
            args,
            output.status,
            stderr.trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    #[test]
    fn extracts_greg_queue_marker() {
        // The canonical first-body-line form on stable-rc/queue/* and
        // released stable trees alike.
        let body = "ipmi:ssif: NULL thread on error\n\n\
             commit a8aebe93a4938c0ca1941eeaae821738f869be3d upstream.\n\n\
             Cleanup code was checking the thread for NULL.\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("a8aebe93a4938c0ca1941eeaae821738f869be3d")
        );
    }

    #[test]
    fn extracts_greg_queue_marker_no_period() {
        let body = "subject\n\ncommit 1111111111111111111111111111111111111111 upstream\n\nbody\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("1111111111111111111111111111111111111111")
        );
    }

    #[test]
    fn extracts_autosel_bracket_marker() {
        // Sasha's AUTOSEL marker.
        let body = "mei: me: add nova lake point H DID\n\n\
             [ Upstream commit a5a1804332afc7035d5c5b880548262e81d796bc ]\n\n\
             Add Nova Lake H device id.\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("a5a1804332afc7035d5c5b880548262e81d796bc")
        );
    }

    #[test]
    fn extracts_autosel_bracket_marker_no_inner_spaces() {
        let body = "subj\n\n\
             [Upstream commit 2222222222222222222222222222222222222222]\n\nbody\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("2222222222222222222222222222222222222222")
        );
    }

    #[test]
    fn header_marker_takes_priority_over_cherry_pick_trailer() {
        let body = "subj\n\n\
             commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa upstream.\n\n\
             body\n\n\
             (cherry picked from commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    #[test]
    fn extracts_single_trailer() {
        let body = "fix something\n\nLong description.\n\n\
             Signed-off-by: Foo <foo@example.com>\n\
             (cherry picked from commit 0123456789abcdef0123456789abcdef01234567)\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
    }

    #[test]
    fn returns_last_when_multiple_trailers() {
        let body = "subject\n\n\
             (cherry picked from commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n\
             (cherry picked from commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        );
    }

    #[test]
    fn normalises_case() {
        let body = "(cherry picked from commit ABCDEF0123456789ABCDEF0123456789ABCDEF01)\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("abcdef0123456789abcdef0123456789abcdef01")
        );
    }

    #[test]
    fn ignores_short_or_malformed_sha() {
        assert_eq!(
            parse_upstream_sha("(cherry picked from commit deadbeef)\n"),
            None
        );
        assert_eq!(
            parse_upstream_sha(
                "(cherry picked from commit zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz)\n"
            ),
            None
        );
    }

    #[test]
    fn returns_none_for_no_trailer() {
        assert_eq!(parse_upstream_sha("subject\n\nbody\n"), None);
    }

    #[test]
    fn tolerates_indented_trailer() {
        let body =
            "subject\n\n    (cherry picked from commit 1111111111111111111111111111111111111111)\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("1111111111111111111111111111111111111111")
        );
    }

    #[test]
    fn tolerates_period_after_trailer() {
        // Some trees terminate trailers with a period.
        let body = "(cherry picked from commit 2222222222222222222222222222222222222222).\n";
        assert_eq!(
            parse_upstream_sha(body).as_deref(),
            Some("2222222222222222222222222222222222222222")
        );
    }

    #[tokio::test]
    async fn walk_range_against_synthetic_repo() {
        // Build a tiny git repo with two commits where the second carries a
        // (cherry picked from commit ...) trailer pointing at a fabricated
        // upstream SHA.
        let tmp = TempDir::new().unwrap();
        let repo = tmp.path();
        let run = |args: &[&str]| {
            let status = StdCommand::new("git")
                .args(["-C", repo.to_str().unwrap()])
                .args(args)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?} failed");
        };
        run(&["init", "-q", "-b", "main"]);
        std::fs::write(repo.join("a"), "a").unwrap();
        run(&["add", "a"]);
        run(&["commit", "-q", "-m", "first"]);
        std::fs::write(repo.join("b"), "b").unwrap();
        run(&["add", "b"]);
        let msg =
            "second\n\n(cherry picked from commit 3333333333333333333333333333333333333333)\n";
        run(&["commit", "-q", "-m", msg]);

        let candidates = walk_range(repo, "main~1..main").await.unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].subject, "second");
        assert_eq!(
            candidates[0].upstream_sha.as_deref(),
            Some("3333333333333333333333333333333333333333")
        );
    }

    #[tokio::test]
    async fn walk_range_rejects_flags() {
        let tmp = TempDir::new().unwrap();
        let err = walk_range(tmp.path(), "--all").await.unwrap_err();
        assert!(format!("{err}").contains("must not start with '-'"));
    }
}
