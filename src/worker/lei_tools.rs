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

//! Tools for searching lore via `lei` and looking up commit discussions
//! via `b4`. Exposed to the LLM through `ToolBox`. All tools degrade
//! gracefully when the corresponding binary is missing — the model gets a
//! structured `{ok: false, error: "..."}` payload instead of a hard error,
//! so a review can continue without lei/b4 installed.

use anyhow::Result;
use serde_json::{Value, json};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

const DEFAULT_LEI_BIN: &str = "lei";
const DEFAULT_B4_BIN: &str = "b4";
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 30;
/// Cap stdout so a runaway tool can't blow out a stage budget.
const MAX_STDOUT_BYTES: usize = 256 * 1024;
/// Cap on returned hits for `lei_search` to keep prompt size sane.
const MAX_LEI_HITS_DEFAULT: usize = 20;

/// Search lore (or any lei-indexed inbox) without modifying the user's
/// saved-search index. Always invokes `lei q --no-save`.
///
/// `query` follows the lei query syntax — e.g.
/// `"f:torvalds@linux-foundation.org dfn:1.week.ago"`,
/// `"\"Fixes: deadbeef0000\""`, or `"mid:<msgid>"`.
///
/// On success returns:
/// ```jsonc
/// { "ok": true, "hits": [{"mid":"…","subject":"…","from":"…","date":"…"}, …],
///   "truncated": false }
/// ```
/// On lei-binary missing or error returns
/// `{ "ok": false, "error": "…" }` so the caller can keep going.
pub async fn lei_search(query: &str, limit: usize, lei_bin: &str) -> Value {
    let limit = if limit == 0 {
        MAX_LEI_HITS_DEFAULT
    } else {
        limit.min(200)
    };
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return json!({ "ok": false, "error": "empty query" });
    }

    let args = vec![
        "q".to_string(),
        "--no-save".to_string(),
        "-f".to_string(),
        "json".to_string(),
        "-o".to_string(),
        "-".to_string(),
        format!("--limit={}", limit),
        trimmed.to_string(),
    ];

    let raw = match run_capture(lei_bin, &args, DEFAULT_TOOL_TIMEOUT_SECS).await {
        Ok(out) => out,
        Err(e) => return json!({ "ok": false, "error": format!("{e}") }),
    };

    parse_lei_json(&raw)
}

/// Look up the lore discussion thread for a commit (`b4 dig -c <sha>`) or
/// a message-id (`b4 dig <mid>`). Returns the raw stdout in
/// `{"ok": true, "output": "…"}` so the model can read whatever b4
/// surfaces (links, related threads, follow-ups).
pub async fn b4_dig(input: &str, b4_bin: &str) -> Value {
    let input = input.trim();
    if input.is_empty() {
        return json!({ "ok": false, "error": "empty input" });
    }

    // b4 distinguishes commit hashes from message-ids implicitly via -c.
    // Use -c when the input looks like a hex SHA (≥7 chars, all hex).
    let looks_like_sha =
        input.len() >= 7 && input.len() <= 64 && input.chars().all(|c| c.is_ascii_hexdigit());
    let args: Vec<String> = if looks_like_sha {
        vec!["dig".to_string(), "-c".to_string(), input.to_string()]
    } else {
        vec!["dig".to_string(), input.to_string()]
    };

    match run_capture(b4_bin, &args, DEFAULT_TOOL_TIMEOUT_SECS).await {
        Ok(out) => json!({ "ok": true, "output": out }),
        Err(e) => json!({ "ok": false, "error": format!("{e}") }),
    }
}

/// Fetch the canonical lore thread for a message-id and return a list of
/// parsed messages. Mirrors the download path used by `api.rs` for
/// thread injection but returns the parsed mbox to the caller instead of
/// pushing to the ingest pipeline.
pub async fn lore_thread(message_id: &str) -> Value {
    let mid = message_id.trim();
    if mid.is_empty() {
        return json!({ "ok": false, "error": "empty message-id" });
    }
    // Strip wrapping <…> if present; lore's URL takes the bare id.
    let mid = mid.trim_start_matches('<').trim_end_matches('>');

    let url = format!("https://lore.kernel.org/all/{}/t.mbox.gz", mid);
    let response = match reqwest::get(&url).await {
        Ok(r) => r,
        Err(e) => return json!({ "ok": false, "error": format!("fetch failed: {e}") }),
    };
    if !response.status().is_success() {
        return json!({
            "ok": false,
            "error": format!("HTTP {} for {}", response.status(), url)
        });
    }
    let bytes = match response.bytes().await {
        Ok(b) => b,
        Err(e) => return json!({ "ok": false, "error": format!("read body failed: {e}") }),
    };

    let raw = match tokio::task::spawn_blocking(move || -> Result<String, std::io::Error> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
        let mut raw = String::new();
        decoder.read_to_string(&mut raw)?;
        Ok(raw)
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return json!({ "ok": false, "error": format!("gunzip: {e}") }),
        Err(e) => return json!({ "ok": false, "error": format!("blocking task: {e}") }),
    };

    let messages = parse_mbox(&raw);
    json!({ "ok": true, "url": url, "count": messages.len(), "messages": messages })
}

/// Bind the configured lei/b4 binary paths plus the tool dispatcher.
#[derive(Debug, Clone)]
pub struct LeiToolConfig {
    pub lei_bin: String,
    pub b4_bin: String,
}

impl Default for LeiToolConfig {
    fn default() -> Self {
        Self {
            lei_bin: std::env::var("SASHIKO_LEI_BIN")
                .unwrap_or_else(|_| DEFAULT_LEI_BIN.to_string()),
            b4_bin: std::env::var("SASHIKO_B4_BIN").unwrap_or_else(|_| DEFAULT_B4_BIN.to_string()),
        }
    }
}

async fn run_capture(bin: &str, args: &[String], timeout_secs: u64) -> Result<String> {
    debug!("spawning {bin} {args:?}");
    let mut child = match Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            // Most useful failure: binary missing. Surface it clearly.
            anyhow::bail!("failed to spawn {}: {} (is it installed?)", bin, e);
        }
    };

    let mut stdout_buf = Vec::with_capacity(8 * 1024);
    let mut stderr_buf = Vec::with_capacity(2 * 1024);

    let collect = async {
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout
                .read_to_end_capped(&mut stdout_buf, MAX_STDOUT_BYTES)
                .await;
        }
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_end_capped(&mut stderr_buf, 32 * 1024).await;
        }
        child.wait().await
    };

    let status = match timeout(Duration::from_secs(timeout_secs), collect).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => anyhow::bail!("{} wait error: {}", bin, e),
        Err(_) => {
            // Timeout: best-effort kill before returning.
            let _ = child.start_kill();
            anyhow::bail!("{} timed out after {}s", bin, timeout_secs);
        }
    };

    let stderr = String::from_utf8_lossy(&stderr_buf);
    if !stderr.trim().is_empty() {
        for line in stderr.lines().take(20) {
            if !line.trim().is_empty() {
                debug!("[{} stderr] {}", bin, line);
            }
        }
    }

    if !status.success() {
        anyhow::bail!(
            "{} exited with {}: {}",
            bin,
            status,
            stderr.trim().chars().take(500).collect::<String>()
        );
    }

    Ok(String::from_utf8_lossy(&stdout_buf).into_owned())
}

/// `lei q -f json` emits a JSON array of message records (with one final
/// `null` element used as an end-of-stream sentinel in some versions).
/// Skip the sentinel and project to the fields we care about. If the
/// output isn't an array (older lei? streaming JSON?), fall back to
/// returning the raw text so the caller still has *something*.
fn parse_lei_json(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return json!({ "ok": true, "hits": [], "truncated": false });
    }

    if let Ok(v) = serde_json::from_str::<Value>(trimmed)
        && let Some(arr) = v.as_array()
    {
        let hits: Vec<Value> = arr
            .iter()
            .filter(|entry| !entry.is_null())
            .map(|entry| {
                json!({
                    "mid": entry.get("m").and_then(|v| v.as_str()),
                    "subject": entry.get("s").and_then(|v| v.as_str()),
                    "from": entry.get("f").and_then(|v| v.as_str()),
                    "date": entry.get("d").and_then(|v| v.as_str()),
                })
            })
            .collect();
        return json!({
            "ok": true,
            "hits": hits,
            "truncated": false,
        });
    }

    // Fallback: just hand the model the raw text.
    warn!("lei output not a JSON array; returning raw");
    json!({
        "ok": true,
        "hits": [],
        "raw": trimmed.chars().take(8000).collect::<String>(),
        "truncated": trimmed.len() > 8000,
    })
}

/// Minimal mbox splitter mirroring `ingestor::split_mbox`'s contract: each
/// message starts with a line beginning `From `. We keep this internal so
/// the public surface stays small; the existing ingestor variant has more
/// guards but pulls in heavier dependencies.
fn parse_mbox(raw: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut current = String::new();
    for line in raw.lines() {
        if line.starts_with("From ") && !current.is_empty() {
            out.push(parse_one_message(&current));
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.is_empty() {
        out.push(parse_one_message(&current));
    }
    out
}

fn parse_one_message(raw: &str) -> Value {
    let mut from = None;
    let mut subject = None;
    let mut date = None;
    let mut message_id = None;
    let mut in_header = true;
    let mut body = String::new();
    for line in raw.lines() {
        if in_header {
            if line.is_empty() {
                in_header = false;
                continue;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("from: ") {
                from = Some(line[6..].trim().to_string());
                let _ = v;
            } else if let Some(v) = lower.strip_prefix("subject: ") {
                subject = Some(line[9..].trim().to_string());
                let _ = v;
            } else if let Some(v) = lower.strip_prefix("date: ") {
                date = Some(line[6..].trim().to_string());
                let _ = v;
            } else if let Some(v) = lower.strip_prefix("message-id: ") {
                message_id = Some(line[12..].trim().to_string());
                let _ = v;
            }
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    json!({
        "from": from,
        "subject": subject,
        "date": date,
        "message_id": message_id,
        "body": body.chars().take(8000).collect::<String>(),
    })
}

/// Tiny extension trait to bound how much we read from a child process.
trait ReadToEndCapped {
    async fn read_to_end_capped(&mut self, buf: &mut Vec<u8>, cap: usize)
    -> std::io::Result<usize>;
}

impl<T: tokio::io::AsyncRead + Unpin> ReadToEndCapped for T {
    async fn read_to_end_capped(
        &mut self,
        buf: &mut Vec<u8>,
        cap: usize,
    ) -> std::io::Result<usize> {
        let mut chunk = [0u8; 8 * 1024];
        let mut total = 0;
        loop {
            let n = self.read(&mut chunk).await?;
            if n == 0 {
                return Ok(total);
            }
            let remaining = cap.saturating_sub(buf.len());
            if remaining == 0 {
                // Drop overflow on the floor; the process keeps writing
                // but we won't grow past cap.
                continue;
            }
            let take = n.min(remaining);
            buf.extend_from_slice(&chunk[..take]);
            total += take;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_binary_is_structured_error() {
        let v = lei_search("anything", 5, "/nonexistent/lei-not-here").await;
        assert_eq!(v["ok"].as_bool(), Some(false));
        assert!(
            v["error"]
                .as_str()
                .unwrap_or("")
                .contains("failed to spawn"),
            "expected spawn failure, got {v}"
        );
    }

    #[tokio::test]
    async fn lei_search_parses_array_payload() {
        let json_body = r#"[{"m":"<a@b>","s":"subj","f":"alice","d":"2026-01-01T00:00:00Z"},null]"#;
        let (_dir, stub) = stub_in_tempdir("lei");
        write_stub(&stub, &format!("#!/bin/sh\nprintf '%s' '{json_body}'\n"));
        let v = lei_search("Fixes:deadbeef", 5, stub.to_str().unwrap()).await;
        assert_eq!(v["ok"].as_bool(), Some(true), "got {v}");
        let hits = v["hits"].as_array().expect("hits array");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["mid"].as_str(), Some("<a@b>"));
        assert_eq!(hits[0]["subject"].as_str(), Some("subj"));
    }

    #[tokio::test]
    async fn lei_search_handles_empty_output() {
        let (_dir, stub) = stub_in_tempdir("lei");
        write_stub(&stub, "#!/bin/sh\nprintf ''\n");
        let v = lei_search("anything", 5, stub.to_str().unwrap()).await;
        assert_eq!(v["ok"].as_bool(), Some(true));
        assert_eq!(v["hits"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn lei_search_rejects_empty_query() {
        let v = lei_search("   ", 5, "lei").await;
        assert_eq!(v["ok"].as_bool(), Some(false));
    }

    #[tokio::test]
    async fn b4_dig_returns_stdout_on_success() {
        let (_dir, stub) = stub_in_tempdir("b4");
        write_stub(&stub, "#!/bin/sh\necho \"called: $*\"\n");
        let v = b4_dig("deadbeefcafe", stub.to_str().unwrap()).await;
        assert_eq!(v["ok"].as_bool(), Some(true), "got {v}");
        let out = v["output"].as_str().unwrap_or("");
        assert!(out.contains("dig"), "expected dig in args; got {out:?}");
        assert!(
            out.contains("-c"),
            "expected -c flag for SHA input; got {out:?}"
        );
    }

    #[tokio::test]
    async fn b4_dig_omits_dash_c_for_message_id() {
        let (_dir, stub) = stub_in_tempdir("b4");
        write_stub(&stub, "#!/bin/sh\necho \"called: $*\"\n");
        let v = b4_dig("<20260101.deadbeef@example.com>", stub.to_str().unwrap()).await;
        let out = v["output"].as_str().unwrap_or("");
        assert!(
            !out.contains("-c"),
            "should not pass -c for non-SHA; got {out:?}"
        );
    }

    /// Returns a fresh `(TempDir, PathBuf)` pair so each test gets a
    /// dedicated, parallel-safe stub script directory. The TempDir handle
    /// keeps the directory alive for the lifetime of the test.
    fn stub_in_tempdir(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(name);
        (dir, path)
    }

    fn write_stub(path: &std::path::Path, content: &str) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, content).unwrap();
        let mut perms = std::fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).unwrap();
    }
}
