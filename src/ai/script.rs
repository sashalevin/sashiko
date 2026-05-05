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

//! AI provider that delegates to a user-supplied wrapper script.
//!
//! Contract:
//! * `argv[1]` of the script is a path to a UTF-8 file containing the prompt
//!   built from the `AiRequest` (system + messages + tool definitions).
//! * The script writes the model response to stdout. Plain text becomes
//!   `AiResponse.content`; JSON matching the `claude_cli` tool-call shape is
//!   parsed into structured tool calls.
//! * Non-zero exit code is treated as an error. Quota / rate-limit retries
//!   are the script's responsibility (see ~/cursor.sh, ~/claude.sh for
//!   reference implementations).

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::io::Write;
use std::process::Stdio;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use crate::ai::{AiProvider, AiRequest, AiResponse, ProviderCapabilities, claude_cli};

pub struct ScriptProvider {
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    pub env: BTreeMap<String, String>,
    pub model: String,
    pub context_window_size: usize,
}

#[async_trait]
impl AiProvider for ScriptProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = claude_cli::build_prompt(&request);
        debug!(
            "script provider: invoking {:?} with prompt length {} chars",
            self.command,
            prompt.len()
        );

        // Block to keep the NamedTempFile sync handle out of the await path.
        let tmp = {
            let mut f = NamedTempFile::new()
                .context("failed to create temp file for script provider prompt")?;
            f.write_all(prompt.as_bytes())
                .context("failed to write prompt to temp file")?;
            f.flush().ok();
            f
        };

        let mut argv: Vec<String> = self.args.clone();
        argv.push(tmp.path().to_string_lossy().into_owned());

        let mut cmd = Command::new(&self.command);
        cmd.args(&argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in &self.env {
            cmd.env(k, v);
        }

        let child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn script provider command {:?} (is the path correct and executable?)",
                self.command
            )
        })?;

        let output = timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "script provider command {:?} timed out after {}s",
                self.command,
                self.timeout_secs
            )
        })?
        .with_context(|| format!("script provider wait error for {:?}", self.command))?;

        // Tempfile drops here, removing the prompt file.
        drop(tmp);

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if !line.trim().is_empty() {
                    debug!("[script-provider stderr] {}", line);
                }
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "script provider command {:?} exited with {}: {}",
                self.command,
                output.status,
                stderr.trim()
            );
        }

        let raw = String::from_utf8_lossy(&output.stdout).into_owned();
        if raw.trim().is_empty() {
            warn!("script provider {:?} returned empty stdout", self.command);
        }

        claude_cli::parse_inner_response(raw.trim(), None)
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        let chars: usize = request
            .messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();
        chars / 4
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: self.context_window_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiRole};

    fn req(text: &str) -> AiRequest {
        AiRequest {
            system: None,
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some(text.to_string()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
        }
    }

    fn provider(command: &str, args: Vec<&str>, timeout_secs: u64) -> ScriptProvider {
        ScriptProvider {
            command: command.to_string(),
            args: args.into_iter().map(String::from).collect(),
            timeout_secs,
            env: BTreeMap::new(),
            model: "test-model".to_string(),
            context_window_size: 100_000,
        }
    }

    #[tokio::test]
    async fn echoes_prompt_back_as_content() {
        // /bin/sh -c 'cat "$1"' _ <tmpfile> -> echoes the prompt file contents.
        let p = provider("/bin/sh", vec!["-c", "cat \"$1\"", "_"], 30);
        let resp = p
            .generate_content(req("hello sashiko"))
            .await
            .expect("script invocation should succeed");
        let content = resp.content.expect("expected text content");
        assert!(
            content.contains("hello sashiko"),
            "content should embed prompt; got: {content:?}"
        );
    }

    #[tokio::test]
    async fn json_tool_calls_are_parsed() {
        let json = r#"{"tool_calls":[{"id":"c1","function_name":"do_thing","arguments":{"x":1}}]}"#;
        let p = provider(
            "/bin/sh",
            vec!["-c", &format!("printf '%s' '{json}'"), "_"],
            30,
        );
        let resp = p
            .generate_content(req("ignored"))
            .await
            .expect("script invocation should succeed");
        let calls = resp.tool_calls.expect("expected structured tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "do_thing");
    }

    #[tokio::test]
    async fn timeout_is_reported() {
        let p = provider("/bin/sh", vec!["-c", "sleep 5", "_"], 1);
        let err = p
            .generate_content(req("anything"))
            .await
            .expect_err("expected timeout error");
        let msg = format!("{err}");
        assert!(
            msg.contains("timed out"),
            "error message should mention timeout; got: {msg}"
        );
    }

    #[tokio::test]
    async fn nonzero_exit_is_error() {
        let p = provider("/bin/sh", vec!["-c", "echo boom >&2; exit 7", "_"], 30);
        let err = p
            .generate_content(req("anything"))
            .await
            .expect_err("expected non-zero exit error");
        let msg = format!("{err}");
        assert!(
            msg.contains("exited with"),
            "error message should mention exit status; got: {msg}"
        );
    }
}
