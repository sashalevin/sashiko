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

//! sashiko-backport: review queued stable backports.
//!
//! Walks a git range in a stable-rc queue branch and emits, per commit,
//! a JSON line indicating whether the commit is a correct/safe/shippable
//! backport on the target stable kernel version. Mirrors the role of
//! `~/stable-ai/reviewer/review-pending.sh` but plugs into sashiko's
//! infrastructure (DB persistence, AI provider, structured findings).

use anyhow::{Context, Result};
use clap::Parser;
use sashiko::ai::create_provider;
use sashiko::backport_reviewer::{BackportReviewer, BackportRunConfig, validate_inputs};
use sashiko::db::Database;
use sashiko::settings::Settings;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(
    name = "sashiko-backport",
    about = "Review queued stable backports for correctness/safety/shippability"
)]
struct Args {
    /// Target stable kernel version (e.g. "6.12"). Used to label
    /// reviews and to derive the canonical target branch name.
    version: String,

    /// Git range to review (e.g.
    /// "stable-rc/linux-6.12.y..stable-rc/queue/6.12"). Must resolve in
    /// the working tree at --repo (defaults to settings.git.repository_path).
    range: String,

    /// Path to the kernel git working tree. Defaults to
    /// `Settings.toml`'s `git.repository_path`.
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Maximum concurrent commit reviews.
    #[arg(short = 'c', long, default_value_t = 4)]
    concurrency: usize,

    /// Re-review every commit even when a verdict is cached in the DB.
    #[arg(long)]
    no_cache: bool,

    /// Restrict to a subset of stages (1-7), comma-separated. Mostly
    /// useful for prompt iteration. Default: all stages.
    #[arg(long, value_delimiter = ',')]
    stages: Option<Vec<u8>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let no_color = std::env::var("NO_COLOR").is_ok();
    let plain_logs = std::env::var("SASHIKO_LOG_PLAIN").is_ok();
    let use_ansi = !no_color && std::io::stderr().is_terminal();
    let builder = tracing_subscriber::fmt()
        .with_writer(sashiko::logging::IgnoreBrokenPipe(std::io::stderr))
        .with_ansi(use_ansi);
    if plain_logs {
        builder
            .with_level(false)
            .with_target(false)
            .without_time()
            .init();
    } else {
        builder.init();
    }

    let args = Args::parse();
    validate_inputs(&args.version, &args.range).context("invalid CLI arguments")?;

    let settings = Settings::new().context("failed to load Settings.toml")?;
    let repo_path = args
        .repo
        .clone()
        .unwrap_or_else(|| PathBuf::from(&settings.git.repository_path));

    if !repo_path.exists() {
        error!(
            "repository path {:?} does not exist; check --repo or git.repository_path in Settings.toml",
            repo_path
        );
        std::process::exit(2);
    }

    info!(
        "sashiko-backport: version={} range={} repo={:?} concurrency={} no_cache={}",
        args.version, args.range, repo_path, args.concurrency, args.no_cache
    );

    let db = Database::new(&settings.database)
        .await
        .context("opening database")?;
    db.migrate().await.context("running migrations")?;
    let db = Arc::new(db);

    let provider = create_provider(&settings).context("creating AI provider")?;
    info!(
        "AI provider model: {}",
        provider.get_capabilities().model_name
    );

    let cfg = BackportRunConfig {
        target_version: args.version.clone(),
        range: args.range.clone(),
        repo_path,
        concurrency: args.concurrency.max(1),
        no_cache: args.no_cache,
        stages: args.stages.clone(),
    };

    let reviewer = BackportReviewer::new(db, provider, cfg.concurrency);
    let summary = reviewer.run(cfg).await.context("running backport review")?;

    eprintln!(
        "Done. total={} yes={} no={} needs_review={} skipped={} cached={} errors={}",
        summary.total,
        summary.yes,
        summary.no,
        summary.needs_review,
        summary.skipped,
        summary.cached,
        summary.errors,
    );

    if summary.errors > 0 {
        std::process::exit(1);
    }
    Ok(())
}
