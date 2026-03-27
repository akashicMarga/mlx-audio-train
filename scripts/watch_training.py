#!/usr/bin/env python3
"""
watch_training.py — Live training dashboard with loss curves.

Opens a Gradio app that reads train_log.jsonl and auto-refreshes every 15s.
Shows: train loss, val loss, learning rate, steps/sec curves.

Usage:
    python scripts/watch_training.py
    python scripts/watch_training.py --log checkpoints/qwen3-hindi/train_log.jsonl
    python scripts/watch_training.py --log checkpoints/qwen3-hindi/train_log.jsonl --refresh 10
"""

import argparse
import json
from pathlib import Path


def read_log(log_path: str):
    """Parse JSONL log → separate train and val entries."""
    train_entries = []
    val_entries   = []

    path = Path(log_path)
    if not path.exists():
        return train_entries, val_entries

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "val_loss" in entry:
                    val_entries.append(entry)
                elif "loss" in entry:
                    train_entries.append(entry)
            except Exception:
                pass

    return train_entries, val_entries


def build_figures(log_path: str):
    """Build matplotlib figures from log file. Returns list of figure objects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    train, val = read_log(log_path)

    if not train:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No training data yet — waiting for first log entry...",
                ha="center", va="center", fontsize=13, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Extract series
    steps       = [e["step"]      for e in train]
    losses      = [e["loss"]      for e in train]
    lrs         = [e.get("lr", 0) for e in train]
    main_losses = [e.get("main_loss", e["loss"]) for e in train]
    sub_losses  = [e.get("sub_loss", 0)          for e in train]

    val_steps  = [e["step"]     for e in val]
    val_losses = [e["val_loss"] for e in val]

    elapsed = train[-1].get("elapsed", 0) if train else 0
    cur_step = steps[-1] if steps else 0
    cur_loss = losses[-1] if losses else 0
    cur_lr   = lrs[-1] if lrs else 0

    # Figure layout
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {
        "train":    "#4fc3f7",
        "val":      "#ff7043",
        "main":     "#81c784",
        "sub":      "#ce93d8",
        "lr":       "#ffd54f",
        "bg":       "#1e2130",
        "grid":     "#2a2d3e",
        "text":     "#e0e0e0",
    }

    def _style(ax, title):
        ax.set_facecolor(COLORS["bg"])
        ax.set_title(title, color=COLORS["text"], fontsize=11, pad=8)
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.grid(True, color=COLORS["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])

    # ── Plot 1: Total loss ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, losses, color=COLORS["train"], linewidth=1.5, label="train loss")
    if val_steps:
        ax1.plot(val_steps, val_losses, "o-", color=COLORS["val"],
                 linewidth=1.5, markersize=4, label="val loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    _style(ax1, "Total Loss")

    # ── Plot 2: Main vs Sub-talker loss ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, main_losses, color=COLORS["main"], linewidth=1.5, label="main (CB0)")
    if any(s > 0 for s in sub_losses):
        ax2.plot(steps, sub_losses, color=COLORS["sub"], linewidth=1.5,
                 linestyle="--", label="sub (CB1-15)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], labelcolor=COLORS["text"])
    _style(ax2, "Main vs Sub-talker Loss")

    # ── Plot 3: Learning rate schedule ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, lrs, color=COLORS["lr"], linewidth=1.5)
    ax3.set_xlabel("step")
    ax3.set_ylabel("lr")
    ax3.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2e"))
    _style(ax3, "Learning Rate")

    # ── Plot 4: Stats text ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS["bg"])
    ax4.axis("off")
    _style(ax4, "Training Status")

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    sps  = cur_step / elapsed if elapsed > 0 else 0

    stats = [
        ("Current step",   f"{cur_step}"),
        ("Current loss",   f"{cur_loss:.5f}"),
        ("Best train loss",f"{min(losses):.5f}"),
        ("Best val loss",  f"{min(val_losses):.5f}" if val_losses else "—"),
        ("Learning rate",  f"{cur_lr:.2e}"),
        ("Elapsed",        f"{mins}m {secs}s"),
        ("Steps/sec",      f"{sps:.3f}"),
        ("Checkpoints",    str(len(val_steps))),
    ]

    y = 0.92
    for label, value in stats:
        ax4.text(0.05, y, label + ":", color="#9e9e9e", fontsize=9,
                 transform=ax4.transAxes, va="top")
        ax4.text(0.55, y, value, color=COLORS["text"], fontsize=9,
                 fontweight="bold", transform=ax4.transAxes, va="top")
        y -= 0.11

    fig.suptitle(
        f"Qwen3-TTS Hindi LoRA — {Path(log_path).parent.name}",
        color=COLORS["text"], fontsize=13, y=0.98
    )

    return fig


def launch_dashboard(log_path: str, refresh_secs: int = 15):
    import gradio as gr

    print(f"[dashboard] Watching: {log_path}")
    print(f"[dashboard] Auto-refresh every {refresh_secs}s")

    def update():
        fig = build_figures(log_path)
        return fig

    with gr.Blocks(title="MLX Training Dashboard", theme=gr.themes.Base()) as demo:
        gr.Markdown("## 📈 MLX Audio Training Dashboard")
        gr.Markdown(f"Watching `{log_path}` — refreshes every **{refresh_secs}s**")

        plot = gr.Plot(label="Training Curves", show_label=False)
        refresh_btn = gr.Button("🔄 Refresh Now", variant="secondary", size="sm")

        # Auto-refresh via timer
        timer = gr.Timer(value=refresh_secs)
        timer.tick(fn=update, outputs=plot)

        # Manual refresh button
        refresh_btn.click(fn=update, outputs=plot)

        # Load on startup
        demo.load(fn=update, outputs=plot)

    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     default="checkpoints/qwen3-hindi/train_log.jsonl")
    parser.add_argument("--refresh", type=int, default=15, help="Auto-refresh interval (seconds)")
    parser.add_argument("--no-ui",   action="store_true",  help="Just print latest stats, no Gradio")
    args = parser.parse_args()

    if args.no_ui:
        train, val = read_log(args.log)
        if not train:
            print("No training data yet.")
            return
        last = train[-1]
        print(f"Step {last['step']}  loss={last['loss']:.5f}  lr={last.get('lr',0):.2e}  "
              f"elapsed={last.get('elapsed',0):.0f}s")
        if val:
            print(f"Best val_loss={min(e['val_loss'] for e in val):.5f}")
        return

    launch_dashboard(args.log, args.refresh)


if __name__ == "__main__":
    main()
