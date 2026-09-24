"""Visualize training metrics from the strategic decision layer.

Reads the JSONL logs generated during training and plots a comprehensive
dashboard showing:
  - Network losses (total, actor, critic, entropy)
  - Money delta performance vs all opponent types
  - Separate breakdown: static opponents vs self-play checkpoints
  - Rolling win rate to detect learning trends
  - Replay buffer growth & training speed
  - Cumulative learning verdict
"""

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────
BG_COLOR    = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR  = "#21262d"
TEXT_COLOR  = "#e6edf3"
ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
RED         = "#f85149"
YELLOW      = "#d29922"
PURPLE      = "#bc8cff"
ORANGE      = "#ffa657"

STATIC_OPPONENTS = {"heuristic", "noisy_0.1", "noisy_0.2", "passive", "starter"}
STATIC_COLORS    = {
    "heuristic": "#58a6ff",
    "noisy_0.1": "#3fb950",
    "noisy_0.2": "#ffa657",
    "passive":   "#bc8cff",
    "starter":   "#f85149",
}

matplotlib.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    PANEL_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_metrics(log_file):
    """Load metrics from JSONL file."""
    if not os.path.exists(log_file):
        print(f"Log file not found at {log_file}")
        return []
    metrics = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metrics.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return metrics


# ── Math helpers ───────────────────────────────────────────────────────────────

def smooth(a, n):
    """Causal rolling mean. Returns (smoothed_values, indices_into_original)."""
    a = np.asarray(a, dtype=float)
    if len(a) < n:
        return a, np.arange(len(a))
    kernel = np.ones(n) / n
    smoothed = np.convolve(a, kernel, mode="valid")
    return smoothed, np.arange(n - 1, len(a))


def rolling_win_rate(deltas, n):
    wins = np.asarray([1.0 if d > 0 else 0.0 for d in deltas])
    return smooth(wins, n)


def fmt_k(x, _):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return str(int(x))


# ── Core plot ──────────────────────────────────────────────────────────────────

def plot_metrics(metrics, eval_metrics=None, output_file=None):
    """Generate and display/save plots for training metrics."""
    if not metrics:
        print("No metrics to plot.")
        return

    # Sort by episode (safety)
    metrics = sorted(metrics, key=lambda m: m["episode"])

    episodes  = np.asarray([m["episode"]           for m in metrics])
    losses    = np.asarray([m["loss"]               for m in metrics])
    actor_l   = np.asarray([m.get("actor_loss",   np.nan) for m in metrics])
    critic_l  = np.asarray([m.get("critic_loss",  np.nan) for m in metrics])
    entropy_l = np.asarray([m.get("entropy_loss", np.nan) for m in metrics])
    deltas    = np.asarray([m["money_delta"]        for m in metrics])
    buf_sizes = np.asarray([m.get("buffer_size", 0) for m in metrics])
    eps_sec   = np.asarray([m.get("eps_per_sec", 0) for m in metrics])
    opponents = [m.get("opponent", "unknown")       for m in metrics]

    # Classify static vs self-play opponents
    # Self-play checkpoints are stored as "ep<number>" (e.g. "ep10000")
    is_static   = np.asarray([o in STATIC_OPPONENTS for o in opponents])
    is_selfplay = ~is_static

    static_eps  = episodes[is_static]
    static_del  = deltas[is_static]
    static_opps = [o for o, s in zip(opponents, is_static) if s]

    sp_eps = episodes[is_selfplay]
    sp_del = deltas[is_selfplay]

    # Window sizes
    n_total = len(episodes)
    win_big = max(20, min(200, n_total // 25))
    win_sm  = max(10, min(50,  n_total // 50))

    # ── Learning analysis ──────────────────────────────────────────────────────
    n5 = max(1, n_total // 5)
    early_del  = deltas[:n5]
    late_del   = deltas[-n5:]
    early_win  = np.mean(early_del > 0)
    late_win   = np.mean(late_del  > 0)
    early_mean = np.mean(early_del)
    late_mean  = np.mean(late_del)
    early_loss = losses[:n5]
    late_loss  = losses[-n5:]

    learned = (late_win - early_win) > 0.05 or (late_mean - early_mean) > 1000

    print("\n" + "=" * 60)
    print("  LEARNING ANALYSIS")
    print("=" * 60)
    print(f"  Total episodes logged : {max(episodes):,}")
    print(f"  Total records         : {n_total:,}")
    print(f"  Static opponent games : {is_static.sum():,}  ({100*is_static.mean():.0f}%)")
    print(f"  Self-play games       : {is_selfplay.sum():,}  ({100*is_selfplay.mean():.0f}%)")
    print()
    print(f"  Early win rate        : {100*early_win:.1f}%  (first 20% of records)")
    print(f"  Late  win rate        : {100*late_win:.1f}%  (last  20% of records)")
    print(f"  Win-rate gain         : {100*(late_win-early_win):+.1f} pp")
    print()
    print(f"  Early mean delta      : {early_mean:+,.0f}")
    print(f"  Late  mean delta      : {late_mean:+,.0f}")
    print()
    print(f"  Early mean loss       : {np.mean(early_loss):.4f}")
    print(f"  Late  mean loss       : {np.mean(late_loss):.4f}")
    print()
    if learned:
        print("  VERDICT: Agent IS learning  (win-rate & delta both improved)")
    else:
        print("  VERDICT: Learning signal is WEAK / not yet conclusive")
    print("=" * 60 + "\n")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 26), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(
        6, 2,
        figure=fig,
        hspace=0.55,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.95,  bottom=0.04,
        height_ratios=[0.6, 1.2, 1.2, 1.0, 0.8, 0.8],
    )

    def ax_style(ax, title, ylabel, xlabel=None):
        ax.set_title(title,  color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
        if xlabel:
            ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    # ── ROW 0: Title banner ────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(PANEL_COLOR)
    verdict_col = GREEN if learned else YELLOW
    verdict_txt = "LEARNING DETECTED" if learned else "WEAK LEARNING SIGNAL"
    ax_title.text(0.5, 0.72, "Strategic Layer — Training Dashboard",
                  ha="center", va="center", fontsize=18, fontweight="bold",
                  color=TEXT_COLOR, transform=ax_title.transAxes)
    ax_title.text(
        0.5, 0.25,
        (f"{verdict_txt}   |   Episodes: {max(episodes):,}   |   "
         f"Late win-rate: {100*late_win:.1f}%  ({100*(late_win-early_win):+.1f} pp)   |   "
         f"Late mean delta: {late_mean:+,.0f}"),
        ha="center", va="center", fontsize=10,
        color=verdict_col, transform=ax_title.transAxes)
    ax_title.axis("off")

    # ── ROW 1 left: Network losses ─────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_style(ax_loss, "Network Losses", "Loss")
    raw_alpha = 0.15

    sm_loss, sm_idx = smooth(losses, win_big)
    ax_loss.plot(episodes, losses, color=TEXT_COLOR, alpha=raw_alpha, lw=0.8, label="_nolegend_")
    ax_loss.plot(episodes[sm_idx], sm_loss, color=TEXT_COLOR, lw=2.0, label=f"Total (avg {win_big})")

    if not np.all(np.isnan(actor_l)):
        sm_a, sm_ai = smooth(actor_l, win_big)
        ax_loss.plot(episodes, actor_l, color=ACCENT,  alpha=raw_alpha, lw=0.6, label="_nolegend_")
        ax_loss.plot(episodes[sm_ai], sm_a, color=ACCENT,  lw=1.5, label="Actor")
    if not np.all(np.isnan(critic_l)):
        sm_c, sm_ci = smooth(critic_l, win_big)
        ax_loss.plot(episodes, critic_l, color=ORANGE, alpha=raw_alpha, lw=0.6, label="_nolegend_")
        ax_loss.plot(episodes[sm_ci], sm_c, color=ORANGE, lw=1.5, label="Critic")
    if not np.all(np.isnan(entropy_l)):
        sm_e, sm_ei = smooth(entropy_l, win_big)
        ax_loss.plot(episodes, entropy_l, color=GREEN, alpha=raw_alpha, lw=0.6, label="_nolegend_")
        ax_loss.plot(episodes[sm_ei], sm_e, color=GREEN, lw=1.5, label="Entropy")

    ax_loss.axhline(0, color=GRID_COLOR, lw=1, ls="--")
    ax_loss.legend(fontsize=7, framealpha=0.7)

    # ── ROW 1 right: Rolling win rate ──────────────────────────────────────────
    ax_wr = fig.add_subplot(gs[1, 1])
    ax_style(ax_wr, "Rolling Win Rate — ALL Games", "Win Rate")

    rw, rw_idx = rolling_win_rate(deltas, win_big)
    ax_wr.plot(episodes[rw_idx], rw, color=ACCENT, lw=2.0, label=f"Win rate (n={win_big})")
    ax_wr.axhline(0.5, color=RED, lw=1.2, ls="--", alpha=0.8, label="50% baseline")
    ax_wr.fill_between(episodes[rw_idx], 0.5, rw,
                       where=(rw >= 0.5), alpha=0.15, color=GREEN)
    ax_wr.fill_between(episodes[rw_idx], 0.5, rw,
                       where=(rw <  0.5), alpha=0.15, color=RED)
    ax_wr.set_ylim(-0.05, 1.05)
    ax_wr.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    ax_wr.legend(fontsize=7, framealpha=0.7)
    # Annotate start / end
    mid = win_big // 2
    ax_wr.annotate(f"{100*early_win:.0f}%", xy=(episodes[mid], early_win),
                   fontsize=9, color=YELLOW, fontweight="bold")
    ax_wr.annotate(f"{100*late_win:.0f}%",  xy=(episodes[-mid], late_win),
                   fontsize=9, color=(GREEN if late_win >= 0.5 else RED), fontweight="bold")

    # ── ROW 2 left: Money delta vs static opponents ────────────────────────────
    ax_st = fig.add_subplot(gs[2, 0])
    ax_style(ax_st, "Money Delta vs Static Opponents", "Money Delta ($)")

    for opp_name in sorted(STATIC_OPPONENTS):
        mask = np.asarray([o == opp_name for o in opponents])
        if mask.sum() == 0:
            continue
        col = STATIC_COLORS.get(opp_name, "#888")
        ax_st.scatter(episodes[mask], deltas[mask],
                      color=col, alpha=0.4, s=10, label=opp_name)
        if mask.sum() > win_sm:
            sm_d, sm_di = smooth(deltas[mask], win_sm)
            ax_st.plot(episodes[mask][sm_di], sm_d, color=col, lw=1.8)

    if len(static_del) > win_big:
        sm_all_st, sm_sti = smooth(static_del, win_big)
        ax_st.plot(static_eps[sm_sti], sm_all_st,
                   color=TEXT_COLOR, lw=2.5, ls="--",
                   label=f"All-static avg (n={win_big})", zorder=5)

    ax_st.axhline(0, color=RED, lw=1.2, ls="--", alpha=0.7)
    ax_st.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_st.legend(fontsize=7, framealpha=0.7, ncol=2)

    # ── ROW 2 right: Money delta vs self-play ──────────────────────────────────
    ax_sp = fig.add_subplot(gs[2, 1])

    if is_selfplay.sum() > 0:
        sp_wr = np.mean(sp_del > 0)
        ax_style(ax_sp,
                 f"Money Delta vs Self-Play Checkpoints  (win rate: {100*sp_wr:.1f}%)",
                 "Money Delta ($)")
        ax_sp.scatter(sp_eps, sp_del, color=PURPLE, alpha=0.3, s=8, label="Self-play game")
        if len(sp_del) > win_big:
            sm_sp, sm_spi = smooth(sp_del, win_big)
            ax_sp.plot(sp_eps[sm_spi], sm_sp, color=PURPLE, lw=2.5,
                       label=f"Self-play avg (n={win_big})")
        ax_sp.axhline(0, color=RED, lw=1.2, ls="--", alpha=0.7)
        ax_sp.yaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax_sp.legend(fontsize=7, framealpha=0.7)
    else:
        ax_style(ax_sp, "Money Delta vs Self-Play Checkpoints", "Money Delta ($)")
        ax_sp.text(0.5, 0.5, "No self-play data",
                   ha="center", va="center", fontsize=12, color=GRID_COLOR,
                   transform=ax_sp.transAxes)

    # ── ROW 3 left: Per-opponent win rate bar chart ────────────────────────────
    ax_bars = fig.add_subplot(gs[3, 0])
    ax_style(ax_bars, "Win Rate per Static Opponent", "Win Rate (%)")

    opp_names, opp_wrs, opp_cols = [], [], []
    for opp_name in sorted(STATIC_OPPONENTS):
        mask = np.asarray([o == opp_name for o in opponents])
        od = deltas[mask]
        if len(od) == 0:
            continue
        opp_names.append(opp_name)
        opp_wrs.append(100 * np.mean(od > 0))
        opp_cols.append(STATIC_COLORS.get(opp_name, "#888"))

    bars = ax_bars.bar(opp_names, opp_wrs, color=opp_cols, alpha=0.85, edgecolor=GRID_COLOR)
    ax_bars.axhline(50, color=RED, lw=1.2, ls="--", alpha=0.8)
    ax_bars.set_ylim(0, 115)
    for bar, wr in zip(bars, opp_wrs):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, wr + 2,
                     f"{wr:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    ax_bars.tick_params(axis="x", labelsize=9)

    # ── ROW 3 right: Critic loss ───────────────────────────────────────────────
    ax_crit = fig.add_subplot(gs[3, 1])
    ax_style(ax_crit, "Critic Loss (Value Estimation Quality)", "MSE Loss")

    if not np.all(np.isnan(critic_l)):
        ax_crit.plot(episodes, critic_l, color=ORANGE, alpha=0.15, lw=0.8, label="_nolegend_")
        sm_c2, sm_c2i = smooth(critic_l, win_big)
        ax_crit.plot(episodes[sm_c2i], sm_c2, color=ORANGE, lw=2.2,
                     label=f"Critic loss (avg {win_big})")
        ax_crit.axhline(0, color=GRID_COLOR, lw=1, ls="--")
        ax_crit.legend(fontsize=7, framealpha=0.7)

    # ── ROW 4 left: Replay buffer ──────────────────────────────────────────────
    ax_buf = fig.add_subplot(gs[4, 0])
    ax_style(ax_buf, "Replay Buffer Size", "Entries", xlabel="Episode")
    ax_buf.plot(episodes, buf_sizes, color=PURPLE, lw=1.8, alpha=0.9)
    ax_buf.fill_between(episodes, buf_sizes, color=PURPLE, alpha=0.12)
    ax_buf.yaxis.set_major_formatter(FuncFormatter(fmt_k))

    # ── ROW 4 right: Training speed ────────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[4, 1])
    ax_style(ax_spd, "Training Speed", "Episodes / sec", xlabel="Episode")
    sm_sp2, sm_sp2i = smooth(eps_sec, win_big)
    ax_spd.plot(episodes, eps_sec, color=GREEN, alpha=0.18, lw=0.8, label="_nolegend_")
    ax_spd.plot(episodes[sm_sp2i], sm_sp2, color=GREEN, lw=2.0, label=f"Smoothed (n={win_big})")
    ax_spd.legend(fontsize=7, framealpha=0.7)

    # ── ROW 5: Cumulative win-rate bars + trend ────────────────────────────────
    ax_cum = fig.add_subplot(gs[5, :])
    ax_style(ax_cum, f"Win Rate Progression ({win_big}-record buckets)", "Win Rate (%)", xlabel="Episode")

    bucket_eps_list, bucket_wr_list = [], []
    for start in range(0, n_total, win_big):
        seg_d = deltas[start:start + win_big]
        seg_e = episodes[start:start + win_big]
        if len(seg_d) == 0:
            continue
        bucket_eps_list.append(float(np.mean(seg_e)))
        bucket_wr_list.append(100 * np.mean(seg_d > 0))

    bucket_eps = np.asarray(bucket_eps_list)
    bucket_wr  = np.asarray(bucket_wr_list)
    bar_width  = (bucket_eps[1] - bucket_eps[0]) * 0.85 if len(bucket_eps) > 1 else win_big

    ax_cum.bar(bucket_eps, bucket_wr, width=bar_width,
               color=[GREEN if w >= 50 else RED for w in bucket_wr],
               alpha=0.65, edgecolor="none")
    ax_cum.axhline(50, color=RED, lw=1.5, ls="--", alpha=0.8, label="50% baseline")

    if len(bucket_eps) > 3:
        z = np.polyfit(bucket_eps, bucket_wr, 1)
        trend = np.poly1d(z)
        ax_cum.plot(bucket_eps, trend(bucket_eps), color=ACCENT, lw=2.5,
                    label=f"Trend ({z[0] * 1e4:+.2f}% / 10k eps)")

    ax_cum.set_ylim(0, 115)
    ax_cum.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_cum.legend(fontsize=8, framealpha=0.7)

    # ── Save / show ────────────────────────────────────────────────────────────
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize strategic-layer training logs")
    parser.add_argument("log", nargs="?", default="./training_metrics.jsonl",
                        help="Path to the training JSONL log file")
    parser.add_argument("--output", "-o", default="training_graphs.png",
                        help="Output image file (default: training_graphs.png)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively (in addition to saving)")
    args = parser.parse_args()

    log_path = args.log
    if not log_path.endswith(".jsonl") and not os.path.exists(log_path):
        if os.path.exists(log_path + ".jsonl"):
            log_path += ".jsonl"

    eval_log_path = os.path.join(os.path.dirname(log_path) or ".", "eval_metrics.jsonl")

    metrics      = load_metrics(log_path)
    eval_metrics = load_metrics(eval_log_path)

    print(f"Loaded {len(metrics)} training metric records from {log_path}")
    if eval_metrics:
        print(f"Loaded {len(eval_metrics)} evaluation metric records.")
    else:
        print("No evaluation metrics found (eval_metrics.jsonl not present).")

    out_file = None if (args.show and not args.output) else args.output
    plot_metrics(metrics, eval_metrics, out_file)

    if args.show and args.output:
        plt.show()
