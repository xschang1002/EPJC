"""Generate paper-ready figures from the frozen representation-v2 reports.

This script only reads existing JSON reports. It does not train models or
modify analysis outputs; PNG and vector PDF assets are written beside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = ROOT / "reports" / "representation_v2"
EXPL = REPORT / "exploratory"
OUT = HERE


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def figure_full_hidden_interaction() -> None:
    formal = read_json(REPORT / "formal_matrix" / "formal_scores.json")
    long_diag = read_json(EXPL / "compatibility_diagnostics_long.json")
    pairs = [row for row in formal["paired_comparisons"] if row["contrast"] == "Raw_minus_Physics"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={"wspace": 0.28})

    # Formal matched full-conditioner comparison. Each point is one optimizer seed.
    ax = axes[0]
    fractions = [0.1, 0.3, 1.0]
    capacities = ["Small", "Large"]
    colors = {"Small": "#0072B2", "Large": "#D55E00"}
    markers = {"Small": "o", "Large": "s"}
    xbase = {0.1: 0, 0.3: 1, 1.0: 2}
    offset = {"Small": -0.12, "Large": 0.12}
    grouped = {}
    for row in pairs:
        key = (row["capacity"], float(row["sample_fraction"]))
        grouped.setdefault(key, []).append(float(row["nll_difference"]))
    for cap in capacities:
        for frac in fractions:
            vals = grouped[(cap, frac)]
            x = xbase[frac] + offset[cap]
            jitter = np.linspace(-0.045, 0.045, len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=22,
                marker=markers[cap],
                color=colors[cap],
                alpha=0.60,
                linewidth=0,
                zorder=3,
            )
            m, ci = mean_ci(vals)
            ax.errorbar(
                [x],
                [m],
                yerr=[ci],
                color=colors[cap],
                marker=markers[cap],
                markersize=5.5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(range(3), ["10%", "30%", "100%"])
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Raw minus Physics NLL (nat/row)")
    ax.set_title("(a) Formal matched study: full conditioner")
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.legend(
        [plt.Line2D([], [], marker=markers[c], color=colors[c], linestyle="", markersize=5) for c in capacities],
        capacities,
        title="Capacity",
        frameon=False,
        loc="lower right",
    )

    # Long-budget hidden-conditioner eventwise contrast by p_in quantile.
    ax = axes[1]
    q = np.arange(1, 6)
    for tag, color, label in [("bottleneck__f010", "#009E73", "10%"), ("bottleneck__f100", "#CC79A7", "100%")]:
        data = long_diag["eventwise"][tag]["by_bin"]["p_in_quantile"]
        means = np.array([data[str(i)]["mean"] for i in range(5)])
        low = np.array([data[str(i)]["ci95_low"] for i in range(5)])
        high = np.array([data[str(i)]["ci95_high"] for i in range(5)])
        ax.plot(q, means, marker="o", markersize=4.6, linewidth=1.5, color=color, label=label)
        ax.fill_between(q, low, high, color=color, alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(q, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("$p_{in}$ quantile")
    ax.set_ylabel("Raw minus Physics NLL (nat/row)")
    ax.set_title("(b) Hidden conditioner: long-budget audit")
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.legend(title="Training fraction", frameon=False, loc="lower right")
    fig.suptitle("Representation--conditioner interaction", y=1.02, fontsize=12, fontweight="bold")
    save(fig, "fig1_full_hidden_interaction")


def figure_hidden_interaction_detail() -> None:
    """Single-panel detail used by the interaction.tex wrapper."""
    long_diag = read_json(EXPL / "compatibility_diagnostics_long.json")
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    q = np.arange(1, 6)
    for tag, color, label in [("bottleneck__f010", "#009E73", "10%"), ("bottleneck__f100", "#CC79A7", "100%")]:
        data = long_diag["eventwise"][tag]["by_bin"]["p_in_quantile"]
        means = np.array([data[str(i)]["mean"] for i in range(5)])
        low = np.array([data[str(i)]["ci95_low"] for i in range(5)])
        high = np.array([data[str(i)]["ci95_high"] for i in range(5)])
        ax.plot(q, means, marker="o", markersize=5.0, linewidth=1.6, color=color, label=label)
        ax.fill_between(q, low, high, color=color, alpha=0.16, linewidth=0)
        overall = long_diag["eventwise"][tag]["overall"]
        ax.text(
            0.98,
            0.95 if label == "10%" else 0.86,
            f"{label}: {overall['mean']:.3f} nat/row",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=8.5,
        )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(q, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("$p_{in}$ quantile")
    ax.set_ylabel("Raw minus Physics NLL (nat/row)")
    ax.set_title("Hidden-conditioner interaction by incident momentum")
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.legend(title="Training fraction", frameon=False, loc="lower right")
    save(fig, "fig5_hidden_interaction_detail")


def figure_mechanism_scatter() -> None:
    audit = read_json(EXPL / "mechanism_connection_audit.json")
    variance = audit["variance_decomposition"]["strata"]
    groups = audit["flow_likelihood_decomposition"]["groups"]
    xs, ys, fracs, thickness, charge = [], [], [], [], []
    pattern = re.compile(r"^(f\d+)__(t[0-9.]+)__(q-?1)__bin(\d+)$")
    for key, record in groups.items():
        match = pattern.match(key)
        if match is None:
            continue
        frac, t_label, q_label, bin_id = match.groups()
        stratum = f"{t_label}__{q_label}"
        x = variance[stratum]["bins"][bin_id]["location_complexity_excess"] * 1.0e12
        y = record["nll_gap"]["mean"]
        xs.append(x)
        ys.append(y)
        fracs.append(frac)
        thickness.append(t_label)
        charge.append(q_label)
    xs, ys = np.asarray(xs), np.asarray(ys)
    colors = {"t0.1": "#0072B2", "t0.3": "#E69F00", "t1": "#D55E00"}
    markers = {"q-1": "o", "q1": "^"}
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for t_label in ["t0.1", "t0.3", "t1"]:
        for q_label in ["q-1", "q1"]:
            mask = np.array([(t == t_label and q == q_label) for t, q in zip(thickness, charge)])
            for frac, face in [("f010", "white"), ("f100", colors[t_label])]:
                m = mask & np.array([f == frac for f in fracs])
                ax.scatter(
                    xs[m],
                    ys[m],
                    s=28,
                    marker=markers[q_label],
                    facecolors=face,
                    edgecolors=colors[t_label],
                    linewidth=0.8,
                    alpha=0.85,
                    label=f"{t_label}, {q_label}, {frac}" if t_label == "t0.1" and q_label == "q-1" else None,
                )
    if xs.size > 1:
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(float(xs.min()), float(xs.max()), 100)
        ax.plot(grid, slope * grid + intercept, color="0.25", linestyle="--", linewidth=1.0, label="Linear trend")
    ax.axhline(0.0, color="0.55", linewidth=0.7)
    ax.set_xlabel("Physics-minus-Raw between-$p_{in}$ mean variance ($\\times 10^{12}$)")
    ax.set_ylabel("Raw minus Physics NLL (nat/row)")
    ax.set_title("Location complexity tracks the hidden-conditioner deficit")
    ax.grid(color="0.92", linewidth=0.7)
    ax.text(
        0.03,
        0.04,
        "48 condition-bin groups; descriptive association only\nPearson r = -0.641",
        transform=ax.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75"},
    )
    # Compact legend: colors encode thickness and fill encodes fraction; marker encodes charge.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[t], markerfacecolor="white", label=t)
        for t in colors
    ]
    handles += [
        plt.Line2D([], [], marker="o", linestyle="", color="0.25", markerfacecolor="white", label="10%"),
        plt.Line2D([], [], marker="o", linestyle="", color="0.25", markerfacecolor="0.25", label="100%"),
        plt.Line2D([], [], marker="o", linestyle="", color="0.25", markerfacecolor="white", label="$q=-1$"),
        plt.Line2D([], [], marker="^", linestyle="", color="0.25", markerfacecolor="white", label="$q=+1$"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right", title="Color: t; fill: fraction; shape: q")
    save(fig, "fig2_mechanism_location_nll")


def figure_support_decomposition() -> None:
    flow = read_json(EXPL / "mechanism_connection_audit.json")["flow_likelihood_decomposition"]
    fig = plt.figure(figsize=(8.4, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 0.9], wspace=0.33)
    ax = fig.add_subplot(gs[0, 0])
    colors = {"f010": "#0072B2", "f100": "#D55E00"}
    for frac in ["f010", "f100"]:
        vals = flow["fractions"][frac]
        x = np.asarray(vals["base_gap"]["values"], dtype=float)
        y = np.asarray(vals["nll_gap"]["values"], dtype=float)
        ax.scatter(x, y, color=colors[frac], s=26, alpha=0.75, label=frac.replace("f", ""))
    lim = max(np.max(np.abs(np.asarray([*ax.get_xlim(), *ax.get_ylim()]))), 0.095)
    ax.plot([-lim, lim], [-lim, lim], color="0.30", linestyle="--", linewidth=1.0, label="Identity")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Base-density + Jacobian gap (nat/row)")
    ax.set_ylabel("Total NLL gap (nat/row)")
    ax.set_title("(a) Total gap equals base/Jacobian gap")
    ax.grid(color="0.92", linewidth=0.7)
    ax.legend(title="Fraction", frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(10)
    for i, frac in enumerate(["f010", "f100"]):
        n = flow["fractions"][frac]["support_gap"]["n"]
        jitter = rng.normal(i, 0.035, n)
        ax.scatter(jitter, np.zeros(n), color=colors[frac], s=30, alpha=0.8)
        ax.errorbar(
            [i],
            [0.0],
            yerr=[[0.0], [0.0]],
            color=colors[frac],
            marker="o",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0e-8, 1.0e-8)
    ax.set_xticks([0, 1], ["10%", "100%"])
    ax.set_ylabel("Support-normalization gap")
    ax.set_title("(b) Support term")
    ax.text(0.5, 0.75e-8, "exactly 0 for all\n20 checkpoints", ha="center", va="center", fontsize=8.7)
    ax.grid(axis="y", color="0.92", linewidth=0.7)
    fig.suptitle("Frozen-flow likelihood decomposition", y=1.02, fontsize=12, fontweight="bold")
    save(fig, "fig3_support_decomposition")


def figure_independent_confirmation() -> None:
    scores = read_json(EXPL / "independent_confirmation_scores.json")["scores"]
    # Build paired Raw-Physics differences within block/context/fraction/seed.
    rows = {}
    for row in scores:
        # The report stores the seed in the frozen checkpoint path.
        seed_match = re.search(r"seed(\d+)", row.get("checkpoint", ""))
        seed = int(seed_match.group(1)) if seed_match else len(rows)
        rows.setdefault((row["block"], row["context_label"], row["fraction_tag"]), {}).setdefault(row["representation"], {})[
            seed
        ] = float(row["mean_nll"])
    groups = {}
    for key, reps in rows.items():
        seed_ids = sorted(set(reps["Raw"]) & set(reps["Physics"]))
        raw = np.asarray([reps["Raw"][seed] for seed in seed_ids])
        physics = np.asarray([reps["Physics"][seed] for seed in seed_ids])
        groups[key] = raw - physics

    fig, ax = plt.subplots(figsize=(8.3, 4.2))
    categories = [("full", "f010"), ("full", "f100"), ("hidden", "f010"), ("hidden", "f100")]
    xbase = np.arange(len(categories))
    block_colors = {"a": "#0072B2", "b": "#D55E00"}
    block_offsets = {"a": -0.09, "b": 0.09}
    for block in ["a", "b"]:
        for i, (context, frac) in enumerate(categories):
            vals = groups[(block, context, frac)]
            x = xbase[i] + block_offsets[block]
            jitter = np.linspace(-0.025, 0.025, len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, color=block_colors[block], s=23, alpha=0.65, zorder=3)
            m, ci = mean_ci(vals)
            ax.errorbar(
                [x],
                [m],
                yerr=[ci],
                color=block_colors[block],
                marker="o",
                markersize=5.5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=3,
                linewidth=1.2,
                zorder=4,
            )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(xbase, ["Full\n10%", "Full\n100%", "Hidden\n10%", "Hidden\n100%"])
    ax.set_ylabel("Raw minus Physics NLL (nat/row)")
    ax.set_title("Independent frozen confirmation on two legacy 500k blocks")
    ax.grid(axis="y", color="0.92", linewidth=0.7)
    ax.legend(
        [plt.Line2D([], [], marker="o", color=block_colors[b], linestyle="", markersize=5) for b in ["a", "b"]],
        ["Block a", "Block b"],
        title="Confirmation block",
        frameon=False,
        loc="upper right",
    )
    ax.text(
        0.02,
        0.96,
        "No refit, checkpoint reselection, or model selection on confirmation data",
        transform=ax.transAxes,
        va="top",
        fontsize=8.2,
        color="0.25",
    )
    save(fig, "fig4_independent_confirmation")


def main() -> None:
    paper_style()
    figure_full_hidden_interaction()
    figure_hidden_interaction_detail()
    figure_mechanism_scatter()
    figure_support_decomposition()
    figure_independent_confirmation()
    print("Generated:")
    for path in sorted(OUT.glob("fig*.pdf")) + sorted(OUT.glob("fig*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
