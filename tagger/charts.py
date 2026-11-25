"""One shared look for every report chart (so figures in the report match each other)."""

import matplotlib

matplotlib.use("Agg")            # draw to files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402

# Fixed colour order: a colourblind-checked categorical palette. Series always take colours
# in this order, so "train" or "tfidf" has the same colour in every chart.
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MODEL_COLORS = {"keyword": COLORS[0], "tfidf": COLORS[1], "semantic": COLORS[2], "hybrid": COLORS[6]}
TEXT = "#191919"
MUTED = "#666666"
GRID = "#e5e5e5"


def new_figure(width=8, height=5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.set_facecolor("white")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig, path, title):
    fig.axes[0].set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"   chart saved: {path.name}")
