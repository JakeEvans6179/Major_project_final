import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Data
# -----------------------------
data = [
    ("LSTM64_32", 0.3438),
    ("CNN-LSTM-Dense", 0.3439),
    ("CNN-LSTM", 0.3441),
    ("CNN-2LSTM-Dense", 0.3446),
    ("2CNN-LSTM", 0.3449),
    ("LSTM64_32-Dense", 0.3451),
    ("LSTM64-Dense", 0.3453),
    ("LSTM64", 0.3453),
    ("Seq2Seq", 0.3458),
    ("LSTM20_20", 0.3461),
    ("2CNN-LSTM-Dense", 0.3463),
    ("Dense", 0.3606),
]

df = pd.DataFrame(data, columns=["Model", "RMSE"])
df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)

shortlisted = ["LSTM64_32", "CNN-LSTM-Dense", "CNN-LSTM", "LSTM64-Dense"]
df_short = df[df["Model"].isin(shortlisted)].copy()
df_short = df_short.sort_values("RMSE", ascending=True).reset_index(drop=True)

# -----------------------------
# Plotting function
# -----------------------------
def make_barh_plot(dataframe, title, filename, shortlist_only=False):
    # make wider so labels do not get cut off
    fig, ax = plt.subplots(figsize=(9.2, 4.8 if len(dataframe) > 6 else 3.0))

    # colours
    if shortlist_only:
        colors = ["tab:blue"] * len(dataframe)
    else:
        colors = ["tab:blue" if m in shortlisted else "lightgray" for m in dataframe["Model"]]

    bars = ax.barh(
        dataframe["Model"],
        dataframe["RMSE"],
        color=colors,
        edgecolor="black",
        height=0.55
    )

    ax.invert_yaxis()

    # start x-axis from 0 as requested
    xmax = dataframe["RMSE"].max() + 0.03
    ax.set_xlim(0, xmax)

    ax.set_xlabel("RMSE")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # value labels
    for bar, value in zip(bars, dataframe["RMSE"]):
        ax.text(
            value + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=8
        )

    # stop truncation of long model names / labels
    plt.subplots_adjust(left=0.28, right=0.97, top=0.88, bottom=0.16)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# -----------------------------
# Plot 1: shortlisted 4 only
# -----------------------------
make_barh_plot(
    df_short,
    title="Centralised Screening: Shortlisted Models",
    filename="centralised_screening_shortlisted4.png",
    shortlist_only=True
)

# -----------------------------
# Plot 2: all models including Dense
# -----------------------------
make_barh_plot(
    df,
    title="Centralised Screening: All Models",
    filename="centralised_screening_all_models.png",
    shortlist_only=False
)