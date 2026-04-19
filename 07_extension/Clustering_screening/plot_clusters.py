import pandas as pd
import matplotlib.pyplot as plt

'''
Plot the 24 hour average profiles of all households within each cluster, for k=2 and k=3.
'''

# Load raw 24D profiles
hourly_profiles = pd.read_csv("household_24h_train_profiles_raw.csv")
features = [f"h{i:02d}" for i in range(24)]

for k in [2, 3]:
    assignments = pd.read_csv(f"kmeans_assignments_rowz_k{k}.csv")
    df_merged = hourly_profiles.merge(assignments, on="house_id")

    cluster_ids = sorted(df_merged["cluster"].unique())
    n_clusters = len(cluster_ids)

    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_df = df_merged[df_merged["cluster"] == cluster_id]

        # plot each household profile faintly
        for _, row in cluster_df.iterrows():
            ax.plot(range(24), row[features].values, alpha=0.2, linewidth=1)

        # plot cluster mean profile
        mean_profile = cluster_df[features].mean()
        ax.plot(range(24), mean_profile.values, linewidth=3, label=f"Mean profile")

        ax.set_title(f"k={k} | Cluster {cluster_id} (n={len(cluster_df)})")
        ax.set_xlabel("Hour of Day")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Scaled kWh Usage")
    fig.suptitle(f"Household Profiles Within Each Cluster for k={k}")
    fig.tight_layout()
    fig.savefig(f"cluster_household_overlay_k{k}.png", dpi=150)
    plt.close(fig)