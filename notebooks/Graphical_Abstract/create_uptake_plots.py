"""
This script loads in the data from the MBP uptake experiments and creates a super minimal plot of uptake for use in the graphical abstract.

The two plots are:

Uptake, coloured by residue index (tab colours)
Uptake, coloured by train/val split (green/orange)

The data is loaded from the MBP uptake experiments, which are stored in the following format:

ResStr ResEnd 30 240 1800 14400
19 30 0.39411 0.66768 0.90249 1.0
19 31 0.3762 0.63567 0.86921 1.0
24 35 0.15136 0.2211 0.46211 1.0
31 53 0.36285 0.47337 0.69324 1.0

for clarity, only select a certain number of entries to plot - keep these the same for both plots

HDX_data = "MBP_wt1.dat"

"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 30,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)
# Set the file path
HDX_data = "BRD4_APO.dat"
data_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
file_path = os.path.join(data_dir, HDX_data)

# Create output directory if it doesn't exist
output_dir = "/home/alexi/Documents/interpretable-hdxer/notebooks/Graphical_Abstract/output"
os.makedirs(output_dir, exist_ok=True)


# Load data
def load_hdx_data(file_path):
    # Read data with space delimiter, skipping the first line (header)
    data = pd.read_csv(file_path, sep="\s+", header=None, skiprows=1)
    # Rename columns
    data.columns = ["ResStr", "ResEnd", "0.0", "15.0", "60.0", "600.0", "3600.0", "14400.0"]

    return data


# Load the data
df = load_hdx_data(file_path)

# Define time points for x-axis
time_points = [1.0, 15.0, 60.0, 600.0, 3600.0, 14400.0]

time_columns = ["0.0", "15.0", "60.0", "600.0", "3600.0", "14400.0"]


# Instead of random sampling, select diverse peptides based on uptake patterns
def select_diverse_peptides(df, n_samples=10):
    # Extract uptake data for clustering
    uptake_data = df[time_columns].values

    # Normalize the data to focus on pattern rather than absolute values
    scaler = StandardScaler()
    normalized_uptake = scaler.fit_transform(uptake_data)

    # Cluster the peptides based on uptake patterns
    kmeans = KMeans(n_clusters=n_samples, random_state=42)
    clusters = kmeans.fit_predict(normalized_uptake)

    # Select one peptide from each cluster (closest to centroid)
    selected_indices = []
    for i in range(n_samples):
        cluster_points = np.where(clusters == i)[0]
        if len(cluster_points) > 0:
            # Find point closest to cluster centroid
            cluster_center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(normalized_uptake[cluster_points] - cluster_center, axis=1)
            closest_point_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(closest_point_idx)

    # If we don't have enough samples (because some clusters might be empty),
    # fill in with random selections from the largest clusters
    while len(selected_indices) < n_samples:
        cluster_sizes = [np.sum(clusters == i) for i in range(n_samples)]
        largest_cluster = np.argmax(cluster_sizes)
        remaining_points = [
            p for p in np.where(clusters == largest_cluster)[0] if p not in selected_indices
        ]
        if remaining_points:
            selected_indices.append(np.random.choice(remaining_points))
        else:
            # If largest cluster is exhausted, just add random points
            remaining_all = [i for i in range(len(df)) if i not in selected_indices]
            if remaining_all:
                selected_indices.append(np.random.choice(remaining_all))
            else:
                break  # Can't select more points

    return selected_indices


# Select diverse peptides
sample_indices = select_diverse_peptides(df, n_samples=10)
df_sample = df.iloc[sample_indices]

# Calculate midpoint of each peptide for plotting
df_sample["Midpoint"] = (df_sample["ResStr"] + df_sample["ResEnd"]) / 2

# Convert wide format to long format for easier plotting
melted_df = pd.melt(
    df_sample,
    id_vars=["ResStr", "ResEnd", "Midpoint"],
    value_vars=time_columns,
    var_name="Time",
    value_name="Uptake",
)

# Convert time labels to numeric values
melted_df["Time"] = melted_df["Time"].map(lambda x: int(float(x)))

# Plot 1: Uptake colored by residue index
plt.figure(figsize=(8, 4))
norm = Normalize(vmin=df_sample["Midpoint"].min(), vmax=df_sample["Midpoint"].max())
cmap = get_cmap("tab10")  # Using tab colors as mentioned

for i, (idx, peptide) in enumerate(df_sample.iterrows()):
    color = cmap(norm(peptide["Midpoint"]))
    plt.plot(
        time_points,
        [
            peptide["0.0"],
            peptide["15.0"],
            peptide["60.0"],
            peptide["600.0"],
            peptide["3600.0"],
            peptide["14400.0"],
        ],
        color=color,
        alpha=0.7,
        # marker="o",
    )
plt.xscale("log")
plt.xlabel("Time (mins)", fontsize=20)
plt.ylabel("Deuterium Uptake", fontsize=20)
plt.title("HDX Uptake by Peptide", fontsize=24)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "uptake_by_residue.png"), dpi=300)
plt.close()

# Plot 2: Uptake colored by train/val split
plt.figure(figsize=(6, 2))

# Define a simple train/val split - first 70% train, last 30% val
train_val_threshold = int(0.5 * len(df_sample))

for i, (idx, peptide) in enumerate(df_sample.iterrows()):
    color = "green" if i < train_val_threshold else "orange"  # Green for train, orange for val
    plt.plot(
        time_points,
        [
            peptide["0.0"],
            peptide["15.0"],
            peptide["60.0"],
            peptide["600.0"],
            peptide["3600.0"],
            peptide["14400.0"],
        ],
        color=color,
        alpha=0.7,
        # marker="o",
    )

# Add legend
plt.plot([], [], color="green", label="Train")
plt.plot([], [], color="orange", label="Validation")
# plt.legend()

plt.xscale("log")
# plt.xlabel("Time (mins)", fontsize=14)
plt.ylabel("Uptake", fontsize=20)
# plt.title("HDX Uptake by Train/Val Split", fontsize=16)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "uptake_by_train_val.png"), dpi=300)
plt.close()

print("Plots saved to:", output_dir)
