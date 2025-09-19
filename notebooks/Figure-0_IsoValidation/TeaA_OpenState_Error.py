"""
Updated HDXer autovalidation analysis script with improved styling and plotting.
Generates plots showing open state recovery and training metrics vs gamma parameters,
plus scatter plots between key metrics including KL divergence against uniform.
Now includes ensemble-specific trajectory analysis and volcano plots.
"""

import glob
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import rms
from scipy import stats

# Set publication-ready style
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 10,
    },
)

# Define color schemes from your styling
full_dataset_colours = {
    "AF2-MSAss": "navy",
    "AF2-Filtered": "dodgerblue",
    "MD-1Start": "fuschia",
    "MD-10Start": "orange",
    "MD-TFES": "purple",
    "ISO-BiModal": "indigo",
    "ISO-TriModal": "saddlebrown",
}

split_name_dataset_colours = {
    "Random": "fuchsia",
    "Sequence": "black",
    "Non-Redundant": "green",
    "Spatial": "grey",
}

split_type_dataset_colours = {"r": "fuchsia", "s": "black", "R3": "green", "Sp": "grey"}

split_name_mapping = {"r": "Random", "s": "Sequence", "R3": "Non-Redundant", "Sp": "Spatial"}
target_ratios = {"Open": 0.4, "Closed": 0.6}  # Target ratios for open/closed states

# Analysis configuration
ensemble_order = ["ISO-BiModal", "ISO-TriModal"]
analysis_split_types = ["R3", "Sp", "r", "s"]  # Split types to include in analysis and plots

selected_dataset_colours = {
    k: full_dataset_colours[k] for k in ensemble_order if k in full_dataset_colours
}
split_name_dataset_order = ["Non-Redundant", "Spatial", "Random", "Sequence"]

# Define paths
base_dir = "/home/alexi/Documents/interpretable-hdxer/data/fig0/RW_recovery/RW_bench"
output_dir = os.path.join(base_dir, "analysis_curve_output")

# Reference structure paths (adjust these to your actual paths)
open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path

# Ensemble-specific trajectory paths
trajectory_paths = {
    "ISO-BiModal": "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc",
    "ISO-TriModal": "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_initial_sliced.xtc",
}


def compute_kl_divergence_uniform(weights, epsilon=1e-10):
    """
    Compute KL divergence between weights distribution and uniform distribution.

    KL(P||U) = Σ p_i * log(p_i / u_i)
    where P is the empirical distribution (weights) and U is uniform distribution.

    Parameters:
    -----------
    weights : np.ndarray
        Normalized weights (should sum to 1)
    epsilon : float
        Small value to add to weights to avoid log(0)

    Returns:
    --------
    float
        KL divergence value
    """
    # Ensure weights are normalized
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        return np.nan

    # Add epsilon to avoid log(0) and remove zero weights
    weights_safe = weights + epsilon
    weights_safe = weights_safe / np.sum(weights_safe)  # Renormalize

    # Uniform distribution
    n = len(weights)
    uniform_prob = 1.0 / n

    # Compute KL divergence
    kl_div = np.sum(weights_safe * np.log(weights_safe / uniform_prob))

    return kl_div


def parse_directory_structure(base_dir):
    """
    Parse the directory structure to extract ensemble, split type, and replicate information.

    Returns:
    --------
    experiment_dirs : list
        List of dictionaries containing experiment information
    """
    experiment_dirs = []

    # Pattern: TeaA_ISO_{ensemble}_RW_bench_{split}_{split_detail}
    for dir_path in glob.glob(os.path.join(base_dir, "TeaA_ISO_*")):
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            parts = dir_name.split("_")

            if len(parts) >= 6:
                ensemble_part = parts[2]  # bi or tri
                split_part = parts[5]  # R3 or Sp

                ensemble = f"ISO-{ensemble_part.capitalize()}Modal"
                split_type = split_part
                split_name = split_name_mapping.get(split_part, split_part)

                # Find train and val directories
                train_dirs = glob.glob(os.path.join(dir_path, "train_*"))
                val_dirs = glob.glob(os.path.join(dir_path, "val_*"))

                for train_dir in train_dirs:
                    train_name = os.path.basename(train_dir)
                    replicate = train_name.split("_")[-1]

                    experiment_dirs.append(
                        {
                            "ensemble": ensemble,
                            "split_type": split_type,
                            "split_name": split_name,
                            "replicate": replicate,
                            "train_dir": train_dir,
                            "val_dir": None,  # Will be matched later
                            "base_name": dir_name,
                        }
                    )

                # Match validation directories
                for exp in experiment_dirs:
                    if exp["base_name"] == dir_name:
                        val_name = f"val_TeaA_ISO_{ensemble_part}_{exp['replicate']}"
                        val_path = os.path.join(dir_path, val_name)
                        if os.path.exists(val_path):
                            exp["val_dir"] = val_path

    return experiment_dirs


def extract_work_data(experiment_dirs):
    """
    Extract work and MSE data from work.dat files.

    Returns:
    --------
    work_df : pd.DataFrame
        DataFrame containing work and MSE data
    """
    work_data = []

    for exp in experiment_dirs:
        train_dir = exp["train_dir"]

        # Find all work.dat files
        work_files = glob.glob(os.path.join(train_dir, "*work.dat"))

        for work_file in work_files:
            filename = os.path.basename(work_file)
            # Parse gamma from filename: reweighting_gamma_5x10^0work.dat
            gamma_part = filename.replace("reweighting_gamma_", "").replace("work.dat", "")

            try:
                # Parse gamma value and exponent
                if "x10^" in gamma_part:
                    gamma_str, exp_str = gamma_part.split("x10^")
                    gamma_value = float(gamma_str)
                    exponent = int(exp_str)
                    gamma_numeric = gamma_value * (10**exponent)
                else:
                    gamma_numeric = float(gamma_part)
                    gamma_value = gamma_numeric
                    exponent = 0

                # Read work data
                if os.path.exists(work_file):
                    data = np.loadtxt(work_file)
                    if data.ndim == 1:
                        # Single line
                        gamma_file, mse, rmse, work = data
                        work_data.append(
                            {
                                "ensemble": exp["ensemble"],
                                "split_type": exp["split_type"],
                                "split_name": exp["split_name"],
                                "replicate": exp["replicate"],
                                "gamma": gamma_numeric,
                                "gamma_value": gamma_value,
                                "exponent": exponent,
                                "mse": mse,
                                "rmse": rmse,
                                "work_kj": work,
                                "train_dir": train_dir,
                            }
                        )
                    elif data.ndim == 2:
                        # Multiple lines (take last one)
                        gamma_file, mse, rmse, work = data[-1]
                        work_data.append(
                            {
                                "ensemble": exp["ensemble"],
                                "split_type": exp["split_type"],
                                "split_name": exp["split_name"],
                                "replicate": exp["replicate"],
                                "gamma": gamma_numeric,
                                "gamma_value": gamma_value,
                                "exponent": exponent,
                                "mse": mse,
                                "rmse": rmse,
                                "work_kj": work,
                                "train_dir": train_dir,
                            }
                        )
            except Exception as e:
                print(f"Error parsing {work_file}: {e}")
                continue

    return pd.DataFrame(work_data)


def extract_weights_and_compute_state_recovery(
    experiment_dirs, trajectory_paths, topology_path, reference_paths
):
    """
    Extract weights and compute open state recovery percentages normalized to target ratio.
    Also computes KL divergence against uniform distribution.
    Uses ensemble-specific trajectory files for accurate clustering.

    Returns:
    --------
    recovery_df : pd.DataFrame
        DataFrame containing state recovery and KL divergence data
    """
    # Pre-compute RMSD and cluster assignments for each ensemble
    ensemble_clustering = {}
    ensemble_unweighted_distributions = {}

    for ensemble, traj_path in trajectory_paths.items():
        print(f"Computing RMSD and clustering for {ensemble}...")
        try:
            rmsd_values = compute_rmsd_to_references(traj_path, topology_path, reference_paths)
            cluster_assignments = cluster_frames_by_rmsd(rmsd_values, threshold=1.0)
            ensemble_clustering[ensemble] = cluster_assignments

            # Calculate true unweighted distribution using uniform weights
            n_frames = len(cluster_assignments)
            uniform_weights = np.ones(n_frames) / n_frames

            n_clusters = 3  # open(0), closed(1), intermediate(2)
            unweighted_ratios = np.zeros(n_clusters)

            for cluster_idx in range(n_clusters):
                mask = cluster_assignments == cluster_idx
                unweighted_ratios[cluster_idx] = np.sum(uniform_weights[mask])

            # Store unweighted distribution
            ensemble_unweighted_distributions[ensemble] = {
                "open_ratio": unweighted_ratios[0],
                "closed_ratio": unweighted_ratios[1],
                "intermediate_ratio": unweighted_ratios[2],
                "open_percentage": unweighted_ratios[0] * 100,
                "closed_percentage": unweighted_ratios[1] * 100,
                "intermediate_percentage": unweighted_ratios[2] * 100,
            }

            # Count clusters
            unique, counts = np.unique(cluster_assignments, return_counts=True)
            cluster_summary = dict(zip(unique, counts))
            print(f"  {ensemble}: {len(cluster_assignments)} frames clustered")
            print(f"    Cluster distribution: {cluster_summary}")
            print(f"    Unweighted open state: {unweighted_ratios[0] * 100:.1f}%")
            print(f"    Unweighted closed state: {unweighted_ratios[1] * 100:.1f}%")
            print(f"    Unweighted intermediate: {unweighted_ratios[2] * 100:.1f}%")

        except Exception as e:
            raise RuntimeError(f"Error processing trajectory for {ensemble}: {e}")
            # Create dummy clustering and unweighted distribution as fallback
            ensemble_clustering[ensemble] = np.zeros(1000, dtype=int)
            ensemble_unweighted_distributions[ensemble] = {
                "open_ratio": 0.5,
                "closed_ratio": 0.3,
                "intermediate_ratio": 0.2,
                "open_percentage": 50.0,
                "closed_percentage": 30.0,
                "intermediate_percentage": 20.0,
            }
            print(f"  Using dummy clustering for {ensemble} (1000 frames, all cluster 0)")

    recovery_data = []

    for exp in experiment_dirs:
        train_dir = exp["train_dir"]
        ensemble = exp["ensemble"]

        # Get the cluster assignments and unweighted distribution for this ensemble
        if ensemble not in ensemble_clustering:
            print(f"Warning: No clustering data for ensemble {ensemble}, skipping...")
            continue

        cluster_assignments = ensemble_clustering[ensemble]
        unweighted_dist = ensemble_unweighted_distributions[ensemble]

        # Find all weight files
        weight_files = glob.glob(os.path.join(train_dir, "*final_weights.dat"))

        for weight_file in weight_files:
            filename = os.path.basename(weight_file)
            # Parse gamma from filename
            gamma_part = filename.replace("reweighting_gamma_", "").replace("final_weights.dat", "")

            try:
                # Parse gamma value and exponent
                if "x10^" in gamma_part:
                    gamma_str, exp_str = gamma_part.split("x10^")
                    gamma_value = float(gamma_str)
                    exponent = int(exp_str)
                    gamma_numeric = gamma_value * (10**exponent)
                else:
                    gamma_numeric = float(gamma_part)
                    gamma_value = gamma_numeric
                    exponent = 0

                # Read weights
                if os.path.exists(weight_file):
                    weights = np.loadtxt(weight_file)

                    # Handle NaN/inf values
                    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

                    # Normalize weights
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                    else:
                        continue

                    # Compute KL divergence against uniform
                    kl_div_uniform = compute_kl_divergence_uniform(weights)

                    # Compute state ratios
                    n_clusters = 3  # open(0), closed(1), intermediate(2)
                    cluster_ratios = np.zeros(n_clusters)

                    # Ensure weights match cluster assignments
                    if len(weights) != len(cluster_assignments):
                        print(
                            f"    Frame count mismatch for {ensemble}: weights={len(weights)}, clusters={len(cluster_assignments)}"
                        )
                        if len(weights) > len(cluster_assignments):
                            weights = weights[: len(cluster_assignments)]
                            print(f"      Truncated weights to {len(weights)} frames")
                        else:
                            padded_weights = np.zeros(len(cluster_assignments))
                            padded_weights[: len(weights)] = weights
                            weights = padded_weights
                            print(f"      Padded weights to {len(weights)} frames")

                        # Renormalize
                        if np.sum(weights) > 0:
                            weights = weights / np.sum(weights)

                    for cluster_idx in range(n_clusters):
                        mask = cluster_assignments == cluster_idx
                        cluster_ratios[cluster_idx] = np.sum(weights[mask])

                    # Calculate raw state percentages
                    raw_open_percentage = cluster_ratios[0] * 100
                    raw_closed_percentage = cluster_ratios[1] * 100
                    raw_intermediate_percentage = cluster_ratios[2] * 100

                    # Normalize to target open state recovery (0-100%)
                    target_open_percentage = target_ratios["Open"] * 100  # 40%
                    open_state_recovery = min(
                        100.0, (raw_open_percentage / target_open_percentage) * 100
                    )

                    recovery_data.append(
                        {
                            "ensemble": exp["ensemble"],
                            "split_type": exp["split_type"],
                            "split_name": exp["split_name"],
                            "replicate": exp["replicate"],
                            "gamma": gamma_numeric,
                            "gamma_value": gamma_value,
                            "exponent": exponent,
                            "open_state_recovery": open_state_recovery,
                            "raw_open_percentage": raw_open_percentage,
                            "closed_state_recovery": raw_closed_percentage,
                            "intermediate_state_recovery": raw_intermediate_percentage,
                            "kl_div_uniform": kl_div_uniform,
                            "train_dir": train_dir,
                            # Add unweighted distribution info for reference
                            "unweighted_open_percentage": unweighted_dist["open_percentage"],
                            "unweighted_closed_percentage": unweighted_dist["closed_percentage"],
                            "unweighted_intermediate_percentage": unweighted_dist[
                                "intermediate_percentage"
                            ],
                        }
                    )

            except Exception as e:
                print(f"Error processing {weight_file}: {e}")
                continue

    return pd.DataFrame(recovery_data)


def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
    """
    Compute RMSD values between each frame in the trajectory and reference structures.
    """
    try:
        traj = mda.Universe(topology_path, trajectory_path)
        n_frames = len(traj.trajectory)
        n_refs = len(reference_paths)
        rmsd_values = np.zeros((n_frames, n_refs))

        for j, ref_path in enumerate(reference_paths):
            mobile = mda.Universe(topology_path, trajectory_path)
            reference = mda.Universe(ref_path)

            mobile_ca = mobile.select_atoms("name CA")
            ref_ca = reference.select_atoms("name CA")

            if len(ref_ca) != len(mobile_ca):
                print(
                    f"Warning: CA atom count mismatch - Trajectory: {len(mobile_ca)}, Reference {j}: {len(ref_ca)}"
                )

            R = rms.RMSD(mobile, reference, select="name CA", ref_frame=0)
            R.run()
            rmsd_values[:, j] = R.rmsd[:, 2]

        return rmsd_values
    except Exception as e:
        raise RuntimeError(
            f"Error computing RMSD: {e}. Ensure trajectory and reference files are valid."
        )


def cluster_frames_by_rmsd(rmsd_values, threshold=1.0):
    """
    Cluster frames based on RMSD to reference structures.

    Args:
        rmsd_values (np.ndarray): RMSD values to reference structures (n_frames, n_refs)
        threshold (float): RMSD threshold for clustering

    Returns:
        np.ndarray: Cluster assignments (0 = open-like, 1 = closed-like, 2 = intermediate)
    """
    # Simple clustering: assign to closest reference if within threshold
    cluster_assignments = np.argmin(rmsd_values, axis=1)

    # Check if frames are within threshold of any reference
    min_rmsd = np.min(rmsd_values, axis=1)
    valid_clusters = min_rmsd <= threshold

    # Set invalid clusters to intermediate state (2)
    cluster_assignments[~valid_clusters] = 2

    return cluster_assignments


def perform_ttest(data, ensemble1, ensemble2, split_type, metric):
    """
    Perform t-test between two ensembles for a given split type.
    """
    group1 = data[(data["ensemble"] == ensemble1) & (data["split_type"] == split_type)][
        metric
    ].dropna()
    group2 = data[(data["ensemble"] == ensemble2) & (data["split_type"] == split_type)][
        metric
    ].dropna()

    if len(group1) > 0 and len(group2) > 0:
        t_stat, p_val = stats.ttest_ind(group1, group2)
        return t_stat, p_val
    else:
        return float("nan"), float("nan")


def plot_mean_std_by_group(data, x, y, group_col, color_dict, ax=None):
    """
    Plot mean values as lines with standard deviation bands.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    x : str
        Column name for x-axis values
    y : str
        Column name for y-axis values
    group_col : str
        Column name for grouping data
    color_dict : dict
        Dictionary mapping group values to colors
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    """
    if ax is None:
        ax = plt.gca()

    for group_val in data[group_col].unique():
        subset = data[data[group_col] == group_val]
        if subset.empty:
            continue

        # Calculate mean and std for each x value
        grouped = subset.groupby(x)[y]
        means = grouped.mean()
        stds = grouped.std().fillna(0)  # Replace NaN with 0 for std
        x_vals = np.array(means.index)

        # Sort by x values for proper line plotting
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = np.array(means.values)[sort_idx]
        std_vals = np.array(stds.values)[sort_idx]

        # Plot mean line
        ax.plot(
            x_vals,
            y_vals,
            color=color_dict[group_val],
            alpha=0.8,
            label=split_name_mapping.get(group_val, group_val),
            linewidth=2,
        )

        # Plot std band
        ax.fill_between(
            x_vals, y_vals - std_vals, y_vals + std_vals, alpha=0.2, color=color_dict[group_val]
        )

    return ax


def plot_training_metrics(work_df, recovery_df, output_dir):
    """
    Plot training MSE, Work, and KL divergence against gamma with line plots and standard deviation bands.
    Creates separate plots for each metric.
    """
    # Merge work and recovery data to get KL divergence
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(
        work_df, recovery_df[merge_cols + ["kl_div_uniform"]], on=merge_cols, how="left"
    )

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for training metrics plot")
        return None

    metrics = [
        {"name": "mse", "title": r"MSE$_{Training}$", "ylabel": r"MSE$_{Training}$"},
        {
            "name": "work_kj",
            "title": r"Apparent Work$_{HDXer}$",
            "ylabel": r"Apparent Work$_{HDXer}$ [kJ/mol]",
        },
        {
            "name": "kl_div_uniform",
            "title": r"KL Divergence vs Uniform",
            "ylabel": r"KL(P||U$_{uniform}$)",
        },
    ]

    figures = []

    # Create separate plots for each metric
    for metric in metrics:
        # Create plot with ensembles in columns, sharing y and x axes
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharey=True, sharex=True
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])  # Ensure axes is iterable

        for col_idx, ensemble in enumerate(ensemble_order):
            ensemble_data = plot_df[plot_df["ensemble"] == ensemble].copy()
            ax = axes[col_idx]

            # If plotting work, ensure non-positive values are replaced with a small epsilon
            if metric["name"] == "work_kj":
                eps = 1e-6
                # Replace non-positive work values before aggregation/plotting to avoid log-scale issues
                ensemble_data["work_kj"] = ensemble_data["work_kj"].apply(
                    lambda v: v if v > 0 else eps
                )

            # Plot metric vs gamma
            plot_mean_std_by_group(
                ensemble_data, "gamma", metric["name"], "split_type", split_type_dataset_colours, ax
            )

            # If plotting work, set y-axis to log scale
            if metric["name"] == "work_kj":
                ax.set_yscale("log")

            ax.set_xlabel(r"$\gamma_{HDXer}$ Parameter", fontsize=20)
            if col_idx == 0:  # Only set y-label for the first subplot in the row
                ax.set_ylabel(metric["ylabel"], fontsize=18)
            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.grid(False, alpha=0)

        # Add legend to the figure
        handles = []
        labels = []
        for split_type in analysis_split_types:
            color = split_type_dataset_colours[split_type]
            handles.append(plt.Line2D([0], [0], color=color, linewidth=2, alpha=0.8))
            labels.append(split_name_mapping[split_type])

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.8,
        )

        # fig.suptitle(metric["title"], fontsize=22)
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Set left=0.2

        # Save the figure
        output_path = os.path.join(output_dir, f"training_{metric['name']}_vs_gamma.png")
        fig.savefig(output_path, dpi=300)
        print(f"Saved {metric['title']} plot to {output_path}")

        figures.append(fig)

    return figures


def plot_scatter_metrics(work_df, recovery_df, output_dir):
    """
    Plot scatter plots between key metrics with statistical annotations.
    Creates scatter plots including KL divergence:
    1. Normalized Open state recovery % vs Apparent Work
    2. MSE Training vs Apparent Work
    3. MSE Training vs Normalized Open state recovery %
    4. KL Divergence vs Apparent Work
    5. KL Divergence vs Open State Recovery
    6. KL Divergence vs MSE Training
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for scatter plots")
        return None

    # Define scatter plot pairs
    scatter_pairs = [
        {
            "x": "open_state_recovery",
            "y": "work_kj",
            "x_label": "Open State Recovery (%)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs Open State Recovery",
            "filename": "scatter_work_vs_recovery.png",
        },
        {
            "x": "work_kj",
            "y": "open_state_recovery",
            "x_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs Apparent Work",
            "filename": "scatter_recovery_vs_work.png",
        },
        {
            "x": "mse",
            "y": "work_kj",
            "x_label": r"MSE$_{Training}$",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs MSE Training",
            "filename": "scatter_work_vs_mse.png",
        },
        {
            "x": "mse",
            "y": "open_state_recovery",
            "x_label": r"MSE$_{Training}$",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs MSE Training",
            "filename": "scatter_recovery_vs_mse.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "work_kj",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs KL Divergence",
            "filename": "scatter_work_vs_kl.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "open_state_recovery",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs KL Divergence",
            "filename": "scatter_recovery_vs_kl.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "mse",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"MSE$_{Training}$",
            "title": "MSE Training vs KL Divergence",
            "filename": "scatter_mse_vs_kl.png",
        },
        {
            "x": "open_state_recovery",
            "y": "mse",
            "x_label": "Open State Recovery (%)",
            "y_label": r"MSE$_{Training}$",
            "title": "MSE Training vs Open State Recovery",
            "filename": "scatter_mse_vs_recovery.png",
        },
    ]

    # Define consistent styles for split types
    split_styles = {}
    for split_type in analysis_split_types:
        split_styles[split_type] = {"marker": "o", "color": split_type_dataset_colours[split_type]}

    figures = []

    for pair in scatter_pairs:
        # Create regular subplots with shared y-axis for better alignment
        fig, axes = plt.subplots(
            1,
            len(ensemble_order),
            figsize=(6 * len(ensemble_order), 4),
            sharey=True,  # This ensures y-axes are aligned
            sharex=False,
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])  # Ensure axes is iterable

        for col_idx, ensemble in enumerate(ensemble_order):
            ax = axes[col_idx]
            ensemble_data = plot_df[plot_df["ensemble"] == ensemble]

            # Plot scatter points for each split type
            for split_type, style in split_styles.items():
                subset = ensemble_data[ensemble_data["split_type"] == split_type]
                if subset.empty:
                    continue

                # Prepare x and y values and filter out non-positive work values when necessary
                x_vals = subset[pair["x"]].to_numpy()
                y_vals = subset[pair["y"]].to_numpy()
                mask = np.ones(len(x_vals), dtype=bool)

                # If either axis is work_kj, filter non-positive values to avoid log scale errors and meaningless points
                if pair["x"] == "work_kj":
                    mask &= x_vals > 0
                if pair["y"] == "work_kj":
                    mask &= y_vals > 0

                if not np.any(mask):
                    continue

                x_plot = x_vals[mask]
                y_plot = y_vals[mask]

                ax.scatter(
                    x_plot,
                    y_plot,
                    marker=style["marker"],
                    color=style["color"],
                    s=100,
                    label=split_name_mapping[split_type],
                    alpha=0.6,
                )

            # Add correlation annotations per split_type using the same filtering logic
            for split_type, style in split_styles.items():
                subset = ensemble_data[ensemble_data["split_type"] == split_type]
                if subset.empty:
                    continue

                x_vals = subset[pair["x"]].to_numpy()
                y_vals = subset[pair["y"]].to_numpy()
                mask = np.ones(len(x_vals), dtype=bool)
                if pair["x"] == "work_kj":
                    mask &= x_vals > 0
                if pair["y"] == "work_kj":
                    mask &= y_vals > 0

                if np.sum(mask) > 1:
                    # Prepare arrays for correlation; transform work values to log10 for fitting
                    x_corr = x_vals[mask].astype(float).copy()
                    y_corr = y_vals[mask].astype(float).copy()

                    if pair["x"] == "work_kj":
                        x_corr = np.log10(x_corr)
                    if pair["y"] == "work_kj":
                        y_corr = np.log10(y_corr)

                    # Ensure finite values remain
                    finite_mask = np.isfinite(x_corr) & np.isfinite(y_corr)
                    if np.sum(finite_mask) > 1:
                        try:
                            corr = np.corrcoef(x_corr[finite_mask], y_corr[finite_mask])[0, 1]
                        except Exception:
                            corr = float("nan")

                        # Compute linear fit (slope) on the same transformed data used for correlation
                        try:
                            if np.sum(finite_mask) > 1:
                                slope, intercept = np.polyfit(
                                    x_corr[finite_mask], y_corr[finite_mask], 1
                                )
                            else:
                                slope = float("nan")
                        except Exception:
                            slope = float("nan")
                    else:
                        corr = float("nan")
                        slope = float("nan")
                else:
                    corr = float("nan")
                    slope = float("nan")

                # Add annotation including Pearson R and linear-fit slope (gradient)
                y_pos = 0.9 - (0.1 * list(split_styles.keys()).index(split_type))
                slope_text = "nan" if not np.isfinite(slope) else f"{slope:.2e}"
                ax.annotate(
                    f"{split_name_mapping[split_type]} R = {np.nan_to_num(corr, nan=0.0):.3f}, slope = {slope_text}",
                    xy=(0.02, y_pos),
                    xycoords="axes fraction",
                    fontsize=10,
                    color=style["color"],
                    bbox=dict(
                        facecolor="white",
                        edgecolor=style["color"],
                        alpha=0.7,
                        boxstyle="round,pad=0.5",
                    ),
                )

            # Set titles and labels
            ax.set_title(
                f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20, pad=0
            )
            ax.set_xlabel(pair["x_label"], fontsize=20)

            # Only set y-label for the first subplot
            if col_idx == 0:
                ax.set_ylabel(pair["y_label"], fontsize=18)

            # Set axis scales: log scale if work_kj is on either axis
            if pair["y"] == "work_kj":
                ax.set_yscale("log")
            if pair["x"] == "work_kj":
                ax.set_xscale("log")

            ax.tick_params(axis="both", which="major")
            ax.grid(False, alpha=0)

            # Set y-axis limits for recovery plots
            if pair["y"] == "open_state_recovery":
                ax.set_ylim(0, 100)
            if pair["x"] == "open_state_recovery":
                ax.set_xlim(0, 100)

        # Create legend
        handles = []
        labels = []
        for split_type, style in split_styles.items():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="None",
                    markersize=10,
                    alpha=0.6,
                )
            )
            labels.append(split_name_mapping[split_type])

        # Add legend to the figure
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.6,
        )

        # Apply consistent layout adjustment - same as your training metrics
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Same rect as training metrics

        # Save figure
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300)
        print(f"Saved {pair['title']} scatter plot to {output_path}")

        figures.append(fig)

    return figures


def plot_volcano_recovery_vs_variables(work_df, recovery_df, output_dir):
    """
    Plot volcano plots with log2 fold change in recovery on y-axis vs other variables on x-axis.
    Creates separate plots for each variable: work, MSE, and KL divergence.
    Ensembles are shown side by side in panels, point sizes proportional to gamma values,
    with error bars and correlation annotations.

    Args:
        work_df (pd.DataFrame): Work data containing gamma parameters, MSE, and work values
        recovery_df (pd.DataFrame): Recovery data containing KL divergence and open state recovery
        output_dir (str): Output directory for plots

    Returns:
        list: List of matplotlib figures created
    """
    plt.style.use("default")
    sns.set_style("ticks")

    print("Creating volcano plots: Recovery fold change vs variables...")

    # Merge work and recovery dataframes
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for analysis ensembles and split types
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if len(plot_df) == 0:
        print("No matching data found for volcano plots")
        return []

    print(f"Merged {len(plot_df)} data points for volcano plots")

    # Calculate fold change relative to unweighted distribution for each combination
    fold_change_data = []

    for ensemble in plot_df["ensemble"].unique():
        for split_type in plot_df["split_type"].unique():
            for replicate in plot_df["replicate"].unique():
                subset = plot_df[
                    (plot_df["ensemble"] == ensemble)
                    & (plot_df["split_type"] == split_type)
                    & (plot_df["replicate"] == replicate)
                ]

                if len(subset) > 0:
                    # Get the true unweighted baseline for this ensemble
                    baseline_recovery = subset["unweighted_open_percentage"].iloc[0]

                    # Calculate fold change for each point relative to unweighted distribution
                    for _, row in subset.iterrows():
                        if baseline_recovery > 0:
                            fold_change = row["raw_open_percentage"] / baseline_recovery
                            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
                        else:
                            fold_change = 1.0
                            log2_fold_change = 0.0

                        fold_change_data.append(
                            {
                                **row.to_dict(),
                                "recovery_fold_change": fold_change,
                                "log2_recovery_fold_change": log2_fold_change,
                                "baseline_recovery": baseline_recovery,
                            }
                        )

    if not fold_change_data:
        print("No fold change data could be calculated for volcano plots")
        return []

    volcano_df = pd.DataFrame(fold_change_data)
    print(f"Calculated fold changes for {len(volcano_df)} data points")

    # Calculate averages and standard deviations across replicates
    grouping_cols = ["ensemble", "split_type", "split_name", "gamma"]

    averaged_df = (
        volcano_df.groupby(grouping_cols)
        .agg(
            {
                "log2_recovery_fold_change": ["mean", "std", "count"],
                "work_kj": ["mean", "std", "count"],
                "mse": ["mean", "std", "count"],
                "kl_div_uniform": ["mean", "std", "count"],
                "open_state_recovery": ["mean", "std"],
                "raw_open_percentage": ["mean", "std"],
                "baseline_recovery": "first",
                "unweighted_open_percentage": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    averaged_df.columns = [
        "ensemble",
        "split_type",
        "split_name",
        "gamma",
        "log2_recovery_fold_change_mean",
        "log2_recovery_fold_change_std",
        "log2_recovery_fold_change_count",
        "work_kj_mean",
        "work_kj_std",
        "work_kj_count",
        "mse_mean",
        "mse_std",
        "mse_count",
        "kl_div_uniform_mean",
        "kl_div_uniform_std",
        "kl_div_uniform_count",
        "open_state_recovery_mean",
        "open_state_recovery_std",
        "raw_open_percentage_mean",
        "raw_open_percentage_std",
        "baseline_recovery",
        "unweighted_open_percentage",
    ]

    # Fill NaN standard deviations with 0 (for cases with only 1 replicate)
    for col in averaged_df.columns:
        if col.endswith("_std"):
            averaged_df[col] = averaged_df[col].fillna(0)

    print(f"Averaged across replicates: {len(averaged_df)} unique parameter combinations")

    # Define the variable pairs for volcano plots
    volcano_pairs = [
        {
            "x_var": "work_kj_mean",
            "x_err": "work_kj_std",
            "x_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Recovery Fold Change vs Apparent Work",
            "filename": "volcano_recovery_vs_work.png",
            "x_scale": "log",
            "x_transform": "log10",  # For correlation calculation
        },
        {
            "x_var": "mse_mean",
            "x_err": "mse_std",
            "x_label": r"MSE$_{Training}$",
            "title": "Recovery Fold Change vs MSE Training",
            "filename": "volcano_recovery_vs_mse.png",
            "x_scale": "linear",
            "x_transform": None,
        },
        {
            "x_var": "kl_div_uniform_mean",
            "x_err": "kl_div_uniform_std",
            "x_label": r"KL(P||U$_{uniform}$)",
            "title": "Recovery Fold Change vs KL Divergence",
            "filename": "volcano_recovery_vs_kl.png",
            "x_scale": "linear",
            "x_transform": None,
        },
    ]

    # Size mapping for gamma values (higher gamma = larger points)
    gamma_values = sorted(averaged_df["gamma"].unique())
    max_size = 200
    min_size = 50

    size_map = {}
    for i, gamma in enumerate(gamma_values):
        size = min_size + (i / max(1, len(gamma_values) - 1)) * (max_size - min_size)
        size_map[gamma] = size

    figures = []

    for pair in volcano_pairs:
        print(f"Creating volcano plot: {pair['title']}")

        # Create figure with ensembles side by side
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(8 * len(ensemble_order), 6), sharey=True
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])

        fig.suptitle(
            f"{pair['title']}\n(vs Unweighted Distribution)",
            fontsize=20,
            fontweight="bold",
            y=0.95,
        )

        # Calculate global axis limits for consistency
        x_data = averaged_df[pair["x_var"]]
        y_data = averaged_df["log2_recovery_fold_change_mean"]

        # Handle work data (filter out non-positive values)
        if pair["x_var"] == "work_kj_mean":
            valid_mask = x_data > 0
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]

        if len(x_data) > 0 and len(y_data) > 0:
            x_margin = (x_data.max() - x_data.min()) * 0.1 if pair["x_scale"] == "linear" else None
            y_margin = (y_data.max() - y_data.min()) * 0.1

            if pair["x_scale"] == "linear":
                global_xlim = [x_data.min() - x_margin, x_data.max() + x_margin]
            else:
                global_xlim = [x_data.min() * 0.5, x_data.max() * 2.0]  # Log scale margins

            global_ylim = [y_data.min() - y_margin, y_data.max() + y_margin]
        else:
            global_xlim = [0, 1]
            global_ylim = [-1, 1]

        for i, ensemble in enumerate(ensemble_order):
            ax = axes[i]
            ensemble_data = averaged_df[averaged_df["ensemble"] == ensemble]

            if len(ensemble_data) > 0:
                # Calculate correlations for each split type
                correlation_annotations = []

                # Plot each split type with different colors
                for split_type in analysis_split_types:
                    split_data = ensemble_data[ensemble_data["split_type"] == split_type]

                    if len(split_data) > 0:
                        x_vals = split_data[pair["x_var"]].values
                        y_vals = split_data["log2_recovery_fold_change_mean"].values
                        x_errs = split_data[pair["x_err"]].values
                        y_errs = split_data["log2_recovery_fold_change_std"].values

                        # Filter out invalid values for work
                        if pair["x_var"] == "work_kj_mean":
                            valid_mask = x_vals > 0
                            x_vals = x_vals[valid_mask]
                            y_vals = y_vals[valid_mask]
                            x_errs = x_errs[valid_mask]
                            y_errs = y_errs[valid_mask]
                            split_data = split_data[valid_mask]

                        if len(x_vals) == 0:
                            continue

                        # Calculate correlation for this split type
                        if len(x_vals) > 1:
                            x_corr = x_vals.copy()

                            # Apply transformation for correlation if specified
                            if pair["x_transform"] == "log10":
                                x_corr = np.log10(x_corr)

                            try:
                                corr_coef = np.corrcoef(x_corr, y_vals)[0, 1]
                                correlation_annotations.append(
                                    f"{split_name_mapping[split_type]}: R = {corr_coef:.3f}"
                                )
                            except:
                                correlation_annotations.append(
                                    f"{split_name_mapping[split_type]}: R = NaN"
                                )
                        else:
                            correlation_annotations.append(
                                f"{split_name_mapping[split_type]}: R = NaN"
                            )

                        # Color by split type
                        color = split_type_dataset_colours[split_type]

                        # Sizes based on gamma value
                        sizes = [size_map.get(gamma, min_size) for gamma in split_data["gamma"]]

                        # Plot points with error bars
                        scatter = ax.scatter(
                            x_vals,
                            y_vals,
                            c=color,
                            s=sizes,
                            alpha=0.7,
                            edgecolors="black",
                            linewidth=0.5,
                            label=split_name_mapping[split_type],
                            zorder=3,
                        )

                        # Add error bars
                        ax.errorbar(
                            x_vals,
                            y_vals,
                            xerr=x_errs,
                            yerr=y_errs,
                            fmt="none",
                            ecolor="black",
                            alpha=0.3,
                            capsize=2,
                            capthick=1,
                            zorder=2,
                        )

                # Add reference lines
                ax.axhline(
                    y=0,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2,
                    label="Unweighted Distribution",
                    zorder=1,
                )

                # Add line at 100% recovery (log2 fold change depends on baseline)
                baseline_recovery = (
                    ensemble_data["baseline_recovery"].iloc[0] if len(ensemble_data) > 0 else 100
                )
                if baseline_recovery > 0:
                    target_fold_change = 100.0 / baseline_recovery  # 100% recovery
                    target_log2_fold_change = np.log2(target_fold_change)
                    ax.axhline(
                        y=target_log2_fold_change,
                        color="green",
                        linestyle=":",
                        alpha=0.8,
                        linewidth=2,
                        label="100% Recovery Target",
                        zorder=1,
                    )

                # Add correlation annotations
                for j, annotation in enumerate(correlation_annotations):
                    ax.text(
                        0.02,
                        0.98 - j * 0.08,
                        annotation,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

                # Set axis properties
                ax.set_xlabel(pair["x_label"], fontsize=16)
                if i == 0:
                    ax.set_ylabel("Log₂ Fold Change\n(Recovery vs Unweighted)", fontsize=16)

                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=18, pad=10
                )

                # Set axis scales and limits
                if pair["x_scale"] == "log":
                    ax.set_xscale("log")

                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)

                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=12)

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=18
                )

        # Create legend
        legend_elements = []

        # Split type colors
        for split_type in analysis_split_types:
            color = split_type_dataset_colours[split_type]
            legend_elements.append(
                plt.scatter(
                    [],
                    [],
                    c=color,
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                    label=split_name_mapping[split_type],
                )
            )

        # Reference lines
        legend_elements.extend(
            [
                plt.Line2D([0], [0], color="red", linestyle="--", label="Unweighted"),
                plt.Line2D([0], [0], color="green", linestyle=":", label="100% Recovery"),
            ]
        )

        # Add legend
        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.98, 0.85),
            loc="upper right",
            title="Split Type & References",
            title_fontsize=12,
            fontsize=11,
            frameon=True,
            framealpha=0.9,
        )

        # Add size legend as text
        size_text = "Point Size ∝ γ Value\n"
        size_text += f"Small: γ = {gamma_values[0]:.0e}\n"
        size_text += f"Large: γ = {gamma_values[-1]:.0e}"

        fig.text(
            0.02,
            0.15,
            size_text,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontsize=10,
            verticalalignment="bottom",
        )

        plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.92])

        # Save the plot
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Volcano plot saved to: {output_path}")

        figures.append(fig)

    # Save the averaged dataset
    output_csv = os.path.join(output_dir, "volcano_recovery_vs_variables_data.csv")
    averaged_df.to_csv(output_csv, index=False)
    print(f"Volcano plot dataset saved to: {output_csv}")

    # Print summary statistics
    print("\nVolcano Plots Summary:")
    print("-" * 40)
    print(f"Individual data points: {len(volcano_df)}")
    print(f"Averaged combinations: {len(averaged_df)}")
    print(f"Variables plotted: {len(volcano_pairs)}")
    print(f"Gamma values: {[f'{g:.0e}' for g in sorted(averaged_df['gamma'].unique())]}")

    fold_change_range = averaged_df["log2_recovery_fold_change_mean"]
    print(
        f"Recovery fold change range: {fold_change_range.min():.2f} to {fold_change_range.max():.2f}"
    )

    return figures


def extract_residue_uptake_data(experiment_dirs):
    """
    Extract residue uptake data from *SUMMARY_residue_fractions.dat files.
    """
    uptake_data = []
    for exp in experiment_dirs:
        train_dir = exp["train_dir"]
        train_name = os.path.basename(train_dir)
        uptake_file = os.path.join(train_dir, f"out__{train_name}SUMMARY_residue_fractions.dat")
        if os.path.exists(uptake_file):
            try:
                # Read the space-separated file, skipping comment lines
                df = pd.read_csv(
                    uptake_file,
                    delim_whitespace=True,
                    comment="#",
                    header=None,
                    names=["ResID", "0.167", "1.0", "10.0", "60.0", "120.0"],
                )

                # Rename ResID column to residue (no need for reset_index)
                df = df.rename(columns={"ResID": "residue"})

                # Melt to long format
                long_df = df.melt(id_vars=["residue"], var_name="time", value_name="uptake")
                long_df["time"] = long_df["time"].astype(float)
                long_df["residue"] = long_df["residue"].astype(int)

                # Add metadata
                long_df["ensemble"] = exp["ensemble"]
                long_df["split_type"] = exp["split_type"]
                long_df["split_name"] = exp["split_name"]
                long_df["replicate"] = exp["replicate"]

                uptake_data.append(long_df)
            except Exception as e:
                print(f"Could not parse {uptake_file}: {e}")

    if not uptake_data:
        return pd.DataFrame()

    return pd.concat(uptake_data, ignore_index=True)


def plot_residue_uptake_comparison(uptake_df, output_dir):
    """
    Generates plots to compare residue uptake curves between ensembles.
    Fixed version to handle pandas compatibility issues.
    """
    if uptake_df.empty:
        print("No uptake data to plot.")
        return

    ensembles = uptake_df["ensemble"].unique()
    n_ensembles = len(ensembles)

    # Ensure data types are consistent and convert to numpy where needed
    uptake_df = uptake_df.copy()
    uptake_df["residue"] = uptake_df["residue"].astype(
        str
    )  # Convert to string to avoid indexing issues
    uptake_df["time"] = pd.to_numeric(uptake_df["time"], errors="coerce")
    uptake_df["uptake"] = pd.to_numeric(uptake_df["uptake"], errors="coerce")

    # Remove any rows with NaN values
    uptake_df = uptake_df.dropna(subset=["time", "uptake"])

    if uptake_df.empty:
        print("No valid uptake data after cleaning.")
        return

    # Plot 1: Uptake curves per residue, hued by residue
    # Limit number of residues to avoid overcrowding and plotting issues
    unique_residues = sorted(uptake_df["residue"].unique())
    max_residues = 500  # Limit to avoid overcrowding and potential plotting issues

    if len(unique_residues) > max_residues:
        selected_residues = unique_residues[:: len(unique_residues) // max_residues][:max_residues]
        plot_uptake_df = uptake_df[uptake_df["residue"].isin(selected_residues)].copy()
        print(
            f"Limiting to {len(selected_residues)} residues for visualization: {selected_residues}"
        )
    else:
        plot_uptake_df = uptake_df.copy()
        selected_residues = unique_residues

    fig1, axes1 = plt.subplots(1, n_ensembles, figsize=(6 * n_ensembles, 5), sharey=True)
    if n_ensembles == 1:
        axes1 = [axes1]
    fig1.suptitle("Residue Uptake Curves", fontsize=24)

    for ax, ensemble in zip(axes1, ensemble_order):
        if ensemble not in ensembles:
            continue
        ensemble_data = plot_uptake_df[plot_uptake_df["ensemble"] == ensemble]

        if ensemble_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ensemble, color=full_dataset_colours.get(ensemble, "black"))
            continue

        # Plot individual residues with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_residues)))
        for i, residue in enumerate(selected_residues):
            residue_data = ensemble_data[ensemble_data["residue"] == residue]
            if not residue_data.empty:
                # Convert to numpy arrays to avoid pandas indexing issues
                time_vals = np.array(residue_data["time"].values, dtype=float)
                uptake_vals = np.array(residue_data["uptake"].values, dtype=float)

                # Sort by time for proper line plotting
                sort_idx = np.argsort(time_vals)
                # Ensure sort_idx is numpy array and do explicit indexing
                sort_idx = np.array(sort_idx)
                time_sorted = time_vals[sort_idx]
                uptake_sorted = uptake_vals[sort_idx]

                ax.plot(
                    time_sorted,
                    uptake_sorted,
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1,
                    label=f"Res {residue}",
                )

        ax.set_title(ensemble, color=full_dataset_colours.get(ensemble, "black"))
        ax.set_xscale("log")
        ax.set_xlabel("Time (s)")
        if ax == axes1[0]:
            ax.set_ylabel("Fractional Uptake")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(os.path.join(output_dir, "residue_uptake_curves.png"), dpi=300)
    print("Saved residue uptake curves plot.")
    plt.close(fig1)

    # Plot 2: Average uptake curves with SEM across residues
    fig2, axes2 = plt.subplots(1, n_ensembles, figsize=(6 * n_ensembles, 5), sharey=True)
    if n_ensembles == 1:
        axes2 = [axes2]
    fig2.suptitle("Average Uptake Curves Across Residues (±SEM)", fontsize=24)

    for ax, ensemble in zip(axes2, ensemble_order):
        if ensemble not in ensembles:
            continue
        ensemble_data = uptake_df[uptake_df["ensemble"] == ensemble]

        if ensemble_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ensemble, color=full_dataset_colours.get(ensemble, "black"))
            continue

        # Calculate mean and SEM across residues for each time point
        # First, get the mean uptake per residue at each time point (to handle multiple replicates)
        residue_means = ensemble_data.groupby(["residue", "time"])["uptake"].mean().reset_index()

        # Then calculate statistics across residues for each time point
        time_stats = (
            residue_means.groupby("time")["uptake"]
            .agg(
                [
                    "mean",  # Mean across residues
                    "std",  # Standard deviation across residues
                    "count",  # Number of residues
                ]
            )
            .reset_index()
        )

        # Calculate SEM across residues
        time_stats["sem"] = time_stats["std"] / np.sqrt(time_stats["count"])

        # Handle cases where std is NaN (single residue)
        time_stats["sem"] = time_stats["sem"].fillna(0)

        # Sort by time
        time_stats = time_stats.sort_values("time")

        # Convert to numpy arrays for plotting
        time_arr = np.array(time_stats["time"].values, dtype=float)
        mean_arr = np.array(time_stats["mean"].values, dtype=float)
        sem_arr = np.array(time_stats["sem"].values, dtype=float)

        # Plot mean with SEM as shaded region
        color = full_dataset_colours.get(ensemble, "black")
        ax.plot(time_arr, mean_arr, color=color, linewidth=2, label="Mean across residues")
        ax.fill_between(
            time_arr,
            mean_arr - sem_arr,
            mean_arr + sem_arr,
            color=color,
            alpha=0.3,
            label="±SEM across residues",
        )

        ax.set_title(ensemble, color=full_dataset_colours.get(ensemble, "black"))
        ax.set_xscale("log")
        ax.set_xlabel("Time (s)")
        if ax == axes2[0]:
            ax.set_ylabel("Average Fractional Uptake")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(os.path.join(output_dir, "average_uptake_curves.png"), dpi=300)
    print("Saved average uptake curves plot with SEM across residues.")
    plt.close(fig2)

    # Difference plots
    if n_ensembles < 2:
        print("Need at least two ensembles to plot differences.")
        return

    # Prepare data for difference calculation
    ensemble1, ensemble2 = ensemble_order[0], ensemble_order[1]

    avg_uptake = uptake_df.groupby(["ensemble", "residue", "time"])["uptake"].mean().reset_index()

    pivot_df = avg_uptake.pivot_table(
        index=["residue", "time"], columns="ensemble", values="uptake"
    ).reset_index()

    if ensemble1 not in pivot_df.columns or ensemble2 not in pivot_df.columns:
        print(f"Cannot find data for both {ensemble1} and {ensemble2} to compute difference.")
        return

    pivot_df["abs_diff"] = (pivot_df[ensemble1] - pivot_df[ensemble2]).abs()
    pivot_df["rel_diff"] = (pivot_df["abs_diff"] / pivot_df[ensemble2].replace(0, 1e-9)) * 100

    # Remove any NaN values
    pivot_df = pivot_df.dropna(subset=["abs_diff", "rel_diff"])

    if pivot_df.empty:
        print("No valid data for difference calculations.")
        return

    # Plot 3: Absolute difference per residue (limited selection)
    if len(unique_residues) > max_residues:
        plot_pivot_df = pivot_df[pivot_df["residue"].isin(selected_residues)].copy()
    else:
        plot_pivot_df = pivot_df.copy()

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Plot individual residue differences
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_residues)))
    for i, residue in enumerate(selected_residues):
        residue_data = plot_pivot_df[plot_pivot_df["residue"] == residue]
        if not residue_data.empty:
            residue_data = residue_data.sort_values("time")
            # Convert to numpy arrays
            time_arr = np.array(residue_data["time"].values, dtype=float)
            diff_arr = np.array(residue_data["abs_diff"].values, dtype=float)
            ax3.plot(
                time_arr, diff_arr, color=colors[i], alpha=0.7, linewidth=1, label=f"Res {residue}"
            )

    ax3.set_title(f"Absolute Difference in Uptake ({ensemble1} - {ensemble2})")
    ax3.set_xscale("log")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Absolute Difference")
    ax3.grid(True, alpha=0.3)
    fig3.savefig(os.path.join(output_dir, "residue_abs_diff_curves.png"), dpi=300)
    print("Saved absolute difference per residue plot.")
    plt.close(fig3)

    # Plot 4: Average absolute difference
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    avg_diff = pivot_df.groupby("time")["abs_diff"].agg(["mean", "std", "count"]).reset_index()
    avg_diff["stderr"] = avg_diff["std"] / np.sqrt(avg_diff["count"])
    avg_diff = avg_diff.sort_values("time")

    # Convert to numpy arrays
    time_arr = np.array(avg_diff["time"].values, dtype=float)
    mean_arr = np.array(avg_diff["mean"].values, dtype=float)
    stderr_arr = np.array(avg_diff["stderr"].values, dtype=float)

    ax4.plot(time_arr, mean_arr, linewidth=2, label="Mean", color="blue")
    ax4.fill_between(
        time_arr,
        mean_arr - stderr_arr,
        mean_arr + stderr_arr,
        alpha=0.3,
        color="blue",
        label="±SEM",
    )

    ax4.set_title(f"Average Absolute Difference in Uptake ({ensemble1} - {ensemble2})")
    ax4.set_xscale("log")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Average Absolute Difference")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.savefig(os.path.join(output_dir, "average_abs_diff_curve.png"), dpi=300)
    print("Saved average absolute difference plot.")
    plt.close(fig4)

    # Plot 5: Relative difference per residue (limited selection)
    fig5, ax5 = plt.subplots(figsize=(8, 6))

    for i, residue in enumerate(selected_residues):
        residue_data = plot_pivot_df[plot_pivot_df["residue"] == residue]
        if not residue_data.empty:
            residue_data = residue_data.sort_values("time")
            # Convert to numpy arrays
            time_arr = np.array(residue_data["time"].values, dtype=float)
            rel_diff_arr = np.array(residue_data["rel_diff"].values, dtype=float)
            ax5.plot(
                time_arr,
                rel_diff_arr,
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Res {residue}",
            )

    ax5.set_title(f"Relative Difference in Uptake ({ensemble1} vs {ensemble2})")
    ax5.set_xscale("log")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Relative Difference (%)")
    ax5.grid(True, alpha=0.3)
    fig5.savefig(os.path.join(output_dir, "residue_rel_diff_curves.png"), dpi=300)
    print("Saved relative difference per residue plot.")
    plt.close(fig5)

    # Plot 6: Average relative difference
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    avg_rel_diff = pivot_df.groupby("time")["rel_diff"].agg(["mean", "std", "count"]).reset_index()
    avg_rel_diff["stderr"] = avg_rel_diff["std"] / np.sqrt(avg_rel_diff["count"])
    avg_rel_diff = avg_rel_diff.sort_values("time")

    # Convert to numpy arrays
    time_arr = np.array(avg_rel_diff["time"].values, dtype=float)
    mean_arr = np.array(avg_rel_diff["mean"].values, dtype=float)
    stderr_arr = np.array(avg_rel_diff["stderr"].values, dtype=float)

    ax6.plot(time_arr, mean_arr, linewidth=2, label="Mean", color="red")
    ax6.fill_between(
        time_arr, mean_arr - stderr_arr, mean_arr + stderr_arr, alpha=0.3, color="red", label="±SEM"
    )

    ax6.set_title(f"Average Relative Difference in Uptake ({ensemble1} vs {ensemble2})")
    ax6.set_xscale("log")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Average Relative Difference (%)")
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    fig6.savefig(os.path.join(output_dir, "average_rel_diff_curve.png"), dpi=300)
    print("Saved average relative difference plot.")
    plt.close(fig6)


def main():
    """
    Main function to run the updated HDXer autovalidation analysis.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing directory structure...")
    experiment_dirs = parse_directory_structure(base_dir)

    if not experiment_dirs:
        print("No experiment directories found!")
        return

    print(f"Found {len(experiment_dirs)} experiment directories")

    # Extract work data
    print("Extracting work and MSE data...")
    work_df = extract_work_data(experiment_dirs)

    if work_df.empty:
        print("No work data found!")
    else:
        print(f"Extracted work data for {len(work_df)} conditions")
        print("Work data summary:")
        print(work_df.groupby(["ensemble", "split_name"]).size())

    # Extract weights and compute state recovery
    print("Computing state recovery and KL divergence...")
    reference_paths = [open_path, closed_path]

    # Verify trajectory files exist
    print("Verifying ensemble-specific trajectory files:")
    for ensemble, traj_path in trajectory_paths.items():
        if os.path.exists(traj_path):
            print(f"  ✓ {ensemble}: {traj_path}")
        else:
            print(f"  ✗ {ensemble}: {traj_path} (FILE NOT FOUND)")

    try:
        recovery_df = extract_weights_and_compute_state_recovery(
            experiment_dirs, trajectory_paths, topology_path, reference_paths
        )

        if recovery_df.empty:
            print("No recovery data found!")
        else:
            print(f"Computed recovery data for {len(recovery_df)} conditions")
            print("Recovery data summary:")
            print(recovery_df.groupby(["ensemble", "split_name"]).size())
    except Exception as e:
        print(f"Error computing state recovery: {e}")
        recovery_df = pd.DataFrame()

    # Extract residue uptake data
    print("Extracting residue uptake data...")
    uptake_df = extract_residue_uptake_data(experiment_dirs)
    if uptake_df.empty:
        print("No residue uptake data found.")
    else:
        print(f"Extracted residue uptake data for {len(uptake_df)} data points.")
        uptake_df.to_csv(os.path.join(output_dir, "residue_uptake_data.csv"), index=False)
        print("Residue uptake data saved to csv.")

    # Create plots
    print("Creating plots...")

    # Create residue uptake comparison plots
    if not uptake_df.empty:
        print("Creating residue uptake comparison plots...")
        plot_residue_uptake_comparison(uptake_df, output_dir)
        print("Residue uptake comparison plots saved.")

    # Save data to CSV for further analysis
    if not work_df.empty:
        work_df.to_csv(os.path.join(output_dir, "work_data.csv"), index=False)

    if not recovery_df.empty:
        recovery_df.to_csv(os.path.join(output_dir, "recovery_data.csv"), index=False)

    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
