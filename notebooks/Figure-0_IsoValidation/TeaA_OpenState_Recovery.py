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
output_dir = os.path.join(base_dir, "analysis_output")

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


def plot_open_state_recovery(recovery_df, output_dir):
    """
    Plot normalized open state recovery % against gamma parameter with ensembles in separate panels.
    Using line plots with standard deviation bands.
    """
    # Filter for the ensembles and split types we want
    plot_df = recovery_df[recovery_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for open state recovery plot")
        return None

    # Create FacetGrid
    g = sns.FacetGrid(plot_df, col="ensemble", height=4, aspect=1.5, col_order=ensemble_order)

    # Define function to apply our custom plotting to each facet
    def plot_facet(data, **kwargs):
        ax = plt.gca()
        return plot_mean_std_by_group(
            data, "gamma", "open_state_recovery", "split_type", split_type_dataset_colours, ax
        )

    # Map the plotting function
    g.map_dataframe(plot_facet)

    # Custom function to set colored titles
    def set_colored_titles(figure):
        for ax in figure.axes:
            title = ax.get_title()
            if title:
                ensemble = title.strip()
                if ensemble in full_dataset_colours:
                    ax.set_title(
                        f"TeaA | {ensemble}",
                        color=full_dataset_colours[ensemble],
                        fontsize=20,
                        pad=0,
                    )

    # Set titles and labels
    g.set_titles(col_template="{col_name}")
    set_colored_titles(g.fig)

    # Set axis labels
    for ax in g.axes.flat:
        ax.set_xlabel(r"$\gamma_{HDXer}$ Parameter", fontsize=20)
        ax.set_ylabel("Open State Recovery (%)", fontsize=18)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)  # Set y-axis range to 0-100%
        ax.grid(False, alpha=0)
        ax.tick_params(axis="both", which="major", labelsize=12)

    # Add legend
    handles = []
    labels = []
    for split_type in analysis_split_types:
        color = split_type_dataset_colours[split_type]
        handles.append(
            plt.Line2D(
                [0], [0], color=color, linewidth=2, alpha=0.8, label=split_name_mapping[split_type]
            )
        )
        labels.append(split_name_mapping[split_type])

    g.fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.8),
        title="Split Type",
        frameon=True,
        framealpha=0.8,
    )

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Set left=0.2

    # Save figure
    output_path = os.path.join(output_dir, "open_state_recovery_vs_gamma.png")
    g.fig.savefig(output_path, dpi=300)

    return g.fig


def plot_training_metrics(work_df, recovery_df, output_dir):
    """
    Plot training MSE, Work, and KL divergence against gamma with line plots and standard deviation bands.
    Creates separate plots for each metric.
    """
    # Merge work and recovery data to get KL divergence
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(
        work_df,
        recovery_df[merge_cols + ["kl_div_uniform", "open_state_recovery"]],
        on=merge_cols,
        how="left",
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
        {
            "name": "open_state_recovery",
            "title": "Open State Recovery",
            "ylabel": "Open State Recovery (%)",
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
            bbox_to_anchor=(0.96, 0.8),
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
            bbox_to_anchor=(0.96, 0.8),
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


def plot_scatter_metrics_as_lines(work_df, recovery_df, output_dir):
    """
    Plot the same metric pairs as plot_scatter_metrics but as line plots vs gamma parameter.
    Each subplot shows both variables from a pair plotted as separate lines against gamma,
    using the same style as plot_training_metrics with mean lines and standard deviation bands.
    Creates dual y-axis plots when variables have different scales/units.
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for line plots of metric relationships")
        return None

    # Define the same pairs as in plot_scatter_metrics
    metric_pairs = [
        {
            "x_var": "open_state_recovery",
            "y_var": "work_kj",
            "x_label": "Open State Recovery (%)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Open State Recovery & Apparent Work vs γ",
            "filename": "lines_recovery_and_work_vs_gamma.png",
            "y_scale": "log",
        },
        {
            "x_var": "work_kj",
            "y_var": "open_state_recovery",
            "x_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "y_label": "Open State Recovery (%)",
            "title": "Apparent Work & Open State Recovery vs γ",
            "filename": "lines_work_and_recovery_vs_gamma.png",
            "x_scale": "log",
        },
        {
            "x_var": "mse",
            "y_var": "work_kj",
            "x_label": r"MSE$_{Training}$",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "MSE Training & Apparent Work vs γ",
            "filename": "lines_mse_and_work_vs_gamma.png",
            "y_scale": "log",
        },
        {
            "x_var": "mse",
            "y_var": "open_state_recovery",
            "x_label": r"MSE$_{Training}$",
            "y_label": "Open State Recovery (%)",
            "title": "MSE Training & Open State Recovery vs γ",
            "filename": "lines_mse_and_recovery_vs_gamma.png",
        },
        {
            "x_var": "kl_div_uniform",
            "y_var": "work_kj",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "KL Divergence & Apparent Work vs γ",
            "filename": "lines_kl_and_work_vs_gamma.png",
            "y_scale": "log",
        },
        {
            "x_var": "kl_div_uniform",
            "y_var": "open_state_recovery",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": "Open State Recovery (%)",
            "title": "KL Divergence & Open State Recovery vs γ",
            "filename": "lines_kl_and_recovery_vs_gamma.png",
        },
        {
            "x_var": "kl_div_uniform",
            "y_var": "mse",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"MSE$_{Training}$",
            "title": "KL Divergence & MSE Training vs γ",
            "filename": "lines_kl_and_mse_vs_gamma.png",
        },
        {
            "x_var": "open_state_recovery",
            "y_var": "mse",
            "x_label": "Open State Recovery (%)",
            "y_label": r"MSE$_{Training}$",
            "title": "Open State Recovery & MSE Training vs γ",
            "filename": "lines_recovery_and_mse_vs_gamma.png",
        },
    ]

    # Define line styles for the two variables in each pair
    line_styles = {
        "primary": {"linestyle": "-", "linewidth": 2.5, "alpha": 0.9},
        "secondary": {"linestyle": "--", "linewidth": 2, "alpha": 0.8},
    }

    figures = []

    for pair in metric_pairs:
        print(f"Creating line plot for {pair['title']}")

        # Create plot with ensembles in columns
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharex=True
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])

        for col_idx, ensemble in enumerate(ensemble_order):
            ax1 = axes[col_idx]
            ensemble_data = plot_df[plot_df["ensemble"] == ensemble].copy()

            if ensemble_data.empty:
                ax1.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                    fontsize=14,
                )
                ax1.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20
                )
                continue

            # Handle non-positive work values for log scale
            if pair["x_var"] == "work_kj" or pair["y_var"] == "work_kj":
                work_col = "work_kj"
                eps = 1e-6
                ensemble_data[work_col] = ensemble_data[work_col].apply(
                    lambda v: v if v > 0 else eps
                )

            # Create secondary y-axis for the second variable
            ax2 = ax1.twinx()

            # Plot first variable (x_var from the pair) on left y-axis
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]
                if split_data.empty:
                    continue

                color = split_type_dataset_colours[split_type]

                # Plot x_var on left axis (ax1)
                plot_mean_std_by_group(
                    split_data, "gamma", pair["x_var"], "split_type", {split_type: color}, ax1
                )

                # Modify the line style for the first variable
                lines = ax1.get_lines()
                if lines:
                    lines[-1].set_linestyle(line_styles["primary"]["linestyle"])
                    lines[-1].set_linewidth(line_styles["primary"]["linewidth"])
                    lines[-1].set_alpha(line_styles["primary"]["alpha"])

            # Plot second variable (y_var from the pair) on right y-axis
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]
                if split_data.empty:
                    continue

                color = split_type_dataset_colours[split_type]

                # Plot y_var on right axis (ax2) with different line style
                plot_mean_std_by_group(
                    split_data, "gamma", pair["y_var"], "split_type", {split_type: color}, ax2
                )

                # Modify the line style for the second variable
                lines = ax2.get_lines()
                if lines:
                    lines[-1].set_linestyle(line_styles["secondary"]["linestyle"])
                    lines[-1].set_linewidth(line_styles["secondary"]["linewidth"])
                    lines[-1].set_alpha(line_styles["secondary"]["alpha"])

            # Set axis properties
            ax1.set_xlabel(r"$\gamma_{HDXer}$ Parameter", fontsize=20)
            ax1.set_xscale("log")

            # Set y-axis labels and colors
            ax1.set_ylabel(pair["x_label"], fontsize=18, color="black")
            ax2.set_ylabel(pair["y_label"], fontsize=18, color="black")

            # Set y-axis scales if specified
            if pair.get("x_scale") == "log":
                ax1.set_yscale("log")
            if pair.get("y_scale") == "log":
                ax2.set_yscale("log")

            # Set specific limits for recovery variables
            if pair["x_var"] == "open_state_recovery":
                ax1.set_ylim(0, 100)
            if pair["y_var"] == "open_state_recovery":
                ax2.set_ylim(0, 100)

            ax1.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            ax1.grid(False, alpha=0)
            ax2.grid(False, alpha=0)

            # Color the y-axis labels to match the line styles
            ax1.tick_params(axis="y", labelcolor="black", labelsize=12)
            ax2.tick_params(axis="y", labelcolor="black", labelsize=12)
            ax1.tick_params(axis="x", labelsize=12)

        # Create comprehensive legend
        legend_elements = []

        # Split type colors with both line styles
        for split_type in analysis_split_types:
            color = split_type_dataset_colours[split_type]
            split_name = split_name_mapping[split_type]

            # Add entry for first variable (solid line)
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=line_styles["primary"]["linestyle"],
                    linewidth=line_styles["primary"]["linewidth"],
                    alpha=line_styles["primary"]["alpha"],
                    label=f"{split_name} - {pair['x_label'].split('(')[0].strip()}",
                )
            )

            # Add entry for second variable (dashed line)
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=line_styles["secondary"]["linestyle"],
                    linewidth=line_styles["secondary"]["linewidth"],
                    alpha=line_styles["secondary"]["alpha"],
                    label=f"{split_name} - {pair['y_label'].split('(')[0].strip()}",
                )
            )

        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Split Type & Variable",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            ncol=1,
        )

        plt.tight_layout()
        # Make room for the legend
        plt.subplots_adjust(right=0.75)

        # Save figure
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved line plot for {pair['title']} to {output_path}")

        figures.append(fig)

    return figures


def plot_line_metrics_variable_pairs(work_df, recovery_df, output_dir):
    """
    Plot line plots between key metrics with mean values and standard deviation bands.
    Creates line plots using the same variable pairs as scatter plots but with aggregated
    data connected by gamma values, similar to the training metrics style.
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for line plots")
        return None

    # Define the same variable pairs as in scatter plots
    line_pairs = [
        {
            "x": "open_state_recovery",
            "y": "work_kj",
            "x_label": "Open State Recovery (%)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs Open State Recovery",
            "filename": "line_work_vs_recovery.png",
        },
        {
            "x": "work_kj",
            "y": "open_state_recovery",
            "x_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs Apparent Work",
            "filename": "line_recovery_vs_work.png",
        },
        {
            "x": "mse",
            "y": "work_kj",
            "x_label": r"MSE$_{Training}$",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs MSE Training",
            "filename": "line_work_vs_mse.png",
        },
        {
            "x": "work_kj",
            "y": "mse",
            "x_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "y_label": r"MSE$_{Training}$",
            "title": "MSE Training vs Apparent Work",
            "filename": "line_mse_vs_work.png",
        },
        {
            "x": "mse",
            "y": "open_state_recovery",
            "x_label": r"MSE$_{Training}$",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs MSE Training",
            "filename": "line_recovery_vs_mse.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "work_kj",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "title": "Apparent Work vs KL Divergence",
            "filename": "line_work_vs_kl.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "open_state_recovery",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": "Open State Recovery (%)",
            "title": "Open State Recovery vs KL Divergence",
            "filename": "line_recovery_vs_kl.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "mse",
            "x_label": r"KL(P||U$_{uniform}$)",
            "y_label": r"MSE$_{Training}$",
            "title": "MSE Training vs KL Divergence",
            "filename": "line_mse_vs_kl.png",
        },
        {
            "x": "open_state_recovery",
            "y": "mse",
            "x_label": "Open State Recovery (%)",
            "y_label": r"MSE$_{Training}$",
            "title": "MSE Training vs Open State Recovery",
            "filename": "line_mse_vs_recovery.png",
        },
    ]

    figures = []

    for pair in line_pairs:
        print(f"Creating line plot: {pair['title']}")

        # Aggregate data by ensemble, split_type, and gamma
        grouping_cols = ["ensemble", "split_type", "split_name", "gamma"]

        # Group and calculate means and standard deviations
        aggregated_df = (
            plot_df.groupby(grouping_cols)
            .agg({pair["x"]: ["mean", "std", "count"], pair["y"]: ["mean", "std", "count"]})
            .reset_index()
        )

        # Flatten column names
        aggregated_df.columns = [
            "ensemble",
            "split_type",
            "split_name",
            "gamma",
            f"{pair['x']}_mean",
            f"{pair['x']}_std",
            f"{pair['x']}_count",
            f"{pair['y']}_mean",
            f"{pair['y']}_std",
            f"{pair['y']}_count",
        ]

        # Fill NaN standard deviations with 0 (for cases with only 1 replicate)
        aggregated_df[f"{pair['x']}_std"] = aggregated_df[f"{pair['x']}_std"].fillna(0)
        aggregated_df[f"{pair['y']}_std"] = aggregated_df[f"{pair['y']}_std"].fillna(0)

        # Size mapping for gamma values (higher gamma = larger points)
        gamma_values = sorted(aggregated_df["gamma"].unique())
        max_size = 100
        min_size = 20

        size_map = {}
        for i, gamma in enumerate(gamma_values):
            # Higher gamma values get larger sizes
            size = min_size + (i / max(1, len(gamma_values) - 1)) * (max_size - min_size)
            size_map[gamma] = size

        # Create regular subplots with shared y-axis for better alignment
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharey=True, sharex=False
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])  # Ensure axes is iterable

        for col_idx, ensemble in enumerate(ensemble_order):
            ax = axes[col_idx]
            ensemble_data = aggregated_df[aggregated_df["ensemble"] == ensemble]

            # Plot lines for each split type
            for split_type in analysis_split_types:
                subset = ensemble_data[ensemble_data["split_type"] == split_type]
                if subset.empty:
                    continue

                # Filter out non-positive work values when necessary
                x_vals = subset[f"{pair['x']}_mean"].to_numpy()
                y_vals = subset[f"{pair['y']}_mean"].to_numpy()
                x_errs = subset[f"{pair['x']}_std"].to_numpy()
                y_errs = subset[f"{pair['y']}_std"].to_numpy()
                gamma_vals = subset["gamma"].to_numpy()

                mask = np.ones(len(x_vals), dtype=bool)

                # If either axis is work_kj, filter non-positive values
                if pair["x"] == "work_kj":
                    mask &= x_vals > 0
                if pair["y"] == "work_kj":
                    mask &= y_vals > 0

                if not np.any(mask):
                    continue

                x_plot = x_vals[mask]
                y_plot = y_vals[mask]
                x_err_plot = x_errs[mask]
                y_err_plot = y_errs[mask]
                gamma_plot = gamma_vals[mask]

                # Sort by gamma to ensure proper line connection
                sort_idx = np.argsort(gamma_plot)
                x_plot = x_plot[sort_idx]
                y_plot = y_plot[sort_idx]
                x_err_plot = x_err_plot[sort_idx]
                y_err_plot = y_err_plot[sort_idx]
                gamma_plot = gamma_plot[sort_idx]

                # Calculate point sizes based on gamma values
                sizes = [size_map.get(gamma, min_size) for gamma in gamma_plot]

                # Plot mean line (without markers first)
                ax.plot(
                    x_plot,
                    y_plot,
                    color=split_type_dataset_colours[split_type],
                    alpha=0.9,
                    linewidth=2,
                )

                # Plot points with gamma-scaled sizes
                ax.scatter(
                    x_plot,
                    y_plot,
                    c=split_type_dataset_colours[split_type],
                    s=sizes,
                    alpha=0.4,
                    edgecolors="white",
                    linewidth=0.25,
                    label=split_name_mapping.get(split_type, split_type),
                    zorder=3,
                )

                # Add error bars
                ax.errorbar(
                    x_plot,
                    y_plot,
                    xerr=x_err_plot,
                    yerr=y_err_plot,
                    fmt="none",
                    ecolor=split_type_dataset_colours[split_type],
                    alpha=0.4,
                    capsize=2,
                    capthick=1,
                    zorder=2,
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

            # Set axis limits for recovery plots
            if pair["y"] == "open_state_recovery":
                ax.set_ylim(0, 100)
            if pair["x"] == "open_state_recovery":
                ax.set_xlim(0, 100)

            ax.tick_params(axis="both", which="major")
            ax.grid(True, alpha=0.3)

        # Add legend to the figure
        handles = []
        labels = []
        for split_type in analysis_split_types:
            color = split_type_dataset_colours[split_type]
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2,
                    alpha=0.6,
                    marker="o",
                    markersize=6,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
            labels.append(split_name_mapping[split_type])

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.6,
        )

        # # Add size legend as text
        # size_text = "Point Size ∝ γ Value\n"
        # size_text += f"Small: γ = {gamma_values[0]:.0e}\n"
        # size_text += f"Large: γ = {gamma_values[-1]:.0e}"

        # fig.text(
        #     0.02,
        #     0.15,
        #     size_text,
        #     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        #     fontsize=10,
        #     verticalalignment="bottom",
        # )

        plt.tight_layout(rect=[0.05, 0, 1, 1])

        # Save figure
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300)
        print(f"Saved {pair['title']} line plot to {output_path}")

        figures.append(fig)

    return figures


def plot_volcano_kl_recovery_hdxer(work_df, recovery_df, output_dir):
    """
    Plot averaged volcano plot for HDXer data with KL divergence vs open recovery fold change.
    Colors by split type, sizes by gamma value, ensembles arranged as rows.

    Args:
        work_df (pd.DataFrame): Work data containing gamma parameters
        recovery_df (pd.DataFrame): Recovery data containing KL divergence and open state recovery
        output_dir (str): Output directory for plots
    """
    plt.style.use("default")  # Reset to avoid conflicts with seaborn style
    sns.set_style("ticks")

    print("Creating HDXer volcano plot...")

    # Merge work and recovery dataframes
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for analysis ensembles and split types
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if len(plot_df) == 0:
        print("No matching data found for HDXer volcano plot")
        return None

    print(f"Merged {len(plot_df)} data points for HDXer volcano plot")

    # Calculate fold change relative to true unweighted distribution for each combination
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
                    # All rows in the subset should have the same unweighted values
                    baseline_recovery = subset["unweighted_open_percentage"].iloc[0]

                    # Calculate fold change for each point relative to unweighted distribution
                    for _, row in subset.iterrows():
                        if baseline_recovery > 0:
                            fold_change = row["raw_open_percentage"] / baseline_recovery
                            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
                        else:
                            fold_change = 1.0  # No change if baseline is 0
                            log2_fold_change = 0.0

                        fold_change_data.append(
                            {
                                **row.to_dict(),
                                "open_recovery_fold_change": fold_change,
                                "log2_fold_change": log2_fold_change,
                                "baseline_recovery": baseline_recovery,
                                "baseline_type": "unweighted",
                            }
                        )

    if not fold_change_data:
        print("No fold change data could be calculated for HDXer volcano plot")
        return None

    volcano_df = pd.DataFrame(fold_change_data)
    print(f"Calculated fold changes for {len(volcano_df)} data points")

    # Calculate averages and standard deviations across replicates
    grouping_cols = ["ensemble", "split_type", "split_name", "gamma"]

    averaged_df = (
        volcano_df.groupby(grouping_cols)
        .agg(
            {
                "log2_fold_change": ["mean", "std", "count"],
                "kl_div_uniform": ["mean", "std", "count"],
                "open_state_recovery": ["mean", "std"],
                "raw_open_percentage": ["mean", "std"],
                "work_kj": ["mean", "std"],
                "mse": ["mean", "std"],
                "baseline_recovery": "first",
                "baseline_type": "first",
                "unweighted_open_percentage": "first",
                "unweighted_closed_percentage": "first",
                "unweighted_intermediate_percentage": "first",
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
        "log2_fold_change_mean",
        "log2_fold_change_std",
        "log2_fold_change_count",
        "kl_div_uniform_mean",
        "kl_div_uniform_std",
        "kl_div_uniform_count",
        "open_state_recovery_mean",
        "open_state_recovery_std",
        "raw_open_percentage_mean",
        "raw_open_percentage_std",
        "work_kj_mean",
        "work_kj_std",
        "mse_mean",
        "mse_std",
        "baseline_recovery",
        "baseline_type",
        "unweighted_open_percentage",
        "unweighted_closed_percentage",
        "unweighted_intermediate_percentage",
    ]

    # Fill NaN standard deviations with 0 (for cases with only 1 replicate)
    averaged_df["log2_fold_change_std"] = averaged_df["log2_fold_change_std"].fillna(0)
    averaged_df["kl_div_uniform_std"] = averaged_df["kl_div_uniform_std"].fillna(0)
    averaged_df["open_state_recovery_std"] = averaged_df["open_state_recovery_std"].fillna(0)

    print(f"Averaged across replicates: {len(averaged_df)} unique parameter combinations")

    # Calculate target fold changes for each ensemble based on target 40:60 ratio
    target_fold_changes = {}

    print("Calculating target fold changes based on 40% open state target...")

    for ensemble in averaged_df["ensemble"].unique():
        # Get the true unweighted distribution for this ensemble
        ensemble_data = plot_df[plot_df["ensemble"] == ensemble]
        if len(ensemble_data) > 0:
            # Get unweighted baseline from the data (all rows should have same unweighted values)
            baseline_recovery = ensemble_data["unweighted_open_percentage"].iloc[0]

            if baseline_recovery > 0:
                target_recovery = 100  # Target is 100% recovery
                target_fold_change = target_recovery / baseline_recovery
                target_log2_fold_change = np.log2(target_fold_change)
                target_fold_changes[ensemble] = target_log2_fold_change
                print(
                    f"  {ensemble}: Unweighted open = {baseline_recovery:.1f}%, "
                    + f"Target fold change = {target_fold_change:.3f} (log2 = {target_log2_fold_change:.3f})"
                )
            else:
                target_fold_changes[ensemble] = 0
                print(
                    f"  {ensemble}: Warning - unweighted open state is 0, setting target fold change to 0"
                )
        else:
            target_fold_changes[ensemble] = 0
            print(f"  {ensemble}: Warning - no data found, setting target fold change to 0")

    # Create figure with ensembles as rows
    fig, axes = plt.subplots(
        len(ensemble_order), 1, figsize=(10, 6 * len(ensemble_order)), squeeze=False
    )

    fig.suptitle(
        "HDXer Volcano Plot: KL Divergence vs Open Recovery Fold Change\n(vs Unweighted Distribution)",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    # Calculate global axis limits for consistency across panels
    all_x_data = averaged_df["log2_fold_change_mean"]
    all_y_data = averaged_df["kl_div_uniform_mean"]

    # Include error bars in range calculation
    x_with_error = np.concatenate(
        [
            all_x_data + averaged_df["log2_fold_change_std"],
            all_x_data - averaged_df["log2_fold_change_std"],
        ]
    )
    y_with_error = np.concatenate(
        [
            all_y_data + averaged_df["kl_div_uniform_std"],
            all_y_data - averaged_df["kl_div_uniform_std"],
        ]
    )

    x_margin = (x_with_error.max() - x_with_error.min()) * 0.05  # 5% margin
    y_margin = (y_with_error.max() - y_with_error.min()) * 0.05  # 5% margin

    global_xlim = [x_with_error.min() - x_margin, x_with_error.max() + x_margin]
    global_ylim = [y_with_error.min() - y_margin, y_with_error.max() + y_margin]

    print(f"  Global X-axis range: {global_xlim[0]:.3f} to {global_xlim[1]:.3f}")
    print(f"  Global Y-axis range: {global_ylim[0]:.4f} to {global_ylim[1]:.4f}")

    # Size mapping for gamma values (higher gamma = larger points)
    gamma_values = sorted(averaged_df["gamma"].unique())
    max_size = 200
    min_size = 50

    size_map = {}
    for i, gamma in enumerate(gamma_values):
        # Higher gamma values get larger sizes
        size = min_size + (i / max(1, len(gamma_values) - 1)) * (max_size - min_size)
        size_map[gamma] = size

    print("  Gamma size mapping (higher gamma = larger points):")
    for i, gamma in enumerate(gamma_values[:5]):  # Show first 5
        print(f"    γ = {gamma:.1e}: size {size_map[gamma]:.0f}")
    if len(gamma_values) > 5:
        print(f"    ... and {len(gamma_values) - 5} more")

    for i, ensemble in enumerate(ensemble_order):
        ax = axes[i, 0]

        ensemble_data = averaged_df[averaged_df["ensemble"] == ensemble]

        if len(ensemble_data) > 0:
            # Plot each split type with different colors
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]

                if len(split_data) > 0:
                    x_vals = split_data["log2_fold_change_mean"]
                    y_vals = split_data["kl_div_uniform_mean"]
                    x_errs = split_data["log2_fold_change_std"]
                    y_errs = split_data["kl_div_uniform_std"]

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
            ax.axvline(
                x=0,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Unweighted Distribution",
            )

            # Add target fold change line for this ensemble
            if ensemble in target_fold_changes:
                target_x = target_fold_changes[ensemble]
                ax.axvline(
                    x=target_x,
                    color="orange",
                    linestyle=":",
                    alpha=0.8,
                    linewidth=2,
                    label="Target (100%)",
                )

                # Add text label for target line
                ax.text(
                    target_x,
                    global_ylim[1] * 0.95,
                    "Target\n(100%)",
                    ha="center",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                )

            # Set consistent axis limits
            ax.set_xlim(-6, 6)
            ax.set_ylim(global_ylim)

            # Add quadrant labels
            ax.text(
                0.98,
                0.98,
                "High KL\nHigh Recovery",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                fontsize=10,
            )

            ax.text(
                0.02,
                0.98,
                "High KL\nLow Recovery",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                fontsize=10,
            )

            # Set axis labels
            ax.set_xlabel("Log₂ Fold Change (Open Recovery vs Unweighted)", fontsize=16)
            ax.set_ylabel("KL Divergence vs Uniform", fontsize=16)
            ax.set_title(
                f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20, pad=10
            )
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
            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)

    # Create comprehensive legend
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

    # Add separator
    legend_elements.append(plt.Line2D([0], [0], color="white", label=""))

    # Reference lines
    legend_elements.append(
        plt.Line2D([0], [0], color="red", linestyle="--", label="Unweighted Distribution")
    )
    legend_elements.append(
        plt.Line2D([0], [0], color="orange", linestyle=":", label="Target (100%)")
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
        0.85,
        size_text,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.92])

    # Save the plot
    output_path = os.path.join(output_dir, "hdxer_volcano_plot_kl_recovery.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"HDXer volcano plot saved to: {output_path}")

    # Save the averaged dataset
    averaged_df["target_log2_fold_change"] = averaged_df["ensemble"].map(target_fold_changes)
    output_csv = os.path.join(output_dir, "hdxer_volcano_plot_data.csv")
    averaged_df.to_csv(output_csv, index=False)
    print(f"HDXer volcano plot dataset saved to: {output_csv}")

    # Print summary statistics
    print("\nHDXer Volcano Plot Summary:")
    print("-" * 40)
    print(f"Individual data points: {len(volcano_df)}")
    print(f"Averaged combinations: {len(averaged_df)}")
    print(f"Gamma values: {[f'{g:.0e}' for g in sorted(averaged_df['gamma'].unique())]}")
    print(
        f"Fold change range (means): {averaged_df['log2_fold_change_mean'].min():.2f} to {averaged_df['log2_fold_change_mean'].max():.2f}"
    )
    print(
        f"KL divergence range (means): {averaged_df['kl_div_uniform_mean'].min():.4f} to {averaged_df['kl_div_uniform_mean'].max():.4f}"
    )

    print("\nTarget fold changes (40% open state vs unweighted distribution):")
    for ensemble, target_fc in target_fold_changes.items():
        # Get unweighted percentage for this ensemble
        ensemble_data = averaged_df[averaged_df["ensemble"] == ensemble]
        if len(ensemble_data) > 0:
            unweighted_open = ensemble_data["unweighted_open_percentage"].iloc[0]
            print(
                f"  {ensemble}: Unweighted = {unweighted_open:.1f}% → Target log₂ fold change = {target_fc:.3f}"
            )
        else:
            print(f"  {ensemble}: Log₂ fold change = {target_fc:.3f}")

    return fig, averaged_df


def plot_r_squared_vs_mean(work_df, recovery_df, output_dir):
    """
    Plot R² values between variable pairs vs mean x-variable values across replicates for each gamma.

    For each scatter plot pair, calculates R² between the two variables within replicates
    of each gamma value, then plots R² (y-axis) vs mean of x-variable (x-axis).

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing open state recovery and KL divergence
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    list of matplotlib.Figure
        List of generated figures
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for R² analysis plots")
        return None

    # Define the same scatter plot pairs as in the original scatter function
    scatter_pairs = [
        {
            "x": "open_state_recovery",
            "y": "work_kj",
            "x_label": "Mean Open State Recovery (%)",
            "y_label": r"R² (Open Recovery vs Work)",
            "title": "R² Analysis: Open Recovery vs Work",
            "filename": "r_squared_recovery_vs_work.png",
        },
        {
            "x": "mse",
            "y": "work_kj",
            "x_label": r"Mean MSE$_{Training}$",
            "y_label": r"R² (MSE vs Work)",
            "title": "R² Analysis: MSE vs Work",
            "filename": "r_squared_mse_vs_work.png",
        },
        {
            "x": "mse",
            "y": "open_state_recovery",
            "x_label": r"Mean MSE$_{Training}$",
            "y_label": r"R² (MSE vs Recovery)",
            "title": "R² Analysis: MSE vs Recovery",
            "filename": "r_squared_mse_vs_recovery.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "work_kj",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"R² (KL vs Work)",
            "title": "R² Analysis: KL vs Work",
            "filename": "r_squared_kl_vs_work.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "open_state_recovery",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"R² (KL vs Recovery)",
            "title": "R² Analysis: KL vs Recovery",
            "filename": "r_squared_kl_vs_recovery.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "mse",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"R² (KL vs MSE)",
            "title": "R² Analysis: KL vs MSE",
            "filename": "r_squared_kl_vs_mse.png",
        },
    ]

    figures = []

    for pair in scatter_pairs:
        print(f"Creating R² plot for {pair['title']}")

        # Calculate R² and mean values for each ensemble, split_type, gamma combination
        r_squared_data = []

        for ensemble in ensemble_order:
            for split_type in analysis_split_types:
                # Get data for this ensemble and split type
                subset = plot_df[
                    (plot_df["ensemble"] == ensemble) & (plot_df["split_type"] == split_type)
                ]

                if subset.empty:
                    continue

                # Group by gamma value
                for gamma in subset["gamma"].unique():
                    gamma_data = subset[subset["gamma"] == gamma]

                    if len(gamma_data) < 2:  # Need at least 2 replicates for R²
                        continue

                    # Get x and y values
                    x_vals = gamma_data[pair["x"]].values
                    y_vals = gamma_data[pair["y"]].values

                    # Handle non-positive values for work_kj (transform to log space)
                    x_vals_calc = x_vals.copy()
                    y_vals_calc = y_vals.copy()

                    # Filter out invalid values and transform if needed
                    valid_mask = np.ones(len(x_vals), dtype=bool)

                    if pair["x"] == "work_kj":
                        valid_mask &= x_vals > 0
                        x_vals_calc = np.log10(x_vals_calc)
                    if pair["y"] == "work_kj":
                        valid_mask &= y_vals > 0
                        y_vals_calc = np.log10(y_vals_calc)

                    # Remove invalid values
                    x_vals_calc = x_vals_calc[valid_mask]
                    y_vals_calc = y_vals_calc[valid_mask]
                    x_vals_mean = x_vals[valid_mask]  # Keep original scale for mean calculation

                    if len(x_vals_calc) < 2:
                        continue

                    # Calculate R²
                    try:
                        corr_matrix = np.corrcoef(x_vals_calc, y_vals_calc)
                        if corr_matrix.shape == (2, 2):
                            r_value = corr_matrix[0, 1]
                            r_squared = r_value**2
                        else:
                            r_squared = 0.0
                    except:
                        r_squared = 0.0

                    # Calculate mean of x-variable (in original scale)
                    x_mean = np.mean(x_vals_mean)

                    r_squared_data.append(
                        {
                            "ensemble": ensemble,
                            "split_type": split_type,
                            "split_name": split_name_mapping[split_type],
                            "gamma": gamma,
                            "r_squared": r_squared,
                            "x_mean": x_mean,
                            "n_replicates": len(x_vals_calc),
                        }
                    )

        if not r_squared_data:
            print(f"No R² data could be calculated for {pair['title']}")
            continue

        r_squared_df = pd.DataFrame(r_squared_data)

        # Create plot with ensembles in columns
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharey=True, sharex=False
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])

        for col_idx, ensemble in enumerate(ensemble_order):
            ax = axes[col_idx]
            ensemble_data = r_squared_df[r_squared_df["ensemble"] == ensemble]

            if ensemble_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20
                )
                continue

            # Plot R² vs mean x-variable for each split type
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]

                if split_data.empty:
                    continue

                # Sort by x_mean for proper line plotting
                split_data = split_data.sort_values("x_mean")

                ax.plot(
                    split_data["x_mean"].to_numpy(),
                    split_data["r_squared"].to_numpy(),
                    color=split_type_dataset_colours[split_type],
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    alpha=0.8,
                    label=split_name_mapping[split_type],
                )

                # Add gamma value annotations on points
                for _, row in split_data.iterrows():
                    ax.annotate(
                        f"{row['gamma']:.0e}",
                        (float(row["x_mean"]), float(row["r_squared"])),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        color=split_type_dataset_colours[split_type],
                    )

            # Set axis properties
            ax.set_xlabel(pair["x_label"], fontsize=20)
            if col_idx == 0:
                ax.set_ylabel(pair["y_label"], fontsize=18)

            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)

            # Set axis scales - log scale for work variables
            if pair["x"] == "work_kj":
                ax.set_xscale("log")

            # Set y-axis limits for R²
            ax.set_ylim(0, 1.1)

            # Set x-axis limits for recovery plots
            if pair["x"] == "open_state_recovery":
                ax.set_xlim(0, 100)

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=12)

        # Add legend
        handles = []
        labels = []
        for split_type in analysis_split_types:
            if split_type in split_type_dataset_colours:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=split_type_dataset_colours[split_type],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        alpha=0.8,
                    )
                )
                labels.append(split_name_mapping[split_type])

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.8,
        )

        plt.tight_layout(rect=[0.05, 0, 1, 1])

        # Save figure
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300)
        print(f"Saved R² analysis plot to {output_path}")

        figures.append(fig)

    # Save R² data to CSV
    if r_squared_data:
        all_r_squared_df = pd.DataFrame(r_squared_data)
        output_csv = os.path.join(output_dir, "r_squared_analysis_data.csv")
        all_r_squared_df.to_csv(output_csv, index=False)
        print(f"Saved R² analysis data to {output_csv}")

    return figures


def plot_multivariate_sd_vs_mean(work_df, recovery_df, output_dir):
    """
    Plot multivariate standard deviation values between variable pairs vs mean x-variable values
    across replicates for each gamma.

    For each scatter plot pair, calculates the multivariate standard deviation (square root of
    determinant of covariance matrix) between the two variables within replicates of each gamma value,
    then plots multivariate SD (y-axis) vs mean of x-variable (x-axis).

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing open state recovery and KL divergence
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    list of matplotlib.Figure
        List of generated figures
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for multivariate SD analysis plots")
        return None

    # Define the same scatter plot pairs as in the original scatter function
    scatter_pairs = [
        {
            "x": "open_state_recovery",
            "y": "work_kj",
            "x_label": "Mean Open State Recovery (%)",
            "y_label": r"Multivariate SD (Recovery, Work)",
            "title": "Multivariate SD Analysis: Recovery vs Work",
            "filename": "multivariate_sd_recovery_vs_work.png",
        },
        {
            "x": "mse",
            "y": "work_kj",
            "x_label": r"Mean MSE$_{Training}$",
            "y_label": r"Multivariate SD (MSE, Work)",
            "title": "Multivariate SD Analysis: MSE vs Work",
            "filename": "multivariate_sd_mse_vs_work.png",
        },
        {
            "x": "mse",
            "y": "open_state_recovery",
            "x_label": r"Mean MSE$_{Training}$",
            "y_label": r"Multivariate SD (MSE, Recovery)",
            "title": "Multivariate SD Analysis: MSE vs Recovery",
            "filename": "multivariate_sd_mse_vs_recovery.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "work_kj",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"Multivariate SD (KL, Work)",
            "title": "Multivariate SD Analysis: KL vs Work",
            "filename": "multivariate_sd_kl_vs_work.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "open_state_recovery",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"Multivariate SD (KL, Recovery)",
            "title": "Multivariate SD Analysis: KL vs Recovery",
            "filename": "multivariate_sd_kl_vs_recovery.png",
        },
        {
            "x": "kl_div_uniform",
            "y": "mse",
            "x_label": r"Mean KL(P||U$_{uniform}$)",
            "y_label": r"Multivariate SD (KL, MSE)",
            "title": "Multivariate SD Analysis: KL vs MSE",
            "filename": "multivariate_sd_kl_vs_mse.png",
        },
    ]

    def calculate_multivariate_sd(x_vals, y_vals):
        """
        Calculate multivariate standard deviation using the square root of the
        determinant of the covariance matrix (generalized standard deviation).

        Parameters:
        -----------
        x_vals : np.array
            Values for first variable
        y_vals : np.array
            Values for second variable

        Returns:
        --------
        float
            Multivariate standard deviation
        """
        if len(x_vals) < 2 or len(y_vals) < 2:
            return np.nan

        try:
            # Create data matrix
            data_matrix = np.column_stack([x_vals, y_vals])

            # Calculate covariance matrix
            cov_matrix = np.cov(data_matrix.T)

            # Calculate determinant
            det_cov = np.linalg.det(cov_matrix)

            # Return square root of determinant (generalized standard deviation)
            if det_cov >= 0:
                return np.sqrt(det_cov)
            else:
                return np.nan

        except Exception:
            return np.nan

    figures = []

    for pair in scatter_pairs:
        print(f"Creating multivariate SD plot for {pair['title']}")

        # Calculate multivariate SD and mean values for each ensemble, split_type, gamma combination
        multivar_sd_data = []

        for ensemble in ensemble_order:
            for split_type in analysis_split_types:
                # Get data for this ensemble and split type
                subset = plot_df[
                    (plot_df["ensemble"] == ensemble) & (plot_df["split_type"] == split_type)
                ]

                if subset.empty:
                    continue

                # Group by gamma value
                for gamma in subset["gamma"].unique():
                    gamma_data = subset[subset["gamma"] == gamma]

                    if len(gamma_data) < 2:  # Need at least 2 replicates for multivariate SD
                        continue

                    # Get x and y values
                    x_vals = gamma_data[pair["x"]].values
                    y_vals = gamma_data[pair["y"]].values

                    # Handle non-positive values for work_kj (transform to log space for calculation)
                    x_vals_calc = x_vals.copy()
                    y_vals_calc = y_vals.copy()

                    # Filter out invalid values and transform if needed
                    valid_mask = np.ones(len(x_vals), dtype=bool)

                    if pair["x"] == "work_kj":
                        valid_mask &= x_vals > 0
                        x_vals_calc = np.log10(x_vals_calc)
                    if pair["y"] == "work_kj":
                        valid_mask &= y_vals > 0
                        y_vals_calc = np.log10(y_vals_calc)

                    # Remove invalid values
                    x_vals_calc = x_vals_calc[valid_mask]
                    y_vals_calc = y_vals_calc[valid_mask]
                    x_vals_mean = x_vals[valid_mask]  # Keep original scale for mean calculation

                    if len(x_vals_calc) < 2:
                        continue

                    # Calculate multivariate SD
                    multivar_sd = calculate_multivariate_sd(x_vals_calc, y_vals_calc)

                    # Calculate mean of x-variable (in original scale)
                    x_mean = np.mean(x_vals_mean)

                    # Also calculate individual standard deviations for comparison
                    x_std = np.std(x_vals_calc, ddof=1) if len(x_vals_calc) > 1 else 0
                    y_std = np.std(y_vals_calc, ddof=1) if len(y_vals_calc) > 1 else 0

                    multivar_sd_data.append(
                        {
                            "ensemble": ensemble,
                            "split_type": split_type,
                            "split_name": split_name_mapping[split_type],
                            "gamma": gamma,
                            "multivar_sd": multivar_sd,
                            "x_mean": x_mean,
                            "x_std": x_std,
                            "y_std": y_std,
                            "n_replicates": len(x_vals_calc),
                        }
                    )

        if not multivar_sd_data:
            print(f"No multivariate SD data could be calculated for {pair['title']}")
            continue

        multivar_sd_df = pd.DataFrame(multivar_sd_data)

        # Create plot with ensembles in columns
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharey=True, sharex=False
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])

        for col_idx, ensemble in enumerate(ensemble_order):
            ax = axes[col_idx]
            ensemble_data = multivar_sd_df[multivar_sd_df["ensemble"] == ensemble]

            if ensemble_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20
                )
                continue

            # Plot multivariate SD vs mean x-variable for each split type
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]

                if split_data.empty:
                    continue

                # Filter out NaN values
                valid_data = split_data[split_data["multivar_sd"].notna()]

                if valid_data.empty:
                    continue

                # Sort by x_mean for proper line plotting
                valid_data = valid_data.sort_values("x_mean")

                ax.plot(
                    valid_data["x_mean"].to_numpy(),
                    valid_data["multivar_sd"].to_numpy(),
                    color=split_type_dataset_colours[split_type],
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    alpha=0.8,
                    label=split_name_mapping[split_type],
                )

                # Add gamma value annotations on points
                for _, row in valid_data.iterrows():
                    ax.annotate(
                        f"{row['gamma']:.0e}",
                        (float(row["x_mean"]), float(row["multivar_sd"])),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        color=split_type_dataset_colours[split_type],
                    )

            # Set axis properties
            ax.set_xlabel(pair["x_label"], fontsize=20)
            if col_idx == 0:
                ax.set_ylabel(pair["y_label"], fontsize=18)

            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)

            # Set axis scales - log scale for work variables
            if pair["x"] == "work_kj":
                ax.set_xscale("log")

            # Set y-axis to log scale if multivariate SD values span multiple orders of magnitude
            y_data = ensemble_data["multivar_sd"].dropna()
            if len(y_data) > 0 and y_data.max() / y_data.min() > 100:
                ax.set_yscale("log")

            # Set x-axis limits for recovery plots
            if pair["x"] == "open_state_recovery":
                ax.set_xlim(0, 100)

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=12)

        # Add legend
        handles = []
        labels = []
        for split_type in analysis_split_types:
            if split_type in split_type_dataset_colours:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=split_type_dataset_colours[split_type],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        alpha=0.8,
                    )
                )
                labels.append(split_name_mapping[split_type])

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.8,
        )

        plt.tight_layout(rect=[0.05, 0, 1, 1])

        # Save figure
        output_path = os.path.join(output_dir, pair["filename"])
        fig.savefig(output_path, dpi=300)
        print(f"Saved multivariate SD analysis plot to {output_path}")

        figures.append(fig)

    # Save multivariate SD data to CSV
    if multivar_sd_data:
        all_multivar_sd_df = pd.DataFrame(multivar_sd_data)
        output_csv = os.path.join(output_dir, "multivariate_sd_analysis_data.csv")
        all_multivar_sd_df.to_csv(output_csv, index=False)
        print(f"Saved multivariate SD analysis data to {output_csv}")

        # Print summary statistics
        print("\nMultivariate SD Analysis Summary:")
        print("-" * 40)
        print(f"Total data points: {len(all_multivar_sd_df)}")
        for pair in scatter_pairs:
            pair_data = all_multivar_sd_df[all_multivar_sd_df["ensemble"].notna()]
            if not pair_data.empty:
                valid_sd = pair_data["multivar_sd"].dropna()
                if len(valid_sd) > 0:
                    print(f"Multivariate SD range: {valid_sd.min():.4f} to {valid_sd.max():.4f}")
                    break

    return figures


def plot_volcano_recovery_vs_variables(work_df, recovery_df, output_dir):
    """
    Plot volcano plots with log2 fold change in recovery on y-axis vs other variables on x-axis.
    Creates separate plots for each variable: work, 1/MSE, MSE, 1-MSE, and KL divergence.
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
                            fold_change = 1.0  # No change if baseline is 0
                            log2_fold_change = 0.0

                        # Calculate reciprocal MSE (1/MSE)
                        reciprocal_mse = 1.0 / row["mse"] if row["mse"] > 0 else np.inf

                        # Calculate 1-MSE
                        one_minus_mse = 1 - row["mse"]

                        fold_change_data.append(
                            {
                                **row.to_dict(),
                                "recovery_fold_change": fold_change,
                                "log2_recovery_fold_change": log2_fold_change,
                                "baseline_recovery": baseline_recovery,
                                "reciprocal_mse": reciprocal_mse,
                                "one_minus_mse": one_minus_mse,
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
                "reciprocal_mse": ["mean", "std", "count"],
                "one_minus_mse": ["mean", "std", "count"],
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
        "reciprocal_mse_mean",
        "reciprocal_mse_std",
        "reciprocal_mse_count",
        "one_minus_mse_mean",
        "one_minus_mse_std",
        "one_minus_mse_count",
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
            "x_var": "reciprocal_mse_mean",
            "x_err": "reciprocal_mse_std",
            "x_label": r"1/MSE$_{Training}$",
            "title": "Recovery Fold Change vs 1/MSE Training",
            "filename": "volcano_recovery_vs_reciprocal_mse.png",
            "x_scale": "linear",
            "x_transform": None,
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
            "x_var": "one_minus_mse_mean",
            "x_err": "one_minus_mse_std",
            "x_label": r"1 - MSE$_{Training}$",
            "title": "Recovery Fold Change vs 1-MSE Training",
            "filename": "volcano_recovery_vs_one_minus_mse.png",
            "x_scale": "linear",
            "x_transform": None,
        },
        {
            "x_var": "kl_div_uniform_mean",
            "x_err": "kl_div_uniform_std",
            "x_label": r"KL(P||U$_{uniform}$)",
            "title": "Recovery Fold Change vs KL Divergence",
            "filename": "volcano_recovery_vs_kl.png",
            "x_scale": "log",
            "x_transform": None,
        },
    ]

    # Size mapping for gamma values (higher gamma = larger points)
    gamma_values = sorted(averaged_df["gamma"].unique())
    max_size = 200
    min_size = 50

    size_map = {}
    for i, gamma in enumerate(gamma_values):
        # Higher gamma values get larger sizes
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
        # Handle reciprocal MSE data (filter out infinite values)
        elif pair["x_var"] == "reciprocal_mse_mean":
            valid_mask = np.isfinite(x_data) & (x_data > 0)
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
        # Handle 1-MSE data (filter out invalid values if needed)
        elif pair["x_var"] == "one_minus_mse_mean":
            valid_mask = np.isfinite(x_data)
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
                        # Filter out invalid values for reciprocal MSE
                        elif pair["x_var"] == "reciprocal_mse_mean":
                            valid_mask = np.isfinite(x_vals) & (x_vals > 0)
                            x_vals = x_vals[valid_mask]
                            y_vals = y_vals[valid_mask]
                            x_errs = x_errs[valid_mask]
                            y_errs = y_errs[valid_mask]
                            split_data = split_data[valid_mask]
                        # Filter out invalid values for 1-MSE
                        elif pair["x_var"] == "one_minus_mse_mean":
                            valid_mask = np.isfinite(x_vals)
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
                                corr_coef = corr_coef**2
                                correlation_annotations.append(
                                    f"{split_name_mapping[split_type]}: R^2 = {corr_coef:.3f}"
                                )
                            except:
                                correlation_annotations.append(
                                    f"{split_name_mapping[split_type]}: R^2 = NaN"
                                )
                        else:
                            correlation_annotations.append(
                                f"{split_name_mapping[split_type]}: R^2 = NaN"
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


def plot_variable_sd_vs_gamma(work_df, recovery_df, output_dir):
    """
    Plot standard deviation of variables (KL divergence, work, MSE) vs gamma parameter.
    Creates scatter plots with gamma on x-axis and SD on y-axis for each variable.

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing KL divergence values
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    list of matplotlib.Figure
        List of generated figures
    """
    # Merge the dataframes on common columns
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for the ensembles and split types we want
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if plot_df.empty:
        print("No data available for SD vs gamma plots")
        return None

    # Define variables to analyze
    variables = [
        {
            "name": "kl_div_uniform",
            "title": r"KL Divergence vs Uniform",
            "ylabel": r"SD of KL(P||U$_{uniform}$)",
            "filename": "sd_kl_vs_gamma.png",
        },
        {
            "name": "work_kj",
            "title": r"Apparent Work$_{HDXer}$",
            "ylabel": r"SD of Apparent Work$_{HDXer}$ [kJ/mol]",
            "filename": "sd_work_vs_gamma.png",
        },
        {
            "name": "mse",
            "title": r"MSE$_{Training}$",
            "ylabel": r"SD of MSE$_{Training}$",
            "filename": "sd_mse_vs_gamma.png",
        },
    ]

    figures = []

    for var_info in variables:
        print(f"Creating SD vs gamma plot for {var_info['title']}")

        # Calculate standard deviation for each ensemble, split_type, gamma combination
        sd_data = []

        for ensemble in ensemble_order:
            for split_type in analysis_split_types:
                # Get data for this ensemble and split type
                subset = plot_df[
                    (plot_df["ensemble"] == ensemble) & (plot_df["split_type"] == split_type)
                ]

                if subset.empty:
                    continue

                # Group by gamma value and calculate standard deviation
                for gamma in subset["gamma"].unique():
                    gamma_data = subset[subset["gamma"] == gamma]

                    if len(gamma_data) < 2:  # Need at least 2 replicates for meaningful SD
                        continue

                    # Get variable values
                    var_values = gamma_data[var_info["name"]].values

                    # Handle non-positive values for work_kj
                    if var_info["name"] == "work_kj":
                        # Filter out non-positive values
                        valid_mask = var_values > 0
                        if not np.any(valid_mask):
                            continue
                        var_values = var_values[valid_mask]

                        # If we still don't have enough values, skip
                        if len(var_values) < 2:
                            continue

                        # Calculate SD in original space
                        var_sd = np.std(var_values, ddof=1)  # Original space SD
                        var_mean = np.mean(var_values)  # Original scale mean
                    else:
                        # Handle NaN values
                        finite_mask = np.isfinite(var_values)
                        if not np.any(finite_mask):
                            continue
                        var_values = var_values[finite_mask]

                        if len(var_values) < 2:
                            continue

                        var_sd = np.std(var_values, ddof=1)
                        var_mean = np.mean(var_values)

                    sd_data.append(
                        {
                            "ensemble": ensemble,
                            "split_type": split_type,
                            "split_name": split_name_mapping[split_type],
                            "gamma": gamma,
                            "variable_sd": var_sd,
                            "variable_mean": var_mean,
                            "n_replicates": len(var_values),
                            "variable_name": var_info["name"],
                        }
                    )

        if not sd_data:
            print(f"No SD data could be calculated for {var_info['title']}")
            continue

        sd_df = pd.DataFrame(sd_data)

        # Create plot with ensembles in columns
        fig, axes = plt.subplots(
            1, len(ensemble_order), figsize=(6 * len(ensemble_order), 4), sharey=True, sharex=True
        )

        if len(ensemble_order) == 1:
            axes = np.array([axes])

        for col_idx, ensemble in enumerate(ensemble_order):
            ax = axes[col_idx]
            ensemble_data = sd_df[sd_df["ensemble"] == ensemble]

            if ensemble_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20
                )
                continue

            # Plot SD vs gamma for each split type
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]

                if split_data.empty:
                    continue

                # Sort by gamma for proper line plotting
                split_data = split_data.sort_values("gamma")

                ax.scatter(
                    split_data["gamma"].to_numpy(),
                    split_data["variable_sd"].to_numpy(),
                    color=split_type_dataset_colours[split_type],
                    s=80,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=split_name_mapping[split_type],
                    zorder=3,
                )

                # Connect points with lines
                ax.plot(
                    split_data["gamma"].to_numpy(),
                    split_data["variable_sd"].to_numpy(),
                    color=split_type_dataset_colours[split_type],
                    alpha=0.5,
                    linewidth=1.5,
                    zorder=2,
                )

                # Add gamma value annotations on points
                for _, row in split_data.iterrows():
                    ax.annotate(
                        f"{row['n_replicates']}",
                        (float(row["gamma"]), float(row["variable_sd"])),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        color=split_type_dataset_colours[split_type],
                    )

            # Set axis properties
            ax.set_xlabel(r"$\gamma_{HDXer}$ Parameter", fontsize=20)
            if col_idx == 0:
                ax.set_ylabel(var_info["ylabel"], fontsize=18)

            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            ax.set_xscale("log")

            # Set y-axis to log scale if SD values span multiple orders of magnitude
            # y_data = ensemble_data["variable_sd"].dropna()
            # # if len(y_data) > 0 and y_data.max() / y_data.min() > 100:
            ax.set_yscale("log")

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=12)

        # Add legend
        handles = []
        labels = []
        for split_type in analysis_split_types:
            if split_type in split_type_dataset_colours:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color=split_type_dataset_colours[split_type],
                        linewidth=1.5,
                        markersize=8,
                        alpha=0.7,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                    )
                )
                labels.append(split_name_mapping[split_type])

        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.8),
            title="Split Type",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            framealpha=0.8,
        )

        # Add annotation about what the numbers on points represent
        fig.text(
            0.02,
            0.15,
            "Numbers on points = # replicates",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            fontsize=10,
            verticalalignment="bottom",
        )

        plt.tight_layout(rect=[0.05, 0, 1, 1])

        # Save figure
        output_path = os.path.join(output_dir, var_info["filename"])
        fig.savefig(output_path, dpi=300)
        print(f"Saved SD vs gamma plot for {var_info['title']} to {output_path}")

        figures.append(fig)

    # Save SD data to CSV
    if sd_data:
        all_sd_df = pd.DataFrame([item for item in sd_data if item])  # Flatten all data
        output_csv = os.path.join(output_dir, "variable_sd_vs_gamma_data.csv")
        all_sd_df.to_csv(output_csv, index=False)
        print(f"Saved SD vs gamma data to {output_csv}")

        # Print summary statistics
        print("\nVariable SD vs Gamma Analysis Summary:")
        print("-" * 40)
        print(f"Total SD calculations: {len(all_sd_df)}")
        for var_name in all_sd_df["variable_name"].unique():
            var_data = all_sd_df[all_sd_df["variable_name"] == var_name]
            if not var_data.empty:
                print(
                    f"{var_name}: SD range = {var_data['variable_sd'].min():.4f} to {var_data['variable_sd'].max():.4f}"
                )

    return figures


def plot_split_type_correlation_heatmaps(work_df, recovery_df, output_dir):
    """
    Plot correlation heatmaps between split types for each variable used in volcano plots.
    Creates separate heatmap plots for each variable with ensembles side by side.

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing KL divergence and open state recovery
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    list of matplotlib.Figure
        List of generated figures
    """
    # Merge the dataframes
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for analysis ensembles and split types
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if len(plot_df) == 0:
        print("No matching data found for split type correlation heatmaps")
        return []

    print(f"Creating correlation heatmaps with {len(plot_df)} data points")

    # Calculate derived variables (similar to volcano plot preparation)
    plot_df = plot_df.copy()

    # Calculate reciprocal MSE (1/MSE)
    plot_df["reciprocal_mse"] = 1.0 / plot_df["mse"]
    plot_df.loc[plot_df["mse"] <= 0, "reciprocal_mse"] = np.inf

    # Calculate 1-MSE
    plot_df["one_minus_mse"] = 1 - plot_df["mse"]

    # Calculate recovery fold change
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
                    baseline_recovery = subset["unweighted_open_percentage"].iloc[0]

                    for _, row in subset.iterrows():
                        if baseline_recovery > 0:
                            fold_change = row["raw_open_percentage"] / baseline_recovery
                            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
                        else:
                            log2_fold_change = 0.0

                        fold_change_data.append(
                            {**row.to_dict(), "log2_recovery_fold_change": log2_fold_change}
                        )

    if not fold_change_data:
        print("No fold change data could be calculated")
        return []

    extended_df = pd.DataFrame(fold_change_data)

    # Define variables to analyze (hardcoded as requested)
    variables_to_analyze = [
        {
            "name": "work_kj",
            "label": r"Apparent Work$_{HDXer}$ [kJ/mol]",
            "transform": "log10",  # Apply log transform for correlation
            "filename": "split_correlation_heatmap_work.png",
        },
        {
            "name": "reciprocal_mse",
            "label": r"1/MSE$_{Training}$",
            "transform": None,
            "filename": "split_correlation_heatmap_reciprocal_mse.png",
        },
        {
            "name": "mse",
            "label": r"MSE$_{Training}$",
            "transform": None,
            "filename": "split_correlation_heatmap_mse.png",
        },
        {
            "name": "one_minus_mse",
            "label": r"1 - MSE$_{Training}$",
            "transform": None,
            "filename": "split_correlation_heatmap_one_minus_mse.png",
        },
        {
            "name": "kl_div_uniform",
            "label": r"KL(P||U$_{uniform}$)",
            "transform": None,
            "filename": "split_correlation_heatmap_kl.png",
        },
        {
            "name": "log2_recovery_fold_change",
            "label": "Log₂ Recovery Fold Change",
            "transform": None,
            "filename": "split_correlation_heatmap_recovery_fold_change.png",
        },
        {
            "name": "open_state_recovery",
            "label": "Open State Recovery [%]",
            "transform": None,
            "filename": "split_correlation_heatmap_open_state_recovery.png",
        },
    ]

    figures = []

    for var_info in variables_to_analyze:
        print(f"Creating split type correlation heatmap for {var_info['label']}")

        # Create figure with ensembles side by side
        fig, axes = plt.subplots(1, len(ensemble_order), figsize=(6 * len(ensemble_order), 5))
        if len(ensemble_order) == 1:
            axes = np.array([axes])

        fig.suptitle(f"Split Type Correlations: {var_info['label']}", fontsize=16, y=0.95)

        for ensemble_idx, ensemble in enumerate(ensemble_order):
            ax = axes[ensemble_idx]

            ensemble_data = extended_df[extended_df["ensemble"] == ensemble]

            if len(ensemble_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=14
                )
                continue

            # Find common gamma values across all split types for this ensemble
            split_gamma_sets = []
            for split_type in analysis_split_types:
                split_data = ensemble_data[ensemble_data["split_type"] == split_type]
                if len(split_data) > 0:
                    split_gamma_sets.append(set(split_data["gamma"].unique()))

            if not split_gamma_sets:
                ax.text(
                    0.5,
                    0.5,
                    "No split data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=14
                )
                continue

            # Get intersection of gamma values
            common_gammas = set.intersection(*split_gamma_sets)

            if not common_gammas:
                ax.text(
                    0.5,
                    0.5,
                    "No common gamma values",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=14
                )
                continue

            print(f"  {ensemble}: Using {len(common_gammas)} common gamma values")

            # Filter data to common gammas only
            common_data = ensemble_data[ensemble_data["gamma"].isin(common_gammas)]

            # Calculate correlation matrix between split types
            correlation_matrix = np.zeros((len(analysis_split_types), len(analysis_split_types)))
            correlation_matrix.fill(np.nan)

            # Collect data for each split type for common gammas
            split_data_arrays = {}

            for split_idx, split_type in enumerate(analysis_split_types):
                split_subset = common_data[common_data["split_type"] == split_type]

                if len(split_subset) > 0:
                    # Sort by gamma to ensure alignment
                    split_subset = split_subset.sort_values("gamma")

                    # Get variable values and handle transformations
                    var_values = split_subset[var_info["name"]].values
                    gamma_values = split_subset["gamma"].values

                    # Apply transformations and filters based on variable type
                    if var_info["name"] == "work_kj":
                        # Filter positive values
                        valid_mask = var_values > 0
                        if np.any(valid_mask):
                            var_values = var_values[valid_mask]
                            gamma_values = gamma_values[valid_mask]
                            if var_info["transform"] == "log10":
                                var_values = np.log10(var_values)
                        else:
                            var_values = np.array([])
                            gamma_values = np.array([])

                    elif var_info["name"] == "reciprocal_mse":
                        # Filter finite values
                        valid_mask = np.isfinite(var_values) & (var_values > 0)
                        if np.any(valid_mask):
                            var_values = var_values[valid_mask]
                            gamma_values = gamma_values[valid_mask]
                        else:
                            var_values = np.array([])
                            gamma_values = np.array([])

                    else:
                        # Remove NaN/inf values
                        valid_mask = np.isfinite(var_values)
                        if np.any(valid_mask):
                            var_values = var_values[valid_mask]
                            gamma_values = gamma_values[valid_mask]
                        else:
                            var_values = np.array([])
                            gamma_values = np.array([])

                    split_data_arrays[split_type] = {"values": var_values, "gammas": gamma_values}
                else:
                    split_data_arrays[split_type] = {"values": np.array([]), "gammas": np.array([])}

            # Calculate pairwise correlations
            for i, split_type_i in enumerate(analysis_split_types):
                for j, split_type_j in enumerate(analysis_split_types):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        data_i = split_data_arrays.get(split_type_i, {})
                        data_j = split_data_arrays.get(split_type_j, {})

                        values_i = data_i.get("values", np.array([]))
                        gammas_i = data_i.get("gammas", np.array([]))
                        values_j = data_j.get("values", np.array([]))
                        gammas_j = data_j.get("gammas", np.array([]))

                        if len(values_i) > 1 and len(values_j) > 1:
                            try:
                                # Find common gamma values for these two split types
                                common_gamma_ij = np.intersect1d(gammas_i, gammas_j)

                                if len(common_gamma_ij) > 1:
                                    # Get indices for common gammas
                                    i_indices = np.where(np.isin(gammas_i, common_gamma_ij))[0]
                                    j_indices = np.where(np.isin(gammas_j, common_gamma_ij))[0]

                                    # Sort both by gamma to ensure alignment
                                    i_sorted_indices = i_indices[np.argsort(gammas_i[i_indices])]
                                    j_sorted_indices = j_indices[np.argsort(gammas_j[j_indices])]

                                    aligned_values_i = values_i[i_sorted_indices]
                                    aligned_values_j = values_j[j_sorted_indices]

                                    # Calculate correlation if we have enough aligned data
                                    if len(aligned_values_i) > 1 and len(aligned_values_j) > 1:
                                        corr_coef = np.corrcoef(aligned_values_i, aligned_values_j)[
                                            0, 1
                                        ]
                                        correlation_matrix[i, j] = corr_coef

                            except Exception as e:
                                print(
                                    f"    Error calculating correlation between {split_type_i} and {split_type_j}: {e}"
                                )
                                correlation_matrix[i, j] = np.nan

            # Create heatmap
            im = ax.imshow(
                correlation_matrix,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="equal",
                interpolation="nearest",
            )

            # Set ticks and labels
            split_labels = [split_name_mapping[st] for st in analysis_split_types]
            ax.set_xticks(range(len(analysis_split_types)))
            ax.set_yticks(range(len(analysis_split_types)))
            ax.set_xticklabels(split_labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(split_labels, fontsize=10)

            # Apply colors to tick labels based on split type colors
            for i, split_type in enumerate(analysis_split_types):
                color = split_type_dataset_colours[split_type]
                # Color x-axis labels
                ax.get_xticklabels()[i].set_color(color)
                # Color y-axis labels
                ax.get_yticklabels()[i].set_color(color)

            # Add correlation values as text
            for i in range(len(analysis_split_types)):
                for j in range(len(analysis_split_types)):
                    if np.isfinite(correlation_matrix[i, j]):
                        text_color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{correlation_matrix[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontweight="bold",
                        )
                    else:
                        ax.text(j, i, "NaN", ha="center", va="center", color="gray", fontsize=10)

            ax.set_title(
                f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=14, pad=10
            )

            # Add grid for better readability
            ax.set_xticks(np.arange(len(analysis_split_types)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(analysis_split_types)) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

        # Add colorbar
        # cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20)
        # cbar.set_label("Pearson Correlation Coefficient", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save figure
        output_path = os.path.join(output_dir, var_info["filename"])
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved split type correlation heatmap for {var_info['label']} to {output_path}")

        figures.append(fig)

    return figures


def plot_variable_correlation_heatmap(work_df, recovery_df, output_dir):
    """
    Plot correlation heatmap between all pairs of variables (upper triangle only).
    Shows variable-to-variable correlations with ensembles side by side.

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing KL divergence and open state recovery
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    matplotlib.Figure
        Generated figure
    """
    # Merge the dataframes
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for analysis ensembles and split types
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if len(plot_df) == 0:
        print("No matching data found for variable correlation heatmap")
        return None

    print(f"Creating variable correlation heatmap with {len(plot_df)} data points")

    # Calculate derived variables (same as previous functions)
    plot_df = plot_df.copy()
    plot_df["reciprocal_mse"] = 1.0 / plot_df["mse"]
    plot_df.loc[plot_df["mse"] <= 0, "reciprocal_mse"] = np.inf
    plot_df["one_minus_mse"] = 1 - plot_df["mse"]

    # Calculate recovery fold change
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
                    baseline_recovery = subset["unweighted_open_percentage"].iloc[0]

                    for _, row in subset.iterrows():
                        if baseline_recovery > 0:
                            fold_change = row["raw_open_percentage"] / baseline_recovery
                            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
                        else:
                            log2_fold_change = 0.0

                        fold_change_data.append(
                            {**row.to_dict(), "log2_recovery_fold_change": log2_fold_change}
                        )

    if not fold_change_data:
        print("No fold change data could be calculated")
        return None

    extended_df = pd.DataFrame(fold_change_data)

    # Define variables to analyze with proper LaTeX formatting and complete names
    variables_info = [
        {
            "name": "work_kj",
            "label": r"Apparent Work$_{\mathrm{HDXer}}$ [kJ/mol]",
            "transform": "log10",
        },
        {"name": "mse", "label": r"MSE$_{\mathrm{Training}}$", "transform": None},
        {"name": "open_state_recovery", "label": "Open State Recovery (%)", "transform": None},
        {
            "name": "log2_recovery_fold_change",
            "label": r"Log$_2$ Recovery Fold Change",
            "transform": None,
        },
        # {"name": "kl_div_uniform", "label": r"KL(P$||$U$_{\mathrm{uniform}}$)", "transform": None},
    ]

    n_vars = len(variables_info)

    # Create figure with ensembles side by side
    fig, axes = plt.subplots(1, len(ensemble_order), figsize=(8 * len(ensemble_order), 7))
    if len(ensemble_order) == 1:
        axes = np.array([axes])

    fig.suptitle("Entire Variable-to-Variable Correlations", fontsize=24, y=0.95)

    for ensemble_idx, ensemble in enumerate(ensemble_order):
        ax = axes[ensemble_idx]

        ensemble_data = extended_df[extended_df["ensemble"] == ensemble]

        if len(ensemble_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            continue

        print(f"  {ensemble}: Processing {len(ensemble_data)} data points")

        # Prepare data arrays for each variable
        variable_arrays = {}

        for var_info in variables_info:
            var_values = ensemble_data[var_info["name"]].values

            # Apply variable-specific filtering and transformations
            if var_info["name"] == "work_kj":
                valid_mask = var_values > 0
                if np.any(valid_mask):
                    var_values = var_values[valid_mask]
                    if var_info["transform"] == "log10":
                        var_values = np.log10(var_values)
                else:
                    var_values = np.array([])

            elif var_info["name"] == "reciprocal_mse":
                valid_mask = np.isfinite(var_values) & (var_values > 0)
                if np.any(valid_mask):
                    var_values = var_values[valid_mask]
                else:
                    var_values = np.array([])

            else:
                valid_mask = np.isfinite(var_values)
                if np.any(valid_mask):
                    var_values = var_values[valid_mask]
                else:
                    var_values = np.array([])

            variable_arrays[var_info["name"]] = var_values

        # Find minimum common length across all variables
        min_length = min([len(arr) for arr in variable_arrays.values()])

        if min_length < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for correlations",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            continue

        # Truncate all arrays to common length (take first N points)
        # This assumes data is similarly ordered, which should be the case for our dataset
        for var_name in variable_arrays:
            variable_arrays[var_name] = variable_arrays[var_name][:min_length]

        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_vars, n_vars))
        correlation_matrix.fill(np.nan)

        for i, var_info_i in enumerate(variables_info):
            for j, var_info_j in enumerate(variables_info):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    values_i = variable_arrays[var_info_i["name"]]
                    values_j = variable_arrays[var_info_j["name"]]

                    if len(values_i) > 1 and len(values_j) > 1:
                        try:
                            # Both arrays should have same length due to truncation above
                            min_len = min(len(values_i), len(values_j))
                            if min_len > 1:
                                corr_coef = np.corrcoef(values_i[:min_len], values_j[:min_len])[
                                    0, 1
                                ]
                                correlation_matrix[i, j] = corr_coef
                        except Exception as e:
                            print(
                                f"    Error calculating correlation between {var_info_i['name']} and {var_info_j['name']}: {e}"
                            )
                            correlation_matrix[i, j] = np.nan

        # Mask lower triangle (keep only upper triangle and diagonal)
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
        masked_corr = np.ma.array(correlation_matrix, mask=mask)

        # Create heatmap
        im = ax.imshow(
            masked_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest"
        )

        # Set ticks and labels with proper LaTeX formatting
        var_labels = [var_info["label"] for var_info in variables_info]
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(var_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(var_labels, fontsize=9)

        # Add correlation values as text (only for upper triangle)
        for i in range(n_vars):
            for j in range(i, n_vars):  # Only upper triangle and diagonal
                if np.isfinite(correlation_matrix[i, j]):
                    text_color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
                    text = "1.00" if i == j else f"{correlation_matrix[i, j]:.2f}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=10,
                    )

        ax.set_title(
            f"TeaA | {ensemble}",
            color=full_dataset_colours[ensemble],
            fontsize=20,
            pad=10,
        )

        # Add grid for better readability
        ax.set_xticks(np.arange(n_vars) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_vars) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

    # Add colorbar centered below all subplots
    # Leave space at bottom for the horizontal colorbar
    plt.tight_layout(rect=[0, 0.12, 1, 0.92])  # reserve bottom area

    # Create a centered horizontal colorbar axis
    cbar_left = 0.15
    cbar_width = 0.7
    cbar_bottom = 0.04
    cbar_height = 0.03
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Pearson Correlation Coefficient", fontsize=16)
    cbar.ax.tick_params(labelsize=10)

    # Save figure
    output_path = os.path.join(output_dir, "variable_correlation_heatmap_upper_triangle.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved variable correlation heatmap to {output_path}")

    # Save correlation matrices as CSV
    for ensemble_idx, ensemble in enumerate(ensemble_order):
        ensemble_data = extended_df[extended_df["ensemble"] == ensemble]

        if len(ensemble_data) > 0:
            # Recalculate for this ensemble to save data
            variable_arrays = {}

            for var_info in variables_info:
                var_values = ensemble_data[var_info["name"]].values

                if var_info["name"] == "work_kj":
                    valid_mask = var_values > 0
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                        if var_info["transform"] == "log10":
                            var_values = np.log10(var_values)
                    else:
                        var_values = np.array([])

                elif var_info["name"] == "reciprocal_mse":
                    valid_mask = np.isfinite(var_values) & (var_values > 0)
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                    else:
                        var_values = np.array([])

                else:
                    valid_mask = np.isfinite(var_values)
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                    else:
                        var_values = np.array([])

                variable_arrays[var_info["name"]] = var_values

            min_length = min([len(arr) for arr in variable_arrays.values()])

            if min_length > 1:
                # Create DataFrame for this ensemble
                data_for_corr = {}
                for var_info in variables_info:
                    data_for_corr[var_info["label"]] = variable_arrays[var_info["name"]][
                        :min_length
                    ]

                corr_df = pd.DataFrame(data_for_corr)
                corr_matrix = corr_df.corr()

                # Save correlation matrix
                output_csv = os.path.join(
                    output_dir, f"variable_correlations_{ensemble.lower().replace('-', '_')}.csv"
                )
                corr_matrix.to_csv(output_csv)
                print(f"Saved {ensemble} correlation matrix to {output_csv}")

    return fig


def plot_variable_pair_correlation_heatmap(work_df, recovery_df, output_dir):
    """
    Plot a meta-correlation heatmap showing how split type correlation patterns
    relate between different variable pairs. Uses upper triangle correlations only.

    Parameters:
    -----------
    work_df : pd.DataFrame
        Work data containing MSE and work values
    recovery_df : pd.DataFrame
        Recovery data containing KL divergence and open state recovery
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    matplotlib.Figure
        Generated meta-correlation figure
    """
    # Merge the dataframes
    merge_cols = ["ensemble", "split_type", "split_name", "replicate", "gamma"]
    merged_df = pd.merge(work_df, recovery_df, on=merge_cols, how="inner")

    # Filter for analysis ensembles and split types
    plot_df = merged_df[merged_df["ensemble"].isin(ensemble_order)]
    plot_df = plot_df[plot_df["split_type"].isin(analysis_split_types)]

    if len(plot_df) == 0:
        print("No matching data found for meta-correlation heatmap")
        return None

    print(f"Creating meta-correlation heatmap with {len(plot_df)} data points")

    # Calculate derived variables (same as in split type correlation function)
    plot_df = plot_df.copy()
    plot_df["reciprocal_mse"] = 1.0 / plot_df["mse"]
    plot_df.loc[plot_df["mse"] <= 0, "reciprocal_mse"] = np.inf
    plot_df["one_minus_mse"] = 1 - plot_df["mse"]

    # Calculate recovery fold change
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
                    baseline_recovery = subset["unweighted_open_percentage"].iloc[0]

                    for _, row in subset.iterrows():
                        if baseline_recovery > 0:
                            fold_change = row["raw_open_percentage"] / baseline_recovery
                            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
                        else:
                            log2_fold_change = 0.0

                        fold_change_data.append(
                            {**row.to_dict(), "log2_recovery_fold_change": log2_fold_change}
                        )

    if not fold_change_data:
        print("No fold change data could be calculated for meta-correlation")
        return None

    extended_df = pd.DataFrame(fold_change_data)

    # Define variables to analyze with proper LaTeX formatting and complete names
    variables_info = [
        {
            "name": "work_kj",
            "label": r"Apparent Work$_{\mathrm{HDXer}}$ [kJ/mol]",
            "transform": "log10",
        },
        {"name": "mse", "label": r"MSE$_{\mathrm{Training}}$", "transform": None},
        {"name": "open_state_recovery", "label": "Open State Recovery (%)", "transform": None},
        {
            "name": "log2_recovery_fold_change",
            "label": r"Log$_2$ Recovery Fold Change",
            "transform": None,
        },
        # {"name": "kl_div_uniform", "label": r"KL(P$||$U$_{\mathrm{uniform}}$)", "transform": None},
    ]

    # Function to get upper triangle correlations as vector
    def get_upper_triangle_correlations(correlation_matrix):
        """Extract upper triangle correlations (excluding diagonal) as 1D array"""
        n = correlation_matrix.shape[0]
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.isfinite(correlation_matrix[i, j]):
                    upper_triangle.append(correlation_matrix[i, j])
                else:
                    upper_triangle.append(np.nan)
        return np.array(upper_triangle)

    # Calculate split type correlation matrices for each variable and ensemble
    def calculate_split_correlation_matrix(data, variable_info):
        """Calculate correlation matrix between split types for a given variable"""
        correlation_matrix = np.zeros((len(analysis_split_types), len(analysis_split_types)))
        correlation_matrix.fill(np.nan)

        # Find common gamma values across all split types
        split_gamma_sets = []
        for split_type in analysis_split_types:
            split_data = data[data["split_type"] == split_type]
            if len(split_data) > 0:
                split_gamma_sets.append(set(split_data["gamma"].unique()))

        if not split_gamma_sets:
            return correlation_matrix

        common_gammas = set.intersection(*split_gamma_sets)
        if not common_gammas:
            return correlation_matrix

        # Filter to common gammas
        common_data = data[data["gamma"].isin(common_gammas)]

        # Collect data for each split type
        split_data_arrays = {}
        for split_idx, split_type in enumerate(analysis_split_types):
            split_subset = common_data[common_data["split_type"] == split_type]

            if len(split_subset) > 0:
                split_subset = split_subset.sort_values("gamma")
                var_values = split_subset[variable_info["name"]].values
                gamma_values = split_subset["gamma"].values

                # Apply transformations and filters
                if variable_info["name"] == "work_kj":
                    valid_mask = var_values > 0
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                        gamma_values = gamma_values[valid_mask]
                        if variable_info["transform"] == "log10":
                            var_values = np.log10(var_values)
                    else:
                        var_values = np.array([])
                        gamma_values = np.array([])

                elif variable_info["name"] == "reciprocal_mse":
                    valid_mask = np.isfinite(var_values) & (var_values > 0)
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                        gamma_values = gamma_values[valid_mask]
                    else:
                        var_values = np.array([])
                        gamma_values = np.array([])

                else:
                    valid_mask = np.isfinite(var_values)
                    if np.any(valid_mask):
                        var_values = var_values[valid_mask]
                        gamma_values = gamma_values[valid_mask]
                    else:
                        var_values = np.array([])
                        gamma_values = np.array([])

                split_data_arrays[split_type] = {"values": var_values, "gammas": gamma_values}
            else:
                split_data_arrays[split_type] = {"values": np.array([]), "gammas": np.array([])}

        # Calculate pairwise correlations
        for i, split_type_i in enumerate(analysis_split_types):
            for j, split_type_j in enumerate(analysis_split_types):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    data_i = split_data_arrays.get(split_type_i, {})
                    data_j = split_data_arrays.get(split_type_j, {})

                    values_i = data_i.get("values", np.array([]))
                    gammas_i = data_i.get("gammas", np.array([]))
                    values_j = data_j.get("values", np.array([]))
                    gammas_j = data_j.get("gammas", np.array([]))

                    if len(values_i) > 1 and len(values_j) > 1:
                        try:
                            common_gamma_ij = np.intersect1d(gammas_i, gammas_j)

                            if len(common_gamma_ij) > 1:
                                i_indices = np.where(np.isin(gammas_i, common_gamma_ij))[0]
                                j_indices = np.where(np.isin(gammas_j, common_gamma_ij))[0]

                                i_sorted_indices = i_indices[np.argsort(gammas_i[i_indices])]
                                j_sorted_indices = j_indices[np.argsort(gammas_j[j_indices])]

                                aligned_values_i = values_i[i_sorted_indices]
                                aligned_values_j = values_j[j_sorted_indices]

                                if len(aligned_values_i) > 1 and len(aligned_values_j) > 1:
                                    corr_coef = np.corrcoef(aligned_values_i, aligned_values_j)[
                                        0, 1
                                    ]
                                    correlation_matrix[i, j] = corr_coef

                        except Exception:
                            correlation_matrix[i, j] = np.nan

        return correlation_matrix

    # Create figure with ensembles side by side
    fig, axes = plt.subplots(1, len(ensemble_order), figsize=(8 * len(ensemble_order), 7))
    if len(ensemble_order) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Meta-Correlation: Split Type Patterns Across Variable Pairs\n(Upper Triangle Correlations Only)",
        fontsize=24,
        y=0.95,
    )

    for ensemble_idx, ensemble in enumerate(ensemble_order):
        ax = axes[ensemble_idx]

        ensemble_data = extended_df[extended_df["ensemble"] == ensemble]

        if len(ensemble_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20)
            continue

        # Calculate correlation matrices for each variable
        variable_correlation_matrices = {}
        variable_upper_triangles = {}

        print(f"  {ensemble}: Calculating split correlations for each variable...")

        for var_info in variables_info:
            corr_matrix = calculate_split_correlation_matrix(ensemble_data, var_info)
            upper_triangle = get_upper_triangle_correlations(corr_matrix)

            variable_correlation_matrices[var_info["name"]] = corr_matrix
            variable_upper_triangles[var_info["name"]] = upper_triangle

            print(
                f"    {var_info['label']}: {np.sum(np.isfinite(upper_triangle))} valid correlations"
            )

        # Create meta-correlation matrix between variable pairs
        var_names = list(variable_upper_triangles.keys())
        n_vars = len(var_names)
        meta_correlation_matrix = np.zeros((n_vars, n_vars))
        meta_correlation_matrix.fill(np.nan)

        print("    Calculating meta-correlations between variable pairs...")

        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i == j:
                    meta_correlation_matrix[i, j] = 1.0
                else:
                    upper_tri1 = variable_upper_triangles[var1]
                    upper_tri2 = variable_upper_triangles[var2]

                    # Find positions where both have finite values
                    valid_mask = np.isfinite(upper_tri1) & np.isfinite(upper_tri2)

                    if np.sum(valid_mask) > 1:  # Need at least 2 points for correlation
                        try:
                            corr_coef = np.corrcoef(upper_tri1[valid_mask], upper_tri2[valid_mask])[
                                0, 1
                            ]
                            meta_correlation_matrix[i, j] = corr_coef
                        except Exception:
                            meta_correlation_matrix[i, j] = np.nan

        # Mask lower triangle (keep only upper triangle and diagonal)
        mask = np.tril(np.ones_like(meta_correlation_matrix, dtype=bool), k=-1)
        masked_meta_corr = np.ma.array(meta_correlation_matrix, mask=mask)

        # Create heatmap
        im = ax.imshow(
            masked_meta_corr,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )

        # Set ticks and labels with proper LaTeX formatting
        var_labels = [var_info["label"] for var_info in variables_info]
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(var_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(var_labels, fontsize=9)

        # Add correlation values as text (only for upper triangle)
        for i in range(n_vars):
            for j in range(i, n_vars):  # Only upper triangle and diagonal
                if np.isfinite(meta_correlation_matrix[i, j]):
                    text_color = "white" if abs(meta_correlation_matrix[i, j]) > 0.5 else "black"
                    text = "1.00" if i == j else f"{meta_correlation_matrix[i, j]:.2f}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=10,
                    )

        ax.set_title(
            f"TeaA | {ensemble}", color=full_dataset_colours[ensemble], fontsize=20, pad=10
        )

        # Add grid for better readability
        ax.set_xticks(np.arange(n_vars) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_vars) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

    # Add colorbar centered below all subplots
    # Leave space at bottom for the horizontal colorbar
    plt.tight_layout(rect=[0, 0.12, 1, 0.92])  # reserve bottom area

    # Create a centered horizontal colorbar axis
    cbar_left = 0.15
    cbar_width = 0.7
    cbar_bottom = 0.04
    cbar_height = 0.03
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Meta-Correlation of Split Type Patterns", fontsize=16)
    cbar.ax.tick_params(labelsize=10)

    # Save figure
    output_path = os.path.join(output_dir, "meta_correlation_variable_pairs_heatmap.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved meta-correlation heatmap to {output_path}")

    # Save the meta-correlation matrices as CSV for further analysis
    for ensemble_idx, ensemble in enumerate(ensemble_order):
        if ensemble_idx < len(axes):
            ensemble_data = extended_df[extended_df["ensemble"] == ensemble]
            if len(ensemble_data) > 0:
                # Calculate and save the meta-correlation matrix for this ensemble
                var_names = [var_info["name"] for var_info in variables_info]
                var_labels = [var_info["label"] for var_info in variables_info]

                # Recalculate for saving (could optimize by storing earlier)
                variable_upper_triangles = {}
                for var_info in variables_info:
                    corr_matrix = calculate_split_correlation_matrix(ensemble_data, var_info)
                    upper_triangle = get_upper_triangle_correlations(corr_matrix)
                    variable_upper_triangles[var_info["name"]] = upper_triangle

                meta_corr_df = pd.DataFrame(index=var_labels, columns=var_labels)

                for i, var1 in enumerate(var_names):
                    for j, var2 in enumerate(var_names):
                        if i == j:
                            meta_corr_df.iloc[i, j] = 1.0
                        else:
                            upper_tri1 = variable_upper_triangles[var1]
                            upper_tri2 = variable_upper_triangles[var2]

                            valid_mask = np.isfinite(upper_tri1) & np.isfinite(upper_tri2)

                            if np.sum(valid_mask) > 1:
                                try:
                                    corr_coef = np.corrcoef(
                                        upper_tri1[valid_mask], upper_tri2[valid_mask]
                                    )[0, 1]
                                    meta_corr_df.iloc[i, j] = corr_coef
                                except Exception:
                                    meta_corr_df.iloc[i, j] = np.nan
                            else:
                                meta_corr_df.iloc[i, j] = np.nan

                csv_path = os.path.join(
                    output_dir, f"meta_correlation_matrix_{ensemble.replace('-', '_')}.csv"
                )
                meta_corr_df.to_csv(csv_path)
                print(f"Saved meta-correlation matrix for {ensemble} to {csv_path}")

    return fig


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

    # Create plots
    print("Creating plots...")

    if not recovery_df.empty:
        recovery_fig = plot_open_state_recovery(recovery_df, output_dir)
        print("Open state recovery plot saved")

    if not work_df.empty and not recovery_df.empty:
        metrics_figs = plot_training_metrics(work_df, recovery_df, output_dir)
        print("Training metrics plots saved")

    # Create scatter plots
    if not work_df.empty and not recovery_df.empty:
        print("Creating scatter plots...")
        scatter_figs = plot_scatter_metrics(work_df, recovery_df, output_dir)
        print("Scatter plots saved")
    if not work_df.empty and not recovery_df.empty:
        print("Creating scatter plots...")
        scatter_figs = plot_scatter_metrics_as_lines(work_df, recovery_df, output_dir)
        print("Scatter plots saved")
    if not work_df.empty and not recovery_df.empty:
        print("Creating scatter plots...")
        scatter_figs = plot_line_metrics_variable_pairs(work_df, recovery_df, output_dir)
        print("Scatter plots saved")

    # Create R² analysis plots
    if not work_df.empty and not recovery_df.empty:
        print("Creating R² analysis plots...")
        r_squared_figs = plot_r_squared_vs_mean(work_df, recovery_df, output_dir)
        if r_squared_figs:
            print("R² analysis plots saved")
        else:
            print("R² analysis plots could not be created")

    # Create multivariate SD analysis plots
    if not work_df.empty and not recovery_df.empty:
        print("Creating multivariate SD analysis plots...")
        multivar_sd_figs = plot_multivariate_sd_vs_mean(work_df, recovery_df, output_dir)
        if multivar_sd_figs:
            print("Multivariate SD analysis plots saved")
        else:
            print("Multivariate SD analysis plots could not be created")

    # Create volcano plot
    if not work_df.empty and not recovery_df.empty:
        print("Creating HDXer volcano plot...")
        volcano_fig, volcano_data = plot_volcano_kl_recovery_hdxer(work_df, recovery_df, output_dir)
        if volcano_fig is not None:
            print("HDXer volcano plot saved")
        else:
            print("HDXer volcano plot could not be created")

    # Create volcano plots for recovery vs variables
    if not work_df.empty and not recovery_df.empty:
        print("Creating volcano plots for recovery vs variables...")
        volcano_var_figs = plot_volcano_recovery_vs_variables(work_df, recovery_df, output_dir)
        if volcano_var_figs:
            print("Volcano plots for recovery vs variables saved")
        else:
            print("Volcano plots for recovery vs variables could not be created")
    # Create SD vs gamma plots
    if not work_df.empty and not recovery_df.empty:
        print("Creating variable SD vs gamma plots...")
        sd_vs_gamma_figs = plot_variable_sd_vs_gamma(work_df, recovery_df, output_dir)
        if sd_vs_gamma_figs:
            print("Variable SD vs gamma plots saved")
        else:
            print("Variable SD vs gamma plots could not be created")
    if not work_df.empty and not recovery_df.empty:
        print("Creating split type correlation heatmaps...")
        correlation_heatmap_figs = plot_split_type_correlation_heatmaps(
            work_df, recovery_df, output_dir
        )

        # plot variable correlation heatmap
        variable_corr_fig = plot_variable_correlation_heatmap(work_df, recovery_df, output_dir)

        # plot variable pair meta-correlation heatmap
        variable_pair_corr_fig = plot_variable_pair_correlation_heatmap(
            work_df, recovery_df, output_dir
        )

        if correlation_heatmap_figs:
            print("Split type correlation heatmaps saved")
        else:
            print("Split type correlation heatmaps could not be created")

    # Save data to CSV for further analysis
    if not work_df.empty:
        work_df.to_csv(os.path.join(output_dir, "work_data.csv"), index=False)

    if not recovery_df.empty:
        recovery_df.to_csv(os.path.join(output_dir, "recovery_data.csv"), index=False)

    print(f"Analysis complete! Results saved to: {output_dir}")

    # Save data to CSV for further analysis
    if not work_df.empty:
        work_df.to_csv(os.path.join(output_dir, "work_data.csv"), index=False)

    if not recovery_df.empty:
        recovery_df.to_csv(os.path.join(output_dir, "recovery_data.csv"), index=False)

    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
