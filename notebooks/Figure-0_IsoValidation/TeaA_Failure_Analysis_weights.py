"""
JAX-ENT Weight Failure Analysis

Summary:
    Analysis pipeline for investigating failure modes of JAX-ENT frame weight optimization.
    The script loads optimization results from HDF5 files, extracts frame weights from different
    maxent regularization values, maps frames to structural states (open/closed/intermediate)
    by RMSD to reference structures, and computes metrics that quantify how concentrated
    or informative the weights are.

Capabilities:
    - Load JAX-ENT optimization results from HDF5 files with different maxent values
    - Extract frame weights from optimization histories at convergence
    - Compute KL divergence of weight distributions vs uniform and pairwise KLD between splits
    - Compute sequential KLD across increasing maxent values
    - Compute per-condition "open state recovery" by mapping frames to clusters using RMSD
    - Produce publication-ready plots analogous to HDXer analysis:
        * Weight distribution line plots and heatmaps (per ensemble / split type)
        * Open-state recovery vs maxent scatter
        * KLD between splits vs maxent and vs uniform
        * Sequential-maxent KLD figures and comparisons

Inputs / Assumptions:
    - results_dir: directory containing split_type subdirectories with HDF5 optimization results
    - trajectory_paths: mapping of ensemble names to trajectory files
    - reference_paths: two PDBs (open, closed) used as cluster references
    - Requires: Python packages numpy, pandas, matplotlib, seaborn, MDAnalysis, h5py

Outputs:
    - PNG figures saved to output_dir (weight distributions, heatmaps, KLD and recovery plots)
    - DataFrames built in-memory and saved as CSV files
    - Summary statistics printed to stdout

Usage:
    - Configure paths and parameters in main() function before running
    - Run as a script: python jaxent_failure_analysis.py
    - The script handles missing data gracefully and provides detailed logging
"""

import os
import re
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import rms

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

# Import the HDF5 loading functions
from jaxent.src.utils.hdf import load_optimization_history_from_file

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

# Define color schemes
ensemble_colours = {
    "ISO_TRI": "purple",
    "ISO_BI": "indigo",
}

split_type_colours = {
    "R3": "green",
    "Sp": "grey",
    "r": "fuchsia",
    "s": "black",
}

split_name_mapping = {"R3": "Non-Redundant", "Sp": "Spatial", "r": "Random", "s": "Sequence"}

# Target ratios for open/closed states
target_ratios = {"open": 0.4, "closed": 0.6}


def extract_maxent_value_from_filename(filename: str) -> Optional[float]:
    """
    Extract maxent value from filename.
    """
    match = re.search(r"maxent(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return None


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    maxent_values: List[float] = None,
) -> Dict:
    """
    Load all optimization results from HDF5 files, including maxent values.
    """
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    print(f"Scanning results directory: {results_dir}")
    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]
    print(f"Found split type directories: {split_types}")

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)
        print(f"\nProcessing split type: {split_type}")

        # List all files in this directory
        all_files = os.listdir(split_type_dir)
        hdf5_files = [f for f in all_files if f.endswith(".hdf5")]
        print(
            f"  Found {len(hdf5_files)} HDF5 files: {hdf5_files[:5]}{'...' if len(hdf5_files) > 5 else ''}"
        )

        for ensemble in ensembles:
            results[split_type][ensemble] = {}
            print(f"  Processing ensemble: {ensemble}")

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}
                print(f"    Processing loss: {loss_name}")

                # Discover all files for this ensemble/loss combination
                pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                files = [f for f in hdf5_files if f.startswith(pattern)]
                print(f"      Found {len(files)} matching files for pattern '{pattern}*'")
                if files:
                    print(f"        Examples: {files[:3]}")

                files_loaded = 0
                files_failed = 0

                for filename in files:
                    # Extract split index and maxent value
                    match = re.search(r"split(\d{3})_maxent(\d+(?:\.\d+)?)", filename)
                    if match:
                        split_idx = int(match.group(1))
                        maxent_val = float(match.group(2))
                    else:
                        # Handle files without maxent value (original optimization)
                        match = re.search(r"split(\d{3})", filename)
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = 0.0  # Use 0 for no maxent regularization
                        else:
                            print(f"        Could not parse filename: {filename}")
                            continue

                    # Initialize nested dict if needed
                    if maxent_val not in results[split_type][ensemble][loss_name]:
                        results[split_type][ensemble][loss_name][maxent_val] = {}

                    filepath = os.path.join(split_type_dir, filename)

                    try:
                        history = load_optimization_history_from_file(filepath)
                        results[split_type][ensemble][loss_name][maxent_val][split_idx] = history
                        files_loaded += 1

                        # Basic validation
                        if history is not None and hasattr(history, "states"):
                            n_states = len(history.states) if history.states else 0
                            print(f"        ✓ {filename}: {n_states} states")
                        else:
                            print(f"        ⚠ {filename}: loaded but no states")

                    except Exception as e:
                        files_failed += 1
                        print(f"        ✗ {filename}: {e}")
                        results[split_type][ensemble][loss_name][maxent_val][split_idx] = None

                print(f"      Summary: {files_loaded} loaded, {files_failed} failed")

    return results


def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
    """
    Compute RMSD of trajectory frames to reference structures.
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
        raise RuntimeError(f"Error computing RMSD: {e}")


def cluster_frames_by_rmsd(rmsd_values, threshold=1.0):
    """
    Cluster frames based on RMSD to reference structures.
    """
    cluster_assignments = np.argmin(rmsd_values, axis=1)
    min_rmsd = np.min(rmsd_values, axis=1)
    valid_clusters = min_rmsd <= threshold
    cluster_assignments[~valid_clusters] = 2  # intermediate state
    return cluster_assignments


def compute_kl_divergence_uniform(weights, epsilon=1e-10):
    """
    Compute KL divergence between weights distribution and uniform distribution.
    """
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        return np.nan

    weights_safe = weights + epsilon
    weights_safe = weights_safe / np.sum(weights_safe)

    n = len(weights)
    uniform_prob = 1.0 / n

    kl_div = np.sum(weights_safe * np.log(weights_safe / uniform_prob))
    return kl_div


def compute_kl_divergence_between_distributions(p, q, epsilon=1e-10):
    """
    Compute KL divergence between two probability distributions.
    """
    if np.sum(p) > 0:
        p = p / np.sum(p)
    else:
        return np.nan

    if np.sum(q) > 0:
        q = q / np.sum(q)
    else:
        return np.nan

    p_safe = p + epsilon
    q_safe = q + epsilon
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)

    kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
    return kl_div


def extract_final_weights_from_results(results_dict):
    """
    Extract final converged frame weights from JAX-ENT optimization results.

    Returns:
    --------
    weights_data : list
        List of dictionaries containing weight distribution data
    """
    weights_data = []
    total_attempts = 0
    successful_extractions = 0

    print("Extracting weights from results...")
    for split_type in results_dict:
        print(f"  Split type: {split_type}")
        for ensemble in results_dict[split_type]:
            print(f"    Ensemble: {ensemble}")
            for loss_name in results_dict[split_type][ensemble]:
                print(f"      Loss: {loss_name}")
                for maxent_val in results_dict[split_type][ensemble][loss_name]:
                    print(f"        MaxEnt: {maxent_val}")
                    for split_idx, history in results_dict[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        total_attempts += 1
                        print(f"          Split {split_idx}: ", end="")

                        if history is None:
                            print("No history")
                            continue

                        if not hasattr(history, "states") or not history.states:
                            print("No states in history")
                            continue

                        print(f"{len(history.states)} states, ", end="")

                        # Get final (last) state
                        final_state = history.states[-1]

                        if not hasattr(final_state, "params"):
                            print("No params in final state")
                            continue

                        if not hasattr(final_state.params, "frame_weights"):
                            print("No frame_weights in params")
                            continue

                        if final_state.params.frame_weights is None:
                            print("frame_weights is None")
                            continue

                        frame_weights = np.array(final_state.params.frame_weights)
                        print(f"weights shape: {frame_weights.shape}, ", end="")

                        # Handle NaN/inf values
                        if np.any(np.isnan(frame_weights)) or np.any(np.isinf(frame_weights)):
                            print("cleaning NaN/inf, ", end="")
                            frame_weights = np.nan_to_num(
                                frame_weights, nan=0.0, posinf=0.0, neginf=0.0
                            )

                        # Normalize weights
                        if np.sum(frame_weights) > 0:
                            frame_weights = frame_weights / np.sum(frame_weights)
                            successful_extractions += 1
                            print("✓ SUCCESS")

                            weights_data.append(
                                {
                                    "ensemble": ensemble,
                                    "split_type": split_type,
                                    "split_name": split_name_mapping.get(split_type, split_type),
                                    "split_idx": split_idx,
                                    "loss_function": loss_name,
                                    "maxent_value": maxent_val,
                                    "weights": frame_weights,
                                    "convergence_step": len(history.states) - 1,
                                }
                            )
                        else:
                            print("zero weight sum")

    print("\nWeight extraction summary:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Successful extractions: {successful_extractions}")
    print(
        f"  Success rate: {successful_extractions / total_attempts * 100:.1f}%"
        if total_attempts > 0
        else "  No attempts made"
    )

    return weights_data


def extract_weights_and_compute_state_recovery(
    weights_data, trajectory_paths, topology_path, reference_paths
):
    """
    Compute open state recovery using frame weights and RMSD-based clustering.
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

            # Calculate true unweighted distribution
            n_frames = len(cluster_assignments)
            uniform_weights = np.ones(n_frames) / n_frames

            n_clusters = 3  # open(0), closed(1), intermediate(2)
            unweighted_ratios = np.zeros(n_clusters)

            for cluster_idx in range(n_clusters):
                mask = cluster_assignments == cluster_idx
                unweighted_ratios[cluster_idx] = np.sum(uniform_weights[mask])

            ensemble_unweighted_distributions[ensemble] = {
                "open_ratio": unweighted_ratios[0],
                "closed_ratio": unweighted_ratios[1],
                "intermediate_ratio": unweighted_ratios[2],
                "open_percentage": unweighted_ratios[0] * 100,
                "closed_percentage": unweighted_ratios[1] * 100,
                "intermediate_percentage": unweighted_ratios[2] * 100,
            }

            unique, counts = np.unique(cluster_assignments, return_counts=True)
            cluster_summary = dict(zip(unique, counts))
            print(f"  {ensemble}: {len(cluster_assignments)} frames clustered")
            print(f"    Cluster distribution: {cluster_summary}")
            print(f"    Unweighted open state: {unweighted_ratios[0] * 100:.1f}%")

        except Exception as e:
            raise RuntimeError(f"Error processing trajectory for {ensemble}: {e}")

    recovery_data = []

    for weight_item in weights_data:
        ensemble = weight_item["ensemble"]

        if ensemble not in ensemble_clustering:
            print(f"Warning: No clustering data for ensemble {ensemble}, skipping...")
            continue

        cluster_assignments = ensemble_clustering[ensemble]
        unweighted_dist = ensemble_unweighted_distributions[ensemble]
        weights = weight_item["weights"]

        # Compute KL divergence against uniform
        kl_div_uniform = compute_kl_divergence_uniform(weights)

        # Ensure weights match cluster assignments
        if len(weights) != len(cluster_assignments):
            print(
                f"Frame count mismatch for {ensemble}: weights={len(weights)}, clusters={len(cluster_assignments)}"
            )
            if len(weights) > len(cluster_assignments):
                weights = weights[: len(cluster_assignments)]
            else:
                padded_weights = np.zeros(len(cluster_assignments))
                padded_weights[: len(weights)] = weights
                weights = padded_weights

            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)

        # Compute state ratios
        n_clusters = 3
        cluster_ratios = np.zeros(n_clusters)

        for cluster_idx in range(n_clusters):
            mask = cluster_assignments == cluster_idx
            cluster_ratios[cluster_idx] = np.sum(weights[mask])

        # Calculate recovery percentages
        raw_open_percentage = cluster_ratios[0] * 100
        raw_closed_percentage = cluster_ratios[1] * 100
        raw_intermediate_percentage = cluster_ratios[2] * 100

        target_open_percentage = target_ratios["open"] * 100
        open_state_recovery = min(100.0, (raw_open_percentage / target_open_percentage) * 100)

        recovery_data.append(
            {
                **weight_item,
                "open_state_recovery": open_state_recovery,
                "raw_open_percentage": raw_open_percentage,
                "closed_state_recovery": raw_closed_percentage,
                "intermediate_state_recovery": raw_intermediate_percentage,
                "kl_div_uniform": kl_div_uniform,
                "unweighted_open_percentage": unweighted_dist["open_percentage"],
                "unweighted_closed_percentage": unweighted_dist["closed_percentage"],
                "unweighted_intermediate_percentage": unweighted_dist["intermediate_percentage"],
            }
        )

    return pd.DataFrame(recovery_data)


def compute_pairwise_kld_between_splits(weights_data):
    """
    Compute pairwise KLD between splits for each ensemble, split_type, loss, and maxent combination.
    """
    print("Computing pairwise KLD between splits...")
    kld_data = []

    # Convert to DataFrame for easier grouping
    weights_df = pd.DataFrame(weights_data)

    # Group by ensemble, split_type, loss_function, and maxent_value
    for (ensemble, split_type, loss_func, maxent_val), group in weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "maxent_value"]
    ):
        splits = group["split_idx"].values
        weights_list = group["weights"].tolist()

        if len(splits) < 2:
            continue

        # Compute pairwise KLD between all pairs of splits
        pairwise_klds = []

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                weights_i = weights_list[i]
                weights_j = weights_list[j]

                # Ensure both weight arrays have the same length
                min_len = min(len(weights_i), len(weights_j))
                weights_i = weights_i[:min_len]
                weights_j = weights_j[:min_len]

                # Compute KLD in both directions and take the average (symmetric)
                kld_ij = compute_kl_divergence_between_distributions(weights_i, weights_j)
                kld_ji = compute_kl_divergence_between_distributions(weights_j, weights_i)

                if not (np.isnan(kld_ij) or np.isnan(kld_ji)):
                    avg_kld = (kld_ij + kld_ji) / 2.0
                    pairwise_klds.append(avg_kld)

        if len(pairwise_klds) > 0:
            mean_kld = np.mean(pairwise_klds)
            std_kld = np.std(pairwise_klds)
            sem_kld = std_kld / np.sqrt(len(pairwise_klds))

            kld_data.append(
                {
                    "ensemble": ensemble,
                    "split_type": split_type,
                    "split_name": split_name_mapping.get(split_type, split_type),
                    "loss_function": loss_func,
                    "maxent_value": maxent_val,
                    "mean_kld_between_splits": mean_kld,
                    "std_kld_between_splits": std_kld,
                    "sem_kld_between_splits": sem_kld,
                    "n_pairs": len(pairwise_klds),
                    "n_splits": len(splits),
                }
            )

    return pd.DataFrame(kld_data)


def compute_sequential_maxent_kld(weights_data):
    """
    Compute KLD between sequential maxent values for each ensemble, split_type, loss, and split combination.
    """
    print("Computing KLD between sequential maxent values...")
    sequential_kld_data = []

    # Convert to DataFrame for easier grouping
    weights_df = pd.DataFrame(weights_data)

    # Group by ensemble, split_type, loss_function, and split_idx
    for (ensemble, split_type, loss_func, split_idx), group in weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "split_idx"]
    ):
        # Sort by maxent_value for proper sequential comparison
        group_sorted = group.sort_values("maxent_value")
        maxent_values = group_sorted["maxent_value"].values
        weights_list = group_sorted["weights"].tolist()

        if len(maxent_values) < 2:
            continue

        # For each maxent (except the first), compute KLD with previous maxent
        for i in range(len(maxent_values)):
            current_maxent = maxent_values[i]
            current_weights = weights_list[i]

            if i == 0:
                # Compare first (lowest) maxent to uniform distribution
                n_frames = len(current_weights)
                uniform_weights = np.ones(n_frames) / n_frames

                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_weights, uniform_weights
                )
                previous_maxent = None
                comparison_type = "vs_uniform"
            else:
                # Compare to previous maxent
                previous_maxent = maxent_values[i - 1]
                previous_weights = weights_list[i - 1]

                # Ensure both weight arrays have the same length
                min_len = min(len(current_weights), len(previous_weights))
                current_weights_trimmed = current_weights[:min_len]
                previous_weights_trimmed = previous_weights[:min_len]

                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_weights_trimmed, previous_weights_trimmed
                )
                comparison_type = "vs_previous_maxent"

            if not np.isnan(kld_to_previous):
                sequential_kld_data.append(
                    {
                        "ensemble": ensemble,
                        "split_type": split_type,
                        "split_name": split_name_mapping.get(split_type, split_type),
                        "loss_function": loss_func,
                        "split_idx": split_idx,
                        "current_maxent": current_maxent,
                        "previous_maxent": previous_maxent,
                        "kld_to_previous": kld_to_previous,
                        "comparison_type": comparison_type,
                    }
                )

    return pd.DataFrame(sequential_kld_data)


def plot_weight_distribution_lines(weights_data, output_dir):
    """
    Plot weight distributions as 2D line plots with maxent values as hue.
    """
    print("Creating weight distribution line plots...")

    # Convert to DataFrame for easier manipulation
    weights_df = pd.DataFrame(weights_data)

    if weights_df.empty:
        print("  No weights data available for plotting")
        return

    print(f"  Available data: {len(weights_df)} weight distributions")
    print(f"  Unique ensembles: {weights_df['ensemble'].unique()}")
    print(f"  Unique split types: {weights_df['split_type'].unique()}")
    print(f"  Unique maxent values: {sorted(weights_df['maxent_value'].unique())}")

    # Create plots for each ensemble
    available_ensembles = weights_df["ensemble"].unique()

    for ensemble in available_ensembles:
        ensemble_data = weights_df[weights_df["ensemble"] == ensemble]
        if ensemble_data.empty:
            continue

        print(f"  Creating plots for ensemble: {ensemble}")

        # Get unique split types for this ensemble
        split_types = sorted(ensemble_data["split_type"].unique())
        print(f"    Split types: {split_types}")

        if not split_types:
            continue

        # Create figure with subplots
        n_plots = min(len(split_types), 4)  # Max 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, split_type in enumerate(split_types[:4]):  # Max 4 subplots
            ax = axes[idx]
            split_data = ensemble_data[ensemble_data["split_type"] == split_type]

            if split_data.empty:
                ax.set_visible(False)
                continue

            print(f"      Split {split_type}: {len(split_data)} data points")

            # Group by maxent and compute average histogram across splits
            maxent_groups = {}
            for _, row in split_data.iterrows():
                maxent = row["maxent_value"]
                if maxent not in maxent_groups:
                    maxent_groups[maxent] = []
                maxent_groups[maxent].append(row["weights"])

            # Create colormap for maxent values
            maxent_values = sorted(maxent_groups.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

            print(f"        MaxEnt values: {maxent_values}")

            # Define weight bins
            weight_bins = np.logspace(-50, 0, 50)
            bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

            for maxent_val, color in zip(maxent_values, colors):
                weights_list = maxent_groups[maxent_val]

                # Compute histogram for each split and average
                hist_counts = []
                for weights in weights_list:
                    counts, _ = np.histogram(weights, bins=weight_bins, density=True)
                    hist_counts.append(counts)

                if len(hist_counts) > 0:
                    # Average across splits
                    mean_counts = np.mean(hist_counts, axis=0)
                    std_counts = (
                        np.std(hist_counts, axis=0)
                        if len(hist_counts) > 1
                        else np.zeros_like(mean_counts)
                    )

                    # Plot line with error band
                    ax.plot(
                        bin_centers,
                        mean_counts,
                        color=color,
                        alpha=0.8,
                        label=f"MaxEnt={maxent_val:.0e}",
                        linewidth=2,
                    )
                    if len(hist_counts) > 1:
                        ax.fill_between(
                            bin_centers,
                            mean_counts - std_counts,
                            mean_counts + std_counts,
                            color=color,
                            alpha=0.2,
                        )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")
            ax.set_title(f"{split_name_mapping.get(split_type, split_type)}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(split_types), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Weight Distributions - {ensemble}", fontsize=16, y=0.98)
        plt.tight_layout()

        # Save figure
        filename = f"weight_distributions_lines_{ensemble.replace('/', '_').replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_weight_recovery_scatter(recovery_df, output_dir):
    """
    Plot scatter plots of open state recovery vs maxent parameters.
    """
    print("Creating open state recovery scatter plots...")

    if recovery_df.empty:
        print("  No recovery data available for plotting.")
        return

    print(f"  Available recovery data: {len(recovery_df)} points")
    print(f"  Unique ensembles: {recovery_df['ensemble'].unique()}")
    print(f"  Unique split types: {recovery_df['split_type'].unique()}")

    # Get unique ensembles from the actual data
    available_ensembles = recovery_df["ensemble"].unique()

    for ensemble in available_ensembles:
        ensemble_data = recovery_df[recovery_df["ensemble"] == ensemble]

        if ensemble_data.empty:
            continue

        print(f"  Creating recovery plot for ensemble: {ensemble}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_used = []

        # Plot for each split type
        available_split_types = ensemble_data["split_type"].unique()

        # Create a color map for split types
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        for i, split_type in enumerate(available_split_types):
            split_data = ensemble_data[ensemble_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Plot scatter
            ax.scatter(
                split_data["maxent_value"],
                split_data["open_state_recovery"],
                c=[color],
                alpha=0.7,
                label=label,
                s=60,
                edgecolors="w",
            )

            # Connect points for each split
            for split_idx in split_data["split_idx"].unique():
                split_idx_data = split_data[split_data["split_idx"] == split_idx]
                if len(split_idx_data) > 1:
                    split_idx_data = split_idx_data.sort_values("maxent_value")
                    ax.plot(
                        split_idx_data["maxent_value"],
                        split_idx_data["open_state_recovery"],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )

        ax.set_xscale("log")
        ax.set_xlabel("MaxEnt Value")
        ax.set_ylabel("Open State Recovery (%)")
        ax.set_title(f"Open State Recovery vs MaxEnt - {ensemble}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = f"open_state_recovery_scatter_{ensemble.replace('/', '_').replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_kld_between_splits(kld_df, output_dir):
    """
    Plot mean KLD between splits across maxent values.
    """
    print("Creating KLD between splits plot...")

    if kld_df.empty:
        print("  No KLD data available for plotting.")
        return

    print(f"  Available KLD data: {len(kld_df)} points")
    print(f"  Unique ensembles: {kld_df['ensemble'].unique()}")
    print(f"  Unique split types: {kld_df['split_type'].unique()}")

    # Get unique ensembles from actual data
    available_ensembles = kld_df["ensemble"].unique()

    # Create figure with subplots for each ensemble
    fig, axes = plt.subplots(1, len(available_ensembles), figsize=(6 * len(available_ensembles), 6))
    if len(available_ensembles) == 1:
        axes = [axes]

    for idx, ensemble in enumerate(available_ensembles):
        ax = axes[idx]
        ensemble_data = kld_df[kld_df["ensemble"] == ensemble]

        if ensemble_data.empty:
            ax.set_visible(False)
            continue

        print(f"  Creating KLD plot for ensemble: {ensemble}")

        # Get available split types
        available_split_types = ensemble_data["split_type"].unique()
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        # Plot each split type
        for split_type in available_split_types:
            split_data = ensemble_data[ensemble_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Sort by maxent for proper line plotting
            split_data = split_data.sort_values("maxent_value")

            x_vals = split_data["maxent_value"].values
            y_vals = split_data["mean_kld_between_splits"].values
            y_err = split_data["sem_kld_between_splits"].values

            # Plot line with error bars
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                color=color,
                alpha=0.8,
                label=label,
                linewidth=2,
                marker="o",
                markersize=4,
                capsize=3,
            )

        ax.set_xscale("log")
        ax.set_xlabel("MaxEnt Value")
        ax.set_ylabel("Mean KLD Between Splits")
        ax.set_title(ensemble)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("KL Divergence Between Splits Across MaxEnt Values", fontsize=16)
    plt.tight_layout()

    # Save figure
    filename = "kld_between_splits_vs_maxent.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def plot_sequential_maxent_kld(sequential_kld_df, output_dir):
    """
    Plot KLD between sequential maxent values.
    """
    print("Creating sequential maxent KLD plot...")

    if sequential_kld_df.empty:
        print("  No sequential KLD data available for plotting.")
        return

    print(f"  Available sequential KLD data: {len(sequential_kld_df)} points")
    print(f"  Unique ensembles: {sequential_kld_df['ensemble'].unique()}")
    print(f"  Unique split types: {sequential_kld_df['split_type'].unique()}")

    # Get unique ensembles from actual data
    available_ensembles = sequential_kld_df["ensemble"].unique()

    # Create figure with subplots for each ensemble
    fig, axes = plt.subplots(1, len(available_ensembles), figsize=(7 * len(available_ensembles), 6))
    if len(available_ensembles) == 1:
        axes = [axes]

    for idx, ensemble in enumerate(available_ensembles):
        ax = axes[idx]
        ensemble_data = sequential_kld_df[sequential_kld_df["ensemble"] == ensemble]

        if ensemble_data.empty:
            ax.set_visible(False)
            continue

        print(f"  Creating sequential KLD plot for ensemble: {ensemble}")

        # Get available split types
        available_split_types = ensemble_data["split_type"].unique()
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        # Plot each split type
        for split_type in available_split_types:
            split_data = ensemble_data[ensemble_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Plot individual splits as light lines
            for split_idx in split_data["split_idx"].unique():
                split_idx_data = split_data[split_data["split_idx"] == split_idx].sort_values(
                    "current_maxent"
                )

                if len(split_idx_data) > 0:
                    x_vals = split_idx_data["current_maxent"].values
                    y_vals = split_idx_data["kld_to_previous"].values

                    ax.plot(
                        x_vals,
                        y_vals,
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                        marker=".",
                        markersize=2,
                    )

            # Compute and plot mean with error bars
            maxent_stats = (
                split_data.groupby("current_maxent")["kld_to_previous"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            if len(maxent_stats) > 0:
                x_vals = maxent_stats["current_maxent"].values
                y_vals = maxent_stats["mean"].values
                y_err = maxent_stats["std"].values / np.sqrt(maxent_stats["count"].values)

                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=y_err,
                    color=color,
                    alpha=0.8,
                    label=label,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    capsize=3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Current MaxEnt")
        ax.set_ylabel("KLD to Previous MaxEnt (or Uniform)")
        ax.set_title(ensemble)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("KL Divergence Between Sequential MaxEnt Values", fontsize=16)
    plt.tight_layout()

    # Save figure
    filename = "sequential_maxent_kld_vs_maxent.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def main():
    """
    Main function to run the JAX-ENT weight failure analysis.
    """
    # Configuration
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    maxent_values = [1, 2, 5, 10, 50, 100, 500, 1000, 10000]

    # Define directories
    results_dir = "../fitting/jaxENT/_optimise_maxent_HDXer"
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    output_dir = "_analysis_maxent_HDXer_failure_weights"
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Trajectory and reference paths
    bi_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc"
    tri_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_initial_sliced.xtc"

    trajectory_paths = {
        "ISO_TRI": tri_path,
        "ISO_BI": bi_path,
    }

    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    topology_path = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    reference_paths = [
        os.path.join(traj_dir, "TeaA_ref_open_state.pdb"),  # Index 0: Open
        os.path.join(traj_dir, "TeaA_ref_closed_state.pdb"),  # Index 1: Closed
    ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Starting JAX-ENT Weight Failure Analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"MaxEnt values: {maxent_values}")
    print("-" * 60)

    # Load optimization results
    print("Loading optimization results...")
    results = load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        maxent_values=maxent_values,
    )

    if not results:
        print("No optimization results loaded!")
        return

    # Debug: Print structure of loaded results
    print("\nLoaded results structure:")
    for split_type in results:
        print(f"  Split type: {split_type}")
        for ensemble in results[split_type]:
            print(f"    Ensemble: {ensemble}")
            for loss_name in results[split_type][ensemble]:
                print(f"      Loss: {loss_name}")
                for maxent_val in results[split_type][ensemble][loss_name]:
                    n_splits = len(results[split_type][ensemble][loss_name][maxent_val])
                    n_valid = sum(
                        1
                        for h in results[split_type][ensemble][loss_name][maxent_val].values()
                        if h is not None
                    )
                    print(f"        MaxEnt {maxent_val}: {n_splits} splits, {n_valid} valid")

    # Extract final weights
    print("\nExtracting final frame weights...")
    weights_data = extract_final_weights_from_results(results)

    if not weights_data:
        print("No weight data extracted!")
        return

    print(f"\nExtracted {len(weights_data)} weight distributions")

    # Debug: Print summary of extracted weights data
    weights_df_debug = pd.DataFrame(weights_data)
    print("\nWeights data summary:")
    print(f"  Unique ensembles: {weights_df_debug['ensemble'].unique()}")
    print(f"  Unique split types: {weights_df_debug['split_type'].unique()}")
    print(f"  Unique loss functions: {weights_df_debug['loss_function'].unique()}")
    print(f"  Unique maxent values: {sorted(weights_df_debug['maxent_value'].unique())}")
    print(f"  Weight array shapes: {[w.shape for w in weights_df_debug['weights'].head()]}")

    # Save raw weights data for debugging
    weights_summary_df = weights_df_debug.drop("weights", axis=1)  # Remove weights column for CSV
    weights_summary_path = os.path.join(output_dir, "weights_summary_debug.csv")
    weights_summary_df.to_csv(weights_summary_path, index=False)
    print(f"Weights summary saved to: {weights_summary_path}")

    # Compute state recovery
    print("\nComputing state recovery...")
    try:
        recovery_df = extract_weights_and_compute_state_recovery(
            weights_data, trajectory_paths, topology_path, reference_paths
        )

        if not recovery_df.empty:
            # Save recovery data
            recovery_path = os.path.join(output_dir, "state_recovery_data.csv")
            recovery_df.to_csv(recovery_path, index=False)
            print(f"State recovery data saved to: {recovery_path}")

            # Debug recovery data
            print("\nRecovery data summary:")
            print(f"  Shape: {recovery_df.shape}")
            print(f"  Columns: {recovery_df.columns.tolist()}")
            print(
                f"  Open recovery range: {recovery_df['open_state_recovery'].min():.1f} - {recovery_df['open_state_recovery'].max():.1f}"
            )
        else:
            print("No recovery data computed!")

    except Exception as e:
        print(f"Error computing state recovery: {e}")
        import traceback

        traceback.print_exc()
        recovery_df = pd.DataFrame()

    # Compute pairwise KLD between splits
    print("Computing pairwise KLD between splits...")
    kld_df = compute_pairwise_kld_between_splits(weights_data)

    if not kld_df.empty:
        # Save KLD data
        kld_path = os.path.join(output_dir, "kld_between_splits_data.csv")
        kld_df.to_csv(kld_path, index=False)
        print(f"KLD between splits data saved to: {kld_path}")

    # Compute sequential maxent KLD
    print("Computing sequential maxent KLD...")
    sequential_kld_df = compute_sequential_maxent_kld(weights_data)

    if not sequential_kld_df.empty:
        # Save sequential KLD data
        seq_kld_path = os.path.join(output_dir, "sequential_maxent_kld_data.csv")
        sequential_kld_df.to_csv(seq_kld_path, index=False)
        print(f"Sequential maxent KLD data saved to: {seq_kld_path}")

    # Create plots
    print("Creating plots...")

    # Weight distribution plots
    plot_weight_distribution_lines(weights_data, output_dir)

    # Recovery scatter plots
    if not recovery_df.empty:
        plot_weight_recovery_scatter(recovery_df, output_dir)

    # KLD plots
    if not kld_df.empty:
        plot_kld_between_splits(kld_df, output_dir)

    # Sequential KLD plots
    if not sequential_kld_df.empty:
        plot_sequential_maxent_kld(sequential_kld_df, output_dir)

    print("\n HDXer Weight Failure Analysis Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
