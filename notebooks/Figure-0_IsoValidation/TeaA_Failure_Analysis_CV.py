""" """

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

# Import the HDF5 loading functions from the provided script
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
full_dataset_colours = {
    "ISO_TRI": "saddlebrown",
    "ISO_BI": "indigo",
}

split_type_colours = {
    "random": "fuchsia",
    "sequence": "black",
    "nonredundant": "green",
    "spatial": "grey",
}


def extract_maxent_value_from_filename(filename: str) -> Optional[float]:
    """
    Extract maxent value from filename.

    Args:
        filename: HDF5 filename containing maxent value

    Returns:
        Maxent value or None if not found
    """
    match = re.search(r"maxent(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return None


def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
    """
    Compute RMSD of trajectory frames to reference structures.

    Args:
        trajectory_path (str): Path to trajectory file
        topology_path (str): Path to topology file
        reference_paths (list): List of paths to reference structures

    Returns:
        np.ndarray: RMSD values (n_frames, n_refs)
    """
    # Load trajectory
    traj = mda.Universe(topology_path, trajectory_path)

    # Initialize RMSD arrays
    n_frames = len(traj.trajectory)
    n_refs = len(reference_paths)
    rmsd_values = np.zeros((n_frames, n_refs))

    # Compute RMSD for each reference structure
    for j, ref_path in enumerate(reference_paths):
        # Create a new Universe with the trajectory and reference selection
        mobile = mda.Universe(topology_path, trajectory_path)
        reference = mda.Universe(ref_path)

        # Select CA atoms
        mobile_ca = mobile.select_atoms("name CA")
        ref_ca = reference.select_atoms("name CA")

        # Ensure selecting same atoms from both
        if len(ref_ca) != len(mobile_ca):
            print(
                f"Warning: CA atom count mismatch - Trajectory: {len(mobile_ca)}, Reference {j}: {len(ref_ca)}"
            )

        # Calculate RMSD
        R = rms.RMSD(mobile, reference, select="name CA", ref_frame=0)
        R.run()

        # Store RMSD values (column 2 has the RMSD after rotation)
        rmsd_values[:, j] = R.rmsd[:, 2]

    return rmsd_values


def cluster_by_rmsd(rmsd_values, rmsd_threshold=1.0):
    """
    Cluster frames based on RMSD to reference structures.

    Args:
        rmsd_values (np.ndarray): RMSD values to reference structures (n_frames, n_refs)
        rmsd_threshold (float): RMSD threshold for clustering

    Returns:
        np.ndarray: Cluster assignments (0 = open-like, 1 = closed-like, 2 = intermediate)
    """
    # Simple clustering: assign to closest reference if within threshold
    cluster_assignments = np.argmin(rmsd_values, axis=1)

    # Check if frames are within threshold of any reference
    min_rmsd = np.min(rmsd_values, axis=1)
    valid_clusters = min_rmsd <= rmsd_threshold

    # Set invalid clusters to intermediate state (2)
    cluster_assignments[~valid_clusters] = 2

    return cluster_assignments


def calculate_cluster_ratios(cluster_assignments, frame_weights=None):
    """
    Calculate ratios of clusters based on assignments and optional frame weights.

    Args:
        cluster_assignments (np.ndarray): Cluster assignments
        frame_weights (np.ndarray, optional): Frame weights from optimization

    Returns:
        dict: Cluster ratios
    """
    if frame_weights is None:
        frame_weights = np.ones(len(cluster_assignments))

    # Normalize frame weights
    frame_weights = frame_weights / np.sum(frame_weights)

    # Calculate weighted ratios
    ratios = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster in unique_clusters:
        if cluster >= 0:  # Skip invalid clusters (-1)
            mask = cluster_assignments == cluster
            ratios[f"cluster_{cluster}"] = np.sum(frame_weights[mask])

    return ratios


def calculate_recovery_percentage(observed_ratios, ground_truth_ratios):
    """
    Calculate recovery percentage of conformational ratios.

    Args:
        observed_ratios (dict): Observed cluster ratios
        ground_truth_ratios (dict): Ground truth ratios (60:40 Open:Closed)

    Returns:
        dict: Recovery percentages
    """
    recovery = {}

    # Assuming cluster_0 is open-like and cluster_1 is closed-like
    open_observed = observed_ratios.get("cluster_0", 0.0)
    closed_observed = observed_ratios.get("cluster_1", 0.0)

    open_truth = ground_truth_ratios.get("open", 0.4)
    closed_truth = ground_truth_ratios.get("closed", 0.6)

    # Calculate recovery as percentage of truth recovered
    if open_truth > 0:
        recovery["open_recovery"] = min(200.0, (open_observed / open_truth) * 100.0)
    else:
        recovery["open_recovery"] = 0.0

    if closed_truth > 0:
        recovery["closed_recovery"] = min(200.0, (closed_observed / closed_truth) * 100.0)
    else:
        recovery["closed_recovery"] = 0.0

    return recovery


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    num_splits: int = 3,
    maxent_values: List[float] = None,
) -> Dict:
    """
    Load all optimization results from HDF5 files, including maxent values.

    Args:
        results_dir: Directory containing subdirectories for each split type.
        ensembles: List of ensemble names.
        loss_functions: List of loss function names.
        num_splits: Number of data splits per type.
        maxent_values: List of expected maxent values (if None, will discover from files).

    Returns:
        Dictionary with results organized by split_type, ensemble, loss, maxent, and split.
    """
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                # Discover all files for this ensemble/loss combination
                pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                files = [
                    f
                    for f in os.listdir(split_type_dir)
                    if f.startswith(pattern) and f.endswith(".hdf5")
                ]

                for filename in files:
                    # Extract split index and maxent value
                    match = re.search(r"split(\d{3})_maxent(\d+(?:\.\d+)?)", filename)
                    if match:
                        split_idx = int(match.group(1))
                        maxent_val = float(match.group(2))

                        # Initialize nested dict if needed
                        if maxent_val not in results[split_type][ensemble][loss_name]:
                            results[split_type][ensemble][loss_name][maxent_val] = {}

                        filepath = os.path.join(split_type_dir, filename)

                        try:
                            history = load_optimization_history_from_file(filepath)
                            results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                history
                            )
                            print(f"Loaded: {filepath}")
                        except Exception as e:
                            print(f"Failed to load {filepath}: {e}")
                            results[split_type][ensemble][loss_name][maxent_val][split_idx] = None
                    else:
                        # Handle files without maxent value (original optimization)
                        match = re.search(r"split(\d{3})", filename)
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = 0.0  # Use 0 for no maxent regularization

                            if maxent_val not in results[split_type][ensemble][loss_name]:
                                results[split_type][ensemble][loss_name][maxent_val] = {}

                            filepath = os.path.join(split_type_dir, filename)

                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                    history
                                )
                                print(f"Loaded (no maxent): {filepath}")
                            except Exception as e:
                                print(f"Failed to load {filepath}: {e}")
                                results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                    None
                                )

    return results


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calculate KL divergence between two probability distributions.

    Args:
        p: First probability distribution (frame_weights)
        q: Second probability distribution (uniform prior)
        eps: Small value to avoid log(0)

    Returns:
        KL divergence KL(p||q)
    """
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Add small epsilon to avoid log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Calculate KL divergence: KL(p||q) = Σ p(i) * log(p(i)/q(i))
    return np.sum(p * np.log(p / q))


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Calculate Effective Sample Size (ESS) as 1/sum(weights^2).

    Args:
        weights: Frame weights (should be normalized to sum to 1)

    Returns:
        Effective sample size
    """
    # Normalize weights to sum to 1
    normalized_weights = weights / np.sum(weights)

    # Calculate ESS = 1 / sum(w_i^2)
    ess = 1.0 / np.sum(normalized_weights**2)

    return ess


# NEW: Collective Variable Analysis Functions (adapted from HDXer script)


def compute_collective_variable(trajectory_paths, topology_path, reference_paths, n_bins=50):
    """
    Compute collective variable based on RMSD to reference states.
    CV ranges from 0 (fully open) to 1 (fully closed) based on actual RMSD distances.

    Parameters:
    -----------
    trajectory_paths : dict
        Dictionary mapping ensemble names to trajectory file paths
    topology_path : str
        Path to topology file
    reference_paths : list
        List of [open_ref_path, closed_ref_path]
    n_bins : int, optional
        Deprecated parameter, kept for compatibility. CV is now computed continuously.

    Returns:
    --------
    cv_data : dict
        Dictionary containing CV data for each ensemble
    """
    print("Computing collective variable from combined ensemble RMSD data...")

    # Store RMSD data for all ensembles
    all_rmsd_data = []
    ensemble_frame_info = []

    for ensemble_name, traj_path in trajectory_paths.items():
        print(f"Computing RMSD for {ensemble_name}...")

        if not os.path.exists(traj_path):
            print(f"Warning: Trajectory file not found: {traj_path}")
            continue

        try:
            # Compute RMSD to both references
            rmsd_values = compute_rmsd_to_references(traj_path, topology_path, reference_paths)

            # Store RMSD data
            n_frames = rmsd_values.shape[0]
            for frame_idx in range(n_frames):
                rmsd_open = rmsd_values[frame_idx, 0]  # RMSD to open state
                rmsd_closed = rmsd_values[frame_idx, 1]  # RMSD to closed state

                all_rmsd_data.append([rmsd_open, rmsd_closed])
                ensemble_frame_info.append(
                    {
                        "ensemble": ensemble_name,
                        "frame_idx": frame_idx,
                        "rmsd_open": rmsd_open,
                        "rmsd_closed": rmsd_closed,
                    }
                )

        except Exception as e:
            print(f"Error processing {ensemble_name}: {e}")
            continue

    if not all_rmsd_data:
        raise RuntimeError("No RMSD data could be computed")

    all_rmsd_data = np.array(all_rmsd_data)

    # Compute CV based on RMSD ratio: CV = rmsd_open / (rmsd_open + rmsd_closed)
    cv_values = all_rmsd_data[:, 0] / (all_rmsd_data[:, 0] + all_rmsd_data[:, 1])

    # Handle any NaN values
    cv_values = np.nan_to_num(cv_values, nan=0.5)

    # Organize results by ensemble
    cv_data = {}
    current_idx = 0

    for ensemble_name in trajectory_paths.keys():
        if not os.path.exists(trajectory_paths[ensemble_name]):
            continue

        # Count frames for this ensemble
        ensemble_frames = [
            info for info in ensemble_frame_info if info["ensemble"] == ensemble_name
        ]
        n_ensemble_frames = len(ensemble_frames)

        if n_ensemble_frames > 0:
            # Extract CV values for this ensemble
            cv_values_ensemble = cv_values[current_idx : current_idx + n_ensemble_frames]
            rmsd_open = all_rmsd_data[current_idx : current_idx + n_ensemble_frames, 0]
            rmsd_closed = all_rmsd_data[current_idx : current_idx + n_ensemble_frames, 1]

            cv_data[ensemble_name] = {
                "cv_values": cv_values_ensemble,
                "rmsd_open": rmsd_open,
                "rmsd_closed": rmsd_closed,
                "n_frames": n_ensemble_frames,
            }

            current_idx += n_ensemble_frames

            print(
                f"  {ensemble_name}: {n_ensemble_frames} frames, CV range: {cv_values_ensemble.min():.3f} - {cv_values_ensemble.max():.3f}"
            )
            print(f"    Mean CV: {cv_values_ensemble.mean():.3f} ± {cv_values_ensemble.std():.3f}")

    return cv_data


def compute_cv_weighted_distributions(
    optimization_results, cv_data, n_cv_bins=25, cv_bin_edges=None
):
    """
    Compute weighted CV distributions for each optimization result.

    Parameters:
    -----------
    optimization_results : dict
        Results from load_all_optimization_results_with_maxent()
    cv_data : dict
        CV data from compute_collective_variable()
    n_cv_bins : int
        Number of CV bins (used if cv_bin_edges is None)
    cv_bin_edges : np.ndarray, optional
        Custom bin edges for CV. If None, uses equally spaced bins from 0 to 1.

    Returns:
    --------
    cv_weights_data : list
        List of dictionaries containing CV weight distribution data
    """
    print("Computing weighted CV distributions...")

    if cv_bin_edges is None:
        cv_bin_edges = np.linspace(0, 1, n_cv_bins + 1)

    cv_weights_data = []

    for split_type in optimization_results:
        for ensemble in optimization_results[split_type]:
            if ensemble not in cv_data:
                print(f"Warning: No CV data for ensemble {ensemble}, skipping...")
                continue

            cv_values = cv_data[ensemble]["cv_values"]

            for loss_name in optimization_results[split_type][ensemble]:
                for maxent_val in optimization_results[split_type][ensemble][loss_name]:
                    for split_idx, history in optimization_results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            for step_idx, state in enumerate(history.states):
                                if (
                                    hasattr(state.params, "frame_weights")
                                    and state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(state.params.frame_weights)

                                    if len(frame_weights) == 0 or np.sum(frame_weights) == 0:
                                        continue

                                    # Handle NaN/inf values
                                    if np.any(np.isnan(frame_weights)) or np.any(
                                        np.isinf(frame_weights)
                                    ):
                                        frame_weights = np.nan_to_num(
                                            frame_weights, nan=0.0, posinf=0.0, neginf=0.0
                                        )

                                    # Normalize weights
                                    if np.sum(frame_weights) > 0:
                                        frame_weights = frame_weights / np.sum(frame_weights)
                                    else:
                                        continue

                                    # Ensure weights match CV values length
                                    if len(frame_weights) != len(cv_values):
                                        print(
                                            f"    Frame count mismatch for {ensemble}: weights={len(frame_weights)}, CV={len(cv_values)}"
                                        )
                                        if len(frame_weights) > len(cv_values):
                                            frame_weights = frame_weights[: len(cv_values)]
                                        else:
                                            padded_weights = np.zeros(len(cv_values))
                                            padded_weights[: len(frame_weights)] = frame_weights
                                            frame_weights = padded_weights

                                        # Renormalize
                                        if np.sum(frame_weights) > 0:
                                            frame_weights = frame_weights / np.sum(frame_weights)

                                    # Compute weighted CV histogram
                                    cv_hist, _ = np.histogram(
                                        cv_values,
                                        bins=cv_bin_edges,
                                        weights=frame_weights,
                                        density=True,
                                    )

                                    cv_weights_data.append(
                                        {
                                            "ensemble": ensemble,
                                            "split_type": split_type,
                                            "loss_function": loss_name,
                                            "maxent_value": maxent_val,
                                            "split_idx": split_idx,
                                            "step_idx": step_idx,
                                            "cv_histogram": cv_hist,
                                            "cv_bin_edges": cv_bin_edges,
                                            "weights": frame_weights,
                                            "cv_values": cv_values,
                                        }
                                    )

    return cv_weights_data


def compute_kl_divergence_between_distributions(p, q, epsilon=1e-10):
    """
    Compute KL divergence between two probability distributions.
    """
    # Ensure both distributions are normalized
    if np.sum(p) > 0:
        p = p / np.sum(p)
    else:
        return np.nan

    if np.sum(q) > 0:
        q = q / np.sum(q)
    else:
        return np.nan

    # Add epsilon to avoid log(0)
    p_safe = p + epsilon
    q_safe = q + epsilon

    # Renormalize
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)

    # Compute KL divergence
    kl_div = np.sum(p_safe * np.log(p_safe / q_safe))

    return kl_div


def compute_cv_kld_between_splits(cv_weights_data, cv_data):
    """
    Compute pairwise KLD between different splits for CV distributions.

    Parameters:
    -----------
    cv_weights_data : list
        List of dictionaries containing CV weight distribution data
    cv_data : dict
        Original CV data for reference distributions

    Returns:
    --------
    pd.DataFrame
        DataFrame containing pairwise CV KLD statistics
    """
    print("Computing pairwise CV KLD between splits...")

    cv_kld_data = []

    # Convert to DataFrame for easier grouping
    cv_weights_df = pd.DataFrame(cv_weights_data)

    # Group by ensemble, split_type, loss_function, maxent_value, and step_idx
    for (ensemble, split_type, loss_function, maxent_val, step_idx), group in cv_weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "maxent_value", "step_idx"]
    ):
        splits = group["split_idx"].values
        cv_hists = group["cv_histogram"].tolist()

        if len(splits) < 2:
            continue

        # Get original (unweighted) CV distribution for this ensemble
        if ensemble not in cv_data:
            continue

        original_cv_values = cv_data[ensemble]["cv_values"]
        cv_bin_edges = group.iloc[0]["cv_bin_edges"]

        # Compute original CV histogram (uniform weights)
        original_cv_hist, _ = np.histogram(original_cv_values, bins=cv_bin_edges, density=True)

        # Compute pairwise KLD between all pairs of splits
        pairwise_klds = []
        klds_vs_original = []

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                cv_hist_i = cv_hists[i]
                cv_hist_j = cv_hists[j]

                # Compute KLD in both directions and take the average (symmetric)
                kld_ij = compute_kl_divergence_between_distributions(cv_hist_i, cv_hist_j)
                kld_ji = compute_kl_divergence_between_distributions(cv_hist_j, cv_hist_i)

                if not (np.isnan(kld_ij) or np.isnan(kld_ji)):
                    avg_kld = (kld_ij + kld_ji) / 2.0
                    pairwise_klds.append(avg_kld)

            # Also compute KLD vs original distribution for each split
            cv_hist_i = cv_hists[i]
            kld_vs_orig = compute_kl_divergence_between_distributions(cv_hist_i, original_cv_hist)
            if not np.isnan(kld_vs_orig):
                klds_vs_original.append(kld_vs_orig)

        if len(pairwise_klds) > 0:
            mean_kld = np.mean(pairwise_klds)
            std_kld = np.std(pairwise_klds)
            sem_kld = std_kld / np.sqrt(len(pairwise_klds))

            mean_kld_vs_orig = np.mean(klds_vs_original) if len(klds_vs_original) > 0 else np.nan
            std_kld_vs_orig = np.std(klds_vs_original) if len(klds_vs_original) > 1 else np.nan
            sem_kld_vs_orig = (
                std_kld_vs_orig / np.sqrt(len(klds_vs_original))
                if len(klds_vs_original) > 1
                else np.nan
            )

            cv_kld_data.append(
                {
                    "ensemble": ensemble,
                    "split_type": split_type,
                    "loss_function": loss_function,
                    "maxent_value": maxent_val,
                    "step_idx": step_idx,
                    "mean_cv_kld_between_splits": mean_kld,
                    "std_cv_kld_between_splits": std_kld,
                    "sem_cv_kld_between_splits": sem_kld,
                    "mean_cv_kld_vs_original": mean_kld_vs_orig,
                    "std_cv_kld_vs_original": std_kld_vs_orig,
                    "sem_cv_kld_vs_original": sem_kld_vs_orig,
                    "n_pairs": len(pairwise_klds),
                    "n_splits": len(splits),
                }
            )

    return pd.DataFrame(cv_kld_data)


def compute_cv_sequential_maxent_kld(cv_weights_data, cv_data):
    """
    Compute KLD between sequential maxent values for CV distributions.

    Parameters:
    -----------
    cv_weights_data : list
        List of dictionaries containing CV weight distribution data
    cv_data : dict
        Original CV data for reference distributions

    Returns:
    --------
    pd.DataFrame
        DataFrame containing sequential maxent CV KLD statistics
    """
    print("Computing CV KLD between sequential maxent values...")

    cv_sequential_kld_data = []

    # Convert to DataFrame for easier grouping
    cv_weights_df = pd.DataFrame(cv_weights_data)

    # Group by ensemble, split_type, loss_function, split_idx, and step_idx
    for (ensemble, split_type, loss_function, split_idx, step_idx), group in cv_weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "split_idx", "step_idx"]
    ):
        # Sort by maxent value for proper sequential comparison
        group_sorted = group.sort_values("maxent_value")
        maxent_values = group_sorted["maxent_value"].values
        cv_hists = group_sorted["cv_histogram"].tolist()

        if len(maxent_values) < 2:
            continue

        # Get original CV distribution for this ensemble
        if ensemble not in cv_data:
            continue

        original_cv_values = cv_data[ensemble]["cv_values"]
        cv_bin_edges = group.iloc[0]["cv_bin_edges"]

        # Compute original CV histogram (uniform weights)
        original_cv_hist, _ = np.histogram(original_cv_values, bins=cv_bin_edges, density=True)

        # For each maxent (except the first), compute KLD with previous maxent
        for i in range(len(maxent_values)):
            current_maxent = maxent_values[i]
            current_cv_hist = cv_hists[i]

            if i == 0:
                # Compare first (lowest) maxent to original distribution
                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_cv_hist, original_cv_hist
                )
                previous_maxent = None
                comparison_type = "vs_original"
            else:
                # Compare to previous maxent
                previous_maxent = maxent_values[i - 1]
                previous_cv_hist = cv_hists[i - 1]

                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_cv_hist, previous_cv_hist
                )
                comparison_type = "vs_previous_maxent"

            if not np.isnan(kld_to_previous):
                cv_sequential_kld_data.append(
                    {
                        "ensemble": ensemble,
                        "split_type": split_type,
                        "loss_function": loss_function,
                        "split_idx": split_idx,
                        "step_idx": step_idx,
                        "current_maxent": current_maxent,
                        "previous_maxent": previous_maxent,
                        "cv_kld_to_previous": kld_to_previous,
                        "comparison_type": comparison_type,
                    }
                )

    return pd.DataFrame(cv_sequential_kld_data)


def plot_cv_distribution_heatmaps(cv_weights_data, output_dir, n_cv_bins=25):
    """
    Plot CV distributions as heatmaps with maxent on x-axis and CV bins on y-axis.
    """
    print("Creating CV distribution heatmaps...")

    # Convert to DataFrame for easier manipulation
    cv_df = pd.DataFrame(cv_weights_data)

    # Create plots for each ensemble and split type combination
    for ensemble in cv_df["ensemble"].unique():
        for split_type in cv_df["split_type"].unique():
            subset_data = cv_df[
                (cv_df["ensemble"] == ensemble) & (cv_df["split_type"] == split_type)
            ]

            if subset_data.empty:
                continue

            # Group by loss function
            loss_functions = subset_data["loss_function"].unique()

            if len(loss_functions) == 0:
                continue

            # Create figure with subplots for each loss function
            n_loss = len(loss_functions)
            fig, axes = plt.subplots(1, n_loss, figsize=(6 * n_loss, 8))
            if n_loss == 1:
                axes = [axes]

            for idx, loss_func in enumerate(loss_functions):
                ax = axes[idx]
                loss_data = subset_data[subset_data["loss_function"] == loss_func]

                # Group by maxent and compute average histogram across splits and steps (take last step)
                last_step_data = (
                    loss_data.groupby(["maxent_value", "split_idx"]).last().reset_index()
                )

                maxent_groups = {}
                for _, row in last_step_data.iterrows():
                    maxent = row["maxent_value"]
                    if maxent not in maxent_groups:
                        maxent_groups[maxent] = []
                    maxent_groups[maxent].append(row["cv_histogram"])

                maxent_values = sorted(maxent_groups.keys())

                if not maxent_values:
                    ax.set_visible(False)
                    continue

                # Get CV bin edges (should be the same for all)
                cv_bin_edges = last_step_data.iloc[0]["cv_bin_edges"]
                n_actual_bins = len(cv_bin_edges) - 1

                # Create heatmap data matrix
                heatmap_data = np.zeros((n_actual_bins, len(maxent_values)))

                for j, maxent_val in enumerate(maxent_values):
                    hist_list = maxent_groups[maxent_val]
                    # Average across splits
                    mean_hist = np.mean(hist_list, axis=0)
                    heatmap_data[:, j] = mean_hist

                # Apply log scale to density values (add small value to avoid log(0))
                heatmap_data_log = np.log10(heatmap_data + 1e-10)

                # Create heatmap
                im = ax.imshow(
                    heatmap_data_log,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    interpolation="nearest",
                )

                # Set labels and ticks
                ax.set_xlabel("MaxEnt Value")
                ax.set_ylabel("CV Bin (0=Open, 1=Closed)")
                ax.set_title(f"{loss_func}")

                # Set tick labels for maxent values
                maxent_labels = [f"{g:.0e}" for g in maxent_values]
                ax.set_xticks(range(len(maxent_values)))
                ax.set_xticklabels(maxent_labels, rotation=45, ha="right")

                # Set tick labels for CV bins
                n_yticks = min(8, n_actual_bins)
                ytick_indices = np.linspace(0, n_actual_bins - 1, n_yticks, dtype=int)
                ytick_labels = [f"{cv_bin_edges[i]:.2f}" for i in ytick_indices]
                ax.set_yticks(ytick_indices)
                ax.set_yticklabels(ytick_labels, fontsize=10)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Log Density")

            plt.suptitle(f"CV Distribution Heatmaps - {ensemble} - {split_type}", fontsize=16)
            plt.tight_layout()

            # Save figure
            filename = f"cv_distributions_heatmap_{ensemble}_{split_type}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filename}")
            plt.close()


def plot_cv_kld_analysis(cv_kld_df, cv_sequential_kld_df, output_dir):
    """
    Plot CV KLD analysis results.
    """
    print("Creating CV KLD analysis plots...")

    if cv_kld_df.empty and cv_sequential_kld_df.empty:
        print("No CV KLD data available for plotting.")
        return

    # Plot 1: KLD between splits vs maxent value
    if not cv_kld_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        plot_idx = 0
        for ensemble in cv_kld_df["ensemble"].unique():
            for split_type in cv_kld_df["split_type"].unique():
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]
                subset_data = cv_kld_df[
                    (cv_kld_df["ensemble"] == ensemble) & (cv_kld_df["split_type"] == split_type)
                ]

                if subset_data.empty:
                    continue

                for loss_func in subset_data["loss_function"].unique():
                    loss_data = subset_data[subset_data["loss_function"] == loss_func]
                    loss_data = loss_data.sort_values("maxent_value")

                    color = "red" if loss_func == "MSE" else "blue"

                    # Take last step for each condition
                    final_data = loss_data.groupby("maxent_value").last().reset_index()

                    ax.errorbar(
                        final_data["maxent_value"],
                        final_data["mean_cv_kld_between_splits"],
                        yerr=final_data["sem_cv_kld_between_splits"],
                        color=color,
                        label=f"{loss_func}",
                        marker="o",
                        linewidth=2,
                        capsize=3,
                    )

                ax.set_xscale("log")
                ax.set_xlabel("MaxEnt Value")
                ax.set_ylabel("Mean CV KLD Between Splits")
                ax.set_title(f"{ensemble} - {split_type}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("CV KL Divergence Between Splits", fontsize=16)
        plt.tight_layout()

        filename = "cv_kld_between_splits.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()

    # Plot 2: Sequential maxent KLD
    if not cv_sequential_kld_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        plot_idx = 0
        for ensemble in cv_sequential_kld_df["ensemble"].unique():
            for split_type in cv_sequential_kld_df["split_type"].unique():
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]
                subset_data = cv_sequential_kld_df[
                    (cv_sequential_kld_df["ensemble"] == ensemble)
                    & (cv_sequential_kld_df["split_type"] == split_type)
                ]

                if subset_data.empty:
                    continue

                for loss_func in subset_data["loss_function"].unique():
                    loss_data = subset_data[subset_data["loss_function"] == loss_func]

                    # Group by current maxent and compute statistics
                    stats_data = (
                        loss_data.groupby("current_maxent")["cv_kld_to_previous"]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )
                    stats_data["sem"] = stats_data["std"] / np.sqrt(stats_data["count"])

                    color = "red" if loss_func == "MSE" else "blue"

                    ax.errorbar(
                        stats_data["current_maxent"],
                        stats_data["mean"],
                        yerr=stats_data["sem"],
                        color=color,
                        label=f"{loss_func}",
                        marker="o",
                        linewidth=2,
                        capsize=3,
                    )

                ax.set_xscale("log")
                ax.set_xlabel("Current MaxEnt Value")
                ax.set_ylabel("Mean CV KLD to Previous")
                ax.set_title(f"{ensemble} - {split_type}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("CV KL Divergence Between Sequential MaxEnt Values", fontsize=16)
        plt.tight_layout()

        filename = "cv_sequential_maxent_kld.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def analyze_conformational_recovery_with_maxent(
    trajectory_paths, topology_path, reference_paths, results_dict
):
    """
    Analyze conformational ratio recovery for trajectories with maxent values.
    """
    ground_truth_ratios = {"open": 0.4, "closed": 0.6}
    recovery_data = []

    for ensemble_name, traj_path in trajectory_paths.items():
        print(f"Analyzing conformational recovery for {ensemble_name}...")

        # Compute RMSD to references
        rmsd_values = compute_rmsd_to_references(traj_path, topology_path, reference_paths)

        # Cluster by RMSD
        cluster_assignments = cluster_by_rmsd(rmsd_values, rmsd_threshold=1.0)

        # Calculate unweighted (original) ratios
        original_ratios = calculate_cluster_ratios(cluster_assignments)
        original_recovery = calculate_recovery_percentage(original_ratios, ground_truth_ratios)

        recovery_data.append(
            {
                "ensemble": ensemble_name,
                "loss_function": "Original",
                "split_type": "N/A",
                "split": "N/A",
                "maxent_value": 0.0,
                "convergence_step": "N/A",
                "open_ratio": original_ratios.get("cluster_0", 0.0),
                "closed_ratio": original_ratios.get("cluster_1", 0.0),
                "open_recovery": original_recovery["open_recovery"],
                "closed_recovery": original_recovery["closed_recovery"],
                "total_frames": len(cluster_assignments),
                "clustered_frames": np.sum(cluster_assignments >= 0),
            }
        )

        # Analyze with optimized frame weights including maxent
        for split_type in results_dict:
            if ensemble_name in results_dict[split_type]:
                for loss_name in results_dict[split_type][ensemble_name]:
                    for maxent_val in results_dict[split_type][ensemble_name][loss_name]:
                        for split_idx, history in results_dict[split_type][ensemble_name][
                            loss_name
                        ][maxent_val].items():
                            if history is not None and history.states:
                                for step_idx, state in enumerate(history.states):
                                    if (
                                        hasattr(state.params, "frame_weights")
                                        and state.params.frame_weights is not None
                                    ):
                                        frame_weights = np.array(state.params.frame_weights)

                                        if (
                                            len(frame_weights) == len(cluster_assignments)
                                            and np.sum(frame_weights) > 0
                                        ):
                                            # Calculate weighted ratios
                                            weighted_ratios = calculate_cluster_ratios(
                                                cluster_assignments, frame_weights
                                            )
                                            weighted_recovery = calculate_recovery_percentage(
                                                weighted_ratios, ground_truth_ratios
                                            )

                                            recovery_data.append(
                                                {
                                                    "ensemble": ensemble_name,
                                                    "loss_function": loss_name,
                                                    "split_type": split_type,
                                                    "split": split_idx,
                                                    "maxent_value": maxent_val,
                                                    "convergence_step": step_idx,
                                                    "open_ratio": weighted_ratios.get(
                                                        "cluster_0", 0.0
                                                    ),
                                                    "closed_ratio": weighted_ratios.get(
                                                        "cluster_1", 0.0
                                                    ),
                                                    "open_recovery": weighted_recovery[
                                                        "open_recovery"
                                                    ],
                                                    "closed_recovery": weighted_recovery[
                                                        "closed_recovery"
                                                    ],
                                                    "total_frames": len(cluster_assignments),
                                                    "clustered_frames": np.sum(
                                                        cluster_assignments >= 0
                                                    ),
                                                }
                                            )

    return pd.DataFrame(recovery_data)


def extract_frame_weights_kl_with_maxent(results: Dict) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence and ESS including maxent values.
    """
    data_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            for step_idx, state in enumerate(history.states):
                                if (
                                    hasattr(state.params, "frame_weights")
                                    and state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(state.params.frame_weights)
                                    if len(frame_weights) == 0 or np.sum(frame_weights) == 0:
                                        continue

                                    uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)
                                    try:
                                        kl_div = kl_divergence(frame_weights, uniform_prior)
                                        ess = effective_sample_size(frame_weights)

                                        data_rows.append(
                                            {
                                                "split_type": split_type,
                                                "ensemble": ensemble,
                                                "loss_function": loss_name,
                                                "maxent_value": maxent_val,
                                                "split": split_idx,
                                                "step": step_idx,
                                                "convergence_threshold_step": step_idx,
                                                "kl_divergence": float(kl_div),
                                                "effective_sample_size": float(ess),
                                                "num_frames": len(frame_weights),
                                                "step_number": state.step
                                                if hasattr(state, "step")
                                                else step_idx,
                                            }
                                        )
                                    except Exception as e:
                                        print(
                                            f"Failed to calculate KL/ESS for {split_type}/{ensemble}_{loss_name}_maxent{maxent_val}_split{split_idx}, step {step_idx}: {e}"
                                        )
                                        continue
    return pd.DataFrame(data_rows)


def main():
    """
    Main function to run the complete analysis including maxent values and CV analysis.
    """
    # Define parameters
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    num_splits = 3
    maxent_values = [
        1,
        2,
        5,
        10,
        50,
        100,
        500,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
    ]

    # Define directories
    results_dir = "../fitting/jaxENT/_optimise_maxent_MAE"
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    output_dir = "_analysis_maxent_MAE_CV"
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Define trajectory and reference paths
    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    bi_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc"
    tri_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_initial_sliced.xtc"

    trajectory_paths = {
        "ISO_TRI": tri_path,
        "ISO_BI": bi_path,
    }

    topology_path = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    reference_paths = [
        os.path.join(traj_dir, "TeaA_ref_open_state.pdb"),  # Index 0: Open
        os.path.join(traj_dir, "TeaA_ref_closed_state.pdb"),  # Index 1: Closed
    ]

    # Check if required directories and files exist
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    for path in [topology_path] + list(trajectory_paths.values()) + reference_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")

    print("Starting Complete JAX-ENT Analysis with MaxEnt Values and CV Analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Trajectory directory: {traj_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print(f"MaxEnt values: {maxent_values}")
    print("-" * 60)

    # Load all optimization results with maxent
    print("Loading optimization results with maxent values...")
    results = load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        num_splits=num_splits,
        maxent_values=maxent_values,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: KL Divergence Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE AND ESS ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Extract KL divergences and ESS
    print("Extracting frame weights and calculating KL divergences and ESS...")
    kl_ess_df = extract_frame_weights_kl_with_maxent(results)

    if len(kl_ess_df) > 0:
        print(
            f"Extracted {len(kl_ess_df)} KL divergence and ESS data points from optimization histories"
        )

        # Save the KL divergence and ESS dataset
        kl_ess_df_path = os.path.join(output_dir, "kl_divergence_ess_analysis_maxent_data.csv")
        kl_ess_df.to_csv(kl_ess_df_path, index=False)
        print(f"KL divergence and ESS dataset saved to: {kl_ess_df_path}")

    # Part 2: Conformational Recovery Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 2: CONFORMATIONAL RECOVERY ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Check if both trajectory files exist
    for name, path in trajectory_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trajectory file not found for {name}: {path}. Please check the paths."
            )
        else:
            print(f"Found: {name} -> {path}")

    print("Analyzing conformational recovery with maxent values...")
    recovery_df = analyze_conformational_recovery_with_maxent(
        trajectory_paths, topology_path, reference_paths, results
    )

    if len(recovery_df) > 0:
        # Save the recovery dataset
        recovery_df_path = os.path.join(output_dir, "conformational_recovery_maxent_data.csv")
        recovery_df.to_csv(recovery_df_path, index=False)
        print(f"Conformational recovery dataset saved to: {recovery_df_path}")

        # Print summary statistics
        print("\nConformational Recovery Summary with MaxEnt:")
        print("-" * 40)

        # Summary by maxent value
        maxent_summary = (
            recovery_df[recovery_df["loss_function"] != "Original"]
            .groupby(["split_type", "ensemble", "loss_function", "maxent_value"])
            .last()
            .reset_index()
        )

        for split_type in maxent_summary["split_type"].unique():
            print(f"\nSplit Type: {split_type}")
            split_summary = maxent_summary[maxent_summary["split_type"] == split_type]

            for _, row in split_summary.iterrows():
                print(
                    f"  {row['ensemble']} - {row['loss_function']} - MaxEnt {row['maxent_value']:.0f}: "
                    f"Open Recovery = {row['open_recovery']:.1f}%, "
                    f"Open Ratio = {row['open_ratio']:.3f}"
                )
    else:
        print("No conformational recovery data generated!")

    # Part 3: NEW - Collective Variable Analysis
    print("\n" + "=" * 60)
    print("PART 3: COLLECTIVE VARIABLE (CV) ANALYSIS")
    print("=" * 60)

    try:
        # Compute collective variable from trajectory data
        print("Computing collective variable from trajectory RMSD data...")
        cv_data = compute_collective_variable(trajectory_paths, topology_path, reference_paths)

        if cv_data:
            print("✓ CV computation completed successfully!")

            # Save CV data
            cv_results = []
            for ensemble_name, data in cv_data.items():
                for i, cv_val in enumerate(data["cv_values"]):
                    cv_results.append(
                        {
                            "ensemble": ensemble_name,
                            "frame_idx": i,
                            "cv_value": cv_val,
                            "rmsd_open": data["rmsd_open"][i],
                            "rmsd_closed": data["rmsd_closed"][i],
                        }
                    )
            cv_df = pd.DataFrame(cv_results)
            cv_df.to_csv(os.path.join(output_dir, "cv_data.csv"), index=False)
            print(f"CV data saved to: {os.path.join(output_dir, 'cv_data.csv')}")

            # Compute weighted CV distributions
            print("Computing weighted CV distributions...")
            cv_weights_data = compute_cv_weighted_distributions(results, cv_data, n_cv_bins=50)

            if cv_weights_data:
                print(f"Computed CV weight distributions for {len(cv_weights_data)} conditions")

                # Create CV distribution visualizations
                plot_cv_distribution_heatmaps(cv_weights_data, output_dir, n_cv_bins=50)

                # Compute CV KLD between splits
                print("Computing CV KLD between splits...")
                cv_kld_df = compute_cv_kld_between_splits(cv_weights_data, cv_data)

                if not cv_kld_df.empty:
                    print(f"Computed CV KLD data for {len(cv_kld_df)} conditions")
                    cv_kld_df.to_csv(
                        os.path.join(output_dir, "cv_kld_between_splits.csv"), index=False
                    )

                # Compute CV sequential maxent KLD
                print("Computing CV KLD between sequential maxent values...")
                cv_sequential_kld_df = compute_cv_sequential_maxent_kld(cv_weights_data, cv_data)

                if not cv_sequential_kld_df.empty:
                    print(
                        f"Computed CV sequential maxent KLD for {len(cv_sequential_kld_df)} conditions"
                    )
                    cv_sequential_kld_df.to_csv(
                        os.path.join(output_dir, "cv_sequential_maxent_kld.csv"), index=False
                    )

                # Create CV KLD analysis plots
                plot_cv_kld_analysis(cv_kld_df, cv_sequential_kld_df, output_dir)

                # Save CV weights data summary
                cv_weights_summary = []
                for item in cv_weights_data:
                    summary_item = {
                        k: v
                        for k, v in item.items()
                        if k not in ["cv_histogram", "cv_bin_edges", "weights", "cv_values"]
                    }
                    summary_item["n_cv_bins"] = len(item["cv_histogram"])
                    summary_item["n_frames"] = len(item["cv_values"])
                    cv_weights_summary.append(summary_item)

                cv_weights_summary_df = pd.DataFrame(cv_weights_summary)
                cv_weights_summary_df.to_csv(
                    os.path.join(output_dir, "cv_weights_summary.csv"), index=False
                )

                print("✓ CV weights analysis completed successfully!")

            else:
                print("No CV weight distribution data computed!")

        else:
            print("No CV data computed!")

    except Exception as e:
        print(f"Error in CV analysis: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ANALYSIS WITH MAXENT VALUES, ESS, AND CV COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
