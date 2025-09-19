import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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


def generate_residue_feature_data(num_frames, num_residues):
    """
    Generate sample RMSD-like data for demonstration
    Parameters:
    -----------
    num_frames : int
        Number of frames (snapshots) in the ensemble
    num_residues : int
        Number of residues to visualize
    Returns:
    --------
    numpy.ndarray
        2D array with shape (num_frames, num_residues)
    """
    # Initialize the data matrix
    data = np.zeros((num_frames, num_residues))
    # Generate RMSD-like pattern (increasing over time for some residues)
    for frame in range(num_frames):
        for residue in range(num_residues):
            # Base value with some randomness
            base_value = 0.5 + (frame / num_frames) * 3 * np.random.random()
            # Add pattern: every 5th residue has higher RMSD
            if residue % 5 == 0:
                base_value += np.random.random() * 3
            # Add slight time-dependent increase
            time_factor = frame / (num_frames - 1 if num_frames > 1 else 1)
            data[frame, residue] = base_value + time_factor * 0.5
    return data


def plot_residue_heatmap(data, output_file="residue_heatmap.png", dpi=300, figsize=None):
    """
    Create a minimal grayscale heatmap visualization of residue features
    Parameters:
    -----------
    data : numpy.ndarray
        2D array with shape (num_frames, num_residues)
    output_file : str
        File path for saving the figure
    dpi : int
        Resolution of the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Get dimensions
    num_frames, num_residues = data.shape
    # Create figure with appropriate size
    if figsize is None:
        fig_width = max(4, num_residues * 0.25)
        fig_height = max(3, num_frames * 0.05)
        figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize)
    # Create custom grayscale colormap (white to black)
    grayscale_cmap = LinearSegmentedColormap.from_list("grayscale", ["#FFFFFF", "#000000"])
    # Plot the heatmap
    heatmap = ax.imshow(data, cmap=grayscale_cmap, aspect="auto", interpolation="none")
    # Remove all ticks and labels from both axes
    ax.set_xticks([])
    ax.set_yticks([])
    # Add labels suitable for graphical abstract (minimal)
    ax.set_xlabel("Residue", fontsize=24)
    ax.set_ylabel("Frame", fontsize=24)
    # Tight layout to maximize space usage
    plt.tight_layout()
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {output_file}")
    return fig


def generate_random_split(num_residues, val_ratio=0.5, random_seed=None):
    """
    Generate a random binary mask for training/validation split

    Parameters:
    -----------
    num_residues : int
        Number of residues
    val_ratio : float
        Ratio of validation set (default: 0.5)
    random_seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    numpy.ndarray
        Binary mask where 0 = training, 1 = validation
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create random permutation of indices
    indices = np.random.permutation(num_residues)

    # Determine number of validation samples
    num_val = int(num_residues * val_ratio)

    # Create binary mask
    split_mask = np.zeros(num_residues)
    split_mask[indices[:num_val]] = 1

    return split_mask


def plot_train_val_split(split_mask, output_file="train_val_split.png", dpi=300, figsize=None):
    """
    Create a simple green/orange heatmap showing training/validation split

    Parameters:
    -----------
    split_mask : numpy.ndarray
        Binary mask where 0 = training, 1 = validation
    output_file : str
        File path for saving the figure
    dpi : int
        Resolution of the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Reshape the mask for plotting
    num_residues = len(split_mask)
    split_data = split_mask.reshape(1, -1)

    # Create figure
    if figsize is None:
        fig_width = max(4, num_residues * 0.25)
        figsize = (fig_width, 0.5)
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom green/orange colormap
    train_val_cmap = LinearSegmentedColormap.from_list("train_val", ["#00CC00", "#FFA500"])

    # Plot the heatmap
    heatmap = ax.imshow(split_data, cmap=train_val_cmap, aspect="auto", interpolation="none")

    # Remove all ticks and labels from both axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Add labels (place them at reasonable positions)
    # Find the center of the largest continuous training region
    train_indices = np.where(split_data[0] == 0)[0]
    if len(train_indices) > 0:
        train_center = int(np.median(train_indices))
        ax.text(train_center, 0, "Training", ha="left", va="center", color="beige", fontsize=20)

    # Find the center of the largest continuous validation region
    val_indices = np.where(split_data[0] == 1)[0]
    if len(val_indices) > 0:
        val_center = int(np.median(val_indices))
        ax.text(val_center, 0, "Validation", ha="right", va="center", color="beige", fontsize=20)
    # Tight layout
    plt.tight_layout(pad=0)

    # Save the figure
    plt.savefig(output_file, dpi=dpi, pad_inches=0)
    plt.close()
    print(f"Train/val split figure saved to {output_file}")
    return fig


def plot_colored_residue_heatmap(
    data, split_mask, output_file="colored_residue_heatmap.png", dpi=300, figsize=None
):
    """
    Create a heatmap visualization of residue features with different colors for training and validation sets

    Parameters:
    -----------
    data : numpy.ndarray
        2D array with shape (num_frames, num_residues)
    split_mask : numpy.ndarray
        Binary mask where 0 = training, 1 = validation
    output_file : str
        File path for saving the figure
    dpi : int
        Resolution of the output figure
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Get dimensions
    num_frames, num_residues = data.shape

    # Create figure with appropriate size
    if figsize is None:
        fig_width = max(4, num_residues * 0.25)
        fig_height = max(3, num_frames * 0.1)
        figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize)

    # Create masked arrays for training and validation sets based on the random split mask
    train_mask = np.ones_like(data, dtype=bool)
    val_mask = np.ones_like(data, dtype=bool)

    for i in range(num_residues):
        if split_mask[i] == 0:  # Training
            val_mask[:, i] = False
        else:  # Validation
            train_mask[:, i] = False

    # Create custom colormaps for training and validation sets
    train_cmap = LinearSegmentedColormap.from_list("train", ["#FFFFFF", "green"])
    val_cmap = LinearSegmentedColormap.from_list("val", ["#FFFFFF", "orange"])

    # Plot the heatmaps with different colormaps for training and validation
    vmin, vmax = data.min(), data.max()

    # Use masked arrays to apply different colormaps
    train_data = np.ma.array(data, mask=~train_mask)
    val_data = np.ma.array(data, mask=~val_mask)

    heatmap1 = ax.imshow(
        train_data, cmap=train_cmap, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax
    )
    heatmap2 = ax.imshow(
        val_data, cmap=val_cmap, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax
    )

    # Add vertical lines at the split boundaries
    for i in range(1, num_residues):
        if split_mask[i] != split_mask[i - 1]:
            ax.axvline(x=i - 0.5, color="red", linestyle="-", linewidth=1)

    # Remove all ticks and labels from both axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Add labels suitable for graphical abstract (minimal)
    ax.set_xlabel("Residue", fontsize=24)
    ax.set_ylabel("Frame", fontsize=24)

    # Find the center of the largest continuous training and validation regions
    train_indices = np.where(split_mask == 0)[0]
    val_indices = np.where(split_mask == 1)[0]

    # Function to find largest continuous block in array
    def find_largest_block(indices):
        if len(indices) == 0:
            return None

        # Find runs of consecutive indices
        runs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        # Get the longest run
        longest_run = max(runs, key=len)
        return int(np.mean(longest_run)) if len(longest_run) > 0 else None

    # Add training/validation labels at the center of the largest blocks
    train_center = find_largest_block(train_indices)
    val_center = find_largest_block(val_indices)

    if train_center is not None:
        ax.text(train_center, -1, "Training", ha="center", va="center", color="green", fontsize=24)

    if val_center is not None:
        ax.text(val_center, -1, "Validation", ha="center", va="center", color="orange", fontsize=24)

    # Tight layout to maximize space usage
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Colored figure saved to {output_file}")
    return fig


def main():
    """Main function to handle command line arguments and create the visualization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a minimal residue feature heatmap for graphical abstract"
    )
    parser.add_argument("--frames", type=int, default=3, help="Number of frames (default: 5)")
    parser.add_argument("--residues", type=int, default=30, help="Number of residues (default: 30)")
    parser.add_argument(
        "--output",
        type=str,
        default="residue_heatmap.png",
        help="Output file path (default: residue_heatmap.png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output figure resolution (default: 300)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.5, help="Validation set ratio (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--figwidth", type=float, default=None, help="Figure width in inches (default: auto)"
    )
    parser.add_argument(
        "--figheight", type=float, default=None, help="Figure height in inches (default: auto)"
    )
    args = parser.parse_args()

    # Generate data
    data = generate_residue_feature_data(args.frames, args.residues)

    # Calculate a consistent figure size if specified
    figsize = None
    if args.figwidth is not None and args.figheight is not None:
        figsize = (args.figwidth, args.figheight)
    else:
        # Calculate a reasonable default size based on data dimensions
        fig_width = max(4, args.residues * 0.5)
        fig_height = max(3, args.frames * 0.1)
        figsize = (fig_width, fig_height)

    # For the split view, use the same width but adjust height
    split_figsize = (figsize[0], 0.5)

    # Create and save the original grayscale visualization
    plot_residue_heatmap(data, args.output, args.dpi, figsize)

    # Generate random split mask
    split_mask = generate_random_split(args.residues, args.val_ratio, args.seed)

    # Create and save the training/validation split visualization
    output_split = args.output.replace(".png", "_train_val_split.png")
    plot_train_val_split(split_mask, output_split, args.dpi, split_figsize)

    # Create and save the colored visualization with training/validation split
    output_colored = args.output.replace(".png", "_colored_split.png")
    plot_colored_residue_heatmap(data, split_mask, output_colored, args.dpi, figsize)


if __name__ == "__main__":
    main()
