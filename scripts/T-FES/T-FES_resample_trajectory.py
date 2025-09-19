import json
import os
import random

import MDAnalysis as mda


def expand_frame_info(frame_info):
    """
    Expands a single frame info entry into individual frame entries.

    Args:
        frame_info (dict): Information about a trajectory segment

    Returns:
        list: List of frame-by-frame information
    """
    expanded_frames = []
    start_frame = frame_info["start_frame"]
    end_frame = frame_info["end_frame"]

    for frame_num in range(start_frame, end_frame):
        expanded_frame = {
            "topology": frame_info["topology"],
            "cycle": frame_info["cycle"],
            "seed": frame_info["seed"],
            "frame_no": frame_num,
            "segment_start": start_frame,
            "segment_end": end_frame,
        }
        expanded_frames.append(expanded_frame)

    return expanded_frames


def get_slice_indices(total_frames, target_frames, flexibility=100):
    """
    Calculate slice indices to get close to target frame count.
    Ensures additional frames are unique and taken from frames not already selected.

    Args:
        total_frames (int): Total number of frames available
        target_frames (int): Desired number of frames
        flexibility (int): Number of frames we can adjust by to hit target

    Returns:
        tuple: (slice_indices, actual_frames)
    """
    # Calculate initial stride to get approximately target frames
    base_stride = max(1, total_frames // target_frames)
    initial_indices = set(range(0, total_frames, base_stride))
    frames_with_base = len(initial_indices)

    # If we're within flexibility range, adjust to hit target exactly
    if abs(frames_with_base - target_frames) <= flexibility:
        if frames_with_base > target_frames:
            # Remove random frames to hit target
            num_to_remove = frames_with_base - target_frames
            frames_to_keep = sorted(
                random.sample(initial_indices, frames_with_base - num_to_remove)
            )
            return frames_to_keep, len(frames_to_keep)
        else:
            # Add random frames from unused frames
            unused_frames = set(range(total_frames)) - initial_indices
            num_to_add = target_frames - frames_with_base
            if len(unused_frames) < num_to_add:
                print(
                    f"Warning: Cannot add {num_to_add} frames - only {len(unused_frames)} unused frames available"
                )
                additional_frames = unused_frames
            else:
                additional_frames = set(random.sample(unused_frames, num_to_add))

            # Combine and sort all frames
            all_frames = sorted(initial_indices.union(additional_frames))
            return all_frames, len(all_frames)

    # If outside flexibility range, just return initial indices
    return sorted(initial_indices), frames_with_base


def process_protein(protein_name, target_frames=500):
    """
    Process protein trajectory with regular interval sampling.

    Args:
        protein_name (str): Name of the protein to process
        target_frames (int): Desired number of frames to sample
    """
    print(f"Processing protein: {protein_name}")
    protein_dir = os.path.join(base_dir, f"{protein_name}", "final")
    traj_path = os.path.join(protein_dir, f"{protein_name}_overall_combined_stripped.xtc")
    top_path = os.path.join(protein_dir, f"{protein_name}_overall_combined_stripped.pdb")
    json_path = os.path.join(protein_dir, f"{protein_name}_trajectory_info.json")

    print(f"Loading trajectory from {traj_path}")
    u = mda.Universe(top_path, traj_path)
    total_frames = len(u.trajectory)

    print(f"Total frames available: {total_frames}")
    print(f"Target frames: {target_frames}")

    print(f"Loading trajectory information from {json_path}")
    with open(json_path, "r") as f:
        traj_info = json.load(f)

    # Expand frame information
    print("Expanding frame information...")
    per_frame_data = []
    for frame_segment in traj_info["frames"]:
        expanded_frames = expand_frame_info(frame_segment)
        per_frame_data.extend(expanded_frames)

    # Verify frame count
    if len(per_frame_data) != total_frames:
        print(
            f"Warning: Expanded frames ({len(per_frame_data)}) doesn't match trajectory length ({total_frames})"
        )
        if len(per_frame_data) > total_frames:
            per_frame_data = per_frame_data[:total_frames]
        else:
            #
            total_frames = len(per_frame_data)
            print(f"Adjusted total frames to {total_frames}")

    print(f"Saving original per-frame JSON file to {protein_dir}")
    original_json_path = os.path.join(protein_dir, f"{protein_name}_per_frame_info.json")
    with open(original_json_path, "w") as f:
        json.dump(per_frame_data, f, indent=4)

    output_dir = os.path.join(protein_dir, "resampled_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at {output_dir}")

    # Get sliced indices with some flexibility to hit target
    sampled_indices, actual_frames = get_slice_indices(total_frames, target_frames)
    print(f"Will sample {actual_frames} frames")

    # Create sampled data using the selected indices
    sampled_data = [per_frame_data[i] for i in sampled_indices]

    # Update frame numbers to reflect new trajectory order
    for new_idx, frame_data in enumerate(sampled_data):
        frame_data["sampled_frame_no"] = new_idx + 1
        frame_data["original_frame_no"] = frame_data["frame_no"]
        frame_data["frame_no"] = new_idx + 1

    print(f"Saving sampled per-frame JSON file to {output_dir}")
    sampled_json_path = os.path.join(
        output_dir, f"{protein_name}_sampled_per_frame_info_{target_frames}.json"
    )
    with open(sampled_json_path, "w") as f:
        json.dump(sampled_data, f, indent=4)

    print(f"Writing sampled trajectory to {output_dir}")
    sampled_traj_path = os.path.join(output_dir, f"{protein_name}_sampled_{target_frames}.xtc")
    with mda.Writer(sampled_traj_path, u.atoms.n_atoms) as W:
        for frame_idx in sampled_indices:
            u.trajectory[frame_idx]
            W.write(u)

    print(f"Finished processing protein: {protein_name}")
    return actual_frames


if __name__ == "__main__":
    base_dir = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/T-FES"
    protein_names = [
        "BPTI",
        "HOIP",
        "BRD4",
        "LXR",
        "MBP",
    ]
    random_seed = 42
    random.seed(random_seed)

    for protein in protein_names:
        actual_frames = process_protein(protein)
        print(f"Successfully sampled {actual_frames} frames for {protein}")
