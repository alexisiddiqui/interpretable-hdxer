# /home/alexi/Documents/interpretable-hdxer/notebooks/SI-Fig-1_AF_MD_PCA_Space/SI-Fig-1_Transfer.py
import json
import multiprocessing as mp
import os
import sys
from itertools import combinations
from typing import Dict, List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.helpful_funcs import segs_to_df

# set plain white style
sns.set_style("ticks")
# set font scales

sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 10,
    },
)

import matplotlib as mpl

# set font sizes for matplotlib
mpl.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
    }
)
dataset_colours = {
    "af_dirty": "navy",
    "af_clean": "cyan",
    "1Start": "orchid",
    "10Start": "orange",
    "TFES": "purple",
    "Shaw": "black",
}

full_dataset_colours = {
    "AF2-MSAss": "navy",
    "AF2-Filtered": "dodgerblue",
    "MD-1Start": "orchid",
    "MD-10Start": "orange",
    "MD-TFES": "purple",
    "MD-Shaw": "black",
}


class Segments:
    """
    takes in a dataframe of experimental hdx segments
    and contains the values of the peptides
    """

    def __init__(
        self, segs_df: pd.DataFrame = None, segs_path: str = None, keys=["ResStr", "ResEnd"]
    ):
        if segs_df is not None:
            self.segs_df = segs_df.copy()
        if segs_path is not None:
            self.segs_df = segs_to_df(segs_path)
        self.keys = keys
        if "peptide" not in self.segs_df.columns:
            self.segs_df["peptide"] = np.arange(len(self.segs_df))
        self.pep_nums = self.get_pep_nums(self.segs_df)
        assert len(self.segs_df["peptide"].unique()) == len(self.pep_nums), (
            "Peptides are not unique"
        )
        print(f"Segments class created with {len(self.pep_nums)} peptides")
        self.res_nums = self.get_resnums(self.segs_df)
        self.residues = self.get_residues(self.segs_df)
        self.peptides = self.get_peptides(self.segs_df)
        self.residue_centrality = self.get_residue_centrality(self.segs_df)
        self.peptide_centrality = self.get_peptide_centrality(self.segs_df)

    def get_resnums(self, df: pd.DataFrame, keys=None):
        """
        Get the residue numbers of the segments in the dataframe
        """
        if df is None:
            df = self.segs_df

        if keys is None:
            keys = self.keys
        res_nums = df.apply(lambda x: np.arange(x[keys[0]] + 1, x[keys[1]] + 1), axis=1).to_numpy()
        print(f"Resnumbers calculated for {len(res_nums)} segments")
        return res_nums

    def get_peptides(self, df: pd.DataFrame = None):
        """
        Get the peptides from the dataframe,
        creates a dictionary with the peptide number as the key and the residues as the values
        """
        if df is None:
            df = self.segs_df
        peptides = {pep: res for pep, res in zip(df["peptide"], self.get_resnums(df))}
        print(f"Peptides calculated for {len(peptides)} segments")
        return peptides

    def get_residues(self, df: pd.DataFrame = None):
        """
        Get the residues from the dataframe
        """
        if df is None:
            df = self.segs_df
        residues = np.unique(np.concatenate(self.get_resnums(df)))
        print(f"Residues calculated for {len(residues)} segments")
        return residues

    def get_pep_nums(self, df: pd.DataFrame = None):
        """
        Get the peptide numbers from the dataframe
        """
        if df is None:
            df = self.segs_df
        return df["peptide"].to_numpy()

    def get_residue_centrality(self, df: pd.DataFrame):
        """
        calculate the number of peptides that contain each residue
        """
        print("Calculating residue centrality")
        if df is None:
            df = self.segs_df

        res_nums = self.get_resnums(df)

        res_cent = {res: 0 for res in self.residues}

        for res in self.residues:
            for res_num in res_nums:
                if res in res_num:
                    res_cent[res] += 1

        return res_cent

    def get_peptide_centrality(self, df: pd.DataFrame):
        """
        calculate the mean residue centrality for each peptide
        """

        print("Calculating peptide centrality")
        if df is None:
            df = self.segs_df
        peptides = self.get_peptides(df)
        res_cent = self.get_residue_centrality(df)

        pep_cent = {
            pep: np.mean([res_cent[res] for res in res_nums]) for pep, res_nums in peptides.items()
        }

        print(f"Peptide centrality calculated for {len(pep_cent)} peptides")
        return pep_cent

    def update(self, df: pd.DataFrame = None):
        """
        Update the values of the class
        """
        if df is not None:
            self.segs_df = df.copy()
        self.residues = self.get_residues(self.segs_df)
        self.res_nums = self.get_resnums(self.segs_df)
        self.peptides = self.get_peptides(self.segs_df)
        self.pep_nums = self.get_pep_nums(self.segs_df)

        self.residue_centrality = self.get_residue_centrality(self.segs_df)
        self.peptide_centrality = self.get_peptide_centrality(self.segs_df)

    def df_from_peptides(self, peptides: dict):
        """
        Update the dataframe with the peptides and residues
        Peptide is the index of the dataframe
        """
        keys = self.keys
        print(f"Creating dataframe from peptides {peptides}")
        df = pd.DataFrame(columns=[keys[0], keys[1], "peptide"])

        for pep, res in peptides.items():
            df = pd.concat(
                [df, pd.DataFrame({keys[0]: res[0], keys[1]: res[-1], "peptide": pep}, index=[pep])]
            )
        print("Dataframe created")
        print(df.head())
        return df

    def df_select_peptides(self, pep_nums: list):
        """
        Select the peptides from the dataframe
        """
        print(f"Selecting {pep_nums} peptides from the dataframe")
        return self.segs_df.loc[pep_nums].copy()

    def df_remove_peptides(self, pep_nums: list):
        """
        Remove the peptides from the dataframe
        """
        print(f"Removing {pep_nums} peptides from the dataframe")
        return self.segs_df.loc[~self.segs_df["peptide"].isin(pep_nums)].copy()

    def df_select_residues(self, residues: list):
        """
        Select peptides from the dataframe based on the residues
        """
        print(f"Selecting peptides with residues {residues}")
        unique_res = np.unique(residues)

        selected_peptides = []
        for pep, res in self.peptides.items():
            if len(np.intersect1d(res, unique_res)) > 0:
                selected_peptides.append(pep)
        return self.df_select_peptides(selected_peptides).copy()

    def df_remove_residues(self, residues: list):
        """
        Remove peptides from the dataframe based on the residues
        """
        print(f"Removing peptides with residues {residues}")
        unique_res = np.unique(residues)

        selected_peptides = []
        for pep, res in self.peptides.items():
            if len(np.intersect1d(res, unique_res)) == 0:
                selected_peptides.append(pep)
        return self.df_select_peptides(selected_peptides).copy()

    def select_segments(
        self,
        peptides: list = None,
        residues: list = None,
        new_segs_df: pd.DataFrame = None,
        remove: bool = False,
    ):
        """
        Update the segments class with new peptides or residues
        Either remove or select the peptides or residues
        First updates based on peptides, then updates based on residues
        """
        if new_segs_df is None:
            new_segs_df = self.segs_df.copy()

        if remove:
            peptide_update_function = self.df_remove_peptides
            residue_update_function = self.df_remove_residues
        else:
            peptide_update_function = self.df_select_peptides
            residue_update_function = self.df_select_residues

        if peptides is not None:
            new_segs_df = peptide_update_function(peptides)

        if residues is not None:
            new_segs_df = residue_update_function(residues)

        print("Updating Segments")
        self.update(new_segs_df)

    def create_train_val_segs(self, train_peps: np.ndarray, val_peps: np.ndarray):
        train_segs = self.df_select_peptides(list(train_peps))
        val_segs = self.df_select_peptides(list(val_peps))

        return train_segs, val_segs


protein_names = ["BPTI", "BRD4", "HOIP", "LXR", "MBP"]
experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start"]  # , "MD-Shaw"]
experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "MD-Shaw"]

af_experiments = ["AF2-MSAss", "AF2-Filtered"]
# protein_names = ["BPTI"]#, "BRD4", "HOIP", "LXR", "MBP"]

BPTI_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/P00974_60_1_af_sample_127_10000_protonated.pdb"
BPTI_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/P00974_60_1_af_sample_127_10000_protonated.xtc"
BPTI_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/P00974_60_1_af_sample_127_10000_ranks.json"
BPTI_segs_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_expt_data/BPTI_residue_segs_trimmed.txt"
BPTI_num_residues = 58


BRD4_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/BRD4_APO_484_1_af_sample_127_10000_protonated.pdb"
BRD4_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/BRD4_APO_484_1_af_sample_127_10000_protonated.xtc"
BRD4_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/BRD4_APO_484_1_af_sample_127_10000_ranks.json"
BRD4_segs_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_segs_trimmed.txt"
BRD4_num_residues = 484

HOIP_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/HOIP_apo697_1_af_sample_127_10000_protonated.pdb"
HOIP_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/HOIP_apo697_1_af_sample_127_10000_protonated.xtc"
HOIP_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/HOIP_apo697_1_af_sample_127_10000_ranks.json"
HOIP_segs_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_APO_segs_trimmed.txt"
HOIP_num_residues = 376

LXR_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/LXRa200_1_af_sample_127_10000_protonated.pdb"
LXR_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/LXRa200_1_af_sample_127_10000_protonated.xtc"
LXR_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/LXRa200_1_af_sample_127_10000_ranks.json"
LXR_segs_path = (
    "/home/alexi/Documents/ValDX/raw_data/LXRalpha/LXRalpha_APO/LXRa_APO_segs200_trimmed.txt"
)
LXR_num_residues = 250


MBP_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_protonated.pdb"
MBP_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_protonated.xtc"
MBP_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_ranks.json"
MBP_segs_path = (
    "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP_wt1_segs_trimmed.txt"
)
MBP_num_residues = 396


AF2_proteins = {
    "BPTI": {
        "top": BPTI_top_path,
        "traj": BPTI_traj_path,
        "json": BPTI_json_path,
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": BRD4_top_path,
        "traj": BRD4_traj_path,
        "json": BRD4_json_path,
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": HOIP_top_path,
        "traj": HOIP_traj_path,
        "json": HOIP_json_path,
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": LXR_top_path,
        "traj": LXR_traj_path,
        "json": LXR_json_path,
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": MBP_top_path,
        "traj": MBP_traj_path,
        "json": MBP_json_path,
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


AF2_json_data = {protein: load_json(AF2_proteins[protein]["json"]) for protein in protein_names}

for protein in protein_names:
    AF2_proteins[protein]["index"] = AF2_json_data[protein]


def extract_val_json(protein_data: Dict, key: str = "plddt") -> np.array:
    protein_data = dict(
        sorted(protein_data.items(), key=lambda x: int(x[0].split("_")[1]), reverse=True)
    )

    val_data = []
    for msa, run in protein_data.items():
        for rank, data in run.items():
            if key in data:
                val_data.append(data[key])

    return np.array(val_data)


plddt_data = {protein: extract_val_json(AF2_json_data[protein]) for protein in protein_names}

max_pae_data = {
    protein: extract_val_json(AF2_json_data[protein], key="max_pae") for protein in protein_names
}

ptm_data = {
    protein: extract_val_json(AF2_json_data[protein], key="ptm") for protein in protein_names
}

for protein in protein_names:
    AF2_proteins[protein]["index"] = plddt_data[protein]


segs_data = {
    protein: Segments(segs_path=AF2_proteins[protein]["segs_path"]) for protein in AF2_proteins
}


res_data = {protein: segs_data[protein].residues for protein in AF2_proteins}
print(res_data)


# cleaned AF
Cleaned_AF2_proteins = {
    "BPTI": {
        "top": BPTI_top_path,
        "traj": os.path.splitext(BPTI_traj_path)[0] + "_all_filtered.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
        "filtered_plddt_json": os.path.splitext(BPTI_traj_path)[0]
        + "_all_filtered_frame_plddt.json",
    },
    "BRD4": {
        "top": BRD4_top_path,
        "traj": os.path.splitext(BRD4_traj_path)[0] + "_all_filtered.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
        "filtered_plddt_json": os.path.splitext(BRD4_traj_path)[0]
        + "_all_filtered_frame_plddt.json",
    },
    "HOIP": {
        "top": HOIP_top_path,
        "traj": os.path.splitext(HOIP_traj_path)[0] + "_all_filtered.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
        "filtered_plddt_json": os.path.splitext(HOIP_traj_path)[0]
        + "_all_filtered_frame_plddt.json",
    },
    "LXR": {
        "top": LXR_top_path,
        "traj": os.path.splitext(LXR_traj_path)[0] + "_all_filtered.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
        "filtered_plddt_json": os.path.splitext(LXR_traj_path)[0]
        + "_all_filtered_frame_plddt.json",
    },
    "MBP": {
        "top": MBP_top_path,
        "traj": os.path.splitext(MBP_traj_path)[0] + "_all_filtered.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
        "filtered_plddt_json": os.path.splitext(MBP_traj_path)[0]
        + "_all_filtered_frame_plddt.json",
    },
}

# filtered_plddt is json of "frame": plddt, extract plddt as a list


def extract_filered_plddt(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [data[frame] for frame in sorted(data.keys())]


filtered_plddt_data = {
    protein: extract_filered_plddt(Cleaned_AF2_proteins[protein]["filtered_plddt_json"])
    for protein in protein_names
}

for protein in protein_names:
    Cleaned_AF2_proteins[protein]["index"] = filtered_plddt_data[protein]


# Function to get number of frames in a trajectory
def get_n_frames(top_path, traj_path):
    u = mda.Universe(top_path, traj_path)
    n_frames = len(u.trajectory)
    return n_frames


# Base path for regular MD
regular_MD_base = "/home/alexi/Documents/ValDX/raw_data/full_length_regular_MD"

# 1Start MD paths and data
onestart_MD_proteins = {
    "BPTI": {
        "top": f"{regular_MD_base}/BPTI_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/BPTI_test_concatenated_stripped.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": f"{regular_MD_base}/BRD4_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/BRD4_test_concatenated_stripped.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": f"{regular_MD_base}/HOIP_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/HOIP_test_concatenated_stripped.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": f"{regular_MD_base}/LXR_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/LXR_test_concatenated_stripped.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": f"{regular_MD_base}/MBP_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/MBP_test_concatenated_stripped.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}

# 10Start MD paths and data
tenstart_MD_proteins = {
    "BPTI": {
        "top": f"{regular_MD_base}/BPTI_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/BPTI_10_c_combined.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": f"{regular_MD_base}/BRD4_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/BRD4_10_c_combined.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": f"{regular_MD_base}/HOIP_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/HOIP_10_c_combined.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": f"{regular_MD_base}/LXR_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/LXR_10_c_combined.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": f"{regular_MD_base}/MBP_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/MBP_10_c_combined.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}

# Regular (Bad) MD paths and data
regular_MD_proteins = {
    "BPTI": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/BadMD_BRD4_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/BadMD_BRD4_r5_15010_concatenated.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/BadMD_HOIP_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/BadMD_HOIP_r5_15010_concatenated.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/BadMD_LXR_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/BadMD_LXR_r5_15010_concatenated.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}

# Better (Good) MD paths and data
better_MD_proteins = {
    "BPTI": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/GoodMD_BRD4_r6_12006_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/GoodMD_BRD4_r6_12006_concatenated.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/GoodMD_HOIP_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/GoodMD_HOIP_r10_10010_concatenated.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/GoodMD_LXR_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/GoodMD_LXR_r10_10010_concatenated.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}

# Add index data for 1Start MD
for protein in protein_names:
    n_frames = get_n_frames(
        onestart_MD_proteins[protein]["top"], onestart_MD_proteins[protein]["traj"]
    )
    onestart_MD_proteins[protein]["index"] = [5 / n_frames * i for i in range(n_frames)]
    print(f"{protein} 1Start MD frames: {n_frames}")

# Add index data for 10Start MD
for protein in protein_names:
    n_frames = get_n_frames(
        tenstart_MD_proteins[protein]["top"], tenstart_MD_proteins[protein]["traj"]
    )
    tenstart_MD_proteins[protein]["index"] = [10 / n_frames * i for i in range(n_frames)]
    print(f"{protein} 10Start MD frames: {n_frames}")

# Add index data for regular MD
for protein in protein_names:
    if protein == "BRD4":
        regular_MD_proteins[protein]["index"] = [5 / 15010 * i for i in range(15010)]
    else:
        regular_MD_proteins[protein]["index"] = [5 / 15010 * i for i in range(15010)]

# Add index data for better MD
for protein in protein_names:
    if protein == "BRD4":
        better_MD_proteins[protein]["index"] = [10 / 12006 * i for i in range(12006)]
    else:
        better_MD_proteins[protein]["index"] = [10 / 10010 * i for i in range(10010)]

shaw_MD_BPTI_top_path = (
    "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
)
shaw_MD_BPTI_traj_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_400.xtc"

test_universe = mda.Universe(shaw_MD_BPTI_top_path, shaw_MD_BPTI_traj_path)


shaw_MD_proteins = {
    "BPTI": {"top": shaw_MD_BPTI_top_path, "traj": shaw_MD_BPTI_traj_path},
    "num_residues": BPTI_num_residues,
    "segs_path": BPTI_segs_path,
}


shaw_MD_proteins["BPTI"]["index"] = [
    100 / len(test_universe.trajectory) * i for i in range(len(test_universe.trajectory))
]


# Update ensemble_experiments dictionary
ensemble_experiments = {
    "AF2-MSAss": AF2_proteins,
    "AF2-Filtered": Cleaned_AF2_proteins,
    "MD-1Start": onestart_MD_proteins,
    "MD-10Start": tenstart_MD_proteins,
    # "Regular-MD": regular_MD_proteins,
    # "Better-MD": better_MD_proteins,
    "MD-Shaw": shaw_MD_proteins,
}


def load_tfes_frame_info(json_path):
    """
    Load and process MD-TFES per-frame information.
    Creates a list where index corresponds to frame number (0-based)
    and value is the cycle number.
    """
    with open(json_path, "r") as f:
        frame_info = json.load(f)

    # Find the maximum frame number to determine list size
    max_frame = max(item["sampled_frame_no"] for item in frame_info)

    # Create a list initialized with zeros
    # Add 1 to max_frame since frame numbers are 1-based but we need 0-based indexing
    cycle_list = [0] * (max_frame)

    # Fill in the cycle numbers
    # Subtract 1 from frame number to convert to 0-based indexing
    for item in frame_info:
        frame_idx = item["sampled_frame_no"] - 1  # Convert to 0-based index
        cycle_list[frame_idx] = item["cycle"]

    return cycle_list


# Base path for MD-TFES data
tfes_base = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/T-FES"

# MD-TFES paths and data structure

tfes_proteins = {
    "BPTI": {
        "top": f"{tfes_base}/BPTI/final/BPTI_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/BPTI/final/resampled_outputs/BPTI_sampled_500.xtc",
        "frame_info": f"{tfes_base}/BPTI/final/resampled_outputs/BPTI_sampled_per_frame_info_500.json",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path,
    },
    "BRD4": {
        "top": f"{tfes_base}/BRD4/final/BRD4_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/BRD4/final/resampled_outputs/BRD4_sampled_500.xtc",
        "frame_info": f"{tfes_base}/BRD4/final/resampled_outputs/BRD4_sampled_per_frame_info_500.json",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path,
    },
    "HOIP": {
        "top": f"{tfes_base}/HOIP/final/HOIP_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/HOIP/final/resampled_outputs/HOIP_sampled_500.xtc",
        "frame_info": f"{tfes_base}/HOIP/final/resampled_outputs/HOIP_sampled_per_frame_info_500.json",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path,
    },
    "LXR": {
        "top": f"{tfes_base}/LXR/final/LXR_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/LXR/final/resampled_outputs/LXR_sampled_500.xtc",
        "frame_info": f"{tfes_base}/LXR/final/resampled_outputs/LXR_sampled_per_frame_info_500.json",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path,
    },
    "MBP": {
        "top": f"{tfes_base}/MBP/final/MBP_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/MBP/final/resampled_outputs/MBP_sampled_500.xtc",
        "frame_info": f"{tfes_base}/MBP/final/resampled_outputs/MBP_sampled_per_frame_info_500.json",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path,
    },
}
# Load frame info for each protein and add to the data structure
for protein in tfes_proteins:
    print(f"Loading MD-TFES frame info for {protein}")
    cycle_list_from_json = load_tfes_frame_info(tfes_proteins[protein]["frame_info"])

    # Get the actual number of frames from the trajectory file
    actual_n_frames = get_n_frames(tfes_proteins[protein]["top"], tfes_proteins[protein]["traj"])

    # Create the final index list, initialized to default (cycle 0)
    final_cycle_list = [0] * actual_n_frames

    # Determine how many cycle numbers to copy from the JSON-derived list
    num_to_copy = min(actual_n_frames, len(cycle_list_from_json))
    final_cycle_list[:num_to_copy] = cycle_list_from_json[:num_to_copy]

    if actual_n_frames > len(cycle_list_from_json):
        print(
            f"Warning: Trajectory for {protein} ({actual_n_frames} frames) is longer than JSON-derived cycle list ({len(cycle_list_from_json)} frames). Extra frames assigned cycle 0."
        )
    elif actual_n_frames < len(cycle_list_from_json):
        print(
            f"Warning: Trajectory for {protein} ({actual_n_frames} frames) is shorter than JSON-derived cycle list ({len(cycle_list_from_json)} frames). JSON data for non-existent frames ignored."
        )

    tfes_proteins[protein]["index"] = final_cycle_list
    print(
        f"Loaded {len(cycle_list_from_json)} cycle entries from JSON for {protein}. Adjusted index list to {len(final_cycle_list)} based on actual trajectory frames."
    )

# Update ensemble_experiments dictionary to include MD-TFES
ensemble_experiments["MD-TFES"] = tfes_proteins

# Update experiments list
experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "MD-TFES"]
experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "MD-TFES"]

# Update experiments list
# experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "Regular-MD", "Better-MD", "MD-Shaw"]
import gc
from typing import Dict, Generator

import MDAnalysis as mda
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


class MemoryEfficientPCA:
    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)

    def process_trajectory_chunk(
        self,
        universe: mda.Universe,
        chunk_size: int,
        residues: np.ndarray,
        experiment_name: str,
        index: List[float],
    ) -> Generator:
        """Process trajectory in chunks to reduce memory usage"""
        atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, residues))}")
        n_atoms = len(atom_group)
        expected_dist_size = (n_atoms * (n_atoms - 1)) // 2  # Size of pairwise distance array

        chunk_distances = []
        chunk_labels = []
        chunk_indexes = []

        print(f"Starting trajectory processing for {experiment_name} in chunks.")
        for chunk_start in tqdm(
            range(0, len(universe.trajectory), chunk_size),
            desc=f"Processing Chunks {experiment_name}",
        ):
            chunk_end = min(chunk_start + chunk_size, len(universe.trajectory))

            for ts in universe.trajectory[chunk_start:chunk_end]:
                # pdist calculates the condensed distance matrix, which contains the
                # upper triangle elements of the pairwise distance matrix between CA atoms.
                distances = pdist(atom_group.positions)

                # Convert to numpy array and ensure correct shape
                distances = np.array(distances, dtype=np.float64)

                # Validate distance array shape
                if len(distances) != expected_dist_size:
                    print(f"Warning: Unexpected distance array size for frame {ts.frame}")
                    print(f"Expected: {expected_dist_size}, Got: {len(distances)}")
                    continue

                chunk_distances.append(distances)
                chunk_labels.append(experiment_name)
                chunk_indexes.append(index[ts.frame])

            if chunk_distances:  # Only yield if we have valid distances
                # Convert chunk to numpy array before yielding
                distances_array = np.vstack(chunk_distances)
                yield distances_array, chunk_labels, chunk_indexes

            # Clear chunk data
            chunk_distances = []
            chunk_labels = []
            chunk_indexes = []
            gc.collect()

    def fit_transform_chunks(
        self,
        universes: List[mda.Universe],
        experiment_names: List[str],
        indexes: List[List[float]],
        residues: np.ndarray,
        chunk_size: int = 1000,
    ) -> pd.DataFrame:
        """Fit and transform PCA in chunks"""
        all_distances = []
        all_labels = []
        all_indexes = []

        print("Starting PCA fit_transform_chunks.")
        # Process each universe in chunks
        for universe, exp_name, index in tqdm(
            zip(universes, experiment_names, indexes),
            total=len(universes),
            desc="Processing Universes for PCA",
        ):
            print(f"Processing {exp_name} with {len(universe.trajectory)} frames for PCA.")
            atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, residues))}")
            print(f"Number of CA atoms selected for {exp_name}: {len(atom_group)}")

            valid_frames = 0
            for distances_chunk, labels_chunk, chunk_indexes_chunk in self.process_trajectory_chunk(
                universe, chunk_size, residues, exp_name, index
            ):
                if len(distances_chunk) > 0:  # Only add if we have valid distances
                    all_distances.append(distances_chunk)
                    all_labels.extend(labels_chunk)
                    all_indexes.extend(chunk_indexes_chunk)
                    valid_frames += len(distances_chunk)

            print(f"Processed {valid_frames} valid frames for {exp_name} in PCA.")
            gc.collect()

        if not all_distances:
            print("Error: No valid distance calculations were performed for PCA.")
            raise ValueError("No valid distance calculations were performed")

        try:
            print("Concatenating all distance arrays for PCA.")
            # Concatenate all distance arrays
            all_distances = np.vstack(all_distances)
            print(f"Final distance array shape for PCA: {all_distances.shape}")

            print("Performing PCA transformation.")
            # Perform PCA
            pca_results = self.pca.fit_transform(all_distances)
            print("PCA transformation complete.")
            explained_variance = self.pca.explained_variance_ratio_

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "PC1": pca_results[:, 0],
                    "PC2": pca_results[:, 1],
                    "Experiment": all_labels,
                    "Index": all_indexes,
                }
            )

            del all_distances, pca_results
            gc.collect()

            return df, explained_variance

        except ValueError as e:
            print("Error in PCA transformation:")
            if len(all_distances) > 0:
                print("First few distance array shapes:")
                for i, dist in enumerate(all_distances[:5]):
                    print(f"Array {i} shape: {dist.shape}")
            raise e


def optimize_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize the dataframe for plotting by reducing memory usage"""
    # Convert string columns to categorical
    df["Experiment"] = df["Experiment"].astype("category")
    if "Protein" in df.columns:
        df["Protein"] = df["Protein"].astype("category")

    # Downsample if too many points
    max_points = 10000
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    return df


def memory_efficient_plot_PCA(
    ensemble_experiments: Dict,
    protein_names: List[str],
    segs: Dict[str, "Segments"],
    experiments: List[str],
    chunk_size: int = 1000,
    contour_ensemble_combo: List[str] = ["AF2-Filtered", "MD-1Start"],
) -> None:
    """Memory-efficient version of PCA calculation and plotting"""
    pca_processor = MemoryEfficientPCA(n_components=10)
    print("Initialized MemoryEfficientPCA.")

    for protein in protein_names:
        print(f"\nStarting PCA processing for protein: {protein}")
        universes = []
        experiment_names_for_protein = []
        indexes_for_protein = []

        # Load data for current protein
        print(f"Loading trajectory data for {protein}...")
        for experiment in tqdm(experiments, desc=f"Loading data for {protein}"):
            if protein in ensemble_experiments[experiment]:
                exp_data = ensemble_experiments[experiment][protein]
                try:
                    print(f"  Loading universe for {experiment}...")
                    universes.append(mda.Universe(exp_data["top"], exp_data["traj"]))
                    experiment_names_for_protein.append(experiment)
                    indexes_for_protein.append(exp_data["index"])
                    print(f"  Successfully loaded {experiment} for {protein}.")
                except Exception as e:
                    print(f"  Error loading universe for {protein}, {experiment}: {e}")
            else:
                print(f"  Skipping {experiment} for {protein}: data not found.")

        if not universes:
            print(f"No valid universes loaded for {protein}. Skipping PCA calculation.")
            continue

        # Process PCA in chunks
        protein_segs = segs[protein]
        print(
            f"Starting PCA calculation for {protein} using {len(protein_segs.residues)} residues."
        )
        pca_data, explained_variance = pca_processor.fit_transform_chunks(
            universes,
            experiment_names_for_protein,
            indexes_for_protein,
            protein_segs.residues,
            chunk_size,
        )
        pca_data["Protein"] = protein
        print(f"PCA calculation finished for {protein}.")

        # Optimize for plotting
        print(f"Optimizing PCA data for plotting for {protein}.")
        plot_data = optimize_plotting(pca_data)

        # Generate plots
        print(f"Generating PCA index plot for {protein}.")
        plot_PCA_index([protein], experiments, plot_data, explained_variance=explained_variance)
        print(f"Generating PCA density plot for {protein}.")
        plot_PCA_density([protein], experiments, plot_data, explained_variance=explained_variance)

        # Call contour comparison plot using the same PCA data
        print(f"Generating PCA contour comparison plot for {protein}.")
        plot_PCA_contour_comparison(
            protein_name=protein,
            pca_data_for_protein=plot_data.copy(),  # plot_data is already optimized and contains PCs
            ensemble_combo=contour_ensemble_combo,
            explained_variance=explained_variance,
            # experiment_window for axis scaling can use its default or be passed if needed
        )

        # Clear memory
        print(f"Clearing PCA data for {protein} from memory.")
        del pca_data, plot_data
        gc.collect()


def calc_PCA(
    universes: List[mda.Universe],
    experiment_names: List[str],
    indexes: List[List[float]],
    segs: "Segments" = None,
) -> pd.DataFrame:
    """
    Calculates the PCA on the pairwise distances of CA coordinates - only on the residues specified in the segments

    Uses 20 dimensions but saves the frame number and the first 2 PCA dimensions labelled by each experiment to a dataframe for later plotting
    """
    if segs is None:
        raise ValueError("Segments object must be provided")

    all_pairwise_distances = []
    all_labels = []
    all_indexes = []

    for universe, exp_name, index in tqdm(
        zip(universes, experiment_names, indexes), total=len(universes), desc="Calculating PCA"
    ):
        # Select CA atoms for specified residues
        atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, segs.residues))}")
        print(
            f"Processing {exp_name} for PCA: {len(universe.trajectory)} frames, {len(atom_group)} atoms."
        )

        for ts in tqdm(universe.trajectory, desc=f"Frames {exp_name}", leave=False):
            # Calculate pairwise distances
            # pdist calculates the condensed distance matrix, which contains the
            # upper triangle elements of the pairwise distance matrix between CA atoms.
            pairwise_distances = pdist(atom_group.positions)
            all_pairwise_distances.append(pairwise_distances)
            all_labels.append(exp_name)

            # Handle different types of index
            if isinstance(index, (list, np.ndarray)):
                all_indexes.append(index[ts.frame])
            elif isinstance(index, dict):
                all_indexes.append(index.get(ts.frame, 0))  # Default to 0 if frame not found
            else:
                all_indexes.append(0)  # Default value if index type is unknown

    # Convert to numpy array
    all_pairwise_distances = np.array(all_pairwise_distances)

    pca = PCA(n_components=20)

    pca_results = pca.fit_transform(all_pairwise_distances)
    explained_variance = pca.explained_variance_ratio_

    # Create DataFrame
    df = pd.DataFrame(
        {
            "PC1": pca_results[:, 0],
            "PC2": pca_results[:, 1],
            "Experiment": all_labels,
            "Index": all_indexes,
        }
    )

    return df, explained_variance


from typing import List

import pandas as pd

sns.set_style("whitegrid")

from typing import List

import pandas as pd
from matplotlib.cm import ScalarMappable


def plot_PCA_index(
    protein_names: List[str],
    experiments: List[str],
    data: pd.DataFrame,
    experiment_window=["MD-TFES"],
    explained_variance: np.ndarray = None,
) -> None:
    """
    Plot the PCA results colour by index, split by experiment (column) and protein (row)
    Each column shares x and y axes, centered on the specified experiment_window
    Colorbar range is fixed from 0 to 100 for all plots
    """
    fig, axes = plt.subplots(
        len(experiments),
        len(protein_names),
        figsize=(6 * len(protein_names), 4 * len(experiments)),
        squeeze=False,
    )

    if experiment_window not in experiments:
        experiment_window = ["MD-TFES"]

    # Custom colormaps
    colors = ["magenta", "blue", "cyan", "lime", "black"]
    AF2_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

    colour10_MD_cmap = LinearSegmentedColormap.from_list(
        "custom",
        [
            "purple",
            "magenta",
            "blue",
            "cyan",
            "green",
            "greenyellow",
            "yellow",
            "orange",
            "red",
            "black",
        ],
        N=10,
    )

    colour5_MD_cmap = LinearSegmentedColormap.from_list(
        "custom", ["magenta", "blue", "cyan", "greenyellow", "gold"], N=5
    )

    # New colormap for MD-TFES
    tfes_cmap = LinearSegmentedColormap.from_list(
        "tfes", ["black", "navy", "purple", "magenta", "pink"], N=20
    )  # Adjust N based on max cycle number

    for j, protein in enumerate(protein_names):
        protein_data = data[data["Protein"] == protein]

        # Calculate shared axes limits based on experiment_window
        window_data = protein_data[protein_data["Experiment"].isin(experiment_window)]
        x_min, x_max = window_data["PC1"].min(), window_data["PC1"].max()
        y_min, y_max = window_data["PC2"].min(), window_data["PC2"].max()

        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        for i, experiment in enumerate(experiments):
            ax = axes[i, j]
            subset = protein_data[protein_data["Experiment"] == experiment]

            # Select appropriate colormap and normalization based on experiment type
            if "AF2-MSAss" in experiment:
                cmap = "viridis"
                norm = Normalize(vmin=40, vmax=100)
                index_label = "pLDDT"
            elif "AF2-Filtered" in experiment:
                cmap = "viridis"
                norm = None  # This will allow the colormap to scale to the data's min/max
                index_label = "pLDDT"
            elif "MD-TFES" in experiment:
                cmap = tfes_cmap
                max_cycle = max(
                    subset["Index"].max(), 20
                )  # Ensure at least 20 cycles for consistent scale
                norm = Normalize(vmin=0, vmax=max_cycle)
                index_label = "Cycle"
            elif "10Start" in experiment:
                cmap = colour10_MD_cmap
                norm = Normalize(vmin=0, vmax=10)
                index_label = "Cluster number"
            elif "1Start" in experiment:
                cmap = colour5_MD_cmap
                norm = Normalize(vmin=0, vmax=5)
                index_label = "Cluster number"
            else:
                cmap = "magma"
                norm = Normalize(vmin=0, vmax=100)
                index_label = "Trajectory %"

            # Add border color based on dataset color
            border_color = full_dataset_colours.get(experiment, "black")
            ax.spines["top"].set_color(border_color)
            ax.spines["bottom"].set_color(border_color)
            ax.spines["left"].set_color(border_color)
            ax.spines["right"].set_color(border_color)
            ax.spines["top"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_linewidth(2)
            ax.spines["right"].set_linewidth(2)

            size = 60 if "MD-TFES" in experiment else 20
            alpha = 0.6 if "MD-TFES" in experiment else 0.6
            scatter = ax.scatter(
                subset["PC1"],
                subset["PC2"],
                c=subset["Index"],
                cmap=cmap,
                alpha=alpha,
                s=size,
                edgecolors="none",
                norm=norm,
            )

            # Add colored title with single line format
            title_color = full_dataset_colours.get(experiment, "black")
            ax.set_title(
                f"{protein} | {experiment}",
                fontsize=mpl.rcParams["axes.titlesize"],
                fontweight="bold",
                color=title_color,
            )
            xlabel = "PC1"
            ylabel = "PC2"
            if explained_variance is not None and len(explained_variance) >= 2:
                xlabel += f" ({explained_variance[0] * 100:.1f}%)"
                ylabel += f" ({explained_variance[1] * 100:.1f}%)"

            ax.set_xlabel(xlabel, fontsize=mpl.rcParams["axes.labelsize"])
            ax.set_ylabel(ylabel, fontsize=mpl.rcParams["axes.labelsize"])
            ax.tick_params(axis="both", which="major", labelsize=mpl.rcParams["xtick.labelsize"])

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            cbar = plt.colorbar(scatter, ax=ax, label=index_label, aspect=40, norm=norm)
            cbar.ax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"])
            cbar.set_label(index_label, fontsize=mpl.rcParams["axes.labelsize"])

        # Share axes within column
        for ax in axes[1:, j]:
            ax.sharex(axes[0, j])
            ax.sharey(axes[0, j])

    plt.tight_layout()
    plt.savefig(
        f"{'_'.join(protein_names)}-{'_'.join(experiments)}_PCA_index_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_PCA_density(
    protein_names: List[str],
    experiments: List[str],
    data: pd.DataFrame,
    experiment_window=["MD-TFES"],
    explained_variance: np.ndarray = None,
) -> None:
    """
    Plot the PCA results as contour density plots, with each experiment on its own set of axes.
    Creates a grid of subplots: each column is a protein, and each row is an experiment.
    The view is centered on the specified experiment_window.
    A colorbar is added to represent the density levels.
    """
    fig, axes = plt.subplots(
        len(experiments),
        len(protein_names),
        figsize=(6 * len(protein_names), 4 * len(experiments)),
        squeeze=False,
    )
    if experiment_window not in experiments:
        experiment_window = ["MD-TFES"]

    # Use the 'cividis' colormap for the ScalarMappable, matching sns.kdeplot
    cmap_for_colorbar = "cividis"

    # Create a ScalarMappable for the colorbar
    # The normalization can be simple (0 to 1) as it's representing the colormap's range
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap_for_colorbar, norm=norm)
    sm.set_array([])  # Dummy array

    for j, protein in enumerate(protein_names):
        protein_data = data[data["Protein"] == protein]

        # Calculate the limits for shared axes based on the experiment_window
        window_data = protein_data[protein_data["Experiment"].isin(experiment_window)]
        x_min, x_max = window_data["PC1"].min(), window_data["PC1"].max()
        y_min, y_max = window_data["PC2"].min(), window_data["PC2"].max()

        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        for i, experiment in enumerate(experiments):
            ax = axes[i, j]
            subset = protein_data[protein_data["Experiment"] == experiment]

            # Get dataset-specific color
            exp_color = full_dataset_colours.get(experiment, "black")

            # Add border with dataset color
            ax.spines["top"].set_color(exp_color)
            ax.spines["bottom"].set_color(exp_color)
            ax.spines["left"].set_color(exp_color)
            ax.spines["right"].set_color(exp_color)
            ax.spines["top"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_linewidth(2)
            ax.spines["right"].set_linewidth(2)

            # Use cividis colormap for density plots
            sns.kdeplot(
                data=subset,
                x="PC1",
                y="PC2",
                ax=ax,
                cmap="cividis",  # This is the colormap used for plotting
                fill=True,
                alpha=0.9,
                levels=15,
                thresh=0.01,
            )

            # Set title with dataset color and single line format
            ax.set_title(
                f"{protein} | {experiment}",
                fontsize=mpl.rcParams["axes.titlesize"],
                fontweight="bold",
                color=exp_color,
            )
            xlabel = "PC1"
            ylabel = "PC2"
            if explained_variance is not None and len(explained_variance) >= 2:
                xlabel += f" ({explained_variance[0] * 100:.1f}%)"
                ylabel += f" ({explained_variance[1] * 100:.1f}%)"

            ax.set_xlabel(xlabel, fontsize=mpl.rcParams["axes.labelsize"])
            ax.set_ylabel(ylabel, fontsize=mpl.rcParams["axes.labelsize"])
            ax.tick_params(axis="both", which="major", labelsize=mpl.rcParams["xtick.labelsize"])

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Share x and y axes for each column
        for ax_col_idx in range(len(axes[0])):  # Iterate through columns
            if len(axes) > 1:  # Ensure there's more than one row to share axes
                for row_idx in range(1, len(axes)):
                    axes[row_idx, ax_col_idx].sharex(axes[0, ax_col_idx])
                    axes[row_idx, ax_col_idx].sharey(axes[0, ax_col_idx])

    # Add colorbar to the right of the subplots
    # Position and size: [left, bottom, width, height] relative to figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Density", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"])
    cbar.set_label("Density", fontsize=mpl.rcParams["axes.labelsize"], rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    filename = f"{'_'.join(protein_names)}-{'_'.join(experiments)}_PCA_density_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {filename}")
    plt.close()


def plot_mean_correlations_by_residue(
    protein: str, ensemble_experiments: Dict, segs: Dict[str, "Segments"], experiments: List[str]
) -> None:
    """
    Plot mean correlation distributions for each residue across different ensembles
    with enhanced visualization features.
    """
    print(f"Starting mean correlation plotting for {protein}.")
    residues = segs[protein].residues
    n_residues = len(residues)
    print(f"Number of residues for {protein}: {n_residues}")

    # Calculate correlations for each ensemble
    ensemble_correlations = {}

    for experiment in tqdm(experiments, desc=f"Calculating Correlations for {protein}"):
        if protein not in ensemble_experiments[experiment]:
            print(f"Skipping {experiment} for {protein}: data not found.")
            continue

        exp_data = ensemble_experiments[experiment][protein]
        if "top" not in exp_data or "traj" not in exp_data:
            print(f"Skipping {experiment} for {protein}: topology or trajectory path missing.")
            continue

        try:
            print(f"  Processing {experiment} for {protein}...")
            universe = mda.Universe(exp_data["top"], exp_data["traj"])
            _, correlations = calculate_local_correlations(universe, residues)
            ensemble_correlations[experiment] = correlations
            print(f"  Successfully calculated correlations for {experiment}, {protein}.")
        except Exception as e:
            print(f"  Error processing {experiment} for {protein} correlations: {str(e)}")
            continue

    if not ensemble_correlations:
        print(f"No valid correlation data calculated for {protein}")
        return

    # Create figure with specific dimensions and DPI
    plt.figure(figsize=(15, 8), dpi=100)

    # Set style
    plt.style.use("seaborn-whitegrid")

    # Use predefined colors from full_dataset_colours
    for exp_name, corr_matrix in ensemble_correlations.items():
        mean_corrs = np.nanmean(corr_matrix, axis=1)
        std_corrs = np.nanstd(corr_matrix, axis=1)

        # Get color from predefined dictionary
        color = full_dataset_colours.get(exp_name, "gray")

        # Plot mean line with larger width and slight transparency
        line = plt.plot(residues, mean_corrs, label=exp_name, color=color, linewidth=2.5, alpha=0.8)

        # Add error band (standard deviation)
        plt.fill_between(
            residues, mean_corrs - std_corrs, mean_corrs + std_corrs, color=color, alpha=0.1
        )

    # Enhance x-axis
    plt.xlabel("Residue Number", fontsize=mpl.rcParams["axes.labelsize"])

    # Set x-ticks to show reasonable number of residues
    tick_spacing = max(len(residues) // 20, 1)  # Show ~20 ticks
    plt.xticks(residues[::tick_spacing], rotation=45)

    # Enhance y-axis
    plt.ylabel("Mean Local Correlation", fontsize=mpl.rcParams["axes.labelsize"])

    # Add horizontal lines at important correlation values
    plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Enhanced title with single line format
    plt.title(
        f"{protein} | Local Motion Correlations (Mean correlation with residues within 8Å)",
        fontsize=mpl.rcParams["axes.titlesize"],
        pad=20,
    )

    # Enhanced legend
    plt.legend(
        # bbox_to_anchor=(1.05, 1),
        loc="lower left",
        fontsize=mpl.rcParams["legend.fontsize"],
        frameon=True,
        fancybox=True,
        framealpha=0.6,
        # shadow=True,
        title="Ensembles",
        title_fontsize=mpl.rcParams["legend.fontsize"],
    )

    # Add grid but make it subtle
    plt.grid(True, alpha=0.3, linestyle=":")

    # Create standardized filename
    experiments_sorted = sorted(experiments)
    experiments_str = "_".join(experiments_sorted)
    filename = f"{protein}_local_correlations_{experiments_str}.png"

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()


def plot_mean_correlations_by_residue_faceted(
    protein: str, ensemble_experiments: Dict, segs: Dict[str, "Segments"], experiments: List[str]
) -> None:
    """
    Plot mean correlation distributions for each residue across different ensembles,
    with each ensemble in its own subplot row.
    """
    print(f"Starting faceted mean correlation plotting for {protein}.")
    residues = segs[protein].residues
    n_residues = len(residues)
    print(f"Number of residues for {protein}: {n_residues}")

    # Calculate correlations for each ensemble
    ensemble_correlations = {}
    valid_experiments = []

    for experiment in tqdm(experiments, desc=f"Calculating Correlations (Faceted) for {protein}"):
        if protein not in ensemble_experiments[experiment]:
            print(f"Skipping {experiment} for {protein}: data not found.")
            continue

        exp_data = ensemble_experiments[experiment][protein]
        if "top" not in exp_data or "traj" not in exp_data:
            print(f"Skipping {experiment} for {protein}: topology or trajectory path missing.")
            continue

        try:
            print(f"  Processing {experiment} for {protein} (faceted)...")
            universe = mda.Universe(exp_data["top"], exp_data["traj"])
            _, correlations = calculate_local_correlations(universe, residues)
            ensemble_correlations[experiment] = correlations
            valid_experiments.append(experiment)
            print(f"  Successfully calculated correlations for {experiment}, {protein} (faceted).")
        except Exception as e:
            print(f"  Error processing {experiment} for {protein} correlations (faceted): {str(e)}")
            continue

    if not ensemble_correlations:
        print(f"No valid correlation data calculated for {protein}")
        return

    # Create figure with subplots
    n_experiments = len(valid_experiments)
    fig, axes = plt.subplots(
        n_experiments,
        1,
        figsize=(15, 4 * n_experiments),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    if n_experiments == 1:
        axes = [axes]

    # Set style
    plt.style.use("seaborn-whitegrid")

    # Plot each ensemble in its own subplot
    for idx, (experiment, ax) in enumerate(zip(valid_experiments, axes)):
        corr_matrix = ensemble_correlations[experiment]
        color = full_dataset_colours.get(experiment, "black")

        # Calculate mean and std
        mean_corrs = np.nanmean(corr_matrix, axis=1)
        std_corrs = np.nanstd(corr_matrix, axis=1)

        # Plot mean line
        ax.plot(
            residues, mean_corrs, color=color, linewidth=2.5, alpha=0.8, label="Mean correlation"
        )

        # Add error band
        ax.fill_between(
            residues,
            mean_corrs - std_corrs,
            mean_corrs + std_corrs,
            color=color,
            alpha=0.2,
            label="±1 std. dev.",
        )

        # Add horizontal reference lines
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

        # Set y-axis label
        ax.set_ylabel("Local Correlation", fontsize=mpl.rcParams["axes.labelsize"])
        ax.set_ylim(0, 1)  # Set y-limits for better visibility
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=":")

        # Add subplot title with single line format
        ax.set_title(
            f"{protein} | {experiment}",
            fontsize=mpl.rcParams["axes.titlesize"],
            pad=10,
            color=color,
        )

        # Customize the axes with the dataset color
        ax.spines["top"].set_color(color)
        ax.spines["right"].set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["top"].set_linewidth(2)
        ax.spines["right"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

        # Add mean correlation value
        mean_overall = np.nanmean(corr_matrix)
        median_overall = np.nanmedian(corr_matrix)
        ax.text(
            0.02,
            0.80,
            f"Mean: {mean_overall:.2f}\nMedian: {median_overall:.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=color, boxstyle="round,pad=0.5"),
            fontsize=14,
        )

        # Add legend
        ax.legend(
            loc="lower left",
            fontsize=mpl.rcParams["legend.fontsize"],
            frameon=True,
            fancybox=True,
            framealpha=0.6,
        )

    # Set x-axis labels on bottom subplot
    axes[-1].set_xlabel("Residue Number", fontsize=mpl.rcParams["axes.labelsize"])

    # Set reasonable number of x-ticks
    tick_spacing = max(len(residues) // 20, 1)
    axes[-1].set_xticks(residues[::tick_spacing])
    axes[-1].tick_params(axis="x", rotation=45)

    # Add overall title with single line format
    fig.suptitle(
        "Local Motion Correlations (Mean correlation with residues within 8Å)",
        fontsize=mpl.rcParams["axes.titlesize"],
        fontweight="bold",
    )

    # Create standardized filename
    experiments_sorted = sorted(experiments)
    experiments_str = "_".join(experiments_sorted)
    filename = f"{protein}_local_correlations_faceted_{experiments_str}.png"
    plt.tight_layout()
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()


def calculate_local_correlations(
    universe: mda.Universe, residues: np.ndarray, cutoff: float = 8.0
) -> tuple:
    """
    Calculate local cross-correlations between CA atoms within cutoff distance.
    Only includes residues covered by HDX-MS data.
    Uses a cutoff of 8 Angstroms for local environment.
    Excludes self-correlation terms from the analysis.

    Args:
        universe: MDAnalysis Universe
        residues: Array of residue numbers to analyze (from HDX-MS data)
        cutoff: Distance cutoff in Angstroms

    Returns:
        Tuple of (mean_distance_matrix, correlation_matrix)
    """
    # Select CA atoms for specified residues
    atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, residues))}")
    n_residues = len(atom_group)

    # Initialize arrays
    all_positions = np.zeros((len(universe.trajectory), n_residues, 3))
    dist_matrices = np.zeros((len(universe.trajectory), n_residues, n_residues))
    print(
        f"Calculating local correlations for {n_residues} residues using {len(universe.trajectory)} frames."
    )

    # Collect positions and calculate distance matrices for each frame
    for ts_idx, ts in tqdm(
        enumerate(universe.trajectory),
        total=len(universe.trajectory),
        desc="Calculating Distances for Correlation",
    ):
        positions = atom_group.positions
        all_positions[ts_idx] = positions
        dist_matrices[ts_idx] = squareform(pdist(positions))

    # Calculate mean distance matrix
    mean_dist_matrix = np.mean(dist_matrices, axis=0)

    # Create contact mask (≤ cutoff)
    contact_mask = mean_dist_matrix <= cutoff

    # Set diagonal to False to exclude self-correlations
    np.fill_diagonal(contact_mask, False)

    # Calculate correlation matrix
    correlation_matrix = np.zeros((n_residues, n_residues))

    # Calculate mean positions
    mean_positions = np.mean(all_positions, axis=0)
    print("Mean positions calculated for correlation.")

    # Calculate fluctuations around mean
    fluctuations = all_positions - mean_positions
    print("Fluctuations calculated for correlation.")

    # Calculate correlations for residue pairs within contact distance
    print("Calculating correlation coefficients...")
    for i in tqdm(range(n_residues), desc="Correlating Residue Pairs"):
        for j in range(i, n_residues):
            if contact_mask[i, j]:
                # Get fluctuations for residues i and j
                fluct_i = fluctuations[:, i]
                fluct_j = fluctuations[:, j]

                # Calculate correlation coefficient
                numerator = np.mean(np.sum(fluct_i * fluct_j, axis=1))
                denominator = np.sqrt(np.mean(np.sum(fluct_i * fluct_i, axis=1))) * np.sqrt(
                    np.mean(np.sum(fluct_j * fluct_j, axis=1))
                )

                if denominator > 0:
                    correlation = numerator / denominator
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation

    # Set diagonal to 1.0 (self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)

    # Mask correlations beyond contact distance
    correlation_matrix = np.where(contact_mask, correlation_matrix, np.nan)

    return mean_dist_matrix, correlation_matrix


def calculate_residue_local_rmsd(
    universe1: mda.Universe, universe2: mda.Universe, cutoff: float = 8.0
) -> tuple:
    """
    Calculate local RMSD for each residue's neighborhood within cutoff distance.
    Compares the local environment (residues within 8 Angstroms) between two universes
    based on the RMSD of average pairwise C-alpha distances within that neighborhood.
    Only compares residues that are represented in HDX-MS data.

    Args:
        universe1: First MDAnalysis Universe
        universe2: Second MDAnalysis Universe
        cutoff: Distance cutoff in Angstroms (default 8.0)



    Returns:
        Tuple of (local_rmsd_values, residue_ids) - local RMSD for each residue and its ID
    """
    # Select CA atoms
    atom_group1 = universe1.select_atoms("name CA")
    atom_group2 = universe2.select_atoms("name CA")

    # Verify the universes have matching atom counts
    if len(atom_group1) != len(atom_group2):
        raise ValueError(
            f"Atom groups have different lengths: {len(atom_group1)} vs {len(atom_group2)}"
        )

    # Verify residue IDs match between universes
    if not np.array_equal(atom_group1.resids, atom_group2.resids):
        raise ValueError("Residue IDs between universes do not match")

    res_ids = atom_group1.resids
    num_residues = len(res_ids)

    # Store all CA positions for each frame
    all_positions1_list = []
    print(
        f"Calculating residue local RMSD of pairwise distances between two universes for {num_residues} residues."
    )

    print("  Extracting positions from universe 1...")
    for ts in tqdm(universe1.trajectory, desc="Frames Universe 1 (RMSD)"):
        all_positions1_list.append(atom_group1.positions.copy())

    if not all_positions1_list:
        print("Warning: Universe 1 trajectory is empty.")
        return np.full(num_residues, np.nan), res_ids
    all_positions1 = np.array(all_positions1_list)

    all_positions2_list = []
    print("  Extracting positions from universe 2...")
    for ts in tqdm(universe2.trajectory, desc="Frames Universe 2 (RMSD)"):
        all_positions2_list.append(atom_group2.positions.copy())

    if not all_positions2_list:
        print("Warning: Universe 2 trajectory is empty.")
        return np.full(num_residues, np.nan), res_ids
    all_positions2 = np.array(all_positions2_list)

    # Calculate mean positions across all frames for each universe
    print("  Calculating mean positions for both universes (RMSD)...")
    mean_pos1 = np.mean(all_positions1, axis=0)
    # mean_pos2 = np.mean(all_positions2, axis=0) # Not strictly needed for this version of RMSD
    print("  Mean positions calculated (RMSD).")

    # Calculate local RMSD for each residue's environment
    local_rmsds = np.zeros(num_residues)
    print("  Calculating local RMSD of pairwise distances per residue...")
    for i in tqdm(range(num_residues), desc="Residue Local RMSD of Pairwise Distances"):
        # Find local neighbors within cutoff distance from mean positions in first universe
        distances_to_center_res = np.linalg.norm(mean_pos1 - mean_pos1[i], axis=1)
        local_indices = np.where(distances_to_center_res <= cutoff)[0]

        if len(local_indices) < 2:  # Need at least 2 atoms to form a pair
            local_rmsds[i] = 0.0  # Or np.nan if preferred for no-pair cases
            continue

        # Calculate average pairwise distances for local_indices in universe1
        sum_pdist1_local = None
        num_frames1 = len(universe1.trajectory)
        if num_frames1 > 0:
            for frame_idx in range(num_frames1):
                local_coords_frame1 = all_positions1[frame_idx, local_indices, :]
                pdist_frame1 = pdist(local_coords_frame1)
                if sum_pdist1_local is None:
                    sum_pdist1_local = np.zeros_like(pdist_frame1)
                sum_pdist1_local += pdist_frame1
            avg_pdist1_local = sum_pdist1_local / num_frames1
        else:  # Should be caught by earlier check, but as safeguard
            local_rmsds[i] = np.nan
            continue

        # Calculate average pairwise distances for local_indices in universe2
        sum_pdist2_local = None
        num_frames2 = len(universe2.trajectory)
        if num_frames2 > 0:
            for frame_idx in range(num_frames2):
                local_coords_frame2 = all_positions2[frame_idx, local_indices, :]
                pdist_frame2 = pdist(local_coords_frame2)
                if sum_pdist2_local is None:
                    sum_pdist2_local = np.zeros_like(pdist_frame2)
                sum_pdist2_local += pdist_frame2
            avg_pdist2_local = sum_pdist2_local / num_frames2
        else:  # Should be caught by earlier check
            local_rmsds[i] = np.nan
            continue

        if len(avg_pdist1_local) == 0:  # Should be caught by len(local_indices) < 2
            local_rmsds[i] = 0.0
            continue

        # Calculate RMSD between the two average pdist vectors
        squared_diffs_pdist = (avg_pdist1_local - avg_pdist2_local) ** 2
        local_rmsds[i] = np.sqrt(np.mean(squared_diffs_pdist))

    return local_rmsds, res_ids


def save_bfactors(
    protein_dir: str,
    protein: str,
    exp1: str,
    exp2: str,
    topology: str,
    values: np.ndarray,
    res_ids: np.ndarray,
    suffix: str,
):
    """Save values as B-factors in PDB format, mapping values to specific residue IDs."""
    u = mda.Universe(topology)

    # Create a mapping from residue ID to value
    value_map = {res_id: val for res_id, val in zip(res_ids, values)}

    bfactors = np.zeros(len(u.atoms))

    for res in u.residues:
        if res.resid in value_map:
            val = value_map[res.resid]
            res_indices = res.atoms.indices
            bfactors[res_indices] = val
        # else: # Optional: handle residues not in res_ids (e.g., set to 0 or a specific marker)
        # res_indices = res.atoms.indices
        # bfactors[res_indices] = 0.0 # Default if residue not in the provided list

    u.add_TopologyAttr("tempfactors", bfactors)
    output_path = os.path.join(protein_dir, f"{protein}_{exp1}_vs_{exp2}_{suffix}.pdb")
    u.atoms.write(output_path)
    print(f"Saved B-factors to {output_path}")


def calculate_single_residue_wasserstein(args):
    """Calculate Wasserstein distance for a single residue"""
    i, res_id, data1, data2, num_points = args

    if len(data1) == 0 or len(data2) == 0:
        return i, res_id, np.nan

    try:
        kde1 = gaussian_kde(data1)
        kde2 = gaussian_kde(data2)

        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        x_eval = np.linspace(min_val, max_val, num=num_points)

        pdf1 = kde1(x_eval)
        pdf2 = kde2(x_eval)

        w1_distance = wasserstein_distance(x_eval, x_eval, pdf1, pdf2)
        return i, res_id, w1_distance
    except Exception as e:
        print(f"Error calculating W1 for residue {res_id} (index {i}): {e}")
        return i, res_id, np.nan


def calculate_residue_wasserstein(
    universe1: mda.Universe,
    universe2: mda.Universe,
    num_points: int = 1000,
    n_processes: int = None,
) -> tuple:
    """Calculate Wasserstein distance for each residue's CA atom pairwise distances to all other CA atoms"""

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 2)  # Limit to 8 processes to avoid memory issues

    atom_group1 = universe1.select_atoms("name CA")
    atom_group2 = universe2.select_atoms("name CA")

    if len(atom_group1) != len(atom_group2):
        raise ValueError(
            f"Atom groups have different lengths: {len(atom_group1)} vs {len(atom_group2)}"
        )
    if not np.array_equal(atom_group1.resids, atom_group2.resids):
        raise ValueError("Residue IDs between universes do not match.")

    res_ids = atom_group1.resids
    num_residues = len(res_ids)

    all_frame_distances1 = [[] for _ in range(num_residues)]
    all_frame_distances2 = [[] for _ in range(num_residues)]
    print(
        f"Calculating residue Wasserstein distance for {num_residues} residues using {n_processes} processes."
    )

    # For universe 1
    print("  Processing universe 1 for Wasserstein distances...")
    for ts in tqdm(universe1.trajectory, desc="Frames Universe 1 (Wasserstein)"):
        positions = atom_group1.positions
        for i in range(num_residues):
            dist_vector = np.linalg.norm(positions - positions[i], axis=1)
            all_frame_distances1[i].extend(dist_vector)

    # For universe 2
    print("  Processing universe 2 for Wasserstein distances...")
    for ts in tqdm(universe2.trajectory, desc="Frames Universe 2 (Wasserstein)"):
        positions = atom_group2.positions
        for i in range(num_residues):
            dist_vector = np.linalg.norm(positions - positions[i], axis=1)
            all_frame_distances2[i].extend(dist_vector)

    # Prepare arguments for multiprocessing
    print("  Preparing data for parallel Wasserstein calculation...")
    args_list = []
    for i in range(num_residues):
        data1 = np.array(all_frame_distances1[i])
        data2 = np.array(all_frame_distances2[i])
        args_list.append((i, res_ids[i], data1, data2, num_points))

    # Calculate Wasserstein distances in parallel
    print(f"  Calculating Wasserstein distances using {n_processes} processes...")
    w1_distances_per_residue = np.zeros(num_residues)

    with mp.Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_single_residue_wasserstein, args_list),
                total=num_residues,
                desc="Residue Wasserstein Calculation",
            )
        )

    # Collect results
    for i, res_id, w1_distance in results:
        w1_distances_per_residue[i] = w1_distance

    return w1_distances_per_residue, res_ids


def compute_and_plot_wasserstein(
    ensemble_experiments: Dict,
    protein_names: List[str],
    experiments: List[str],
    output_dir: str = "wasserstein_results",
) -> None:
    """Compute and visualize Wasserstein distance analysis"""

    os.makedirs(output_dir, exist_ok=True)

    for protein in protein_names:
        print(f"\nProcessing Wasserstein distances for {protein}")

        protein_specific_output_dir = os.path.join(output_dir, protein)
        os.makedirs(protein_specific_output_dir, exist_ok=True)

        valid_experiments = [
            exp
            for exp in experiments
            if protein in ensemble_experiments[exp]
            and "top" in ensemble_experiments[exp][protein]
            and "traj" in ensemble_experiments[exp][protein]
        ]

        if len(valid_experiments) < 2:
            print(f"Insufficient data for {protein} to compare for Wasserstein distances.")
            continue

        # Initialize matrices for mean and median W1 distances
        mean_w1_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )
        median_w1_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )

        for exp1, exp2 in tqdm(
            list(combinations(valid_experiments, 2)), desc=f"Wasserstein {protein}"
        ):
            try:
                print(f"  Calculating W1 distance between {exp1} and {exp2} for {protein}")
                universe1 = mda.Universe(
                    ensemble_experiments[exp1][protein]["top"],
                    ensemble_experiments[exp1][protein]["traj"],
                )
                universe2 = mda.Universe(
                    ensemble_experiments[exp2][protein]["top"],
                    ensemble_experiments[exp2][protein]["traj"],
                )

                w1_distances_per_res, res_ids = calculate_residue_wasserstein(universe1, universe2)

                # Save B-factors
                # Ensure AF2-MSAss topology exists for the protein
                if (
                    "AF2-MSAss" in ensemble_experiments
                    and protein in ensemble_experiments["AF2-MSAss"]
                ):
                    af2_top = ensemble_experiments["AF2-MSAss"][protein]["top"]
                    save_bfactors(
                        protein_specific_output_dir,
                        protein,
                        exp1,
                        exp2,
                        af2_top,
                        w1_distances_per_res,
                        res_ids,
                        suffix="w1",
                    )
                else:
                    print(
                        f"AF2-MSAss topology not found for {protein}. Skipping B-factor saving for W1 {exp1} vs {exp2}."
                    )

                # Plot residue-level W1 distances
                plt.figure(figsize=(8, 6))
                plt.plot(res_ids, w1_distances_per_res)
                plt.xlabel("Residue", fontsize=mpl.rcParams["axes.labelsize"])
                plt.ylabel("Wasserstein Distance", fontsize=mpl.rcParams["axes.labelsize"])
                plt.title(
                    f"{protein} | {exp1} vs {exp2} W1 Distances",
                    fontsize=mpl.rcParams["axes.titlesize"],
                )
                plt.savefig(
                    os.path.join(
                        protein_specific_output_dir, f"{protein}_{exp1}_vs_{exp2}_w1_plot.png"
                    )
                )
                plt.close()

                mean_w1 = np.nanmean(w1_distances_per_res)
                median_w1 = np.nanmedian(w1_distances_per_res)

                mean_w1_matrix.loc[exp1, exp2] = mean_w1
                mean_w1_matrix.loc[exp2, exp1] = mean_w1
                median_w1_matrix.loc[exp1, exp2] = median_w1
                median_w1_matrix.loc[exp2, exp1] = median_w1

            except Exception as e:
                print(
                    f"Error calculating Wasserstein distance between {exp1} and {exp2} for {protein}: {str(e)}"
                )
                mean_w1_matrix.loc[exp1, exp2] = np.nan
                mean_w1_matrix.loc[exp2, exp1] = np.nan
                median_w1_matrix.loc[exp1, exp2] = np.nan
                median_w1_matrix.loc[exp2, exp1] = np.nan

        np.fill_diagonal(mean_w1_matrix.values, 0)
        np.fill_diagonal(median_w1_matrix.values, 0)

        # Plot Mean W1 Distance Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            mean_w1_matrix,
            annot=True,
            fmt=".3f",
            cmap="gray",
            cbar_kws={"label": "Mean W1 Distance"},
        )
        plt.title(f"{protein} | Mean W1 Distance Heatmap", fontsize=mpl.rcParams["axes.titlesize"])
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Mean_W1_heatmap.png"), dpi=300
        )
        plt.close()

        # Plot Median W1 Distance Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            median_w1_matrix,
            annot=True,
            fmt=".3f",
            cmap="gray",
            cbar_kws={"label": "Median W1 Distance"},
        )  # Using same cmap for consistency
        plt.title(
            f"{protein} | Median W1 Distance Heatmap", fontsize=mpl.rcParams["axes.titlesize"]
        )
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Median_W1_heatmap.png"), dpi=300
        )
        plt.close()


import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def sample_for_global_minmax(
    ensemble_experiments: Dict, protein: str, experiments: List[str], n_samples: int = 500
) -> Tuple[float, float]:
    """Sample structures from all ensembles to find global min/max distances"""

    all_sampled_distances = []

    for experiment in experiments:
        if protein not in ensemble_experiments[experiment]:
            continue

        exp_data = ensemble_experiments[experiment][protein]
        if "top" not in exp_data or "traj" not in exp_data:
            continue

        try:
            import MDAnalysis as mda

            universe = mda.Universe(exp_data["top"], exp_data["traj"])
            atom_group = universe.select_atoms("name CA")
            n_frames = len(universe.trajectory)

            # Sample frames
            sample_frames = np.random.choice(n_frames, min(n_samples, n_frames), replace=False)

            print(f"  Sampling {len(sample_frames)} frames from {experiment}")

            for frame_idx in tqdm(sample_frames):
                universe.trajectory[frame_idx]
                positions = atom_group.positions.astype(np.float16)

                # Sample a subset of residue pairs to save computation
                n_residues = len(positions)
                sample_residues = np.random.choice(n_residues, min(1000, n_residues), replace=False)

                for i in sample_residues[:]:  # Sample fewer residues for speed
                    distances = np.linalg.norm(positions - positions[i], axis=1).astype(np.float16)
                    all_sampled_distances.extend(distances[distances > 0])

                # if len(all_sampled_distances) > 10000:  # Prevent memory issues
                #     break

            del universe
            gc.collect()

        except Exception as e:
            print(f"Error sampling from {experiment}: {e}")
            continue

    if len(all_sampled_distances) == 0:
        return 0.0, 100.0  # Default fallback

    global_min = float(np.min(all_sampled_distances))
    global_max = float(np.max(all_sampled_distances))

    print(f"Global distance range: {global_min:.2f} - {global_max:.2f}")
    return global_min, global_max


def calculate_kde_batch(args):
    """Calculate KDEs for a batch of residues"""
    (
        residue_indices,
        universe_path_top,
        universe_path_traj,
        chunk_size,
        global_min,
        global_max,
        num_points,
    ) = args

    try:
        import MDAnalysis as mda

        universe = mda.Universe(universe_path_top, universe_path_traj)
        atom_group = universe.select_atoms("name CA")
        n_residues = len(atom_group)

        x_eval = np.linspace(global_min, global_max, num_points, dtype=np.float32)
        batch_pdfs = np.zeros((len(residue_indices), num_points), dtype=np.float32)

        for batch_idx, residue_idx in enumerate(residue_indices):
            residue_distances = []

            # Collect distances for this residue
            for start_frame in range(0, len(universe.trajectory), chunk_size):
                end_frame = min(start_frame + chunk_size, len(universe.trajectory))

                for ts in universe.trajectory[start_frame:end_frame]:
                    positions = atom_group.positions.astype(np.float32)
                    distances_from_i = np.linalg.norm(
                        positions - positions[residue_idx], axis=1
                    ).astype(np.float32)
                    other_distances = np.concatenate(
                        [distances_from_i[:residue_idx], distances_from_i[residue_idx + 1 :]]
                    )
                    residue_distances.extend(other_distances[other_distances > 0])

            # Calculate KDE
            if len(residue_distances) > 10:
                try:
                    # Convert to float32 for KDE calculation
                    # residue_distances = np.array(residue_distances, dtype=np.float32)
                    # use histogram sampling if too many points
                    if len(residue_distances) > 500 * n_residues:
                        residue_distances = np.random.choice(
                            residue_distances,
                            size=min(100 * n_residues, len(residue_distances)),
                            replace=False,
                        )
                        kde = gaussian_kde(residue_distances)
                        pdf = kde(x_eval).astype(np.float32)

                    else:
                        # print(residue_distances.shape)
                        kde = gaussian_kde(residue_distances)
                        pdf = kde(x_eval).astype(np.float32)
                    batch_pdfs[batch_idx, :] = pdf
                except Exception as e:
                    print(f"KDE error for residue {residue_idx}: {e}")
                    batch_pdfs[batch_idx, :] = np.zeros(num_points, dtype=np.float32)
            else:
                batch_pdfs[batch_idx, :] = np.zeros(num_points, dtype=np.float32)

        del universe
        gc.collect()

        return residue_indices, batch_pdfs

    except Exception as e:
        print(f"Error in KDE batch calculation: {e}")
        return residue_indices, np.zeros((len(residue_indices), num_points), dtype=np.float32)


def calculate_ensemble_kdes(
    universe_path_top: str,
    universe_path_traj: str,
    pdf_file: str,
    global_min: float,
    global_max: float,
    chunk_size: int = 500,
    num_points: int = 1000,
    batch_size: int = 5,
    n_processes: int = None,
) -> np.ndarray:
    """Calculate KDEs for all residues in an ensemble using parallel batching"""

    if n_processes is None:
        n_processes = min(mp.cpu_count() // 2, 8)

    import MDAnalysis as mda

    universe = mda.Universe(universe_path_top, universe_path_traj)
    atom_group = universe.select_atoms("name CA")
    n_residues = len(atom_group)
    res_ids = atom_group.resids
    del universe

    print(f"  Calculating KDEs for {n_residues} residues using {n_processes} processes")

    # Create memory-mapped file for PDFs
    pdf_memmap = np.memmap(pdf_file, dtype=np.float32, mode="w+", shape=(n_residues, num_points))

    # Create batches of residues
    residue_batches = []
    for i in range(0, n_residues, batch_size):
        end_idx = min(i + batch_size, n_residues)
        residue_batches.append(list(range(i, end_idx)))

    # Prepare arguments for parallel processing
    args_list = [
        (
            batch,
            universe_path_top,
            universe_path_traj,
            chunk_size,
            global_min,
            global_max,
            num_points,
        )
        for batch in residue_batches
    ]

    # Calculate KDEs in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_kde_batch, args_list),
                total=len(args_list),
                desc="KDE batches",
                leave=False,
            )
        )

    # Store results in memory-mapped file
    for residue_indices, batch_pdfs in results:
        for i, residue_idx in enumerate(residue_indices):
            pdf_memmap[residue_idx, :] = batch_pdfs[i, :]

    del pdf_memmap
    gc.collect()

    return res_ids


def calculate_wasserstein_from_ensemble_pdfs(args):
    """Calculate Wasserstein distance using ensemble-level PDFs"""
    (
        i,
        res_id,
        pdf_file1,
        pdf_file2,
        num_points,
        n_residues1,
        n_residues2,
        global_min,
        global_max,
    ) = args  # Add global_min, global_max

    try:
        # Load PDFs from memory-mapped files
        pdf_memmap1 = np.memmap(
            pdf_file1, dtype=np.float32, mode="r", shape=(n_residues1, num_points)
        )
        pdf_memmap2 = np.memmap(
            pdf_file2, dtype=np.float32, mode="r", shape=(n_residues2, num_points)
        )

        pdf1 = pdf_memmap1[i, :].copy()
        pdf2 = pdf_memmap2[i, :].copy()

        del pdf_memmap1, pdf_memmap2

        # Check if PDFs are valid
        if np.sum(pdf1) == 0 or np.sum(pdf2) == 0:
            return i, res_id, np.nan

        # Create evaluation points using the actual distance range (FIXED!)
        x_eval = np.linspace(global_min, global_max, num_points, dtype=np.float32)

        # Calculate Wasserstein distance
        w1_distance = wasserstein_distance(x_eval, x_eval, pdf1, pdf2)

        return i, res_id, w1_distance

    except Exception as e:
        print(f"Error calculating W1 for residue {res_id} (index {i}): {e}")
        return i, res_id, np.nan


def compute_and_plot_wasserstein_efficient(
    ensemble_experiments: Dict,
    protein_names: List[str],
    experiments: List[str],
    chunk_size: int = 500,
    batch_size: int = 10,
    output_dir: str = "wasserstein_results",
    temp_dir: str = None,
) -> None:
    """Memory-efficient computation with ensemble-level KDE caching"""

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    os.makedirs(output_dir, exist_ok=True)

    for protein in protein_names:
        print(f"\nProcessing Wasserstein distances for {protein}")

        protein_specific_output_dir = os.path.join(output_dir, protein)
        os.makedirs(protein_specific_output_dir, exist_ok=True)

        valid_experiments = [
            exp
            for exp in experiments
            if protein in ensemble_experiments[exp]
            and "top" in ensemble_experiments[exp][protein]
            and "traj" in ensemble_experiments[exp][protein]
        ]

        if len(valid_experiments) < 2:
            print(f"Insufficient data for {protein} to compare for Wasserstein distances.")
            continue

        # Phase 1: Find global min/max across all ensembles
        print("Phase 1: Finding global distance range...")
        global_min, global_max = sample_for_global_minmax(
            ensemble_experiments, protein, valid_experiments
        )

        # Phase 2: Calculate KDEs for each ensemble
        print("Phase 2: Computing ensemble KDEs...")
        ensemble_pdf_files = {}
        ensemble_res_ids = {}

        for experiment in valid_experiments:
            print(f"  Computing KDEs for {experiment}")
            exp_data = ensemble_experiments[experiment][protein]
            pdf_file = os.path.join(temp_dir, f"{protein}_{experiment}_{os.getpid()}_pdfs.dat")
            res_ids = calculate_ensemble_kdes(
                exp_data["top"],
                exp_data["traj"],
                pdf_file,
                global_min,
                global_max,
                chunk_size,
                batch_size=batch_size,
            )
            ensemble_pdf_files[experiment] = pdf_file
            ensemble_res_ids[experiment] = res_ids

        # Phase 3: Calculate pairwise Wasserstein distances
        print("Phase 3: Computing pairwise Wasserstein distances...")
        mean_w1_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )
        median_w1_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )

        from itertools import combinations

        for exp1, exp2 in tqdm(
            list(combinations(valid_experiments, 2)), desc=f"Wasserstein {protein}"
        ):
            try:
                print(f"  Calculating W1 distance between {exp1} and {exp2}")
                res_ids = ensemble_res_ids[exp1]
                n_residues = len(res_ids)

                args_list = [
                    (
                        i,
                        res_ids[i],
                        ensemble_pdf_files[exp1],
                        ensemble_pdf_files[exp2],
                        1000,
                        n_residues,
                        n_residues,
                        global_min,  # Add this
                        global_max,  # Add this
                    )
                    for i in range(n_residues)
                ]

                w1_distances = np.zeros(n_residues, dtype=np.float32)
                with mp.Pool(processes=mp.cpu_count() // 2) as pool:
                    results = list(
                        tqdm(
                            pool.imap(calculate_wasserstein_from_ensemble_pdfs, args_list),
                            total=n_residues,
                            desc=f"{exp1} vs {exp2}",
                            leave=False,
                        )
                    )

                for i, res_id, w1_distance in results:
                    w1_distances[i] = w1_distance

                # Save per-residue plot
                plt.figure(figsize=(8, 6))
                plt.plot(res_ids, w1_distances)
                plt.xlabel("Residue")
                plt.ylabel("Wasserstein Distance")
                plt.title(f"{protein} | {exp1} vs {exp2} W1 Distances")
                plt.savefig(
                    os.path.join(
                        protein_specific_output_dir, f"{protein}_{exp1}_vs_{exp2}_w1_plot.png"
                    )
                )
                plt.close()

                # Fill matrix entries
                mean_w1_matrix.loc[exp1, exp2] = np.nanmean(w1_distances)
                mean_w1_matrix.loc[exp2, exp1] = mean_w1_matrix.loc[exp1, exp2]
                median_w1_matrix.loc[exp1, exp2] = np.nanmedian(w1_distances)
                median_w1_matrix.loc[exp2, exp1] = median_w1_matrix.loc[exp1, exp2]
                gc.collect()

            except Exception as e:
                print(
                    f"Error calculating Wasserstein distance between {exp1} and {exp2} for {protein}: {e}"
                )
                mean_w1_matrix.loc[exp1, exp2] = np.nan
                mean_w1_matrix.loc[exp2, exp1] = np.nan
                median_w1_matrix.loc[exp1, exp2] = np.nan
                median_w1_matrix.loc[exp2, exp1] = np.nan

        # Clean up PDF files
        for pdf_file in ensemble_pdf_files.values():
            if os.path.exists(pdf_file):
                try:
                    os.remove(pdf_file)
                except:
                    pass

        np.fill_diagonal(mean_w1_matrix.values, 0)
        np.fill_diagonal(median_w1_matrix.values, 0)

        # Plot Mean W1 Distance Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            mean_w1_matrix,
            annot=True,
            fmt=".2f",
            cmap="gray",
            cbar_kws={"label": "Mean W1 Distance (Å)"},
        )
        plt.title(f"{protein} | Mean W1 Distance Heatmap", fontsize=mpl.rcParams["axes.titlesize"])
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Mean_W1_heatmap.png"), dpi=300
        )
        plt.close()

        # Plot Median W1 Distance Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            median_w1_matrix,
            annot=True,
            fmt=".2f",
            cmap="gray",
            cbar_kws={"label": "Median W1 Distance (Å)"},
        )  # Using same cmap for consistency
        plt.title(
            f"{protein} | Median W1 Distance Heatmap", fontsize=mpl.rcParams["axes.titlesize"]
        )
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Median_W1_heatmap.png"), dpi=300
        )
        plt.close()


def compute_and_plot_local_rmsd(
    ensemble_experiments: Dict,
    protein_names: List[str],
    experiments: List[str],
    output_dir: str = "local_rmsd_results",
) -> None:
    """Compute and visualize local RMSD analysis"""

    os.makedirs(output_dir, exist_ok=True)

    for protein in protein_names:
        print(f"\nProcessing local RMSD for {protein}")

        protein_specific_output_dir = os.path.join(output_dir, protein)
        os.makedirs(protein_specific_output_dir, exist_ok=True)

        valid_experiments = [
            exp
            for exp in experiments
            if protein in ensemble_experiments[exp]
            and "top" in ensemble_experiments[exp][protein]
            and "traj" in ensemble_experiments[exp][protein]
        ]

        if len(valid_experiments) < 2:
            print(f"Insufficient data for {protein} to compare for local RMSD.")
            continue

        mean_distance_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )
        median_distance_matrix = pd.DataFrame(
            index=valid_experiments, columns=valid_experiments, dtype=float
        )

        for exp1, exp2 in tqdm(
            list(combinations(valid_experiments, 2)), desc=f"Local RMSD {protein}"
        ):
            try:
                print(f"  Calculating local RMSD between {exp1} and {exp2} for {protein}")
                universe1 = mda.Universe(
                    ensemble_experiments[exp1][protein]["top"],
                    ensemble_experiments[exp1][protein]["traj"],
                )
                universe2 = mda.Universe(
                    ensemble_experiments[exp2][protein]["top"],
                    ensemble_experiments[exp2][protein]["traj"],
                )

                local_rmsds, res_ids = calculate_residue_local_rmsd(universe1, universe2)

                # Save B-factors using the generic save_bfactors function
                if (
                    "AF2-MSAss" in ensemble_experiments
                    and protein in ensemble_experiments["AF2-MSAss"]
                ):
                    af2_top = ensemble_experiments["AF2-MSAss"][protein]["top"]
                    save_bfactors(
                        protein_specific_output_dir,
                        protein,
                        exp1,
                        exp2,
                        af2_top,
                        local_rmsds,
                        res_ids,
                        suffix="rmsd",
                    )
                else:
                    print(
                        f"AF2-MSAss topology not found for {protein}. Skipping B-factor saving for RMSD {exp1} vs {exp2}."
                    )

                plt.figure(figsize=(8, 6))
                plt.plot(res_ids, local_rmsds)
                plt.xlabel("Residue", fontsize=mpl.rcParams["axes.labelsize"])
                plt.ylabel("Local RMSD (Å)", fontsize=mpl.rcParams["axes.labelsize"])
                plt.title(
                    f"{protein} | {exp1} vs {exp2} Local RMSD",
                    fontsize=mpl.rcParams["axes.titlesize"],
                )
                plt.savefig(
                    os.path.join(
                        protein_specific_output_dir, f"{protein}_{exp1}_vs_{exp2}_rmsd_plot.png"
                    )
                )
                plt.close()

                mean_rmsd = np.nanmean(local_rmsds)
                median_rmsd = np.nanmedian(local_rmsds)

                mean_distance_matrix.loc[exp1, exp2] = mean_rmsd
                mean_distance_matrix.loc[exp2, exp1] = mean_rmsd
                median_distance_matrix.loc[exp1, exp2] = median_rmsd
                median_distance_matrix.loc[exp2, exp1] = median_rmsd

            except Exception as e:
                print(f"Error calculating RMSD between {exp1} and {exp2} for {protein}: {str(e)}")
                mean_distance_matrix.loc[exp1, exp2] = np.nan
                mean_distance_matrix.loc[exp2, exp1] = np.nan
                median_distance_matrix.loc[exp1, exp2] = np.nan
                median_distance_matrix.loc[exp2, exp1] = np.nan

        np.fill_diagonal(mean_distance_matrix.values, 0)
        np.fill_diagonal(median_distance_matrix.values, 0)

        # Plot Mean Local RMSD Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            mean_distance_matrix,
            annot=True,
            fmt=".2f",
            cmap="gray",
            cbar_kws={"label": "Mean Local RMSD (Å)"},
        )
        plt.title(f"{protein} | Mean Local RMSD Heatmap", fontsize=mpl.rcParams["axes.titlesize"])
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Mean_RMSD_heatmap.png"), dpi=300
        )
        plt.close()

        # Plot Median Local RMSD Heatmap
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            median_distance_matrix,
            annot=True,
            fmt=".2f",
            cmap="gray",
            cbar_kws={"label": "Median Local RMSD (Å)"},
        )
        plt.title(f"{protein} | Median Local RMSD Heatmap", fontsize=mpl.rcParams["axes.titlesize"])
        plt.xlabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        plt.ylabel("Ensembles", fontsize=mpl.rcParams["axes.labelsize"])
        # Color tick labels
        for tick_label in ax.get_xticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(full_dataset_colours.get(tick_label.get_text(), "black"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(protein_specific_output_dir, f"{protein}_Median_RMSD_heatmap.png"), dpi=300
        )
        plt.close()


def plot_PCA_contour_comparison(
    protein_name: str,
    pca_data_for_protein: pd.DataFrame,
    ensemble_combo: List[str] = ["AF2-Filtered", "MD-1Start"],
    explained_variance: np.ndarray = None,
    experiment_window: List[str] = ["AF2-Filtered"],
) -> None:
    """
    Plot PCA contour distributions for specified ensemble combinations for a single protein,
    using pre-calculated PCA data.
    Creates unfilled contour lines with thickness and transparency gradients.

    Args:
        protein_name: Single protein to analyze.
        pca_data_for_protein: DataFrame containing PC1, PC2, Experiment, Protein, Index
                              for the protein, derived from a PCA fitted on all its experiments.
        ensemble_combo: List of ensembles to compare from pca_data_for_protein.
        explained_variance: Optional explained variance array for axis labels.
        experiment_window: Reference experiment(s) in pca_data_for_protein for setting axis limits.
    """
    # Contour plot parameters
    density_threshold = 0.0001  # Minimum threshold for density to include in contours
    num_levels = 7  # Number of contour levels
    main_contour_alphas = [0.4, 0.5, 0.6, 0.9]  # Alpha values for main contours
    main_contour_widths = [3, 3.5, 4.0, 5.0]  # Line widths for main contours
    outer_contour_alphas = [0.4, 0.4]  # Alpha values for outer contours
    outer_contour_widths = [2.5, 3.0]  # Line widths for outer contours

    print(f"\nStarting PCA contour comparison for protein: {protein_name}")

    # Determine which of the requested ensemble_combo are actually available in the pca_data
    available_ensembles_in_data = pca_data_for_protein["Experiment"].unique()
    available_ensembles = [exp for exp in ensemble_combo if exp in available_ensembles_in_data]

    if len(available_ensembles) < 1:  # Allow even a single ensemble for contour
        print(
            f"No ensembles from combo {ensemble_combo} found in PCA data for {protein_name}. Skipping contour plot."
        )
        return
    if len(available_ensembles) < 2:
        print(
            f"Warning: Only {len(available_ensembles)} ensemble(s) from {ensemble_combo} available for contour plot for {protein_name}."
        )

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Calculate axis limits based on experiment_window or all data within pca_data_for_protein
    # Ensure experiment_window ensembles are present in the pca_data_for_protein
    valid_experiment_window = [
        exp for exp in experiment_window if exp in available_ensembles_in_data
    ]

    if valid_experiment_window:
        window_data = pca_data_for_protein[
            pca_data_for_protein["Experiment"].isin(valid_experiment_window)
        ]
        if not window_data.empty:
            x_min, x_max = window_data["PC1"].min(), window_data["PC1"].max()
            y_min, y_max = window_data["PC2"].min(), window_data["PC2"].max()
        else:  # Fallback if window_data is empty (e.g. experiment_window not in current protein's data)
            x_min, x_max = pca_data_for_protein["PC1"].min(), pca_data_for_protein["PC1"].max()
            y_min, y_max = pca_data_for_protein["PC2"].min(), pca_data_for_protein["PC2"].max()
    else:  # Fallback if no valid experiment_window
        x_min, x_max = pca_data_for_protein["PC1"].min(), pca_data_for_protein["PC1"].max()
        y_min, y_max = pca_data_for_protein["PC2"].min(), pca_data_for_protein["PC2"].max()

    # Add padding
    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding + 3

    # Create contour plots for each ensemble
    from matplotlib import patheffects

    for i, experiment in enumerate(available_ensembles):
        subset = pca_data_for_protein[pca_data_for_protein["Experiment"] == experiment]

        if len(subset) < 10:  # Need minimum points for KDE
            print(f"Warning: Too few points ({len(subset)}) for {experiment}. Skipping.")
            continue

        # Get ensemble color
        exp_color = full_dataset_colours.get(experiment, "black")

        # Create KDE for contour lines
        try:
            kde = gaussian_kde(np.vstack([subset["PC1"], subset["PC2"]]))

            # Create evaluation grid
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(xx.shape)

            # Create contour levels with threshold
            min_density = max(density.min(), density.max() * density_threshold)
            levels = np.linspace(min_density, density.max(), num_levels)

            # Split levels for main and outer contours
            main_levels = levels[len(levels) // 2 :]  # Higher density levels for main contours
            outer_levels = levels[: len(levels) // 2]  # Lower density levels for outer contours

            # Create main contour lines with custom thickness and alpha
            if len(main_levels) > len(main_contour_widths):
                # If we have more levels than specified widths, adjust widths list
                main_contour_widths = np.linspace(
                    min(main_contour_widths), max(main_contour_widths), len(main_levels)
                ).tolist()

            if len(main_levels) > len(main_contour_alphas):
                # If we have more levels than specified alphas, adjust alphas list
                main_contour_alphas = np.linspace(
                    min(main_contour_alphas), max(main_contour_alphas), len(main_levels)
                ).tolist()

            # Create main contour lines
            contour_set = ax.contour(
                xx,
                yy,
                density,
                levels=main_levels,
                colors=[exp_color],
                alpha=main_contour_alphas[-1],  # Use highest alpha as base
                linewidths=main_contour_widths,
            )

            # Add path effects for better visibility
            for i, collection in enumerate(contour_set.collections):
                # Get appropriate alpha based on index or use last value if out of bounds
                alpha_idx = min(i, len(main_contour_alphas) - 1)
                collection.set_alpha(main_contour_alphas[alpha_idx])
                collection.set_path_effects(
                    [
                        patheffects.withStroke(
                            linewidth=collection.get_linewidth()[0] + 1,
                            foreground="white",
                            alpha=0.5,
                        )
                    ]
                )

            # Create outer contour lines with different alpha for gradient effect
            if len(outer_levels) > 0:  # Only if we have outer levels
                if len(outer_levels) > len(outer_contour_widths):
                    outer_contour_widths = np.linspace(
                        min(outer_contour_widths), max(outer_contour_widths), len(outer_levels)
                    ).tolist()

                if len(outer_levels) > len(outer_contour_alphas):
                    outer_contour_alphas = np.linspace(
                        min(outer_contour_alphas), max(outer_contour_alphas), len(outer_levels)
                    ).tolist()

                contour_set_light = ax.contour(
                    xx,
                    yy,
                    density,
                    levels=outer_levels,
                    colors=[exp_color],
                    linewidths=outer_contour_widths,
                    linestyles="--",
                )

                # Set alphas for outer contours
                for i, collection in enumerate(contour_set_light.collections):
                    alpha_idx = min(i, len(outer_contour_alphas) - 1)
                    collection.set_alpha(outer_contour_alphas[alpha_idx])

        except Exception as e:
            print(f"Error creating contours for {experiment}: {e}")
            continue

    # Customize the plot
    xlabel = "PC1"
    ylabel = "PC2"
    if explained_variance is not None and len(explained_variance) >= 2:
        xlabel += f" ({explained_variance[0] * 100:.1f}%)"
        ylabel += f" ({explained_variance[1] * 100:.1f}%)"

    ax.set_xlabel(xlabel, fontsize=mpl.rcParams["axes.labelsize"])
    ax.set_ylabel(ylabel, fontsize=mpl.rcParams["axes.labelsize"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add title
    ax.set_title(
        f"{protein_name} | PCA | {' vs '.join(available_ensembles)}",
        fontsize=mpl.rcParams["axes.titlesize"] - 4,
        pad=20,
    )

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = []
    for experiment in available_ensembles:
        exp_color = full_dataset_colours.get(experiment, "black")
        legend_elements.append(Line2D([0], [0], color=exp_color, linewidth=4, label=experiment))

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=mpl.rcParams["legend.fontsize"],
        frameon=True,
        fancybox=True,
        framealpha=0.6,
        # title="Ensembles",
        title_fontsize=mpl.rcParams["legend.fontsize"],
    )

    # Add grid
    ax.grid(False, alpha=0.0, linestyle=":")

    # Style the axes
    ax.tick_params(axis="both", which="major", labelsize=mpl.rcParams["xtick.labelsize"])

    # Save the plot
    ensemble_str = "_vs_".join(available_ensembles)
    filename = f"{protein_name}_{ensemble_str}_PCA_contour_comparison.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"Contour comparison plot saved as {filename}")

    # Clean up memory
    # del pca_data_for_protein # This is an argument, so don't delete here
    gc.collect()


# def plot_all_contour_comparisons(
#     protein_names: List[str],
#     ensemble_experiments: Dict,
#     segs: Dict[str, "Segments"],
#     ensemble_combo: List[str] = ["AF2-Filtered", "MD-1Start"],
#     chunk_size: int = 1000,
# ) -> None:
#     """
#     Plot contour comparisons for all specified proteins.
#     """
#     for protein in protein_names:
#         plot_PCA_contour_comparison(
#             protein_name=protein,
#             ensemble_experiments=ensemble_experiments,
#             segs=segs,
#             ensemble_combo=ensemble_combo,
#             chunk_size=chunk_size,
#         )


# Example usage:
if __name__ == "__main__":
    protein_names = ["MBP", "LXR", "BRD4"]
    protein_names = ["HOIP"]
    protein_names = ["BPTI", "HOIP", "MBP", "LXR", "BRD4"]
    # protein_names = ["BPTI"]

    experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "MD-Shaw", "MD-TFES"]
    experiments = ["AF2-MSAss", "AF2-Filtered", "MD-1Start", "MD-10Start", "MD-TFES"]

    os.chdir("/home/alexi/Documents/interpretable-hdxer/notebooks/SI-Fig-1_AF_MD_PCA_Space")
    print(f"Current working directory: {os.getcwd()}")

    print("Starting memory efficient PCA plotting.")
    # memory_efficient_plot_PCA(
    #     ensemble_experiments=ensemble_experiments,
    #     protein_names=protein_names,
    #     segs=segs_data,
    #     experiments=experiments,
    #     chunk_size=100,
    #     contour_ensemble_combo=["AF2-Filtered", "MD-1Start"],  # Example combo
    # )

    for protein in protein_names[:]:
        print(f"Plotting stats for {protein}")
        # plot_mean_correlations_by_residue(protein, ensemble_experiments, segs_data, experiments)

        # print(f"  Plotting faceted mean correlations for {protein}.")
        # plot_mean_correlations_by_residue_faceted(
        #     protein, ensemble_experiments, segs_data, experiments
        # )

        # The call to plot_PCA_contour_comparison is now inside memory_efficient_plot_PCA
        # So, this specific call is removed from here.
        # plot_PCA_contour_comparison(
        #     pca
        #     protein_name=protein,
        #     ensemble_experiments=ensemble_experiments,
        #     segs=segs_data,
        #     ensemble_combo=["AF2-Filtered", "MD-1Start"],
        #     chunk_size=100,
        # )
        print(protein)

    print(f"  Computing and plotting Wasserstein distances for {protein_names}.")
    compute_and_plot_wasserstein_efficient(
        ensemble_experiments=ensemble_experiments,
        protein_names=protein_names,
        experiments=experiments,
        chunk_size=500,
    )
    print(f"  Finished computing and plotting Wasserstein distances for {protein_names}.")
    # compute_and_plot_local_rmsd(
    #     ensemble_experiments=ensemble_experiments,
    #     protein_names=protein_names,
    #     experiments=experiments,
    # )
    print(f"  Finished computing and plotting local RMSD for {protein_names}.")
