# /home/alexi/Documents/interpretable-hdxer/notebooks/Figure_5_Poisoned_Ensemble/Ensemble_Poison.py
import json
import os 
import numpy as np
import MDAnalysis as mda
import pandas as pd
from typing import List
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.XTC import XTCWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px
import sys
from tqdm import tqdm  # Added tqdm for progress bars
import random
import shutil
sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.helpful_funcs import segs_to_df

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class Segments():
    """
    takes in a dataframe of experimental hdx segments 
    and contains the values of the peptides
    """
    def __init__(self, segs_df: pd.DataFrame=None, segs_path:str=None, keys=['ResStr', 'ResEnd']):
        if segs_df is not None:
            self.segs_df = segs_df.copy()
        if segs_path is not None:
            self.segs_df = segs_to_df(segs_path)
        self.keys = keys
        if "peptide" not in self.segs_df.columns:
            self.segs_df["peptide"] = np.arange(len(self.segs_df))
        self.pep_nums = self.get_pep_nums(self.segs_df)
        assert len(self.segs_df["peptide"].unique()) == len(self.pep_nums), "Peptides are not unique"
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
        res_nums = df.apply(lambda x: np.arange(x[keys[0]]+1, x[keys[1]]+1), 
                            axis=1).to_numpy()
        print(f"Resnumbers calculated for {len(res_nums)} segments")
        return res_nums
    
    def get_peptides(self, df: pd.DataFrame=None):
        """
        Get the peptides from the dataframe,
        creates a dictionary with the peptide number as the key and the residues as the values
        """
        if df is None:
            df = self.segs_df
        peptides = {pep: res for pep, res in zip(df["peptide"], 
                                                 self.get_resnums(df))}
        print(f"Peptides calculated for {len(peptides)} segments")
        return peptides
    
    def get_residues(self, df: pd.DataFrame=None):
        """
        Get the residues from the dataframe
        """
        if df is None:
            df = self.segs_df
        residues = np.unique(np.concatenate(self.get_resnums(df)))
        print(f"Residues calculated for {len(residues)} segments")
        return residues
    
    def get_pep_nums(self, df: pd.DataFrame=None):
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

        pep_cent = {pep: np.mean([res_cent[res] for res in res_nums]) 
                    for pep, res_nums in peptides.items()}
        
        print(f"Peptide centrality calculated for {len(pep_cent)} peptides")
        return pep_cent
    
    def update(self, df: pd.DataFrame=None):
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
            df = pd.concat([df, pd.DataFrame({keys[0]: res[0], 
                                              keys[1]: res[-1], 
                                              "peptide": pep}, 
                                              index=[pep])])
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

    
    def select_segments(self, 
                        peptides: list=None, 
                        residues: list=None, 
                        new_segs_df: pd.DataFrame=None,
                        remove: bool=False):
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
            new_segs_df =  residue_update_function(residues)

        print("Updating Segments")
        self.update(new_segs_df)


    def create_train_val_segs(self,
                               train_peps: np.ndarray,
                                 val_peps: np.ndarray):
        
        train_segs = self.df_select_peptides(list(train_peps))
        val_segs = self.df_select_peptides(list(val_peps))

        return train_segs, val_segs



protein_names = ["BPTI", "BRD4", "HOIP", "LXR", "MBP"]
experiments = ["AF2-MSAss", "AF2-Cleaned", "1Start-MD", "10Start-MD"]#, "Shaw-MD"]
experiments = ["AF2-MSAss", "AF2-Cleaned", "1Start-MD", "10Start-MD", "Shaw-MD"]

af_experiments = ["AF2-MSAss", "AF2-Cleaned"]
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
LXR_segs_path = "/home/alexi/Documents/ValDX/raw_data/LXRalpha/LXRalpha_APO/LXRa_APO_segs200_trimmed.txt"
LXR_num_residues = 250


MBP_top_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_protonated.pdb"
MBP_traj_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_protonated.xtc" 
MBP_json_path = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/json_corrected/MBP_wt_1_af_sample_127_10000_ranks.json"
MBP_segs_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP_wt1_segs_trimmed.txt"
MBP_num_residues = 396





AF2_proteins = {"BPTI": {"top": BPTI_top_path, "traj": BPTI_traj_path, "json": BPTI_json_path, "num_residues": BPTI_num_residues, "segs_path":BPTI_segs_path},
            "BRD4": {"top": BRD4_top_path, "traj": BRD4_traj_path, "json": BRD4_json_path, "num_residues": BRD4_num_residues, "segs_path":BRD4_segs_path},
            "HOIP": {"top": HOIP_top_path, "traj": HOIP_traj_path, "json": HOIP_json_path, "num_residues": HOIP_num_residues, "segs_path":HOIP_segs_path},
            "LXR": {"top": LXR_top_path, "traj": LXR_traj_path, "json": LXR_json_path, "num_residues": LXR_num_residues, "segs_path":LXR_segs_path},
            "MBP": {"top": MBP_top_path, "traj": MBP_traj_path, "json": MBP_json_path, "num_residues": MBP_num_residues, "segs_path":MBP_segs_path}}



def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


AF2_json_data = {protein: load_json(AF2_proteins[protein]["json"]) for protein in protein_names}

for protein in protein_names:

    AF2_proteins[protein]["index"] = AF2_json_data[protein]


from typing import Dict

def extract_val_json(protein_data: Dict, key: str="plddt")->np.array:
    
    protein_data = dict(sorted(protein_data.items(), key=lambda x: int(x[0].split("_")[1]), reverse=True))

    val_data = []
    for msa, run in protein_data.items():
        for rank, data in run.items():
            if key in data:
                val_data.append(data[key])

    return np.array(val_data)

plddt_data = {protein: extract_val_json(AF2_json_data[protein]) for protein in protein_names}

max_pae_data = {protein: extract_val_json(AF2_json_data[protein], key='max_pae') for protein in protein_names}

ptm_data = {protein: extract_val_json(AF2_json_data[protein], key='ptm') for protein in protein_names}

for protein in protein_names:
    AF2_proteins[protein]["index"] = plddt_data[protein]



segs_data = {protein:Segments(segs_path=AF2_proteins[protein]["segs_path"]) for protein in AF2_proteins}


res_data = {protein:segs_data[protein].residues for protein in AF2_proteins}
print(res_data)


# cleaned AF 
Cleaned_AF2_proteins = {"BPTI": {"top": BPTI_top_path, "traj": os.path.splitext(BPTI_traj_path)[0]+"_all_filtered.xtc", "num_residues": BPTI_num_residues, "segs_path":BPTI_segs_path, "filtered_plddt_json": os.path.splitext(BPTI_traj_path)[0]+"_all_filtered_frame_plddt.json"},
            "BRD4": {"top": BRD4_top_path, "traj": os.path.splitext(BRD4_traj_path)[0]+"_all_filtered.xtc", "num_residues": BRD4_num_residues, "segs_path":BRD4_segs_path, "filtered_plddt_json": os.path.splitext(BRD4_traj_path)[0]+"_all_filtered_frame_plddt.json"},
            "HOIP": {"top": HOIP_top_path, "traj": os.path.splitext(HOIP_traj_path)[0]+"_all_filtered.xtc", "num_residues": HOIP_num_residues, "segs_path":HOIP_segs_path, "filtered_plddt_json": os.path.splitext(HOIP_traj_path)[0]+"_all_filtered_frame_plddt.json"},
            "LXR": {"top": LXR_top_path, "traj": os.path.splitext(LXR_traj_path)[0]+"_all_filtered.xtc", "num_residues": LXR_num_residues, "segs_path":LXR_segs_path, "filtered_plddt_json": os.path.splitext(LXR_traj_path)[0]+"_all_filtered_frame_plddt.json"},
            "MBP": {"top": MBP_top_path, "traj": os.path.splitext(MBP_traj_path)[0]+"_all_filtered.xtc", "num_residues": MBP_num_residues, "segs_path":MBP_segs_path, "filtered_plddt_json": os.path.splitext(MBP_traj_path)[0]+"_all_filtered_frame_plddt.json"}}

# filtered_plddt is json of "frame": plddt, extract plddt as a list 

def extract_filered_plddt(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [data[frame] for frame in sorted(data.keys())]

filtered_plddt_data = {protein: extract_filered_plddt(Cleaned_AF2_proteins[protein]["filtered_plddt_json"]) for protein in protein_names}

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
        "segs_path": BPTI_segs_path
    },
    "BRD4": {
        "top": f"{regular_MD_base}/BRD4_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/BRD4_test_concatenated_stripped.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path
    },
    "HOIP": {
        "top": f"{regular_MD_base}/HOIP_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/HOIP_test_concatenated_stripped.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path
    },
    "LXR": {
        "top": f"{regular_MD_base}/LXR_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/LXR_test_concatenated_stripped.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path
    },
    "MBP": {
        "top": f"{regular_MD_base}/MBP_test_concatenated_stripped.pdb",
        "traj": f"{regular_MD_base}/MBP_test_concatenated_stripped.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path
    }
}

# 10Start MD paths and data
tenstart_MD_proteins = {
    "BPTI": {
        "top": f"{regular_MD_base}/BPTI_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/BPTI_10_c_combined.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path
    },
    "BRD4": {
        "top": f"{regular_MD_base}/BRD4_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/BRD4_10_c_combined.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path
    },
    "HOIP": {
        "top": f"{regular_MD_base}/HOIP_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/HOIP_10_c_combined.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path
    },
    "LXR": {
        "top": f"{regular_MD_base}/LXR_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/LXR_10_c_combined.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path
    },
    "MBP": {
        "top": f"{regular_MD_base}/MBP_10_c_combined.pdb",
        "traj": f"{regular_MD_base}/MBP_10_c_combined.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path
    }
}

# Regular (Bad) MD paths and data
regular_MD_proteins = {
    "BPTI": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/BadMD_BPTI_r5_15010_concatenated.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path
    },
    "BRD4": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/BadMD_BRD4_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/BadMD_BRD4_r5_15010_concatenated.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path
    },
    "HOIP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/BadMD_HOIP_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/BadMD_HOIP_r5_15010_concatenated.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path
    },
    "LXR": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/BadMD_LXR_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/BadMD_LXR_r5_15010_concatenated.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path
    },
    "MBP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/BadMD_MBP_r5_15010_concatenated.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path
    }
}

# Better (Good) MD paths and data
better_MD_proteins = {
    "BPTI": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BPTI/GoodMD_BPTI_r10_10010_concatenated.xtc",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path
    },
    "BRD4": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/GoodMD_BRD4_r6_12006_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/BRD4/GoodMD_BRD4_r6_12006_concatenated.xtc",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path
    },
    "HOIP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/GoodMD_HOIP_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/HOIP/GoodMD_HOIP_r10_10010_concatenated.xtc",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path
    },
    "LXR": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/GoodMD_LXR_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/LXRa/GoodMD_LXR_r10_10010_concatenated.xtc",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path
    },
    "MBP": {
        "top": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.pdb",
        "traj": "/home/alexi/Documents/ValDX/raw_data/good_bad_MD/MBP/GoodMD_MBP_r10_10010_concatenated.xtc",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path
    }
}

# Add index data for 1Start MD
for protein in protein_names:
    n_frames = get_n_frames(onestart_MD_proteins[protein]["top"], 
                           onestart_MD_proteins[protein]["traj"])
    onestart_MD_proteins[protein]["index"] = [100/n_frames*i for i in range(n_frames)]
    print(f"{protein} 1Start MD frames: {n_frames}")

# Add index data for 10Start MD
for protein in protein_names:
    n_frames = get_n_frames(tenstart_MD_proteins[protein]["top"], 
                           tenstart_MD_proteins[protein]["traj"])
    tenstart_MD_proteins[protein]["index"] = [100/n_frames*i for i in range(n_frames)]
    print(f"{protein} 10Start MD frames: {n_frames}")

# Add index data for regular MD
for protein in protein_names:
    if protein == "BRD4":
        regular_MD_proteins[protein]["index"] = [100/15010*i for i in range(15010)]
    else:
        regular_MD_proteins[protein]["index"] = [100/15010*i for i in range(15010)]

# Add index data for better MD
for protein in protein_names:
    if protein == "BRD4":
        better_MD_proteins[protein]["index"] = [100/12006*i for i in range(12006)]
    else:
        better_MD_proteins[protein]["index"] = [100/10010*i for i in range(10010)]

shaw_MD_BPTI_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
shaw_MD_BPTI_traj_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_400.xtc"

test_universe = mda.Universe(shaw_MD_BPTI_top_path, shaw_MD_BPTI_traj_path)


shaw_MD_proteins = {"BPTI": {"top": shaw_MD_BPTI_top_path, "traj": shaw_MD_BPTI_traj_path}, "num_residues": BPTI_num_residues, "segs_path":BPTI_segs_path}


shaw_MD_proteins["BPTI"]["index"] = [100/len(test_universe.trajectory)*i for i in range(len(test_universe.trajectory))]


# Update ensemble_experiments dictionary
ensemble_experiments = {
    "AF2-MSAss": AF2_proteins, 
    "AF2-Cleaned": Cleaned_AF2_proteins, 
    "1Start-MD": onestart_MD_proteins,
    "10Start-MD": tenstart_MD_proteins,
    # "Regular-MD": regular_MD_proteins,
    # "Better-MD": better_MD_proteins,
    # "Shaw-MD": shaw_MD_proteins
}

def load_tfes_frame_info(json_path):
    """
    Load and process T-FES per-frame information.
    Creates a list where index corresponds to frame number (0-based)
    and value is the cycle number.
    """
    with open(json_path, 'r') as f:
        frame_info = json.load(f)
    
    # Find the maximum frame number to determine list size
    max_frame = max(item['sampled_frame_no'] for item in frame_info)
    
    # Create a list initialized with zeros
    # Add 1 to max_frame since frame numbers are 1-based but we need 0-based indexing
    cycle_list = [0] * (max_frame)
    
    # Fill in the cycle numbers
    # Subtract 1 from frame number to convert to 0-based indexing
    for item in frame_info:
        frame_idx = item['sampled_frame_no'] - 1  # Convert to 0-based index
        cycle_list[frame_idx] = item['cycle']
    
    return cycle_list

# Base path for T-FES data
tfes_base = "/home/alexi/Documents/interpretable-hdxer/data/si_ifg1/T-FES"

# T-FES paths and data structure
tfes_proteins = {
    "BPTI": {
        "top": f"{tfes_base}/BPTI/final/BPTI_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/BPTI/final/resampled_outputs/BPTI_sampled.xtc",
        "frame_info": f"{tfes_base}/BPTI/final/resampled_outputs/BPTI_sampled_per_frame_info.json",
        "num_residues": BPTI_num_residues,
        "segs_path": BPTI_segs_path
    },
    "BRD4": {
        "top": f"{tfes_base}/BRD4/final/BRD4_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/BRD4/final/resampled_outputs/BRD4_sampled.xtc",
        "frame_info": f"{tfes_base}/BRD4/final/resampled_outputs/BRD4_sampled_per_frame_info.json",
        "num_residues": BRD4_num_residues,
        "segs_path": BRD4_segs_path
    },


    "HOIP": {
        "top": f"{tfes_base}/HOIP/final/HOIP_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/HOIP/final/resampled_outputs/HOIP_sampled.xtc",
        "frame_info": f"{tfes_base}/HOIP/final/resampled_outputs/HOIP_sampled_per_frame_info.json",
        "num_residues": HOIP_num_residues,
        "segs_path": HOIP_segs_path
    },
    "LXR": {
        "top": f"{tfes_base}/LXR/final/LXR_combined.pdb",
        "traj": f"{tfes_base}/LXR/final/resampled_outputs/LXR_sampled.xtc",
        "frame_info": f"{tfes_base}/LXR/final/resampled_outputs/LXR_sampled_per_frame_info.json",
        "num_residues": LXR_num_residues,
        "segs_path": LXR_segs_path
    },
    "MBP": {
        "top": f"{tfes_base}/MBP/final/MBP_overall_combined_stripped.pdb",
        "traj": f"{tfes_base}/MBP/final/resampled_outputs/MBP_sampled.xtc",
        "frame_info": f"{tfes_base}/MBP/resampled_outputs/MBP_sampled_per_frame_info.json",
        "num_residues": MBP_num_residues,
        "segs_path": MBP_segs_path
    }
}
# Load frame info for each protein and add to the data structure
for protein in tfes_proteins:
    try:
        print(f"Loading T-FES frame info for {protein}")
        frame_info = load_tfes_frame_info(tfes_proteins[protein]["frame_info"])
        print(f"Loaded {len(frame_info)} frames for {protein}")
        tfes_proteins[protein]["index"] = frame_info
    except:
        Warning(f"Failed to load T-FES frame info for {protein}")

# Update ensemble_experiments dictionary to include T-FES
ensemble_experiments["T-FES"] = tfes_proteins

# Update experiments list
experiments = ["AF2-MSAss", "AF2-Cleaned", "1Start-MD", "10Start-MD", "Shaw-MD", "T-FES"]
experiments = ["AF2-MSAss", "AF2-Cleaned", "1Start-MD", "10Start-MD",  "T-FES"]
import os
import numpy as np
import MDAnalysis as mda
from typing import Dict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

def shuffle_backbone_protons(universe: mda.Universe, output_path: str):
    """Shuffle the positions of backbone protons in place."""
    with mda.Writer(output_path, universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            backbone = universe.select_atoms("backbone and name H")
            coords = backbone.positions.copy()
            np.random.shuffle(coords)
            backbone.positions = coords
            W.write(universe.atoms)

def mix_coordinates(universe: mda.Universe, output_path: str):
    """Randomly permute atomic positions in each frame in place."""
    with mda.Writer(output_path, universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            universe.atoms.positions = np.random.permutation(universe.atoms.positions)
            W.write(universe.atoms)

def add_noise(universe: mda.Universe, output_path: str):
    """Add Gaussian noise to atomic positions based on residue-specific variance in place."""
    # Calculate residue-specific variance first
    residue_variances = {}
    for ts in universe.trajectory:
        for residue in universe.residues:
            if residue.resid not in residue_variances:
                residue_variances[residue.resid] = []
            residue_variances[residue.resid].append(residue.atoms.positions.copy())
    
    for resid, positions in residue_variances.items():
        positions = np.array(positions)
        residue_variances[resid] = np.var(positions, axis=0)
    
    # Reset trajectory and add noise frame by frame
    universe.trajectory[0]
    with mda.Writer(output_path, universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            for residue in universe.residues:
                noise = np.random.normal(0, np.sqrt(residue_variances[residue.resid]), residue.atoms.positions.shape)
                residue.atoms.positions += noise
            W.write(universe.atoms)

def cluster_and_save_frames(universe: mda.Universe, original_pca_mean: np.ndarray, n_clusters: int = 1000, output_path: str = None):
    """Cluster frames using PCA and K-means, then save selected frames ordered by MSD to original PCA mean."""
    # Extract coordinates
    coords = []
    for ts in universe.trajectory:
        coords.append(ts.positions.flatten())
    coords = np.array(coords)

    
    # PCA reduction
    pca = PCA(n_components=min(50, coords.shape[0], coords.shape[1]))
    reduced = pca.fit_transform(coords)
    
    # K-means clustering with fixed random state
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    
    # Select representative frames
    clustered_indices = []
    for cluster in range(n_clusters):
        indices = np.where(clusters == cluster)[0]
        if len(indices) > 0:
            clustered_indices.append(indices[0])
    
    # Calculate MSD to original average PCA coordinates
    msd = np.sum((reduced[clustered_indices] - original_pca_mean) ** 2, axis=1)
    ordered_indices = [clustered_indices[i] for i in np.argsort(msd)]
    
    # Save ordered frames
    with mda.Writer(output_path, universe.atoms.n_atoms) as W:
        for idx in ordered_indices:
            universe.trajectory[idx]
            W.write(universe.atoms)
    
    return ordered_indices

def process_poisoned_ensemble(protein: str, ensemble: Dict, poison_func, save_path: str, segs_data: Dict):
    """Process and save poisoned ensemble trajectories with PCA analysis using HDX residue distances."""
    print(f"Processing protein: {protein} with {poison_func.__name__}")
    universe = mda.Universe(ensemble["top"], ensemble["traj"])
    
    # Get HDX residues from segments data
    hdx_residues = segs_data[protein].residues
    
    # Calculate pairwise distances for a trajectory
    def calc_hdx_distances(traj_universe):
        distances = []
        ca_atoms = traj_universe.select_atoms(f"name CA and resid {' '.join(map(str, hdx_residues))}")
        for ts in traj_universe.trajectory:
            dist_matrix = pdist(ca_atoms.positions)
            distances.append(dist_matrix)
        return np.array(distances)
    
    # Get original trajectory distances and PCA
    original_distances = calc_hdx_distances(universe)
    pca = PCA(n_components=min(50, original_distances.shape[0], original_distances.shape[1]))
    original_reduced = pca.fit_transform(original_distances)
    original_pca_mean = np.mean(original_reduced, axis=0)
    
    # Create and save poisoned trajectory
    poison_name = poison_func.__name__.replace('_', '-')
    poisoned_file = os.path.join(save_path, f"{protein}_poisoned_{poison_name}.xtc")
    clustered_file = os.path.join(save_path, f"{protein}_clustered_{poison_name}.xtc")
    poison_func(universe, poisoned_file)
    
    # Process poisoned trajectory
    poisoned_universe = mda.Universe(ensemble["top"], poisoned_file)
    ordered_indices = cluster_and_save_frames(poisoned_universe, original_pca_mean, output_path=clustered_file)
    
    # Calculate distances for poisoned and clustered trajectories
    poisoned_distances = calc_hdx_distances(poisoned_universe)
    clustered_universe = mda.Universe(ensemble["top"], clustered_file)
    clustered_distances = calc_hdx_distances(clustered_universe)
    
    all_distances = np.vstack([original_distances, poisoned_distances, clustered_distances])
    pca_results = pca.fit_transform(all_distances)
    
    # Plot results
    plot_pca_results(pca_results, original_distances, poisoned_distances, protein, poison_name, save_path)
    
    return ordered_indices

def create_combined_ensemble(top_path: str,
                           orig_traj_path: str, 
                           poison_traj_path: str,
                           output_path: str,
                           poison_frames: List[int],
                           n_clusters: int = 1000):
    """
    Combines original ensemble (clustered) with specific poisoned frames.
    Args:
        top_path: Path to topology file
        orig_traj_path: Path to original trajectory
        poison_traj_path: Path to poisoned trajectory
        output_path: Path to save combined trajectory
        poison_frames: List of frame indices to select from poisoned trajectory
        n_clusters: Number of clusters for original trajectory
    """
    # Load original trajectory and calculate PCA mean
    orig_universe = mda.Universe(top_path, orig_traj_path)
    
    # Extract coordinates for PCA
    coords = []
    for ts in orig_universe.trajectory:
        coords.append(ts.positions.flatten())
    coords = np.array(coords)
    
    # Calculate PCA mean
    pca = PCA(n_components=min(50, coords.shape[0], coords.shape[1]))
    original_reduced = pca.fit_transform(coords)
    original_pca_mean = np.mean(original_reduced, axis=0)
    
    # Cluster the original trajectory
    clustered_path = output_path.replace('.xtc', '_clustered_temp.xtc')
    
    # Use existing clustering function with the PCA mean
    cluster_and_save_frames(orig_universe, 
                          original_pca_mean=original_pca_mean,
                          n_clusters=n_clusters, 
                          output_path=clustered_path)
    
    # Load poisoned trajectory and select frames
    poison_universe = mda.Universe(top_path, poison_traj_path)
    
    # Combine trajectories
    with mda.Writer(output_path, orig_universe.atoms.n_atoms) as W:
        # Write clustered original frames
        clustered_universe = mda.Universe(top_path, clustered_path)
        for ts in clustered_universe.trajectory:
            W.write(clustered_universe.atoms)
            
        # Write selected poisoned frames
        for frame in poison_frames:
            if frame is not None:
                poison_universe.trajectory[frame]
                W.write(poison_universe.atoms)
    
    # Cleanup temporary file
    os.remove(clustered_path)

def create_poisoned_ensembles(ensemble_experiments: Dict, 
                            save_path: str, 
                            segs_data: Dict,
                            protein_names: List[str]=["BPTI"]):
    """Create poisoned ensembles with explicit segs_data parameter."""
    print("Starting creation of poisoned ensembles")
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    for ensemble_name, proteins in tqdm(ensemble_experiments.items(), desc="Ensembles"):
        for protein, data in tqdm(proteins.items(), desc=f"Proteins in {ensemble_name}", leave=False):
            if protein not in protein_names or not isinstance(data, dict) or 'top' not in data or 'traj' not in data:
                continue
                
            protein_save_path = os.path.join(save_path, protein, ensemble_name)
            os.makedirs(protein_save_path, exist_ok=True)
            print(f"Creating poisoned ensembles for {protein} in {ensemble_name}")
            
            for poison_func in [shuffle_backbone_protons, mix_coordinates, add_noise]:
                try:
                    process_poisoned_ensemble(protein, data, poison_func,
                                           protein_save_path, segs_data)
                except Exception as e:
                    print(f"Error processing {protein} with {poison_func.__name__}: {str(e)}")

def process_all_combinations(ensemble_experiments: Dict,
                           poison_base_path: str,
                           output_base_path: str,
                           segs_data: Dict,
                           protein_names: List[str] = ["BPTI"],
                           poison_types: List[str] = ['shuffle-backbone-protons',
                                                    'mix-coordinates',
                                                    'add-noise'],
                           frame_sets: List[List[int]] = [[],
                                                        [0],
                                                        list(range(0, 10)),
                                                        list(range(0, 20)),
                                                        list(range(0, 50)),
                                                        list(range(0, 100)),
                                                        list(range(0, 500))]):
    """Process all combinations with explicit segs_data parameter."""
    os.makedirs(output_base_path, exist_ok=True)
    
    for ensemble_name, proteins in tqdm(ensemble_experiments.items(), desc="Processing ensembles"):
        for protein, data in tqdm(proteins.items(), desc=f"Processing {ensemble_name}", leave=False):
            if protein not in protein_names or not isinstance(data, dict) or 'top' not in data or 'traj' not in data:
                continue
                
            for poison_type in poison_types:
                for frame_set in frame_sets:
                    poison_path = os.path.join(poison_base_path, protein, ensemble_name,
                                             f"{protein}_clustered_{poison_type}.xtc")
                    output_dir = os.path.join(output_base_path, protein, ensemble_name)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    frame_suffix = f"poison{len(frame_set)}"
                    output_path = os.path.join(output_dir,
                                             f"{protein}_{poison_type}_{frame_suffix}.xtc")
                    
                    try:
                        create_combined_ensemble(
                            top_path=data['top'],
                            orig_traj_path=data['traj'],
                            poison_traj_path=poison_path,
                            output_path=output_path,
                            poison_frames=frame_set
                        )
                    except Exception as e:
                        print(f"Error processing {protein} {poison_type} {frame_suffix}: {str(e)}")

def plot_pca_results(pca_results, original_distances, poisoned_distances, protein, poison_name, save_path):
    """Helper function to plot PCA results."""
    n_orig = len(original_distances)
    n_poisoned = len(poisoned_distances)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[:n_orig, 0], pca_results[:n_orig, 1], label='Original', alpha=0.6)
    plt.scatter(pca_results[n_orig:n_orig+n_poisoned, 0], pca_results[n_orig:n_orig+n_poisoned, 1],
               label='Poisoned', alpha=0.6)
    plt.scatter(pca_results[n_orig+n_poisoned:, 0], pca_results[n_orig+n_poisoned:, 1],
               label='Clustered', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{protein} - {poison_name} PCA')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{protein}_{poison_name}_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    protein_names = ["BRD4"]
    experiments = ["T-FES", "AF2-Cleaned", "1Start-MD", "AF2-MSAss", "10Start-MD"]
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working directory: {os.getcwd()}")
    
    # First create the poisoned ensembles
    create_poisoned_ensembles(ensemble_experiments=ensemble_experiments,
                            save_path="poisoned_ensembles",
                            segs_data=segs_data,
                            protein_names=protein_names)
    
    # Then process the combinations using the created poisoned ensembles
    process_all_combinations(ensemble_experiments=ensemble_experiments,
                           protein_names=protein_names,
                           poison_base_path="poisoned_ensembles",
                           segs_data=segs_data,
                           output_base_path="combined_ensembles")