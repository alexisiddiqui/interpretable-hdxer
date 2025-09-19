# /home/alexi/Documents/interpretable-hdxer/notebooks/SI-Fig-1_AF_MD_PCA_Space/SI-Fig-1_Transfer.py
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
sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.helpful_funcs import segs_to_df


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
    "Regular-MD": regular_MD_proteins,
    "Better-MD": better_MD_proteins,
    "Shaw-MD": shaw_MD_proteins
}

# Update experiments list
experiments = ["AF2-MSAss", "AF2-Cleaned", "1Start-MD", "10Start-MD", "Regular-MD", "Better-MD", "Shaw-MD"]
import MDAnalysis as mda
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from typing import List, Dict, Generator
import gc

class MemoryEfficientPCA:
    def __init__(self, n_components=20):
        self.pca = PCA(n_components=n_components)
        
    def process_trajectory_chunk(self, universe: mda.Universe, 
                               chunk_size: int, 
                               residues: np.ndarray,
                               experiment_name: str,
                               index: List[float]) -> Generator:
        """Process trajectory in chunks to reduce memory usage"""
        atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, residues))}")
        
        for chunk_start in range(0, len(universe.trajectory), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(universe.trajectory))
            chunk_distances = []
            chunk_labels = []
            chunk_indexes = []
            
            for ts in universe.trajectory[chunk_start:chunk_end]:
                distances = pdist(atom_group.positions)
                chunk_distances.append(distances)
                chunk_labels.append(experiment_name)
                chunk_indexes.append(index[ts.frame])
            
            yield np.array(chunk_distances), chunk_labels, chunk_indexes
            
            # Force garbage collection
            del chunk_distances, chunk_labels, chunk_indexes
            gc.collect()

    def fit_transform_chunks(self, universes: List[mda.Universe],
                           experiment_names: List[str],
                           indexes: List[List[float]],
                           residues: np.ndarray,
                           chunk_size: int = 1000) -> pd.DataFrame:
        """Fit and transform PCA in chunks"""
        all_distances = []
        all_labels = []
        all_indexes = []
        
        # Process each universe in chunks
        for universe, exp_name, index in zip(universes, experiment_names, indexes):
            print(f"Processing {exp_name} with {len(universe.trajectory)} frames")
            
            for distances, labels, chunk_indexes in self.process_trajectory_chunk(
                universe, chunk_size, residues, exp_name, index):
                all_distances.extend(distances)
                all_labels.extend(labels)
                all_indexes.extend(chunk_indexes)
                
            # Clear memory after each universe
            gc.collect()
        
        # Convert to numpy array and perform PCA
        all_distances = np.array(all_distances)
        pca_results = self.pca.fit_transform(all_distances)
        
        # Create DataFrame and clean up
        df = pd.DataFrame({
            'PC1': pca_results[:, 0],
            'PC2': pca_results[:, 1],
            'Experiment': all_labels,
            'Index': all_indexes
        })
        
        del all_distances, pca_results
        gc.collect()
        
        return df

def optimize_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize the dataframe for plotting by reducing memory usage"""
    # Convert string columns to categorical
    df['Experiment'] = df['Experiment'].astype('category')
    if 'Protein' in df.columns:
        df['Protein'] = df['Protein'].astype('category')
    
    # Downsample if too many points
    max_points = 10000
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    
    return df

def memory_efficient_plot_PCA(ensemble_experiments: Dict, 
                            protein_names: List[str],
                            segs: Dict[str, 'Segments'],
                            experiments: List[str],
                            chunk_size: int = 1000) -> None:
    """Memory-efficient version of PCA calculation and plotting"""
    pca_processor = MemoryEfficientPCA(n_components=20)
    
    for protein in protein_names:
        print(f"\nProcessing {protein}")
        universes = []
        experiment_names = []
        indexes = []
        
        # Load data for current protein
        for experiment in experiments:
            exp_data = ensemble_experiments[experiment][protein]
            universes.append(mda.Universe(exp_data['top'], exp_data['traj']))
            experiment_names.append(experiment)
            indexes.append(exp_data['index'])
        
        # Process PCA in chunks
        protein_segs = segs[protein]
        pca_data = pca_processor.fit_transform_chunks(
            universes, experiment_names, indexes,
            protein_segs.residues, chunk_size
        )
        pca_data['Protein'] = protein
        
        # Optimize for plotting
        plot_data = optimize_plotting(pca_data)
        
        # Generate plots
        plot_PCA_index([protein], experiments, plot_data)
        plot_PCA_density([protein], experiments, plot_data)
        
        # Clear memory
        del pca_data, plot_data
        gc.collect()

def calc_PCA(universes: List[mda.Universe],
                      experiment_names: List[str],
                      indexes: List[List[float]],
                      segs: 'Segments' = None) -> pd.DataFrame:
    """
    Calculates the PCA on the pairwise distances of CA coordinates - only on the residues specified in the segments

    Uses 20 dimensions but saves the frame number and the first 2 PCA dimensions labelled by each experiment to a dataframe for later plotting
    """
    if segs is None:
        raise ValueError("Segments object must be provided")

    all_pairwise_distances = []
    all_labels = []
    all_indexes = []

    for universe, exp_name, index in zip(universes, experiment_names, indexes):
        # Select CA atoms for specified residues
        atom_group = universe.select_atoms(f"name CA and resid {' '.join(map(str, segs.residues))}")
        
        for ts in universe.trajectory:
            # Calculate pairwise distances
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
    
    # Create DataFrame
    df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Experiment': all_labels,
        'Index': all_indexes
    })

    return df

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
from matplotlib.colors import LinearSegmentedColormap


sns.set_style("whitegrid")

import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_PCA_index(protein_names: List[str], experiments: List[str], data: pd.DataFrame, experiment_window=["10Start-MD"]) -> None:
    """
    Plot the PCA results colour by index, split by experiment (column) and protein (row)
    Each column shares x and y axes, centered on the specified experiment_window
    Colorbar range is fixed from 0 to 100 for all plots
    """
    fig, axes = plt.subplots(len(experiments), len(protein_names), figsize=(6*len(protein_names), 5*len(experiments)), squeeze=False)
    
    if experiment_window not in experiments:
        experiment_window = ["10Start-MD"]

    # Custom colormap
    colors = ['magenta', 'blue', 'cyan', 'lime', 'black']
    n_bins = 100
    AF2_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # pick 10 colours for the 10 rep MD
    colour10_MD_cmap = LinearSegmentedColormap.from_list('custom', ['purple', 'magenta', 'blue', 'cyan', 'green', 'greenyellow', 'yellow', 'orange', 'red', 'black'], N=10)
    
    # pick 5 colours for the 5 rep MD
    colour5_MD_cmap = LinearSegmentedColormap.from_list('custom', ['magenta', 'blue', 'cyan', 'greenyellow', 'gold'], N=5)

    for j, protein in enumerate(protein_names):
        protein_data = data[data['Protein'] == protein]
        
        # Calculate the limits for shared axes based on the experiment_window
        window_data = protein_data[protein_data['Experiment'].isin(experiment_window)]
        x_min, x_max = window_data['PC1'].min(), window_data['PC1'].max()
        y_min, y_max = window_data['PC2'].min(), window_data['PC2'].max()
        
        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        for i, experiment in enumerate(experiments):
            if "AF2" in experiment:
                cmap = 'viridis'
                index = 'pLDDT'
            else:
                cmap = 'magma'
                index = 'Trajectory %'
            if "10Start" in experiment:
                cmap = colour10_MD_cmap
            elif "1Start" in experiment:
                cmap = colour5_MD_cmap

            ax = axes[i, j]
            subset = protein_data[protein_data['Experiment'] == experiment]
            
            # Use a fixed normalization from 0 to 100 for all plots
            norm = Normalize(vmin=0, vmax=100)
    

    

            if "AF2"  in experiment:
                scatter = ax.scatter(subset['PC1'], subset['PC2'], c=subset['Index'], cmap=cmap, alpha=0.7, s=10, edgecolors='none')
            else:
                scatter = ax.scatter(subset['PC1'], subset['PC2'], c=subset['Index'], cmap=cmap, alpha=0.7, s=10, edgecolors='none', norm=norm)
            ax.set_title(f"{protein}\n{experiment}", fontsize=12, fontweight='bold')
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set the same limits for all plots in this column
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add colorbar with fixed range
            cbar = plt.colorbar(scatter, ax=ax, label=index, aspect=40, norm=norm, alpha=1)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(index, fontsize=10)
        
        # Share x and y axes for each column
        for ax in axes[1:, j]:
            ax.sharex(axes[0, j])
            ax.sharey(axes[0, j])

    plt.tight_layout()
    plt.savefig(f"{'_'.join(protein_names)}-{'_'.join(experiments)}_PCA_index_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_PCA_density(protein_names: List[str], experiments: List[str], data: pd.DataFrame, experiment_window=["10Start-MD"]) -> None:
    """
    Plot the PCA results as contour density plots, with each experiment on its own set of axes.
    Creates a grid of subplots: each column is a protein, and each row is an experiment.
    The view is centered on the specified experiment_window.
    A colorbar is added to represent the density levels.
    """
    fig, axes = plt.subplots(len(experiments), len(protein_names),
                             figsize=(6*len(protein_names), 5*len(experiments)),
                             squeeze=False)
    if experiment_window not in experiments:
        experiment_window = ["10Start-MD"]

    # Custom colormap
    colors = ['purple', 'blue', 'white', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    for j, protein in enumerate(protein_names):
        protein_data = data[data['Protein'] == protein]

        # Calculate the limits for shared axes based on the experiment_window
        window_data = protein_data[protein_data['Experiment'].isin(experiment_window)]
        x_min, x_max = window_data['PC1'].min(), window_data['PC1'].max()
        y_min, y_max = window_data['PC2'].min(), window_data['PC2'].max()

        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        for i, experiment in enumerate(experiments):
            ax = axes[i, j]
            subset = protein_data[protein_data['Experiment'] == experiment]

            sns.kdeplot(
                data=subset,
                x='PC1',
                y='PC2',
                ax=ax,
                cmap=cmap,
                fill=True,
                alpha=0.7,
                levels=15,
                thresh=0.05,
            )

            ax.set_title(f"{protein}\n{experiment}", fontsize=12, fontweight='bold')
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Set the same limits for all plots in this column
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Share x and y axes for each column
        for ax in axes[1:, j]:
            ax.sharex(axes[0, j])
            ax.sharey(axes[0, j])

    # Add colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Density', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    filename = f"{'_'.join(protein_names)}-{'_'.join(experiments)}_PCA_density_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.close()

# Example usage:
# protein_names = ["ProteinA", "ProteinB"]
# experiments = ["Exp1", "Exp2", "Exp3"]
# data = pd.DataFrame(...)  # Your data here
# plot_PCA_index(protein_names, experiments, data)
# plot_PCA_density(protein_names, experiments, data)
def plot_PCA_by_protein(ensemble_experiments: Dict, protein_names: List[str], segs: Dict[str, Segments], experiments: List[str]) -> None:
    """
    Compute PCA space over experiments and plot results
    """
    all_data = []

    for protein in protein_names:
        universes = []
        experiment_names = []
        indexes = []
        
        for experiment in experiments:
            exp_data = ensemble_experiments[experiment][protein]
            universes.append(mda.Universe(exp_data['top'], exp_data['traj']))
            print(f"Loaded {experiment} data for {protein}")
            print(f"Number of frames: {len(universes[-1].trajectory)}")
            experiment_names.append(experiment)
            indexes.append(exp_data['index'])
        
        protein_segs = segs[protein]
        pca_data = calc_PCA(universes, experiment_names, indexes, protein_segs)
        pca_data['Protein'] = protein
        all_data.append(pca_data)

    combined_data = pd.concat(all_data, ignore_index=True)

    plot_PCA_index(protein_names, experiments, combined_data)
    plot_PCA_density(protein_names, experiments, combined_data)

# Example usage:
if __name__ == "__main__":

    os.chdir("/home/alexi/Documents/interpretable-hdxer/notebooks/SI-Fig-1_AF_MD_PCA_Space")
    print(f"Current working directory: {os.getcwd()}")

    # plot_PCA_by_protein(ensemble_experiments=ensemble_experiments,
    #                     protein_names=["BPTI"],
    #                     segs=segs_data,
    #                     experiments=experiments)

    for protein in protein_names[:1]:
        print(f"Plotting PCA for {protein}")
        # plot_PCA_by_protein(ensemble_experiments=ensemble_experiments,
        #                     protein_names=[protein],
        #                     segs=segs_data,
        #                     experiments=experiments)

        memory_efficient_plot_PCA(ensemble_experiments=ensemble_experiments,
                                    protein_names=[protein],
                                    segs=segs_data,
                                    experiments=experiments,
                                    chunk_size=100)