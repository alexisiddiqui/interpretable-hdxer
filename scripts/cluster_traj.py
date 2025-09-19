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
from sklearn.cluster import KMeans

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

segs_data = {protein:Segments(segs_path=Cleaned_AF2_proteins[protein]["segs_path"]) for protein in Cleaned_AF2_proteins}


res_data = {protein:segs_data[protein].residues for protein in Cleaned_AF2_proteins}
print(res_data)




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

    # Perform PCA
    pca = PCA(n_components=20)
    pca_results = pca.fit_transform(all_pairwise_distances)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Experiment': all_labels,
        'Index': all_indexes
    })

    return df, pca_results






# if __name__ == "__main__":
#     # using the trajectory information from the cleaned AF2 data - kmeans cluster the ensembles on the PCA space with n_clusters for each protein
#     # write the cluster centres of the trajectories to both an xtc as well as a multiframe pdb file - use clear names for the files - these could be saved in the same directory as the trajectory files
#     # save the plots of the PCA sapce with the clusters labelled with cluster number as well as the frame index for each protein sperately

#     n_clusters = 10
def cluster_and_save_trajectories(protein_names, Cleaned_AF2_proteins, segs_data, n_clusters=10):
    """
    Cluster trajectories in PCA space and save representative structures.
    Aligns structures using CA atoms of selected residues before saving.
    """
    for protein in protein_names:
        print(f"\nProcessing {protein}...")
        
        # Load trajectory
        u = mda.Universe(Cleaned_AF2_proteins[protein]["top"],
                        Cleaned_AF2_proteins[protein]["traj"])
        
        # Create a reference from the first frame
        reference = mda.Universe(Cleaned_AF2_proteins[protein]["top"])
        
        # Select CA atoms for alignment from the segments
        residues = segs_data[protein].residues
        alignment_sel = f"name CA and resid {' '.join(map(str, residues))}"
        mobile = u.select_atoms(alignment_sel)
        ref = reference.select_atoms(alignment_sel)
        
        # Calculate PCA
        pca_df, pca_results = calc_PCA(
            universes=[u],
            experiment_names=[protein],
            indexes=[Cleaned_AF2_proteins[protein]["index"]],
            segs=segs_data[protein]
        )
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(pca_results)
        pca_df['Cluster'] = clusters
        
        # Find frames closest to cluster centers
        center_frames = []
        for i in range(n_clusters):
            cluster_points = pca_results[clusters == i]
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            center_frame_idx = np.where(clusters == i)[0][np.argmin(distances)]
            center_frames.append(int(center_frame_idx))
        
        # Save cluster centers as trajectories
        output_dir = os.path.dirname(Cleaned_AF2_proteins[protein]["traj"])
        base_name = f"{protein}_cluster_centers_{n_clusters}"
        
        # Set up alignment
        aligner = align.AlignTraj(u, reference,
                                 select=alignment_sel,
                                 in_memory=True)
        print(f"Aligning frames to reference structure...")
        aligner.run()
        
        # Save as XTC
        print(f"Saving aligned cluster centers to XTC...")
        with mda.Writer(os.path.join(output_dir, f"{base_name}.xtc"), u.atoms.n_atoms) as W:
            for frame_idx in center_frames:
                u.trajectory[frame_idx]
                W.write(u.atoms)
        
        # Save as PDB
        print(f"Saving aligned cluster centers to PDB...")
        with mda.Writer(os.path.join(output_dir, f"{base_name}.pdb"), multiframe=True) as W:
            for frame_idx in center_frames:
                u.trajectory[frame_idx]
                W.write(u.atoms)
        
        # Create PCA plot with clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                            c=pca_df['Cluster'], cmap='tab10',
                            alpha=0.6)
        
        # Mark cluster centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black',
                   marker='x', s=200, linewidths=3,
                   label='Cluster Centers')
        
        plt.title(f'{protein} - PCA Space with {n_clusters} Clusters')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{protein}_pca_clusters.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save frame indices for reference
        cluster_info = {
            'cluster_centers': centers.tolist(),
            'center_frames': [int(frame) for frame in center_frames],
            'frame_clusters': [int(cluster) for cluster in clusters]
        }
        
        with open(os.path.join(output_dir, f"{protein}_cluster_info.json"), 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        print(f"Saved aligned cluster data for {protein}")

if __name__ == "__main__":
    # Set number of clusters
    n_clusters = 10
    
    # Run clustering for all proteins
    cluster_and_save_trajectories(protein_names, Cleaned_AF2_proteins, 
                                segs_data, n_clusters)
    
    print("\nClustering complete for all proteins!")