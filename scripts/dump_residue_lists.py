#!/usr/bin/env python3


"""
# Load individual protein data
import numpy as np
bpti_residues = np.load('hdx_residues/BPTI_hdx_residues.npy')

# Load all proteins at once
data = np.load('hdx_residues/all_hdx_residues.npz')
hoip_residues = data['HOIP']

# Load JSON with metadata
import json
with open('hdx_residues/all_hdx_residues.json', 'r') as f:
    hdx_data = json.load(f)
    hoip_residues = hdx_data['proteins']['HOIP']['residues']


with open('hdx_residues/BPTI_hdx_residues.json', 'r') as f:
    bpti_data = json.load(f)
    bpti_residues = bpti_data['residues']

with open('hdx_residues/BPTI_hdx_residues.txt', 'r') as f:
    bpti_residues = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]


    
"""





import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import sys
sys.path.append("/home/alexi/Documents/ValDX/")
from ValDX.helpful_funcs import segs_to_df

# Define protein data and paths
protein_names = ["BPTI", "BRD4", "HOIP", "LXR", "MBP"]

protein_data = {
    "BPTI": {"segs_path": "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_expt_data/BPTI_residue_segs_trimmed.txt"},
    "BRD4": {"segs_path": "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_segs_trimmed.txt"},
    "HOIP": {"segs_path": "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_APO_segs_trimmed.txt"},
    "LXR": {"segs_path": "/home/alexi/Documents/ValDX/raw_data/LXRalpha/LXRalpha_APO/LXRa_APO_segs200_trimmed.txt"},
    "MBP": {"segs_path": "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP_wt1_segs_trimmed.txt"}
}

class Segments:
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

    def get_resnums(self, df: pd.DataFrame, keys=None):
        if df is None:
            df = self.segs_df
        if keys is None:
            keys = self.keys
        res_nums = df.apply(lambda x: np.arange(x[keys[0]]+1, x[keys[1]]+1), axis=1).to_numpy()
        return res_nums
    
    def get_residues(self, df: pd.DataFrame=None):
        if df is None:
            df = self.segs_df
        residues = np.unique(np.concatenate(self.get_resnums(df)))
        return residues
    
    def get_peptides(self, df: pd.DataFrame=None):
        if df is None:
            df = self.segs_df
        peptides = {pep: res for pep, res in zip(df["peptide"], self.get_resnums(df))}
        return peptides
    
    def get_pep_nums(self, df: pd.DataFrame=None):
        if df is None:
            df = self.segs_df
        return df["peptide"].to_numpy()

def extract_residues_from_segments(segs_data: Dict, 
                                 output_dir: str = "hdx_residues",
                                 save_json: bool = True,
                                 save_numpy: bool = True) -> Dict[str, List[int]]:
    """
    Extract HDX-monitored residues from Segments class instances and save to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    hdx_residues = {}
    
    for protein_name, segments in segs_data.items():
        residues = sorted(segments.residues.tolist())
        hdx_residues[protein_name] = residues
        
        base_filename = os.path.join(output_dir, f"{protein_name}_hdx_residues")
        
        if save_json:
            json_path = f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "protein": protein_name,
                    "residues": residues,
                    "num_residues": len(residues)
                }, f, indent=2)
            print(f"Saved JSON file: {json_path}")
        
        if save_numpy:
            npy_path = f"{base_filename}.npy"
            np.save(npy_path, np.array(residues))
            print(f"Saved NumPy file: {npy_path}")
            
        txt_path = f"{base_filename}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"# HDX-monitored residues for {protein_name}\n")
            f.write(f"# Total residues: {len(residues)}\n")
            f.write("# Format: one residue number per line\n")
            for res in residues:
                f.write(f"{res}\n")
        print(f"Saved text file: {txt_path}")
    
    if hdx_residues:
        combined_path = os.path.join(output_dir, "all_hdx_residues")
        
        if save_json:
            combined_data = {
                "proteins": {},
                "summary": {
                    "total_proteins": len(hdx_residues),
                    "proteins_analyzed": list(hdx_residues.keys())
                }
            }
            
            for protein, residues in hdx_residues.items():
                combined_data["proteins"][protein] = {
                    "residues": residues,
                    "num_residues": len(residues)
                }
            
            with open(f"{combined_path}.json", 'w') as f:
                json.dump(combined_data, f, indent=2)
            print(f"Saved combined JSON file: {combined_path}.json")
            
        if save_numpy:
            np.savez(f"{combined_path}.npz", **hdx_residues)
            print(f"Saved combined NumPy file: {combined_path}.npz")
    
    return hdx_residues

if __name__ == "__main__":
    # Create Segments instances for each protein
    segs_data = {}
    for protein in protein_names:
        print(f"\nProcessing {protein}...")
        segs_data[protein] = Segments(segs_path=protein_data[protein]["segs_path"])
    
    # Extract and save residue lists
    hdx_residues = extract_residues_from_segments(segs_data, 
                                                output_dir="hdx_residues",
                                                save_json=True,
                                                save_numpy=True)
    
    # Print summary
    print("\nSummary of HDX-monitored residues:")
    for protein, residues in hdx_residues.items():
        print(f"{protein}: {len(residues)} residues")
        print(f"First few residues: {residues[:5]}...")