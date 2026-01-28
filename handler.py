"""Example handler file."""

import time
import shutil

import runpod
import subprocess
import time
from ablang2 import pretrained
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import re

motifs = ""
motifs += "N[GSA]|"
motifs += "D[GS]|"
motifs += "N[^P][ST]"
cdrmotifs = "D[DGHSTP]"
motifs = re.compile(motifs)
cdrmotifs = re.compile(cdrmotifs)


def check_sequence(seq):
    motif_present = motifs.search(seq)
    if motif_present:
        return False
    motif_present = cdrmotifs.search(seq)
    if motif_present:
        return False
    return True


def get_seq(file_location):
    # Parse the PDB file
    parser = PDBParser()
    print(file_location)
    structure = parser.get_structure("protein", file_location)

    # Get the heavy chain
    model = structure[0]
    heavy_chain = model["H"]  # Change 'H' to your chain ID

    # Extract sequence
    sequence = ""
    for residue in heavy_chain:
        if residue.id[0] == " ":  # Only standard amino acids
            sequence += seq1(residue.get_resname())

    return sequence


def write_csv_from_folder(parent_dir, outfile):
    out = []
    # Get all immediate child directories
    child_dirs = [
        d
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    # For each child directory
    for child_dir in child_dirs:
        child_path = os.path.join(parent_dir, child_dir)
        print(f"\n=== Directory: {child_path} ===")

        # List all files in this child directory
        for root, dirs, files in os.walk(child_path):
            for file in files:
                if ".pdb" not in file:
                    continue
                file_path = os.path.join(root, file)
                out.append(
                    {
                        "dir": child_dir,
                        "name": child_dir
                        + "_"
                        + "".join(file.split("_")[:-1]),
                        "seq": get_seq(file_path),
                    }
                )
    pd.DataFrame(out).to_csv(outfile)
    return pd.DataFrame(out)


wt_lc = "QIVLTQSPATLSLSPGERATMSCTASSSVSSSYLHWYQQKPGKAPKLWIYSTSNLASGVPSRFSGSGSGTDYTLTISSLQPEDFATYYCHQYYRLPPITFGQGTKLEIK"
wt_hc_prefix = "EVQLVESGGGLVKPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVATISSGGSHTYYLDSVKGRFTISRDNSKNTLYLQMNSLRAEDTALYYCA"
wt_hc_post = "WGQGTLVTVSS"


def has_required_pattern(sequence):
    """Check if sequence has H next to H, R, or K"""
    required_pairs = ["HH", "HR", "RH", "HK", "KH"]
    return any(pair in sequence for pair in required_pairs)


def find_positions_next_to_HRK(sequence):
    """Find all positions adjacent to R or K that could be mutated to H"""
    indices = []
    for index, character in enumerate(sequence):
        if character in "HKR":
            indices.append(index)
    h_pos = []
    for i in indices:
        if i == 0:
            h_pos.append(1)
        elif i == 9:
            h_pos.append(8)
        else:
            for j in [i + 1, i - 1]:
                if j not in h_pos:
                    h_pos.append(j)
    return h_pos


def fix_seq(hc):
    hcdr3 = hc[97:107]
    if has_required_pattern(hcdr3):
        return wt_hc_prefix + hcdr3 + wt_hc_post, float("inf")
    if not any(i in hcdr3 for i in "HRK"):
        return None
    else:
        targets = find_positions_next_to_HRK(hcdr3)
        modified_hcdr3s = [hcdr3[:i] + "H" + hcdr3[i + 1 :] for i in targets]
        modified_hcdr3s = [
            hcdr3
            for hcdr3 in modified_hcdr3s
            if check_sequence("A" + hcdr3 + "W")
        ]
        if len(modified_hcdr3s) == 0:
            return None
        modified_hcs = [wt_hc_prefix + i + wt_hc_post for i in modified_hcdr3s]
        model = pretrained()
        results = [
            model([(modified_hc, wt_lc)], mode="confidence")[0]
            for modified_hc in modified_hcs
        ]
        max_con = max(results)
        return modified_hcs[results.index(max_con)], max_con


def adjust_seqs(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    df["conseq"] = df.seq.apply(fix_seq)
    df0 = df[df.conseq.apply(lambda r: r is not None)].copy()
    df0["end_seq"] = df0.conseq.apply(lambda r: r[0])
    df0["conf"] = df0.conseq.apply(lambda r: r[1])
    names = np.unique(df0.name)
    outdf = []
    for cname in names:
        cdf = df0[df0.name == cname]
        cdf = cdf.sort_values(by="conf", ascending=False)
        d = cdf.iloc[0].to_dict()
        outdf.append(
            {
                "dir": d["dir"],
                "name": d["name"],
                "hseq": d["end_seq"],
                "ablang_conf": d["conf"],
            }
        )
    pd.DataFrame(outdf).to_csv(out_csv)


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    name = job_input.get("name", "World")
    if "John" in name:
        return f"Hello, {name}!"

    subprocess.call("cp -a /runpod-volume/transfer/. /home", shell=True)
    os.chdir("/home")
    subprocess.call("python3.10 -m pip install poetry", shell=True)
    subprocess.call("bash /home/include/setup.sh", shell=True)
    subprocess.call(
        "export PYTHONPATH=$PYTHONPATH:/home/src/rfantibody/rfdiffusion",
        shell=True,
    )
    print("not entering sleep")

    for i in range(2):
        name = str(time.time()).split(".")[0][2:]
        cmd1 = f"""OMP_NUM_THREADS=4 MKL_NUM_THREADS=4  poetry run python  /home/scripts/rfdiffusion_inference.py \
        --config-name antibody \
        antibody.target_pdb=/home/f_8mn_T.pdb \
        antibody.framework_pdb=/home/su_try_HLT.pdb \
        inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
        'ppi.hotspot_res=[]' \
        'antibody.design_loops=[H3:9]' \
        inference.num_designs=25 \
        inference.output_prefix=/home/c1s_ep2_{name}/c1s_ep2_antibody"""
        subprocess.call(cmd1, shell=True)
        cmd2 = f"""poetry run python /home/scripts/proteinmpnn_interface_design.py \
        -seqs_per_struct 5 \
        -pdbdir /home/c1s_ep2_{name} \
        -outpdbdir /home/protien_out_ep2_{name}/c1s_ep2_multi"""
        subprocess.call(cmd2, shell=True)

        write_csv_from_folder(
            f"/home/protien_out_ep2_{name}",
            f"/runpod-volume/original_anti_ep2_{name}.csv",
        )
        """
        adjust_seqs(
            f"/runpod-volume/original_anti_ep2_{name}.csv",
            f"/runpod-volume/modified_anti_ep2_{name}.csv",
        )
        """
        shutil.rmtree(f"/home/protien_out_ep2_{name}")
        shutil.rmtree(f"/home/c1s_ep2_{name}")

    return f"Hello, {name}!"


runpod.serverless.start({"handler": handler})
