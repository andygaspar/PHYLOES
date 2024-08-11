import os
import re
import shutil
import subprocess

from Likelihood.tree import PhyloTree
from Utils.suppress_prints import suppress_stdout_stderr


def get_likelihood(tree: PhyloTree, verbose=False, keep_output_files=False):

    if not os.path.exists(tree.tree_file):
        raise FileNotFoundError(f"The Newick file {tree.tree_file} does not exist.")

    if not os.path.exists(tree.alignment_file):
        raise FileNotFoundError(f"The alignment file {tree.alignment_file} does not exist.")


    phyml_executable = "Likelihood/phyml-master/src/phyml"
    phyml_command = [
        phyml_executable,
        "-i", tree.alignment_file,  # Specify the alignment file
        "-u", tree.tree_file,  # Specify the tree file
        "-m", "GTR",  # Specify the substitution model (adjust if necessary)
        "-s", "0",
        "-o", "n",
        "--quiet"
    ]

    if not verbose:
        with suppress_stdout_stderr():
            subprocess.run(phyml_command, check=True)
    else:
        subprocess.run(phyml_command, check=True)

    info_file = "sequence.phy_phyml_stats.txt"
    log_likelihood = None
    with open(info_file, 'r') as f:
        for line in f:
            if ". Log-likelihood: 			" in line:
                log_likelihood = line

    log_likelihood = float(log_likelihood.strip().replace("\t", " ").split(" ")[-1])
    output_dir = "Likelihood/phyml/results/"

    output_files = [
        f"{tree.alignment_file}_phyml_tree.txt",
        f"{tree.alignment_file}_phyml_stats.txt",
    ]

    if keep_output_files:
        for filename in output_files:
            if os.path.exists(filename):
                shutil.move(filename, output_dir)
    else:
        for filename in output_files:
            subprocess.run("rm " + filename, shell=True)

    return log_likelihood

