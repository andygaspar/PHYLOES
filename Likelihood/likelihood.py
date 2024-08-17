import os
import re
import shutil
import subprocess
import time

from Likelihood.tree import PhyloTree
from Utils.suppress_prints import suppress_stdout_stderr


def get_likelihood(tree: PhyloTree, model, verbose=False, keep_output_files=False, no_opt=True):

    if not os.path.exists(tree.tree_file):
        raise FileNotFoundError(f"The Newick file {tree.tree_file} does not exist.")

    if not os.path.exists(tree.alignment_file):
        raise FileNotFoundError(f"The alignment file {tree.alignment_file} does not exist.")



    if not verbose:
        with suppress_stdout_stderr():
            os.system("iqtree2 -s " + tree.alignment_file +" -te " + tree.tree_file + " -m JC69  -n 0 --redo --redo-tree")
    else:
        os.system("iqtree2 -s " + tree.alignment_file +" -te " + tree.tree_file + " -m JC69  -n 0 --redo --redo-tree")

    info_file = "sequence.phy.iqtree"
    log_likelihood = None
    with open(info_file, 'r') as f:
        for line in f:
            if "Log-likelihood of the tree: " in line:
                log_likelihood = line
    pattern = r":(.*?)(?=\()"

    # Find all matches
    log_likelihood = float(re.findall(pattern, log_likelihood)[0])

    return log_likelihood


def get_likelihood_phylm(tree: PhyloTree, model, verbose=False, keep_output_files=False, no_opt=True):
    if not os.path.exists(tree.tree_file):
        raise FileNotFoundError(f"The Newick file {tree.tree_file} does not exist.")

    if not os.path.exists(tree.alignment_file):
        raise FileNotFoundError(f"The alignment file {tree.alignment_file} does not exist.")
    phyml_executable = "Likelihood/phyml-master/src/phyml"
    phyml_command = [
        phyml_executable,
        "-i", tree.alignment_file,  # Specify the alignment file
        "-u", tree.tree_file,  # Specify the tree file
        "-m", model]

    if no_opt:
        phyml_command += ["-o n", "-s 0", "-b 0", "-v e"]
    if not verbose:
        with suppress_stdout_stderr():
            os.system(" ".join(phyml_command))
    else:
        os.system(" ".join(phyml_command))

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
    # print(t - time.time())
    return log_likelihood
