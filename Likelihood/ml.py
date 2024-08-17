import os
import re
import subprocess

from Bio import Phylo

from Likelihood.likelihood import get_likelihood
from Likelihood.tree import PhyloTree
from Utils.suppress_prints import suppress_stdout_stderr


def run_raxml(tree: PhyloTree, model, verbose= False,  keep_output_files=False):

    if os.path.exists("Likelihood/raxml/results/RAxML_info.output"):
        subprocess.run("rm Likelihood/raxml/results/*", shell=True)

    raxml_command = [
        "./Likelihood/standard-RAxML-master/raxmlHPC-PTHREADS",  # Ensure this is the correct RAxML command for your setup
        # "-f", "h ",  # Calculate likelihood of a given user tree
        "-m", "GTRGAMMA",  # Substitution model
        "-s", "/home/andrea/Scrivania/PHYLOES/" + tree.alignment_file,  # Path to the temporary alignment file
        "-n", "output",  # Output prefix
        "-w", "/home/andrea/Scrivania/PHYLOES/Likelihood/raxml/results",  # Use the temporary directory for RAxML output
        "-p 1234", "--JC69"
    ]
    raxml_command = " ".join(raxml_command)

    # Run RAxML and capture output
    if not verbose:
        with suppress_stdout_stderr():
            subprocess.run(raxml_command, shell=True, check=True)
    else:
        subprocess.run(raxml_command, shell=True, check=True)

    info_file = "Likelihood/raxml/results/RAxML_info.output"
    log_likelihood = None
    with open(info_file, 'r') as f:
        for line in f:
            if "Final GAMMA-based Score of best tree" in line:
                log_likelihood = line
    log_likelihood = float(log_likelihood.strip().split(" ")[-1])
    # if not keep_output_files:
    #     subprocess.run("rm Likelihood/raxml/results/*", shell=True)

    new_tree = Phylo.read("Likelihood/raxml/results/RAxML_bestTree.output", "newick")
    new_phylo_tree = PhyloTree(tree.n_taxa, tree.alignment_file, tree.labels)
    # new_phylo_tree.set_file("Likelihood/raxml/results/RAxML_bestTree.output")
    new_phylo_tree.set_phylo_tree(new_tree)
    log_likelihood = get_likelihood(new_phylo_tree, model)
    return log_likelihood, new_tree


def run_phyml(tree: PhyloTree, model, verbose=False, keep_output_files=False):

    if not os.path.exists(tree.tree_file):
        raise FileNotFoundError(f"The Newick file {tree.tree_file} does not exist.")

    if not os.path.exists(tree.alignment_file):
        raise FileNotFoundError(f"The alignment file {tree.alignment_file} does not exist.")


    phyml_executable = "Likelihood/phyml-master/src/phyml"
    phyml_command = [
        phyml_executable,
        "-i", tree.alignment_file,  # Specify the alignment file
        "-m", model,  # Specify the substitution model (adjust if necessary)
    ]

    if not verbose:
        with suppress_stdout_stderr():
            subprocess.run(phyml_command, check=True)
    else:
        subprocess.run(phyml_command, check=True)

    new_tree = Phylo.read("sequence.phy_phyml_tree.txt", "newick")
    new_phylo_tree = PhyloTree(tree.n_taxa, tree.alignment_file, tree.labels)
    # new_phylo_tree.set_file("Likelihood/raxml/results/RAxML_bestTree.output")
    new_phylo_tree.set_phylo_tree(new_tree)
    log_likelihood = get_likelihood(new_phylo_tree, model)
    return log_likelihood, new_tree


def run_iq3(tree, verbose=False):
    if verbose:
        os.system("iqtree2 -s " + tree.alignment_file + " -m JC69 --redo --redo-tree")
    else:
        with suppress_stdout_stderr():
            os.system("iqtree2 -s " + tree.alignment_file + " -m JC69 --redo --redo-tree")
    info_file = "sequence.phy.iqtree"
    log_likelihood = None
    with open(info_file, 'r') as f:
        for line in f:
            if "Log-likelihood of the tree: " in line:
                log_likelihood = line
    pattern = r":(.*?)(?=\()"

    # Find all matches
    log_likelihood = float(re.findall(pattern, log_likelihood)[0])
    new_tree = Phylo.read("sequence.phy.treefile", "newick")
    new_phylo_tree = PhyloTree(tree.n_taxa, tree.alignment_file, tree.labels)
    # new_phylo_tree.set_file("Likelihood/raxml/results/RAxML_bestTree.output")
    new_phylo_tree.set_phylo_tree(new_tree)

    return log_likelihood, new_phylo_tree
