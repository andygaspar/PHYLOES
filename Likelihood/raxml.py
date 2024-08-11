import os
import subprocess

from Likelihood.tree import PhyloTree
from Utils.suppress_prints import suppress_stdout_stderr


def run_raxml(tree: PhyloTree, verbose= False,  keep_output_files=False):

    if os.path.exists("Likelihood/raxml/results/RAxML_info.output"):
        subprocess.run("rm Likelihood/raxml/results/*", shell=True)

    raxml_command = [
        "./Likelihood/standard-RAxML-master/raxmlHPC-PTHREADS",  # Ensure this is the correct RAxML command for your setup
        # "-f", "h ",  # Calculate likelihood of a given user tree
        "-m", "GTRGAMMA",  # Substitution model
        "-s", "/home/andrea/Scrivania/PHYLOES/" + tree.alignment_file,  # Path to the temporary alignment file
        "-n", "output",  # Output prefix
        "-w", "/home/andrea/Scrivania/PHYLOES/Likelihood/raxml/results",  # Use the temporary directory for RAxML output
        "-p 1234"
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
    if not keep_output_files:
        subprocess.run("rm Likelihood/raxml/results/*", shell=True)

    return log_likelihood