import os
import subprocess

import dendropy
import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner

from Likelihood.tree import PhyloTree
from Utils.suppress_prints import suppress_stdout_stderr

models = ["JC69", "K80 ", "F81 ", "F84", "TN93", "GTR"]

class Instance(object):

    def __init__(self, fasta_file, model):

        if model not in models:
            raise ValueError("Model must be one of {0}".format(models))
        else:
            self.model = model
        self.sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.labels = [seq.name for seq in self.sequences]

    def make_instance(self, n_taxa, compute_dist_matrix=False):
        # Compute distance matrix

        n_taxa = n_taxa
        labels = self.labels[:n_taxa]

        taxon_namespace = dendropy.TaxonNamespace()

        # Create an empty DnaCharacterMatrix with the taxon namespace
        dna_matrix = dendropy.DnaCharacterMatrix(taxon_namespace=taxon_namespace)

        # Populate the DnaCharacterMatrix with sequences from SeqRecord
        for seq_record in self.sequences[:n_taxa]:
            # Get or create a Taxon object for this sequence
            taxon = taxon_namespace.require_taxon(label=seq_record.id)
            # Insert sequence data into the matrix
            dna_matrix.update_taxon_namespace()
            dna_matrix[taxon] = str(seq_record.seq)

        alignment_file = "sequence.phy"
        # Write data to the temporary files
        dna_matrix.write_to_path(dest=alignment_file, schema="phylip")

        if not compute_dist_matrix:
            return alignment_file, labels

        else:
            with suppress_stdout_stderr():
                os.system('Solvers/FastME/fastme -i ' + alignment_file + ' -O mat.txt -d J -r')
            mat = np.loadtxt('mat.txt', skiprows=1, usecols=range(1, n_taxa+1))
            # phyml_executable = "Likelihood/phyml-master/src/phyml"
            # phyml_command = [
            #     phyml_executable,
            #     "-i", alignment_file,  # Specify the alignment file
            #     "-m", self.model,  # Specify the substitution model (adjust if necessary)
            #     "-o", "n",
            # ]
            #
            # with suppress_stdout_stderr():
            #     subprocess.run(phyml_command, check=True)
            # mat = np.loadtxt('mat.txt')
            return alignment_file, labels, mat

    @staticmethod
    def compute_distance_matrix(sequences):
        num_seqs = len(sequences)
        distance_matrix = np.zeros((num_seqs, num_seqs))

        aligner = PairwiseAligner()
        aligner.mode = 'global'

        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                # Align sequences
                score = aligner.score(sequences[i].seq, sequences[j].seq)
                max_len = max(len(sequences[i]), len(sequences[j]))
                # Compute distance as 1 - normalized score
                distance = 1 - score / max_len

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix
