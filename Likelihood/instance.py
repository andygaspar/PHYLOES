import dendropy
import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner

from Likelihood.tree import PhyloTree


class Instance(object):

    def __init__(self, fasta_file, n_taxa, compute_dist_matrix=False):

        sequences = list(SeqIO.parse(fasta_file, "fasta"))

        # Compute distance matrix
        self.d = self.compute_distance_matrix(sequences[:n_taxa]) if compute_dist_matrix else None
        self.n_taxa = n_taxa
        self.labels = [seq.name for seq in sequences[:n_taxa]]

        tree = PhyloTree(n_taxa, self.labels)
        tree.set_random_tree()
        self.tree_file = tree.tree_file
        taxon_namespace = dendropy.TaxonNamespace()

        # Create an empty DnaCharacterMatrix with the taxon namespace
        dna_matrix = dendropy.DnaCharacterMatrix(taxon_namespace=taxon_namespace)

        # Populate the DnaCharacterMatrix with sequences from SeqRecord
        for seq_record in sequences[:n_taxa]:
            # Get or create a Taxon object for this sequence
            taxon = taxon_namespace.require_taxon(label=seq_record.id)
            # Insert sequence data into the matrix
            dna_matrix.update_taxon_namespace()
            dna_matrix[taxon] = str(seq_record.seq)

        self.alignment_file = "sequence.phy"
        # Write data to the temporary files
        dna_matrix.write_to_path(dest=self.alignment_file, schema="phylip")

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
