def txt_to_fasta(input_file, output_file):

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():
                parts = line.split()
                header = f">{parts[0]}"
                sequence = parts[1]
                outfile.write(f"{header}\n{sequence}\n")

