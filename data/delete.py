def save_first_2000_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            if i >= 2500:
                break
            outfile.write(line)

# Example usage:
save_first_2000_lines("train_original.txt", "train.txt")
