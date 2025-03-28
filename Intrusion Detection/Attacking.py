import csv
import random
import string


def generate_random_hash():
    """Generate a random hash of 64 characters for simulation."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=64))


def modify_hash_data(input_file, output_file, rows_to_modify):

    with open(input_file, 'r') as infile:
        reader = list(csv.reader(infile))

        # Modify specific rows
        for row_index, new_hash in rows_to_modify.items():
            if 0 <= row_index < len(reader):
                reader[row_index][1] = new_hash  # Assuming the hash is in the 2nd column

        # Write back to a new file
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(reader)


# Example usage
input_csv = 'hash_data.csv'
output_csv = 'modified_hash_data.csv'

# Generate 100 rows to modify (indices 1 to 100 and random hash values)
rows_to_modify = {i: generate_random_hash() for i in range(100)}

modify_hash_data(input_csv, output_csv, rows_to_modify)
print(f"Modified CSV saved to {output_csv}")
