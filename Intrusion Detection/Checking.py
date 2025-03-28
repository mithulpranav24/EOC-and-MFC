import csv
import matplotlib.pyplot as plt

def read_hash_data(file_path):
    """Reads hash data from a CSV file."""
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            data.append(row[1])  # Assuming the hash is in the 2nd column
    return data

def compare_hash_data(original_data, modified_data):

    differences = []
    unchanged_count = 0
    for index, (orig, mod) in enumerate(zip(original_data, modified_data)):
        if orig == mod:
            unchanged_count += 1
        else:
            differences.append((index, orig, mod))
    changed_count = len(original_data) - unchanged_count
    return unchanged_count, changed_count, differences

def plot_logarithmic_comparison(unchanged, changed):
    """Plots a logarithmic bar chart comparing unchanged and changed hash values."""
    labels = ['Unchanged', 'Changed']
    values = [unchanged, changed]

    plt.bar(labels, values, color=['green', 'red'])
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Logarithmic Comparison of Hash Data')
    plt.ylabel('Logarithmic Count')
    plt.show()

def plot_pie_chart(unchanged, changed):
    """Plots a pie chart showing the proportion of unchanged vs. changed hashes with increased text size."""
    sizes = [unchanged, changed]
    colors = ['green', 'red']
    explode = (0.1, 0)  # Slightly explode the 'Unchanged' slice

    plt.figure(figsize=(10, 10))  # Larger figure for better spacing

    wedges, _, autotexts = plt.pie(
        sizes,
        explode=explode,
        labels=[None, None],  # No labels on the chart itself
        colors=colors,
        autopct='%1.1f%%',
        shadow=False,
        startangle=140,
        textprops={'fontsize': 25}  # Larger font size for percentages
    )

    # Adjust percentage text font size further if needed
    for autotext in autotexts:
        autotext.set_fontsize(25)

    # Move the legend further down
    plt.legend(
        ['Unchanged', 'Changed'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),  # Move legend further down (adjusted here)
        fontsize=25
    )

    #plt.title('Modified Data Distribution', fontsize=24)
    plt.axis('equal')  # Ensure pie is drawn as a circle
    plt.tight_layout()  # Adjusts plot elements to fit within figure
    plt.savefig("intrusion_detection.png", dpi=300, bbox_inches="tight")
    plt.show()



# Paths to files
original_file = 'hash_data.csv'
modified_file = 'modified_hash_data.csv'

# Read data
original_hashes = read_hash_data(original_file)
modified_hashes = read_hash_data(modified_file)

# Compare data
unchanged_count, changed_count, differences = compare_hash_data(original_hashes, modified_hashes)

# Print differences
print("Differences found:")
for index, orig, mod in differences:
    print(f"Row {index + 1}:")
    print(f"  Original Hash: {orig}")
    print(f"  Modified Hash: {mod}")

# Plot graphs
plot_logarithmic_comparison(unchanged_count, changed_count)
plot_pie_chart(unchanged_count, changed_count)


