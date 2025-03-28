import pandas as pd
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import json
import matplotlib.pyplot as plt

# Function to generate a hash of a given value
def generate_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()

# Function to encrypt data using AES-GCM
def encrypt_aes_gcm(key, data):
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

# Load the dataset
input_file = 'normal_data.csv'
data = pd.read_csv(input_file)

# Generate a 16-byte AES key
key = get_random_bytes(16)

# Store the key in a file
key_file = 'encryption_key.json'
with open(key_file, 'w') as kf:
    json.dump({'key': base64.b64encode(key).decode()}, kf)

# Prepare encrypted data and hashes
encrypted_data = []
hashes = []

# Iterate through each row in the dataset
for index, row in data.iterrows():
    encrypted_row = {}
    hash_row = {}
    for col in data.columns:
        original_value = str(row[col])
        # Generate hash
        hash_value = generate_hash(original_value)
        hash_row[col] = hash_value
        # Encrypt value
        encrypted_value = encrypt_aes_gcm(key, original_value)
        encrypted_row[col] = encrypted_value
    encrypted_data.append(encrypted_row)
    hashes.append(hash_row)

# Create DataFrames for encrypted data and hashes
encrypted_df = pd.DataFrame(encrypted_data)
hash_df = pd.DataFrame(hashes)

# Store the encrypted dataset and hash values
encrypted_file = 'encrypted_data.csv'
hash_file = 'hash_data.csv'
encrypted_df.to_csv(encrypted_file, index=False)
hash_df.to_csv(hash_file, index=False)

print(f"Encrypted dataset saved to {encrypted_file}")
print(f"Hash values saved to {hash_file}")
print(f"Encryption key saved to {key_file}")

# Convert all data to strings and calculate lengths for each column
lengths = data.astype(str).apply(lambda col: col.dropna().map(len))

# Calculate the average lengths per column
avg_lengths = lengths.mean()

# Plot the average lengths with logarithmic scaling
plt.figure(figsize=(10, 6))
avg_lengths.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Length of Original Values per Column (Logarithmic Scale)', fontsize=14)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Average Length (Log Scale)', fontsize=12)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 2: Comparison of data size before and after encryption
original_size = data.memory_usage(deep=True).sum()
encrypted_size = encrypted_df.memory_usage(deep=True).sum()
plt.figure(figsize=(8, 6))
plt.bar(['Original Data', 'Encrypted Data'], [original_size, encrypted_size], color=['green', 'blue'])
plt.title('Data Size Comparison', fontsize=14)
plt.ylabel('Size in Bytes', fontsize=12)
plt.tight_layout()
plt.show()

# Plot 3: Distribution of hash lengths (should be constant as SHA-256 produces fixed-length output)
hash_lengths = hash_df.astype(str).apply(lambda col: col.map(len)).iloc[0]

plt.figure(figsize=(10, 6))
hash_lengths.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Length of Hash Values per Column', fontsize=14)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Hash Length', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Convert encrypted data to strings and calculate lengths for each column
encrypted_lengths = encrypted_df.astype(str).apply(lambda col: col.dropna().map(len))

# Calculate the average lengths per column
avg_encrypted_lengths = encrypted_lengths.mean()

# Plot the average lengths with logarithmic scaling
plt.figure(figsize=(10, 6))
avg_encrypted_lengths.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Length of Encrypted Values per Column (Logarithmic Scale)', fontsize=14)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Average Length (Log Scale)', fontsize=12)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
