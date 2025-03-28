import pandas as pd
import base64
from Crypto.Cipher import AES
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Function to decrypt data using AES-GCM
def decrypt_aes_gcm(key, encrypted_data):
    encrypted_bytes = base64.b64decode(encrypted_data)
    nonce, tag, ciphertext = encrypted_bytes[:16], encrypted_bytes[16:32], encrypted_bytes[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
    return decrypted_bytes.decode()

# Load the encryption key
key_file = 'encryption_key.json'
with open(key_file, 'r') as kf:
    key_data = json.load(kf)
key = base64.b64decode(key_data['key'])

# Load the encrypted dataset
encrypted_file = 'encrypted_data.csv'
encrypted_df = pd.read_csv(encrypted_file)

# Prepare decrypted data
decrypted_data = []

# Iterate through each row in the encrypted dataset
for index, row in encrypted_df.iterrows():
    decrypted_row = {}
    for col in encrypted_df.columns:
        encrypted_value = row[col]
        # Decrypt value
        decrypted_value = decrypt_aes_gcm(key, encrypted_value)
        decrypted_row[col] = decrypted_value
    decrypted_data.append(decrypted_row)

# Create a DataFrame for the decrypted data
decrypted_df = pd.DataFrame(decrypted_data)

# Store the decrypted dataset
decrypted_file = 'decrypted_data.csv'
decrypted_df.to_csv(decrypted_file, index=False)

print(f"Decrypted dataset saved to {decrypted_file}")
