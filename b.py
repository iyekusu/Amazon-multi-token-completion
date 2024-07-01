import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
file_path = '/home/s2299194/amazon-multi-token-completion/wiki_lmn.csv'
df = pd.read_csv(file_path)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the split ratio (18:1)
train_size = int(18 / 19 * len(df))

# Split the data
train_df = df.iloc[:train_size]
dev_df = df.iloc[train_size:]

output_dir = '/home/s2299194/amazon-multi-token-completion/wiki_lmn'

# Save the splits into separate CSV files in the specified directory
train_file_path = os.path.join(output_dir, 'train.csv')
dev_file_path = os.path.join(output_dir, 'dev.csv')

train_df.to_csv(train_file_path, index=False)
dev_df.to_csv(dev_file_path, index=False)

print(f'Training data saved to {train_file_path} ({len(train_df)} rows)')
print(f'Development data saved to {dev_file_path} ({len(dev_df)} rows)')
