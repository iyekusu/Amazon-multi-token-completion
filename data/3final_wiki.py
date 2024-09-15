import pandas as pd

file_path = 'data/wiki/train_comma.csv'
df = pd.read_csv(file_path)

df['span'] = df['span'].str.replace(',', '')
df['span_lower'] = df['span_lower'].str.replace(',', '')

span_frequency = df['span'].value_counts().to_dict()

df['freq'] = df['span'].map(span_frequency)

output_path = 'data/wiki/train_total.csv'
df.to_csv(output_path, index=False, columns = ['span', 'span_lower', 'range', 'text', 'freq', 'masked_text'])

print("CSV file has been modified and saved successfully.")
