import pandas as pd

# Baca dataset hasil labeling
df = pd.read_csv("data/cleaned_dataset_final.csv")

# Hitung jumlah data per label
distribusi = df['label'].value_counts()

print("📊 Distribusi Label Sentimen:")
print(distribusi)
