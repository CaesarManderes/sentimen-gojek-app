import pandas as pd
from modules.preprocessing import clean_text
from tqdm import tqdm
import os

# Inisialisasi tqdm untuk .apply
tqdm.pandas()

# Buat folder output jika belum ada
os.makedirs("data", exist_ok=True)

# === [1] Baca dataset mentah
print("📥 Membaca file: data/cleaned_dataset.csv")
df = pd.read_csv("data/cleaned_dataset.csv")
print(f"   ✅ Jumlah data: {len(df)}")

# === [2] Preprocessing
print("🧼 Memproses ulasan dengan clean_text()...")
df['cleaned_review'] = df['content'].progress_apply(clean_text)

# === [3] Simpan hasil ke file baru
output_path = "data/cleaned_dataset_final.csv"
df[['content', 'cleaned_review', 'label']].to_csv(output_path, index=False)
print(f"✅ Dataset akhir berhasil disimpan di: {output_path}")
