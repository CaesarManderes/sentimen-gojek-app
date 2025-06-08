import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
import os
import time
from tqdm import tqdm

# Aktifkan tqdm untuk progress bar
tqdm.pandas()

# === [1] Baca data final ===
print("🚀 [1/5] Membaca data dari 'data/cleaned_dataset_final.csv'...")
df = pd.read_csv("data/cleaned_dataset_final.csv")

# Pastikan tidak ada data kosong
df = df.dropna(subset=['cleaned_review', 'label'])
df['clean'] = df['cleaned_review'].astype(str)

print(f"   ✅ Jumlah data: {len(df)}")

# === [2] Split data
print("📊 [2/5] Membagi data latih dan uji...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)
print(f"   ✅ Data latih: {len(X_train)}, Data uji: {len(X_test)}")

# === [3] TF-IDF Vectorizer
print("🔠 [3/5] Mengubah teks ke fitur TF-IDF...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
print(f"   ✅ Jumlah fitur: {X_train_tfidf.shape[1]}")

# === [4] Latih model SVM
print("🧠 [4/5] Melatih model SVM dengan class_weight='balanced'...")
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train_tfidf, y_train)
print("   ✅ Model SVM berhasil dilatih.")

# === [5] Simpan model & vectorizer
print("💾 [5/5] Menyimpan model dan vectorizer...")
os.makedirs("models", exist_ok=True)

with open("models/model_svm.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 SELESAI: Model SVM dan vectorizer disimpan di folder 'models/' dan siap digunakan.")
