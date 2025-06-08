import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Aktifkan progress bar
tqdm.pandas()

# === [1] Baca dataset final
print("📥 Membaca data dari 'data/cleaned_dataset_final.csv'...")
df = pd.read_csv("data/cleaned_dataset_final.csv")

# Hapus baris kosong
df = df.dropna(subset=['cleaned_review', 'label'])
df['clean'] = df['cleaned_review'].astype(str)

# === [2] Split data
print("📊 Membagi data latih dan uji...")
X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['label'], test_size=0.2, random_state=42)
print(f"   ✅ Data latih: {len(X_train)}, Data uji: {len(X_test)}")

# === [3] Load vectorizer dan model
print("📦 Memuat model dan TF-IDF vectorizer dari folder 'models/'...")
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

# === [4] Transformasi data uji
X_test_tfidf = vectorizer.transform(X_test)

# === [5] Prediksi
print("🔍 Melakukan prediksi pada data uji...")
y_pred = svm_model.predict(X_test_tfidf)

# === [6] Evaluasi hasil prediksi
print("\n📊 Classification Report (SVM):\n")
print(classification_report(y_test, y_pred))
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# === [7] Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["positif", "netral", "negatif"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["positif", "netral", "negatif"],
            yticklabels=["positif", "netral", "negatif"])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.show()
print("🖼️ Confusion Matrix disimpan sebagai 'svm_confusion_matrix.png'")

# === [8] Cross-validation (opsional tapi disarankan)
print("\n🔁 5-Fold Cross Validation (F1-Score Macro)...")
X_all = vectorizer.transform(df['clean'])
y_all = df['label']
scores = cross_val_score(svm_model, X_all, y_all, cv=5, scoring='f1_macro')
print(f"📈 Rata-rata F1-Score (5-Fold CV): {scores.mean():.4f}")
