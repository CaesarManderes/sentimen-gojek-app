import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Konstanta
MAX_SEQUENCE_LENGTH = 250
LABELS = ['negatif', 'netral', 'positif']

tqdm.pandas()

# === [1] Baca dataset final
print("📥 Membaca data dari 'data/cleaned_dataset_final.csv'...")
df = pd.read_csv("data/cleaned_dataset_final.csv")
df = df.dropna(subset=['cleaned_review', 'label'])

df['clean'] = df['cleaned_review'].astype(str)
df['label_index'] = df['label'].map({'negatif': 0, 'netral': 1, 'positif': 2})
y_true = df['label']

# === [2] Load tokenizer dan model
print("📦 Memuat tokenizer dan model LSTM dari folder 'models/'...")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("models/model_lstm.h5")

# === [3] Tokenisasi dan padding
print("🔠 Melakukan tokenisasi dan padding...")
sequences = list(tqdm(tokenizer.texts_to_sequences(df['clean'].values), desc="Tokenizing"))
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# === [4] Prediksi
print("🔍 Melakukan prediksi dengan model LSTM...")
y_pred_probs = []
for i in tqdm(range(0, len(X), 512), desc="Predicting"):
    batch = X[i:i+512]
    preds = model.predict(batch, verbose=0)
    y_pred_probs.extend(preds)

y_pred_probs = np.array(y_pred_probs)
y_pred_index = np.argmax(y_pred_probs, axis=1)
y_pred_labels = [LABELS[i] for i in y_pred_index]

# === [5] Evaluasi performa
print("\n📊 Classification Report (LSTM - Final Model):\n")
print(classification_report(y_true, y_pred_labels))
print("✅ Accuracy:", accuracy_score(y_true, y_pred_labels))

# === [6] Confusion Matrix
cm = confusion_matrix(y_true, y_pred_labels, labels=LABELS)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png")
plt.show()

print("🖼️ Confusion Matrix disimpan sebagai 'lstm_confusion_matrix.png'")
print("🎉 Evaluasi model LSTM selesai.")
