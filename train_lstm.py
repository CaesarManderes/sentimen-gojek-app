import pandas as pd
import numpy as np
import pickle
import os
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

tqdm.pandas()

# === [1] Baca data final ===
print("🚀 [1/5] Membaca data dari 'data/cleaned_dataset_final.csv'...")
df = pd.read_csv("data/cleaned_dataset_final.csv")
df = df.dropna(subset=['cleaned_review', 'label'])

print(f"   ✅ Jumlah data: {len(df)}")

# === [2] Gunakan kolom cleaned_review yang sudah diproses
df['clean'] = df['cleaned_review'].astype(str)

# === [3] Tokenisasi dan Padding
print("🔠 [2/5] Tokenisasi dan padding...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['clean'].values)

X = tokenizer.texts_to_sequences(df['clean'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Konversi label ke one-hot
y = pd.get_dummies(df['label']).values

# Mapping label ke integer untuk class weight
label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
df['label_int'] = df['label'].map(label_map)
y_labels = df['label_int'].values

# === [3b] Class weight
print("⚖️  [3b] Menghitung class weight...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
class_weights_dict = dict(enumerate(class_weights))
print(f"   ✅ Class Weight: {class_weights_dict}")

# === [4] Split data
print("📊 [4/5] Membagi data latih dan uji...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✅ Data latih: {len(X_train)}, Data uji: {len(X_test)}")

# === [5] Bangun dan latih model
print("🧠 [5/5] Membangun dan melatih model LSTM...")
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights_dict,
    verbose=1
)

# === [6] Simpan model & tokenizer
os.makedirs("models", exist_ok=True)
model.save("models/model_lstm.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("💾 Model dan tokenizer berhasil disimpan di folder 'models/'")
print("🎉 SELESAI: Model LSTM siap digunakan!")
