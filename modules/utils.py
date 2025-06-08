import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.preprocessing import clean_text

# === Konstanta LSTM ===
MAX_SEQUENCE_LENGTH = 250
LABELS = ['negatif', 'netral', 'positif']

# === Load Model SVM ===
with open("models/model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Load Model LSTM ===
lstm_model = load_model("models/model_lstm.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_svm(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label_index = svm_model.predict(X)[0]
    label = label_index  # label sudah berupa string 'negatif', 'netral', 'positif'

    # === Confidence score per label (pakai decision_function → sigmoid)
    try:
        raw_scores = svm_model.decision_function(X)[0]  # bentuk: array [3 kelas]
        if raw_scores.ndim == 0:
            raw_scores = [raw_scores]  # fallback
        conf_soft = sigmoid(raw_scores)
        conf_sum = np.sum(conf_soft)
        conf_norm = conf_soft / conf_sum  # Normalisasi agar total = 1
        confidence_dict = {LABELS[i]: float(conf_norm[i]) for i in range(len(LABELS))}
    except Exception:
        confidence_dict = {l: 1.0 if l == label else 0.0 for l in LABELS}

    # === Kata penting
    tokens = cleaned.split()
    influence_words = []
    if hasattr(svm_model, "coef_"):
        coef = svm_model.coef_[LABELS.index(label)]
        feature_names = vectorizer.get_feature_names_out()
        for word in tokens:
            if word in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[word]
                score = abs(round(coef[idx], 3))
                influence_words.append((word, score))
    influence_words = sorted(influence_words, key=lambda x: -x[1])[:7]

    return label, confidence_dict, influence_words, tokens


# === Fungsi Prediksi LSTM ===
def predict_lstm(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    probs = lstm_model.predict(padded)[0]
    label_index = np.argmax(probs)
    label = LABELS[label_index]

    # Ambil kata-kata penting (top N kata dari kalimat yang dikenal tokenizer)
    tokens = cleaned.split()
    influence_words = []
    for word in tokens:
        if word in tokenizer.word_index:
            score = probs[label_index] * (1 / (1 + tokenizer.word_index[word]))  # dummy pengaruh
            influence_words.append((word, round(score, 3)))
    
    influence_words = sorted(influence_words, key=lambda x: -x[1])[:7]  # Top 7
    return label, float(probs[label_index]), probs, influence_words, tokens
