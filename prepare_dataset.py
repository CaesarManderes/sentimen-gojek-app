import pandas as pd

# Baca file CSV
df = pd.read_csv("data/GojekAppReview_1.csv")

# Ambil kolom penting dan buang nilai kosong
df = df[['content', 'score']].dropna()

# Buat label berdasarkan skor
def label_sentiment(score):
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    else:
        return 'positif'

df['label'] = df['score'].apply(label_sentiment)

# Simpan ke file baru
df.to_csv("data/cleaned_dataset.csv", index=False)

print("✅ Data telah dilabeli dan disimpan sebagai 'data/cleaned_dataset.csv'")
