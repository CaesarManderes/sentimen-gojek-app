import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Unduh stopwords Bahasa Indonesia
nltk.download('stopwords')

# Inisialisasi stopwords dan stemmer
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# Kamus normalisasi kata tidak baku (slang → baku)
normalization_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "ngga": "tidak",
    "yg": "yang",
    "ok": "oke",
    "ya": "iya",
    "aja": "saja",
    "udah": "sudah",
    "udh": "sudah",
    "pake": "pakai",
    "kasih": "terima kasih",
    "makasih": "terima kasih",
    "makasi": "terima kasih",
    "maaci": "terima kasih",
    "trims": "terima kasih",
    "apk": "aplikasi",
    "aplikasinya": "aplikasi",
    "kalo": "kalau",
    "klo": "kalau",
    "bgt": "sangat",
    "dong": "",
    "drivernya": "driver",
    "lah": "",
    "sih": "",
    "deh": "",
    "kok": "kenapa",
    "dgn": "dengan",
    "dr": "dari",
    "mo": "mau",
    "gmna": "bagaimana",
    "kmrn": "kemarin",
    "trus": "terus",
    "trs": "terus",
    "bener": "benar",
    "mksd": "maksud",
    "sm": "sama",
    "sy": "saya",
    "aku": "saya",
    "lg": "lagi",
    "lgi": "lagi",
    "dlu": "dulu",
    "cm": "cuma",
    "cuma": "hanya",
    "doank": "saja",
    "jd": "jadi",
    "dah": "sudah",
    "cs": "customer service",
    "sdh": "sudah",
    "gw": "saya",
    "min": "admin",
    "eror": "error",
    "mulu": "terus-menerus",
    "app": "aplikasi",
    "hp": "handphone",
    "up": "unggah"
}

def clean_text(text):
    # Ubah ke huruf kecil
    text = text.lower()

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # Normalisasi kata tidak baku (slang)
    text = " ".join([normalization_dict.get(word, word) for word in text.split()])

    # Stopword removal
    tokens = [word for word in text.split() if word not in stop_words]

    # Stemming
    stemmed = [stemmer.stem(word) for word in tokens]

    # Gabungkan kembali
    return " ".join(stemmed)
