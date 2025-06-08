import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.utils import predict_svm, predict_lstm

st.set_page_config(page_title="Analisis Sentimen Gojek", layout="wide")
st.title("💬 Aplikasi Analisis Sentimen Ulasan Gojek")

# ==============================
# INPUT USER
# ==============================
ulasan = st.text_area("📝 Masukkan ulasan pengguna:")
model_pilihan = st.selectbox("📌 Pilih model:", ["LSTM (Word Embedding)", "SVM (TF-IDF)"])

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Sentimen"):
    if not ulasan.strip():
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("⏳ Menganalisis..."):
            if model_pilihan == "LSTM (Word Embedding)":
                label, confidence, probs, influence_words, tokens = predict_lstm(ulasan)
            else:
                label, confidence_dict, influence_words, tokens = predict_svm(ulasan)

        # === HEADER HASIL
        st.markdown(f"## ✅ Hasil Prediksi: **:green[{label.upper()}]**")
        if model_pilihan == "LSTM (Word Embedding)":
            st.markdown(f"📊 Confidence Score: `{confidence:.2f}`")
        else:
            st.markdown(f"📊 Confidence Score: `{confidence_dict.get(label, 0):.2f}`")
        st.divider()

        # === VISUAL UNTUK LSTM
        if model_pilihan == "LSTM (Word Embedding)":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 🔢 Confidence per Label")
                labels = ['negatif', 'netral', 'positif']
                for i, l in enumerate(labels):
                    percent = float(probs[i]) * 100
                    st.markdown(f"""
                    **{l.capitalize()}**: {percent:.2f}%
                    <div style='background-color: #eee; border-radius: 10px; height: 20px;'>
                        <div style='width: {percent}%; background-color: {"#f94144" if l=="negatif" else "#f9c74f" if l=="netral" else "#90be6d"}; height: 100%; border-radius: 10px;'></div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🧠 Influential Words")
                if influence_words:
                    df_words = pd.DataFrame(influence_words, columns=["word", "score"])
                    fig, ax = plt.subplots()
                    ax.barh(df_words["word"][::-1], df_words["score"][::-1], color='green')
                    ax.set_title("Words Influencing the Sentiment")
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada kata penting terdeteksi.")

            st.markdown("### 🔦 Highlighted Review")
            highlight_set = set(w for w, s in influence_words)
            highlighted = [
                f"<span style='background-color:#a0f1a0; padding:3px'>{t}</span>" if t in highlight_set else t
                for t in tokens
            ]
            st.markdown(" ".join(highlighted), unsafe_allow_html=True)

        # === VISUAL UNTUK SVM
        elif model_pilihan == "SVM (TF-IDF)":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### 🔢 Confidence per Label")
                labels = ['negatif', 'netral', 'positif']
                for l in labels:
                    percent = float(confidence_dict.get(l, 0)) * 100
                    st.markdown(f"""
                    **{l.capitalize()}**: {percent:.2f}%
                    <div style='background-color: #eee; border-radius: 10px; height: 20px;'>
                        <div style='width: {percent}%; background-color: {"#f94144" if l=="negatif" else "#f9c74f" if l=="netral" else "#90be6d"}; height: 100%; border-radius: 10px;'></div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🧠 Kata Berpengaruh")
                if influence_words:
                    df_words = pd.DataFrame(influence_words, columns=["word", "score"])
                    fig, ax = plt.subplots()
                    ax.barh(df_words["word"][::-1], df_words["score"][::-1], color='purple')
                    ax.set_title("SVM - Kata Berpengaruh")
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada kata penting terdeteksi.")

            st.markdown("### 🔦 Highlighted Review")
            highlight_set = set(w for w, s in influence_words)
            highlighted = [
                f"<span style='background-color:#fddde6; padding:3px'>{t}</span>" if t in highlight_set else t
                for t in tokens
            ]
            st.markdown(" ".join(highlighted), unsafe_allow_html=True)

        # === Penjelasan simpulan
        if label == "positif":
            st.success("✅ Pelanggan kemungkinan besar **puas**.")
        elif label == "negatif":
            st.error("❌ Pelanggan **tidak puas**.")
        else:
            st.info("⚖️ Ulasan bersifat **netral**.")

# ==============================
# FOOTER
# ==============================
st.divider()
st.markdown("*Dibuat dengan ❤️ oleh Tim NLP — Model SVM & LSTM untuk klasifikasi sentimen ulasan pengguna Gojek.*")
