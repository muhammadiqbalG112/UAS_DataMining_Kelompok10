import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Layoff Risk Predictor", page_icon="🏢", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD MODEL (LOGISTIC REGRESSION) ---
@st.cache_resource
def load_champion_model():
    model_path = os.path.join(BASE_DIR, 'model_final.pkl')
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        return None

model = load_champion_model()

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, 'ai-impact-jobs-layoff-risk-dataset.csv')
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pergi ke halaman:", 
                        ["1. Home", "2. Dataset Overview", "3. Prediction Analysis", "4. Visualization", "5. About"])

# --- MENU 1 & 2 ---
if menu == "1. Home":
    st.title("🏢 Sistem Prediksi Risiko Layoff Karyawan")
    st.write("Aplikasi ini menggunakan algoritma **Logistic Regression** yang dioptimasi untuk menganalisis probabilitas pemutusan hubungan kerja berdasarkan parameter historis, dipadukan dengan wawasan asosiasi makro **FP-Growth**.")
    st.info("Anggota Kelompok:\n1. Muhammad Iqbal Hanif (23051214161)")

elif menu == "2. Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("Menampilkan sampel data historis korporasi yang digunakan sebagai basis pelatihan kecerdasan buatan.")
    if not df.empty:
        st.dataframe(df.head(15), use_container_width=True)
        st.write("**Statistik Deskriptif:**")
        st.dataframe(df.describe(), use_container_width=True)
        st.write("**Distribusi Data:**")
        st.bar_chart(df['Layoff_Risk'].value_counts())

# --- MENU 3: PREDICTION ---
elif menu == "3. Prediction Analysis":
    st.title("🔮 Form Prediksi Risiko Layoff")
    st.markdown("Masukkan data profil karyawan untuk memicu inferensi model logistik dan menghitung tingkat kerentanan secara *real-time*.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox("Pilih Sektor Industri", ["Tech", "Finance", "Healthcare", "Retail", "Manufacturing", "Education", "Logistics"])
            job_level = st.selectbox("Level Jabatan Struktural", ["Entry", "Junior", "Mid", "Senior", "Executive"])
        with col2:
            ai_adoption = st.selectbox("Tingkat Adopsi AI di Divisi", ["Low", "Medium", "High"])
            ai_hours = st.slider("Total Jam Pelatihan AI (Per Bulan)", 0, 100, 20)
            
        submitted = st.form_submit_button("Mulai Analisis Risiko 🚀")

    if submitted:
        if model is None:
            st.error("⚠️ Model biner (Pickle) tidak ditemukan di direktori lokal.")
        else:
            input_data = pd.DataFrame([[industry, job_level, ai_adoption, ai_hours]], 
                                      columns=['Industry', 'Job_Level', 'AI_Adoption_Level', 'AI_Training_Hours'])
            
            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
            max_prob = max(probs) * 100
            
            st.divider()
            st.subheader("📋 HASIL KLASIFIKASI SISTEM (INFERENSI MATEMATIS)")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if "High" in str(prediction):
                    st.error(f"🚨 STATUS KEPUTUSAN: **{prediction.upper()}**")
                elif "Medium" in str(prediction):
                    st.warning(f"⚠️ STATUS KEPUTUSAN: **{prediction.upper()}**")
                else:
                    st.success(f"✅ STATUS KEPUTUSAN: **{prediction.upper()}**")
                    
            with res_col2:
                st.metric(label="Confidence Score (Tingkat Keyakinan)", value=f"{max_prob:.2f}%")
            
            st.markdown("### 🧠 Insight Algoritma & Rekomendasi HR")
            if max_prob < 55:
                st.info("Sistem mendeteksi ambiguitas tinggi pada profil matriks ini. Risiko berada di area batas keputusan (*decision boundary*). Wajib dilakukan evaluasi kualitatif lanjutan oleh manajer SDM.")
            elif "High" in str(prediction):
                st.warning("Peringatan Sistem: Kombinasi latar belakang industri, level jabatan, dan rasio adopsi AI karyawan ini memicu identifikasi pola rentan (*Red Flag*) historis. Prioritaskan program pelatihan ulang (*reskilling*).")
            else:
                st.success("Profil kandidat diproyeksikan stabil terhadap disrupsi otomasi. Tidak ditemukan pola anomali risiko layoff yang signifikan.")

# --- MENU 4: VISUALIZATION (FP-GROWTH INTEGRATION) ---
elif menu == "4. Visualization":
    st.title("📈 Visualisasi Analitik & Pola Asosiasi")
    st.markdown("Halaman ini menyajikan wawasan makro tingkat korporat yang diekstrak dari mesin *Big Data* PySpark beserta visualisasi performa model klasifikasi.")
    
    # BAGIAN 1: TABEL ATURAN ASOSIASI
    st.subheader("A. Tabel Ekstraksi Association Rules (FP-Growth)")
    csv_rules_path = os.path.join(BASE_DIR, 'layoff_risk_COMPLETE_rules.csv')
    try:
        rules_df = pd.read_csv(csv_rules_path)
        if not rules_df.empty:
            if 'lift_str' in rules_df.columns:
                rules_df['lift_val'] = pd.to_numeric(rules_df['lift_str'], errors='coerce')
                rules_df = rules_df.sort_values(by='lift_val', ascending=False).drop(columns=['lift_val']).reset_index(drop=True)
            
            st.dataframe(rules_df, use_container_width=True)
        else:
            st.warning("File CSV aturan asosiasi kosong.")
    except Exception as e:
        st.error(f"Gagal memuat file CSV. Error: {e}")

    st.divider()

    # BAGIAN 2: GALERI VISUALISASI GRAFIK
    st.subheader("B. Galeri Visualisasi Komputasi")
    st.write("Buka panel di bawah ini untuk melihat interpretasi grafis dari hasil komputasi *Machine Learning*.")

    # Expander 1: Network Web
    with st.expander("🕸️ 1. Topologi Jaringan Asosiasi (Network Web)"):
        st.markdown("Grafik ini memetakan alur kausalitas dari atribut karyawan (biru) menuju tingkat risiko pemutusan kerja (merah).")
        img_network_path = os.path.join(BASE_DIR, 'network_web.png') # Sesuaikan nama file gambar Anda!
        if os.path.exists(img_network_path):
            img_net = Image.open(img_network_path)
            st.image(img_net, caption="Pemusatan Risiko Tinggi Berdasarkan Algoritma FP-Growth", use_container_width=True)
        else:
            st.info("💡 Simpan gambar Anda dengan nama 'network_web.png' di folder proyek untuk menampilkannya di sini.")

    # Expander 2: Scatter Plot
    with st.expander("🫧 2. Sebaran Kekuatan Aturan (Scatter Plot)"):
        st.markdown("Analisis spasial yang membuktikan bahwa aturan asosiasi yang ditemukan memiliki validitas statistik yang tinggi (Lift Ratio ekstrem).")
        img_scatter_path = os.path.join(BASE_DIR, 'scatter_plot.png') # Sesuaikan nama file gambar Anda!
        if os.path.exists(img_scatter_path):
            img_scat = Image.open(img_scatter_path)
            st.image(img_scat, caption="Pemetaan Metrik Support, Confidence, dan Lift Ratio", use_container_width=True)
        else:
            st.info("💡 Simpan gambar Anda dengan nama 'scatter_plot.png' di folder proyek untuk menampilkannya di sini.")

    # Expander 3: Confusion Matrix
    with st.expander("🎯 3. Evaluasi Prediksi (Confusion Matrix)"):
        st.markdown("Pembuktian empiris akurasi model tunggal *Logistic Regression* dalam menebak ketiga kelas target risiko karyawan.")
        img_matrix_path = os.path.join(BASE_DIR, 'confusion_matrix.png') # Sesuaikan nama file gambar Anda!
        if os.path.exists(img_matrix_path):
            img_mat = Image.open(img_matrix_path)
            st.image(img_mat, caption="Sebaran Ketepatan Klasifikasi Aktual vs Prediksi", use_container_width=True)
        else:
            st.info("💡 Simpan gambar Anda dengan nama 'confusion_matrix.png' di folder proyek untuk menampilkannya di sini.")

# --- MENU 5: ABOUT ---
elif menu == "5. About":
    st.title("ℹ️ Tentang Arsitektur Proyek")
    st.write("Sistem pendukung keputusan hibrida ini dirancang dengan mengadopsi metodologi **CRISP-DM**. Arsitektur ini membuktikan keberhasilan prinsip *Separation of Concerns*, di mana parameter optimasi kompleks dieksekusi pada mesin **PySpark MLlib**, kemudian garis regresi terbaiknya diserialisasikan ke format Pickle untuk menghasilkan inferensi *zero-latency* menggunakan pustaka **Scikit-Learn** dan **Streamlit**.")
    st.markdown("Halaman ini memuat dokumentasi teknis dan informasi arsitektural dari pengembangan purwarupa sistem pendukung keputusan hibrida ini.")
    
    with st.expander("⚙️ 1. Penjelasan Metode"):
        st.write("Sistem ini dibangun dengan mengadopsi secara ketat kerangka kerja *Cross-Industry Standard Process for Data Mining* (CRISP-DM), mengintegrasikan fungsionalitas hibrida antara analisis deskriptif dan prediktif. Pemrosesan analitik berskala masif, termasuk penjelajahan *Hyperparameter Tuning* dan ekstraksi *Association Rule Mining* (FP-Growth), dieksekusi secara paralel di dalam klaster memori *Apache Spark* (PySpark) guna mencegah *bottleneck* komputasi. Guna menjamin reliabilitas dan latensi inferensi yang mendekati nol pada *environment* produksi, parameter dari model juara (*Logistic Regression*) diserialisasikan ke dalam format biner *Pickle*. Arsitektur *Separation of Concerns* ini secara elegan memitigasi isu dependensi *Native I/O Windows* pada *Java Virtual Machine* (JVM), memungkinkan *framework* Scikit-Learn dan Streamlit untuk mengeksekusi kalkulasi probabilitas klasifikasi secara mandiri, instan, dan terisolasi.")
        
    with st.expander("📁 2. Dataset"):
        st.write("Dataset yang digunakan merupakan data sintetis realistis dengan total data sebesar 20.000. dibangun dengan tujuan meninjau bagaimana kecerdasan buatan diadopsi, otomatisasi, pekerjaan, dan skill pegawai mempengaruhi ketenagakerjaan pada ekonomi saat ini.")
        st.write("dataset: S. Singh, ""AI Impact on Jobs and Layoff Risk Dataset,"" Kaggle, 2026. [Online]. Available: https://www.kaggle.com/datasets/shivasingh4945/ai-impact-on-jobs-and-layoff-risk-dataset. (Accessed: Jun. 5, 2026).")

    with st.expander("🚀 3. Informasi Proyek"):
        st.write("Proyek simulasi eksperimental ini diinisiasi sebagai respons strategis terhadap fenomena disrupsi otomasi *Artificial Intelligence* (AI) yang memicu pergeseran peran tenaga kerja secara global. Bertindak sebagai purwarupa *Early Warning System* (EWS) bagi divisi *Human Resources* (HR), aplikasi antarmuka ini dirancang untuk mendemokratisasi akses terhadap wawasan analitik tingkat tinggi. Dengan menyajikan metrik *Confidence Score* secara transparan dan memvisualisasikan korelasi kausalitas melalui topologi graf, proyek ini memberdayakan pengambil kebijakan untuk bertransisi dari pengambilan keputusan kualitatif yang rentan bias menuju strategi retensi talenta berbasis data (*data-driven*), sehingga mengoptimalkan alokasi anggaran untuk program *reskilling* organisasi secara presisi.")