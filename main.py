import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


disease_data = {
    'Tomato Bacterial spot': {
        "cause": "Disebabkan oleh bakteri Xanthomonas campestris yang menyebar melalui air hujan atau irigasi overhead.",
        "video": "https://youtu.be/2QA3TaR0nFk"
    },
    'Tomato Early blight': {
        "cause": "Disebabkan oleh jamur Alternaria solani yang berkembang di lingkungan lembab.",
        "video": "https://youtu.be/OSNdDYF1llw"
    },
    'Tomato Late blight': {
        "cause": "Disebabkan oleh jamur Phytophthora infestans yang berkembang di kondisi basah dan dingin.",
        "video": "https://youtu.be/BJZ1KREO2Cs"
    },
    'Tomato Leaf Mold': {
        "cause": "Disebabkan oleh jamur Passalora fulva yang sering menyerang rumah kaca.",
        "video": "https://youtu.be/CmpyDg3Bwl4"
    },
    'Tomato Septoria leaf spot': {
        "cause": "Disebabkan oleh jamur Septoria lycopersici yang menyebar melalui percikan air.",
        "video": "https://youtu.be/ywIMhFTXeXw"
    },
    'Tomato Spider mites Two-spotted spider mite': {
        "cause": "Disebabkan oleh serangan tungau laba-laba dua bercak (Tetranychus urticae).",
        "video": "https://youtu.be/bqCBIP9TmcY?si=aLjOIp9w0NZo0XQy"
    },
    'Tomato Target Spot': {
        "cause": "Disebabkan oleh jamur Corynespora cassiicola yang menyerang daun dan buah.",
        "video": "https://youtu.be/2QA3TaR0nFk"
    },
    'Tomato Yellow Leaf Curl Virus': {
        "cause": "Disebabkan oleh virus yang ditularkan oleh kutu kebul (whiteflies).",
        "video": "https://youtu.be/SiNDPbeEPIg"
    },
    'Tomato mosaic virus': {
        "cause": "Disebabkan oleh virus yang menyebar melalui kontak langsung atau alat berkebun yang terkontaminasi.",
        "video": "https://youtu.be/d_ylwAUu7OY?si=GK8gcxcz5AWXuaOl"
    },
    'Tomato healthy': {
        "cause": "Tanaman sehat, kamu bisa menonton video ini untuk referensi cara bertanam yang lebih baik.",
        "video": "https://youtu.be/ok95URMnPcw?si=blVBYta5jod7tEAy"
    }
}

# Fungsi prediksi model
def model_prediction(test_image):
    model = load_model('model.h5')
    image = load_img(test_image, target_size=(130, 130))
    input_arr = img_to_array(image)
    input_arr = preprocess_input(input_arr)  
    input_arr = np.expand_dims(input_arr, axis=0)  
    predictions = model.predict(input_arr)  
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("Sistem Pendeteksi Penyakit Tanaman ")
    st.markdown("""
    ## Selamat Datang di Sistem Pendeteksi Penyakit Tanaman!

    Sistem ini memanfaatkan teknik Machine Learning yang canggih untuk mengidentifikasi berbagai penyakit tanaman dari gambar daun tanaman. Berikut adalah fitur utama dari aplikasi kami:

    - **Deteksi Penyakit**: Unggah gambar daun tanaman, dan model kami akan memprediksi penyakit yang menyerang tanaman tersebut.
    - **Rekomendasi Solusi**: Bersamaan dengan prediksi penyakit, sistem memberikan solusi praktis untuk mengelola atau mengobati penyakit yang teridentifikasi.

    ### Cara Menggunakan
    1. Navigasikan ke halaman "Disease Recognition" pada sidebar.
    2. Unggah gambar daun tanaman yang ingin Anda analisis.
    3. Klik tombol "Prediksi" untuk mendapatkan prediksi penyakit dan rekomendasi solusi.

    Kami berharap alat ini membantu Anda dalam menjaga kesehatan tanaman secara efektif.
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    ## Tentang Sistem Pendeteksi Penyakit Tanaman

    Sistem Pendeteksi Penyakit Tanaman adalah proyek yang bertujuan untuk membantu petani dan pekebun dalam mengidentifikasi dan mengelola penyakit tanaman menggunakan kekuatan kecerdasan buatan.

    ### Tujuan
    - **Deteksi Penyakit yang Akurat**: Menyediakan prediksi yang andal untuk berbagai penyakit tanaman berdasarkan gambar daun.
    - **Solusi Efektif**: Menawarkan solusi yang dapat diterapkan untuk mengobati atau mengelola penyakit yang terdeteksi.
    - **Teknologi yang Mudah Diakses**: Memastikan teknologi ini mudah diakses dan digunakan oleh semua orang, dari petani profesional hingga pekebun hobi.

    ### Teknologi yang Digunakan
    - **Model Machine Learning**: Dibangun menggunakan Jaringan Saraf Konvolusional (CNN) dengan TensorFlow.
    - **Dataset**: Menggunakan 'New Plant Datasets' dari Kaggle yang dibuat oleh Vipoool. Dataset ini menyediakan lebih dari 80 ribu gambar daun dari 14 jenis tanaman.
    - **Deploy**: Di-deploy menggunakan Streamlit untuk antarmuka berbasis web yang interaktif dan sederhana.
    """)

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Pilih gambar berjenis .jpg, .jpeg, .png", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    if st.button("Mulai Prediksi"):
        if test_image is not None:
            with st.spinner("Model sedang memprediksi..."):
                class_name = ['Tomato Bacterial spot',
                              'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
                              'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite',
                              'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
                              'Tomato healthy']
                result_index = model_prediction(test_image)
                predicted_class = class_name[result_index]
                st.success("Hasil dari prediksi adalah {}".format(predicted_class))
                cause_data = disease_data.get(predicted_class, {"cause": "Informasi penyebab tidak ditemukan", "video": ""})
                st.success("Penyebab penyakit: {}".format(cause_data["cause"]))
                if cause_data["video"]:
                    st.markdown(f"### Video Penjelasan")
                    st.video(cause_data["video"])
        else:
            st.warning("Anda belum mengunggah gambar, silahkan unggah gambar.")
