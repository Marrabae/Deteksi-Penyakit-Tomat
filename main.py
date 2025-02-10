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

# Fungsi Deteksi model
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
    "Select Page", ["Home", "Tentang", "Deteksi Penyakit"])

# Main Page
if (app_mode == "Home"):
    st.header("Sistem Pendeteksi Penyakit Tanaman Tomat")
    st.markdown("""
    ## Selamat Datang di Sistem Pendeteksi Penyakit Tanaman Tomat!

    Sistem ini memanfaatkan teknik Machine Learning yang canggih untuk mengidentifikasi berbagai penyakit tanaman tomat dari gambar daunnya. Berikut adalah fitur utama dari aplikasi kami:

    - **Deteksi Penyakit**: Unggah gambar daun tanaman tomat, dan model akan mendeteksi penyakit yang menyerang tanaman tersebut.
    - **Rekomendasi Solusi**: Bersamaan dengan Deteksi penyakit, sistem memberikan penyebab dan video yang bisa membantu menangani penyakit yang menimpa tanaman tomat.

    ### Cara Menggunakan
    1. Navigasikan ke halaman "Deteksi Penyakit" pada sidebar.
    2. Unggah gambar daun tanaman yang ingin Anda analisis.
    3. Klik tombol "Deteksi" untuk mendapatkan Deteksi penyakit dan rekomendasi solusi.

    Kami berharap alat ini membantu Anda dalam menjaga kesehatan tanaman secara efektif.
    """)

# About Project
elif (app_mode == "Tentang"):
    st.header("Tentang")
    st.markdown("""
    ## Tentang Sistem Pendeteksi Penyakit Tanaman

    Sistem Pendeteksi Penyakit Tanaman adalah proyek yang bertujuan untuk membantu pekebun rumahan dalam mengidentifikasi dan mengelola penyakit tanaman menggunakan kekuatan kecerdasan buatan.

    ### Tujuan
    - **Deteksi Penyakit**: Menyediakan Deteksi yang andal untuk berbagai penyakit tanaman tomat berdasarkan gambar daun.
    - **Solusi**: Menawarkan solusi yang dapat diterapkan untuk mengobati atau mengelola penyakit yang terdeteksi.
    - **Mudah Diakses**: Memastikan teknologi ini mudah diakses dan digunakan oleh semua orang.

    ### Teknologi yang Digunakan
    - **Model Machine Learning**: Dibangun menggunakan Jaringan Saraf Konvolusional (CNN) dengan TensorFlow.
    - **Dataset**: Menggunakan 'tomatoleaf' dari Kaggle yang dibuat oleh kaustubh b. Dataset ini menyediakan lebih dari 10 ribu gambar daun dari 9 jenis penyakit tanaman tomat.
    - **Deploy**: Di-deploy menggunakan Streamlit untuk antarmuka berbasis web yang sederhana.
    """)

# Prediction Page
elif (app_mode == "Deteksi Penyakit"):
    st.header("Deteksi Penyakit")
    test_image = st.file_uploader("Pilih gambar berjenis .jpg, .jpeg, .png", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_container_width=True)
    if st.button("Mulai Deteksi"):
        if test_image is not None:
            with st.spinner("Model sedang mendeteksi..."):
                class_name = ['Tomato Bacterial spot',
                              'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
                              'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite',
                              'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
                              'Tomato healthy']
                result_index = model_prediction(test_image)
                predicted_class = class_name[result_index]
                st.success("Hasil dari Deteksi adalah {}".format(predicted_class))
                cause_data = disease_data.get(predicted_class, {"cause": "Informasi penyebab tidak ditemukan", "video": ""})
                st.success("Penyebab penyakit: {}".format(cause_data["cause"]))
                if cause_data["video"]:
                    st.markdown(f"### Video Penjelasan")
                    st.video(cause_data["video"])
        else:
            st.warning("Anda belum mengunggah gambar, silahkan unggah gambar.")
