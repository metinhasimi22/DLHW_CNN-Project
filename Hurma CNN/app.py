import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Hurma çeşitleri
hurma_cesidi = ['Galaxy', 'Rutab', 'Sugaey', 'Medjool', 'Nabtat Ali',
                'Ajwa', 'Sokari', 'Shaishe', 'Meneifi']

# Modeli yükleme
model = load_model('model92.h5')
model.summary()

def process_img(img):
    img = img.resize((170, 170), Image.LANCZOS)  # 170x170 piksel boyutuna dönüştürme
    img = np.array(img) / 255.0  # Normalize etme
    img = np.expand_dims(img, axis=0)  # Resme boyut ekleme
    return img

st.title("Hurma Çeşidi Sınıflandırması :date:")
st.write(
    "Bir hurma resmi seçin ve modelimiz, bu resmin hangi hurma çeşidinden olduğunu tahmin etsin. 🖼️📊\n"
    "Upload an image and the model will predict which date variety your image shows."
)

# Stil ayarları
st.markdown("""
<style>
    .reportview-container {
        background: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background: #E0E0E0;
    }
    .css-18e3th9 {
        font-size: 1.25em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

file = st.file_uploader("Resim Yükle & Bir resim seçiniz", type=['png', 'jpg', 'jpeg'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Yüklenen Resim", use_column_width=True)
    
    result = process_img(img)
    prediction = model.predict(result)
    prediction_class = np.argmax(prediction)  # En yüksek tahmin edilen sınıf

    # Sınıf isimleri
    result_text = hurma_cesidi[prediction_class]

    st.write(f"**Sonuç:** {result_text}")
else:
    st.write("Lütfen bir resim yükleyin.")
