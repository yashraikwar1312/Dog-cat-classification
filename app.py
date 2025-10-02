import os

UPLOAD_FOLDER = "uploads"

# Create folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
    }
    h1 {
        color: #ff4b5c;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg,#ff758c,#ff7eb3);
        color:white;
        font-weight:bold;
        padding:10px 20px;
        border-radius:10px;
        border:none;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("Upload an image and let AI tell whether it's a Cat or Dog!")

# ----------------------------
# CREATE UPLOADS FOLDER
# ----------------------------
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model('cat_dog_classifier.h5')
    return model

model = load_trained_model()

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Save uploaded image to uploads folder
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(Image.open(save_path), caption='Uploaded Image', use_column_width=True)

    # Preprocess for model
    IMG_SIZE = 224
    img_array = image.img_to_array(Image.open(save_path).resize((IMG_SIZE, IMG_SIZE)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # LOADING ANIMATION
    # ----------------------------
    with st.spinner('Predicting...'):
        time.sleep(1)
        pred_prob = model.predict(img_array)[0,0]

    # ----------------------------
    # DISPLAY RESULT
    # ----------------------------
    if pred_prob > 0.5:
        st.success(f"Prediction: Dog 🐶 ({pred_prob*100:.2f}% confidence)")
        st.balloons()
    else:
        st.success(f"Prediction: Cat 🐱 ({(1-pred_prob)*100:.2f}% confidence)")
        st.snow()
        import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

model = load_model(
    'cat_dog_classifier.h5',
    custom_objects={
        'Functional': tf.keras.Model,
        'Sequential': tf.keras.Sequential,
        'RandomFlip': layers.RandomFlip,
        'RandomRotation': layers.RandomRotation,
        'RandomZoom': layers.RandomZoom
    }
)

