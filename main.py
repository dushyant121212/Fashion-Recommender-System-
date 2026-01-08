
import streamlit as st
import tensorflow
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
import tensorflow as tf

# Apply custom CSS for a better UI
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #ff5a5f;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            color: white;
            padding: 10px;
            background-color: #333;
            margin-top: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load precomputed image features and file paths
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

# Load ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Title
st.markdown("<h1 class='title'>👓Fashion Recommender System</h1>", unsafe_allow_html=True)

# Sidebar Instructions
st.sidebar.title("📌 Instructions")
st.sidebar.info(
    "1. Upload an image of fashion apparel.\n"
    "2. The system will recommend similar items.\n"
    "3. Click on the suggested images to explore more."
    "⚓ Made By DUSHYANT VERMA(2022ITB055), RAKHI SAREN(2022ITB010), SAIDEEP LAMA(2022ITB002)."   
)

# Function to save uploaded file
def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract image features
def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

# Function to find nearest neighbors
def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Upload file
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if save_file(uploaded_file):
        # Display uploaded image with improved clarity
        show_image = Image.open(uploaded_file)
        st.image(show_image, caption="📌 Uploaded Image", use_container_width=True)

        # Extract features
        features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
        img_indices = recommend(features, features_list)

        # Display recommendations
        st.subheader("🎯 Recommended Items")
        cols = st.columns(5)
        captions = ["✨ Recommended 1", "🔥 Recommended 2", "🌟 Recommended 3", "💖 Recommended 4", "💎 Recommended 5"]

        for i in range(5):
            with cols[i]:
                st.image(img_files_list[img_indices[0][i]], use_container_width=True)
                st.caption(captions[i])
    else:
        st.error("⚠️ An error occurred while uploading the image. Please try again.")

# Footer
st.markdown(
    """
    <div class='footer'>
        💡 IIEST, Shibpur | 📧 E-Mail- iiest24111856howrah@gmail.com 
    </div>
    """,
    unsafe_allow_html=True
)
