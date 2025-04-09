import streamlit as st
import numpy as np
import pickle
import os
from deepface import DeepFace
from PIL import Image
import cv2

# Load features and filenames
@st.cache_resource
def load_embeddings():
    with open("vggface_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("vggface_filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
    return features, filenames

# Preprocess and get embedding from uploaded image
def get_embedding_from_image(image):
    try:
        # Save temporary image
        temp_path = "temp.jpg"
        image.save(temp_path)

        # Extract embedding
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True
        )[0]["embedding"]

        return embedding
    except Exception as e:
        st.error(f"❌ Error detecting face: {e}")
        return None

# Find best match using cosine similarity
def find_best_match(user_embedding, features, filenames, top_k=3):
    distances = np.linalg.norm(features - user_embedding, axis=1)
    top_indices = distances.argsort()[:top_k]
    return [(filenames[i], distances[i]) for i in top_indices]

# Streamlit UI
def main():
    st.set_page_config(page_title="Bollywood Lookalike Finder", layout="centered")
    st.title("🎬 Bollywood Celebrity Lookalike Finder")
    st.markdown("Upload your image and find which Bollywood celebrity you resemble!")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        with st.spinner("Analyzing image..."):
            user_embedding = get_embedding_from_image(image)

        if user_embedding is not None:
            features, filenames = load_embeddings()
            matches = find_best_match(user_embedding, features, filenames)

            st.success("Here are your top matches:")

            # Create columns: one for uploaded image, others for matches
            cols = st.columns(len(matches) + 1)
            cols[0].image(image, caption="Your Image", use_container_width=True)

            for i, (match_path, distance) in enumerate(matches):
                celeb_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
                cols[i+1].image(match_path, caption=f"{celeb_name}\nDist: {distance:.2f}", use_container_width=True)

if __name__ == "__main__":
    main()
