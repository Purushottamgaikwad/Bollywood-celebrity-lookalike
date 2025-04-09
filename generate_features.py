import numpy as np
import pickle
import cv2
import os
from deepface import DeepFace
from tqdm import tqdm
import warnings

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def get_image_paths(dataset_path):
    """Collect all image paths from celebrity subfolders"""
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_features(image_paths):
    """Extract face embeddings using VGGFace with ResNet50 backbone"""
    print("Initializing VGGFace (ResNet50) model...")
    
    # Available DeepFace models: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib
    # Using VGGFace with ResNet50 backbone
    model_name = "ArcFace"
    
    features = []
    valid_paths = []
    failed_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Get face embedding
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend="retinaface",
                enforce_detection=True,
                align=True
            )[0]["embedding"]
            
            features.append(embedding)
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"Skipped {img_path}: {str(e)}")
            failed_count += 1
            continue
    
    print(f"\nFailed to process {failed_count} images")
    return np.array(features), valid_paths

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "data"  # Folder containing celebrity subfolders
    OUTPUT_FEATURES = "vggface_features.pkl"
    OUTPUT_FILENAMES = "vggface_filenames.pkl"
    
    # Verify paths
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    # Get image paths
    image_paths = get_image_paths(DATASET_PATH)
    
    if not image_paths:
        raise ValueError(f"No images found in {DATASET_PATH}")
    
    print(f"Found {len(image_paths)} images in dataset")
    
    # Extract features
    features, valid_paths = extract_features(image_paths)
    
    # Save results
    with open(OUTPUT_FEATURES, "wb") as f:
        pickle.dump(features, f)
    with open(OUTPUT_FILENAMES, "wb") as f:
        pickle.dump(valid_paths, f)
    
    print(f"\nSuccessfully processed {len(features)}/{len(image_paths)} images")
    print(f"Features saved to {OUTPUT_FEATURES}")
    print(f"Filenames saved to {OUTPUT_FILENAMES}")