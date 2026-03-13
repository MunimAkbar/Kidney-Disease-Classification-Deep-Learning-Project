"""
One-time script: Compute mean feature vector from training CT scan images.

This builds a reference for OOD (out-of-distribution) detection.
Run once after training:
    .\env\python.exe compute_features.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from pathlib import Path

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    print("=" * 60)
    print("  Computing Feature Reference for OOD Detection")
    print("=" * 60)

    # 1. Load the trained model
    model_path = os.path.join("model", "model.h5")
    print(f"\n[1/4] Loading model from {model_path}...")
    model = load_model(model_path)
    model.summary()

    # 2. Create feature extractor from the flatten layer (penultimate)
    # The model structure is: VGG16 base → Flatten → Dense(2, softmax)
    # We want features from the Flatten layer (before the final Dense)
    print("\n[2/4] Building feature extractor...")
    
    # Find the flatten layer
    flatten_layer = None
    for layer in model.layers:
        if 'flatten' in layer.name.lower():
            flatten_layer = layer
            break
    
    if flatten_layer is None:
        # Fallback: use the second-to-last layer
        flatten_layer = model.layers[-2]
        print(f"  > Using layer: {flatten_layer.name} (fallback)")
    else:
        print(f"  > Using layer: {flatten_layer.name}")

    feature_extractor = Model(
        inputs=model.input,
        outputs=flatten_layer.output
    )
    
    # Quick test to see feature dimension
    dummy = np.zeros((1, 224, 224, 3))
    feature_dim = feature_extractor.predict(dummy, verbose=0).shape[1]
    print(f"  > Feature dimension: {feature_dim}")

    # 3. Extract features from all training images
    print("\n[3/4] Extracting features from training images...")
    data_dir = os.path.join("artifacts", "data_ingestion", "kidney-ct-scan-image")
    
    all_features = []
    total_images = 0
    
    for class_name in ["Normal", "Tumor"]:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  [!] Directory not found: {class_dir}")
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  > Processing {len(image_files)} images from '{class_name}'...")
        
        batch_images = []
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                batch_images.append(img_array)
            except Exception as e:
                print(f"    [X] Error loading {img_file}: {e}")
                continue
            
            # Process in batches of 32
            if len(batch_images) == 32 or i == len(image_files) - 1:
                batch = np.array(batch_images)
                features = feature_extractor.predict(batch, verbose=0)
                all_features.append(features)
                total_images += len(batch_images)
                batch_images = []
                
                # Progress
                if (i + 1) % 100 == 0 or i == len(image_files) - 1:
                    print(f"    Processed {i + 1}/{len(image_files)}")
    
    if total_images == 0:
        print("\n[X] No images found! Check the data directory.")
        return
    
    # Stack all features
    all_features = np.vstack(all_features)
    print(f"\n  Total images processed: {total_images}")
    print(f"  Feature matrix shape: {all_features.shape}")

    # 4. Compute mean and distance threshold
    print("\n[4/4] Computing reference statistics...")
    
    # Mean feature vector
    feature_mean = np.mean(all_features, axis=0)
    
    # Compute cosine distances of all training images from the mean
    # Cosine distance = 1 - cosine_similarity
    def cosine_distance(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))
    
    distances = np.array([cosine_distance(f, feature_mean) for f in all_features])
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Threshold = mean + 3 * std  (covers ~99.7% of training data)
    threshold = mean_dist + 3 * std_dist
    
    print(f"  Mean cosine distance:  {mean_dist:.6f}")
    print(f"  Std cosine distance:   {std_dist:.6f}")
    print(f"  OOD threshold (mean+3*std): {threshold:.6f}")
    
    # Save to model directory
    os.makedirs("model", exist_ok=True)
    np.save(os.path.join("model", "feature_mean.npy"), feature_mean)
    np.save(os.path.join("model", "ood_threshold.npy"), np.array([threshold]))
    
    print(f"\n[OK] Saved: model/feature_mean.npy  ({feature_mean.shape})")
    print(f"[OK] Saved: model/ood_threshold.npy  (threshold={threshold:.6f})")
    print("\n" + "=" * 60)
    print("  Done! OOD detection reference is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
