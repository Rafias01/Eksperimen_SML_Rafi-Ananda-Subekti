import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

RAW_DATA_DIR = "../vehicle-classification_raw/train"
OUTPUT_DIR = "vehicle_preprocessing"

IMG_SIZE = (224, 224)
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return img

def load_dataset(data_dir):
    X, y = [], []

    classes = os.listdir(data_dir)
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                X.append(load_and_preprocess_image(img_path))
                y.append(cls)
            except Exception as e:
                print(f"Skip {img_path}: {e}")

    return np.array(X), np.array(y)

def main():
    print("Loading dataset...")
    X, y = load_dataset(RAW_DATA_DIR)

    print("Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print("Saving preprocessed data...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

    joblib.dump(encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()