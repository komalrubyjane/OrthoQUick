import os
import cv2
import numpy as np
import pandas as pd
import pickle
import shutil
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_1_FOLDER = 'datasets/FracAtlas'
DATASET_2_FOLDER = 'datasets/Dataset'
PREPROCESSED_FOLDER = 'datasets/preprocessed_combined'
ANNOTATION_FILE = os.path.join(PREPROCESSED_FOLDER, 'annotations.csv')
MODEL_FOLDER = 'model'

labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']
bone_info = {
    'Chest': {'bones': [{'name': 'Ribs', 'roi': (50, 50, 350, 350), 'description': 'Protects heart and lungs.'}, {'name': 'Sternum', 'roi': (180, 100, 220, 300), 'description': 'Breastbone in chest center.'}]},
    'Elbow': {'bones': [{'name': 'Humerus', 'roi': (50, 50, 150, 200), 'description': 'Upper arm bone.'}, {'name': 'Radius', 'roi': (150, 200, 250, 300), 'description': 'Forearm bone (thumb side).'}, {'name': 'Ulna', 'roi': (100, 200, 200, 350), 'description': 'Forearm bone (pinky side).'}]},
    'Finger': {'bones': [{'name': 'Phalanges', 'roi': (100, 100, 300, 300), 'description': 'Finger bones.'}]},
    'Hand': {'bones': [{'name': 'Metacarpals', 'roi': (100, 50, 300, 150), 'description': 'Palm bones.'}, {'name': 'Carpals', 'roi': (50, 150, 200, 250), 'description': 'Wrist bones.'}]},
    'Head': {'bones': [{'name': 'Skull', 'roi': (50, 50, 300, 300), 'description': 'Protects the brain.'}, {'name': 'Mandible', 'roi': (100, 300, 250, 350), 'description': 'Lower jawbone.'}]},
    'Sholder': {'bones': [{'name': 'Clavicle', 'roi': (50, 50, 350, 100), 'description': 'Collarbone.'}, {'name': 'Scapula', 'roi': (50, 100, 200, 300), 'description': 'Shoulder blade.'}]},
    'Wrist': {'bones': [{'name': 'Carpals', 'roi': (100, 100, 300, 300), 'description': 'Wrist bones.'}, {'name': 'Distal Radius', 'roi': (50, 50, 150, 200), 'description': 'Radius end near wrist.'}, {'name': 'Distal Ulna', 'roi': (150, 50, 250, 200), 'description': 'Ulna end near wrist.'}]}
}

# Initialize models
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
fracture_clf_bone = RandomForestClassifier(n_estimators=100, random_state=42)
fracture_clf_type = RandomForestClassifier(n_estimators=100, random_state=42)
fracture_reg_x = RandomForestRegressor(n_estimators=100, random_state=42)
fracture_reg_y = RandomForestRegressor(n_estimators=100, random_state=42)

# Preprocessing functions
def preprocess_fracatlas():
    logger.info("Preprocessing FracAtlas dataset...")
    if not os.path.exists(DATASET_1_FOLDER):
        logger.error(f"FracAtlas dataset not found in {DATASET_1_FOLDER}")
        return pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity'])

    dataset_csv_path = os.path.join(DATASET_1_FOLDER, 'dataset.csv')
    if not os.path.exists(dataset_csv_path):
        logger.error(f"dataset.csv not found in {DATASET_1_FOLDER}")
        return pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity'])

    df = pd.read_csv(dataset_csv_path)
    logger.info(f"Loaded dataset.csv with {len(df)} entries")
    df_fractured = df[df['fractured'] == 1].copy()
    logger.info(f"Filtered {len(df_fractured)} fractured images")

    def map_category(row):
        if row['hand'] == 1:
            return 'Wrist'
        elif row['shoulder'] == 1:
            return 'Shoulder'
        return None

    df_fractured['category'] = df_fractured.apply(map_category, axis=1)
    df_fractured = df_fractured[df_fractured['category'].notnull()]
    logger.info(f"After mapping, {len(df_fractured)} images remain in FracAtlas")

    def map_bone(category):
        if category == 'Wrist':
            return 'Distal Radius'
        elif category == 'Shoulder':
            return 'Clavicle'
        return 'Unknown'

    df_fractured['bone'] = df_fractured['category'].apply(map_bone)
    df_fractured['fracture_type'] = df_fractured['fracture_count'].apply(lambda x: 'Simple' if x == 1 else 'Comminuted')
    df_fractured['severity'] = df_fractured['fracture_type'].apply(lambda x: 50 if x == 'Simple' else 80)

    def estimate_coordinates(image_id):
        img_path = os.path.join(DATASET_1_FOLDER, 'images/Fractured', image_id)
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            return 200, 200
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            return 200, 200
        center_x, center_y = detect_fracture_coordinates(img)
        return center_x, center_y

    df_fractured[['fracture_x', 'fracture_y']] = df_fractured['image_id'].apply(
        lambda x: pd.Series(estimate_coordinates(x))
    )
    df_fractured = df_fractured[['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity']]
    df_fractured['source'] = 'FracAtlas'
    return df_fractured

def preprocess_dataset():
    logger.info("Preprocessing new Dataset...")
    if not os.path.exists(DATASET_2_FOLDER):
        logger.error(f"Dataset not found in {DATASET_2_FOLDER}")
        return pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity'])

    annotations = []
    for category in labels:
        category_path = os.path.join(DATASET_2_FOLDER, category)
        if not os.path.exists(category_path):
            logger.warning(f"Category folder {category} not found in Dataset")
            continue

        for filename in os.listdir(category_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            fracture_center, _, severity, fracture_type = detect_fracture(img_path)
            fracture_x, fracture_y = fracture_center if fracture_center else (200, 200)
            if fracture_type == "None":
                logger.warning(f"No fracture detected in {img_path}, skipping for training.")
                continue

            bone = identify_fractured_bone(category, fracture_center, image_path=img_path)
            if bone == "None":
                bones = bone_info[category]['bones']
                bone = bones[0]['name']
                logger.warning(f"Could not identify specific bone in {img_path}, defaulting to {bone}")

            annotations.append({
                'image_id': filename,
                'category': category,
                'bone': bone,
                'fracture_type': fracture_type,
                'fracture_x': fracture_x,
                'fracture_y': fracture_y,
                'severity': severity,
                'source': 'Dataset'
            })

    df_dataset = pd.DataFrame(annotations)
    logger.info(f"Processed {len(df_dataset)} images from Dataset")
    return df_dataset

def detect_fracture_coordinates(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy
        return 200, 200
    except Exception as e:
        logger.error(f"Error detecting fracture coordinates: {str(e)}")
        return 200, 200

def detect_fracture(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None, 0, "None"
        
        img = cv2.resize(img, (400, 400))
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(thresh, 20, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fracture_centers = []
        fracture_rois = []
        severities = []
        fracture_type = "None"
        
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > 3000:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2
                corner_size = 50
                if (center_x < corner_size or center_x > 400 - corner_size) or \
                   (center_y < corner_size or center_y > 400 - corner_size):
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:
                    continue
                
                fracture_centers.append((center_x, center_y))
                fracture_rois.append((x, y, x + w, y + h))
                severity = min(int(area / 30), 100)
                severities.append(severity)
        
        if not fracture_centers:
            return None, None, 0, "None"
        
        max_severity_idx = np.argmax(severities)
        fracture_type = "Comminuted" if severities[max_severity_idx] > 70 else "Simple"
        return fracture_centers[max_severity_idx], fracture_rois[max_severity_idx], severities[max_severity_idx], fracture_type
    except Exception as e:
        logger.error(f"Fracture detection failed: {str(e)}")
        return None, None, 0, "None"

def identify_fractured_bone(category, fracture_center, image_path=None):
    if not fracture_center:
        return "None"
    
    fx, fy = fracture_center
    bones = bone_info[category]['bones']
    
    for bone in bones:
        x1, y1, x2, y2 = bone['roi']
        if x1 - 30 <= fx <= x2 + 30 and y1 - 30 <= fy <= y2 + 30:
            logger.debug(f"Fracture at {fx}, {fy} in {bone['name']}")
            return bone['name']
    
    min_distance = float('inf')
    nearest_bone = "None"
    for bone in bones:
        x1, y1, x2, y2 = bone['roi']
        bone_center_x = (x1 + x2) / 2
        bone_center_y = (y1 + y2) / 2
        distance = np.sqrt((fx - bone_center_x) ** 2 + (fy - bone_center_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_bone = bone['name']
    
    return nearest_bone

def combine_and_save_datasets():
    df_fracatlas = preprocess_fracatlas()
    df_dataset = preprocess_dataset()
    if df_fracatlas.empty and df_dataset.empty:
        logger.error("No data available from either dataset after preprocessing")
        return pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity', 'source'])

    df_combined = pd.concat([df_fracatlas, df_dataset], ignore_index=True)
    logger.info(f"Combined {len(df_combined)} annotations from both datasets")

    simple_folder = os.path.join(PREPROCESSED_FOLDER, 'Simple Bone Fracture')
    comminuted_folder = os.path.join(PREPROCESSED_FOLDER, 'Comminuted Bone Fracture')
    os.makedirs(simple_folder, exist_ok=True)
    os.makedirs(comminuted_folder, exist_ok=True)

    for _, row in df_combined.iterrows():
        image_id = row['image_id']
        fracture_type = row['fracture_type']
        source = row['source']
        if source == 'FracAtlas':
            src_path = os.path.join(DATASET_1_FOLDER, 'images/Fractured', image_id)
        else:
            src_path = os.path.join(DATASET_2_FOLDER, row['category'], image_id)
        dest_folder = simple_folder if fracture_type == 'Simple' else comminuted_folder
        dest_path = os.path.join(dest_folder, f"{source}_{image_id}")
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            logger.warning(f"Image not found for copying: {src_path}")

    df_combined.to_csv(ANNOTATION_FILE, index=False)
    logger.info(f"Saved combined annotations to {ANNOTATION_FILE}")
    return df_combined

def train_bone_classifier(dataset_annotations):
    X, y = [], []
    for _, ann in dataset_annotations.iterrows():
        folder = "Comminuted Bone Fracture" if ann["fracture_type"] == "Comminuted" else "Simple Bone Fracture"
        img_path = os.path.join(PREPROCESSED_FOLDER, folder, f"{ann['source']}_{ann['image_id']}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load image for training: {img_path}")
            continue
        img_resized = cv2.resize(img, (32, 32))
        X.append(img_resized.ravel() / 255)
        y.append(ann["category"])
    
    if not X:
        logger.error("No valid images for bone classifier training")
        return False
    
    classifier.fit(X, y)
    with open(os.path.join(MODEL_FOLDER, "model.pkl"), "wb") as f:
        pickle.dump(classifier, f)
    logger.info("Bone classifier trained and saved.")
    return True

def train_fracture_model(dataset_annotations):
    X_train, y_bone, y_type, y_x, y_y = [], [], [], [], []
    for _, ann in dataset_annotations.iterrows():
        folder = "Comminuted Bone Fracture" if ann["fracture_type"] == "Comminuted" else "Simple Bone Fracture"
        img_path = os.path.join(PREPROCESSED_FOLDER, folder, f"{ann['source']}_{ann['image_id']}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load image for training: {img_path}")
            continue
        img_resized = cv2.resize(img, (32, 32))
        X_train.append(img_resized.ravel() / 255)
        y_bone.append(ann["bone"])
        y_type.append(ann["fracture_type"])
        y_x.append(ann["fracture_x"])
        y_y.append(ann["fracture_y"])
    
    if not X_train:
        logger.error("No valid images for fracture model training.")
        return False
    
    fracture_clf_bone.fit(X_train, y_bone)
    fracture_clf_type.fit(X_train, y_type)
    fracture_reg_x.fit(X_train, y_x)
    fracture_reg_y.fit(X_train, y_y)
    
    with open(os.path.join(MODEL_FOLDER, "fracture_clf_bone.pkl"), "wb") as f:
        pickle.dump(fracture_clf_bone, f)
    with open(os.path.join(MODEL_FOLDER, "fracture_clf_type.pkl"), "wb") as f:
        pickle.dump(fracture_clf_type, f)
    with open(os.path.join(MODEL_FOLDER, "fracture_reg_x.pkl"), "wb") as f:
        pickle.dump(fracture_reg_x, f)
    with open(os.path.join(MODEL_FOLDER, "fracture_reg_y.pkl"), "wb") as f:
        pickle.dump(fracture_reg_y, f)
    logger.info("Fracture models trained and saved.")
    return True

def main():
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

    dataset_annotations = combine_and_save_datasets()
    if dataset_annotations.empty:
        logger.error("Failed to preprocess datasets.")
        return

    if not train_bone_classifier(dataset_annotations):
        logger.error("Bone classifier training failed.")
        return

    if not train_fracture_model(dataset_annotations):
        logger.error("Fracture model training failed.")
        return

    logger.info("Training completed successfully.")

if __name__ == '__main__':
    main()