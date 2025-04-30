from flask import Flask, request, render_template, jsonify, send_file
from flask_compress import Compress
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from werkzeug.utils import secure_filename
import pickle
from PIL import Image
import logging
import json
from datetime import datetime
import uuid
import csv
import io
import sys
import pandas as pd
import time
import importlib.metadata

# Set up logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')  # Log to a file for debugging
    ]
)
logger = logging.getLogger(__name__)

# Log dependency versions
logger.info(f"Python version: {sys.version}")
try:
    logger.info(f"Flask version: {importlib.metadata.version('flask')}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Scikit-learn version: {importlib.metadata.version('scikit-learn')}")
    logger.info(f"Pillow version: {Image.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
except ImportError as e:
    logger.error(f"Missing dependency: {str(e)}")
    sys.exit(1)  # Exit if dependencies are missing

app = Flask(__name__)
Compress(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
MODEL_FOLDER = 'model'
DATASET_FOLDER = r'C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal'
ANNOTATION_FILE = os.path.normpath(os.path.join(DATASET_FOLDER, 'annotations.csv'))
HISTORY_FILE = 'history.json'
IMPROVEMENT_FILE = 'improvement.json'
FEEDBACK_FILE = 'feedback.json'

# Ensure directories exist with proper permissions
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o755)
        logger.info(f"Ensured directory exists: {folder}")
    except Exception as e:
        logger.error(f"Failed to create directory {folder}: {str(e)}")
        sys.exit(1)  # Exit if directory creation fails

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Labels and bone information
labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']
bone_info = {
    'Chest': {'bones': [
        {'name': 'Ribs', 'roi': (50, 50, 350, 350), 'description': 'Protects heart and lungs.'},
        {'name': 'Sternum', 'roi': (180, 100, 220, 300), 'description': 'Breastbone in chest center.'}
    ]},
    'Elbow': {'bones': [
        {'name': 'Humerus', 'roi': (50, 50, 150, 200), 'description': 'Upper arm bone.'},
        {'name': 'Radius', 'roi': (150, 200, 250, 300), 'description': 'Forearm bone (thumb side).'},
        {'name': 'Ulna', 'roi': (100, 200, 200, 350), 'description': 'Forearm bone (pinky side).'}
    ]},
    'Finger': {'bones': [
        {'name': 'Phalanges', 'roi': (100, 100, 300, 300), 'description': 'Finger bones.'}
    ]},
    'Hand': {'bones': [
        {'name': 'Metacarpals', 'roi': (100, 50, 300, 150), 'description': 'Palm bones.'},
        {'name': 'Carpals', 'roi': (50, 150, 200, 250), 'description': 'Wrist bones.'}
    ]},
    'Head': {'bones': [
        {'name': 'Skull', 'roi': (50, 50, 300, 300), 'description': 'Protects the brain.'},
        {'name': 'Mandible', 'roi': (100, 300, 250, 350), 'description': 'Lower jawbone.'}
    ]},
    'Shoulder': {'bones': [
        {'name': 'Clavicle', 'roi': (50, 50, 350, 100), 'description': 'Collarbone.'},
        {'name': 'Scapula', 'roi': (50, 100, 200, 300), 'description': 'Shoulder blade.'}
    ]},
    'Wrist': {'bones': [
        {'name': 'Carpals', 'roi': (100, 100, 300, 300), 'description': 'Wrist bones.'},
        {'name': 'Distal Radius', 'roi': (50, 50, 150, 200), 'description': 'Radius end near wrist.'},
        {'name': 'Distal Ulna', 'roi': (150, 50, 250, 200), 'description': 'Ulna end near wrist.'}
    ]}
}

# Global variables
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
model_trained = False
fracture_clf_bone = RandomForestClassifier(n_estimators=100, random_state=42)
fracture_clf_type = RandomForestClassifier(n_estimators=100, random_state=42)
fracture_reg_x = RandomForestRegressor(n_estimators=100, random_state=42)
fracture_reg_y = RandomForestRegressor(n_estimators=100, random_state=42)
fracture_model_trained = False
history = []
improvement_data = {}
feedback_data = []
CONFIDENCE_THRESHOLD = 60.0
dataset_annotations = None

def load_dataset_annotations():
    """Load or create dataset annotations for training."""
    global dataset_annotations
    simple_folder = os.path.normpath(os.path.join(DATASET_FOLDER, 'Simple Bone Fracture'))
    comminuted_folder = os.path.normpath(os.path.join(DATASET_FOLDER, 'Comminuted Bone Fracture'))
    annotations = []
    
    try:
        if os.path.exists(ANNOTATION_FILE):
            dataset_annotations = pd.read_csv(ANNOTATION_FILE)
            logger.info(f"Loaded dataset annotations from {ANNOTATION_FILE}")
            return True
        
        # Create mock annotations if the annotation file doesn't exist
        for folder, fracture_type in [(simple_folder, 'Simple'), (comminuted_folder, 'Comminuted')]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder, filename)
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.warning(f"Failed to load image: {img_path}")
                            continue
                        fracture_x, fracture_y = detect_fracture_coordinates(img)
                        annotations.append({
                            'image_id': filename,
                            'category': 'Wrist',  # Default to Wrist as per dataset context
                            'bone': 'Distal Radius',
                            'fracture_type': fracture_type,
                            'fracture_x': fracture_x,
                            'fracture_y': fracture_y,
                            'severity': 50 if fracture_type == 'Simple' else 80
                        })
        
        dataset_annotations = pd.DataFrame(annotations)
        logger.info(f"Created {len(dataset_annotations)} mock annotations from dataset folders")
        if dataset_annotations.empty:
            logger.warning("No images found in dataset folders")
            dataset_annotations = pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity'])
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to load dataset annotations: {str(e)}")
        dataset_annotations = pd.DataFrame(columns=['image_id', 'category', 'bone', 'fracture_type', 'fracture_x', 'fracture_y', 'severity'])
        return False

def safe_remove_file(file_path):
    """Safely remove a file if it exists."""
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {str(e)}")

def load_history():
    """Load history data from file."""
    global history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    history = [entry for entry in data if isinstance(entry, dict) and 'id' in entry]
                    if len(history) != len(data):
                        logger.warning("Some history entries were invalid and skipped")
                else:
                    logger.warning("Invalid history.json format, initializing empty history")
                    history = []
        except Exception as e:
            logger.error(f"Failed to load history: {str(e)}")
            history = []
    return history

def save_history():
    """Save history data to file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save history: {str(e)}")

def load_improvement_data():
    """Load improvement data from file."""
    global improvement_data
    if os.path.exists(IMPROVEMENT_FILE):
        try:
            with open(IMPROVEMENT_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    improvement_data = data
                else:
                    logger.warning("Invalid improvement.json format")
                    improvement_data = {}
        except Exception as e:
            logger.error(f"Failed to load improvement data: {str(e)}")
            improvement_data = {}
    return improvement_data

def save_improvement_data():
    """Save improvement data to file."""
    try:
        with open(IMPROVEMENT_FILE, 'w') as f:
            json.dump(improvement_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save improvement data: {str(e)}")

def load_feedback_data():
    """Load feedback data from file."""
    global feedback_data
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    feedback_data = data
                else:
                    logger.warning("Invalid feedback.json format")
                    feedback_data = []
        except Exception as e:
            logger.error(f"Failed to load feedback data: {str(e)}")
            feedback_data = []
    return feedback_data

def save_feedback_data():
    """Save feedback data to file."""
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save feedback data: {str(e)}")

def load_model():
    """Load pre-trained models or train new ones if necessary."""
    global classifier, model_trained, fracture_clf_bone, fracture_clf_type, fracture_reg_x, fracture_reg_y, fracture_model_trained
    model_path = os.path.normpath(os.path.join(MODEL_FOLDER, 'model.pkl'))
    fracture_model_path = os.path.normpath(os.path.join(MODEL_FOLDER, 'fracture_model.pkl'))
    log = []
    
    try:
        # Ensure dataset annotations are loaded
        if dataset_annotations is None:
            logger.info("Loading dataset annotations before model load...")
            if not load_dataset_annotations():
                log.append("Failed to load dataset annotations")
        
        # Load bone classifier model
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                with open(model_path, 'rb') as file:
                    classifier = pickle.load(file)
                if not hasattr(classifier, 'predict') or not hasattr(classifier, 'predict_proba'):
                    raise ValueError("Loaded model is not a valid classifier")
                log.append("Loaded pre-trained classifier model")
                model_trained = True
            except Exception as e:
                logger.error(f"Failed to load classifier model from {model_path}: {str(e)}")
                log.append(f"Error loading classifier model: {str(e)}")
                model_trained = False
        else:
            log.append(f"Classifier model file not found or empty at {model_path}. Training new model.")
            model_trained = train_bone_classifier()
        
        # Load fracture detection models
        if os.path.exists(fracture_model_path) and os.path.getsize(fracture_model_path) > 0:
            try:
                with open(fracture_model_path, 'rb') as f:
                    fracture_clf_bone = pickle.load(f)
                    fracture_clf_type = pickle.load(f)
                    fracture_reg_x = pickle.load(f)
                    fracture_reg_y = pickle.load(f)
                log.append("Loaded pre-trained fracture models")
                fracture_model_trained = True
            except Exception as e:
                logger.error(f"Failed to load fracture models from {fracture_model_path}: {str(e)}")
                log.append(f"Error loading fracture models: {str(e)}")
                fracture_model_trained = train_fracture_model()
        else:
            log.append(f"Fracture model file not found or empty at {fracture_model_path}. Training new fracture models.")
            fracture_model_trained = train_fracture_model()
        
        return model_trained and fracture_model_trained, log
    except Exception as e:
        logger.error(f"Unexpected error in load_model: {str(e)}")
        log.append(f"Unexpected error: {str(e)}")
        return False, log

def train_bone_classifier():
    """Train the bone classifier using dataset annotations."""
    global dataset_annotations, classifier
    if dataset_annotations is None or dataset_annotations.empty:
        logger.error("No annotations for bone classifier training")
        return False
    
    X, y = [], []
    for _, ann in dataset_annotations.iterrows():
        folder = "Comminuted Bone Fracture" if ann["fracture_type"] == "Comminuted" else "Simple Bone Fracture"
        img_path = os.path.join(DATASET_FOLDER, folder, ann["image_id"])
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
    try:
        with open(os.path.join(MODEL_FOLDER, "model.pkl"), "wb") as f:
            pickle.dump(classifier, f)
        logger.info("Bone classifier trained and saved.")
        return True
    except Exception as e:
        logger.error(f"Failed to save bone classifier model: {str(e)}")
        return False

def detect_fracture_coordinates(img):
    """Detect fracture coordinates using contour analysis."""
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
        return 200, 200  # Default center if no contours found
    except Exception as e:
        logger.error(f"Error detecting fracture coordinates: {str(e)}")
        return 200, 200

def train_fracture_model():
    """Train the fracture detection models using dataset annotations."""
    global dataset_annotations, fracture_clf_bone, fracture_clf_type, fracture_reg_x, fracture_reg_y
    if dataset_annotations is None or dataset_annotations.empty:
        logger.error("No valid annotations available for training fracture models.")
        return False
    
    X_train, y_bone, y_type, y_x, y_y = [], [], [], [], []
    for _, ann in dataset_annotations.iterrows():
        folder = "Comminuted Bone Fracture" if ann["fracture_type"] == "Comminuted" else "Simple Bone Fracture"
        img_path = os.path.join(DATASET_FOLDER, folder, ann["image_id"])
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
    
    try:
        with open(os.path.join(MODEL_FOLDER, "fracture_model.pkl"), "wb") as f:
            pickle.dump(fracture_clf_bone, f)
            pickle.dump(fracture_clf_type, f)
            pickle.dump(fracture_reg_x, f)
            pickle.dump(fracture_reg_y, f)
        logger.info("Fracture models trained and saved.")
        return True
    except Exception as e:
        logger.error(f"Failed to save fracture models: {str(e)}")
        return False

def is_xray_image(image_path):
    """Validate if the image is a valid X-ray with relaxed criteria."""
    try:
        with Image.open(image_path) as img:
            if img.format not in ['PNG', 'JPEG']:
                logger.debug(f"Unsupported image format: {img.format}")
                return False
            if img.size[0] < 10 or img.size[1] < 10:
                logger.debug(f"Image too small: {image_path}")
                return False
            img = img.convert('RGB')
            img_array = np.array(img)
        
        img = cv2.imread(image_path)
        if img is None:
            logger.debug("OpenCV failed, using PIL image")
            img = img_array
        
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            color_diff = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(r - b))
            if color_diff > 15:
                logger.debug("Image has too much color variation")
                return False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        if edge_density < 0.005:
            logger.debug("Image lacks edge density for X-ray")
            return False
        
        return True
    except Exception as e:
        logger.error(f"X-ray validation failed: {str(e)}")
        return False

def enhance_image(image):
    """Enhance image for better fracture detection."""
    try:
        img = cv2.GaussianBlur(image, (3, 3), 0)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return img
    except Exception as e:
        logger.error(f"Image enhancement failed: {str(e)}")
        return image

def remove_text(image, image_size=(400, 400)):
    """Remove text from corners of the image."""
    try:
        img = cv2.resize(image, image_size)
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        corner_size = 50
        mask[0:corner_size, 0:corner_size] = 0
        mask[0:corner_size, -corner_size:] = 0
        mask[-corner_size:, 0:corner_size] = 0
        mask[-corner_size:, -corner_size:] = 0
        img_cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return img_cleaned
    except Exception as e:
        logger.error(f"Text removal failed: {str(e)}")
        return image

def detect_fracture(image_path):
    """Detect fractures in the X-ray image using multiple methods."""
    if not is_xray_image(image_path):
        logger.debug("Not a valid X-ray")
        return None, None, 0, "None"
    
    try:
        logger.debug(f"Detecting fracture: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None, 0, "None"
        
        img = cv2.resize(img, (400, 400))
        img_cleaned = remove_text(img)
        img_enhanced = enhance_image(img_cleaned)
        
        # Check dataset annotations for ground truth
        image_name = os.path.basename(image_path)
        if dataset_annotations is not None and not dataset_annotations.empty:
            annotation = dataset_annotations[dataset_annotations['image_id'] == image_name]
            if not annotation.empty:
                logger.debug(f"Found annotation for {image_name}")
                fracture_center = (int(annotation['fracture_x'].iloc[0]), int(annotation['fracture_y'].iloc[0]))
                fracture_roi = (fracture_center[0]-50, fracture_center[1]-50, fracture_center[0]+50, fracture_center[1]+50)
                severity = int(annotation['severity'].iloc[0])
                fracture_type = annotation['fracture_type'].iloc[0]
                return fracture_center, fracture_roi, severity, fracture_type
        
        # ML-based fracture detection if models are trained
        if fracture_model_trained:
            img_resized = cv2.resize(img_enhanced, (32, 32))
            features = img_resized.ravel() / 255
            features = features.reshape(1, -1)  # Ensure correct shape
            try:
                fracture_type = fracture_clf_type.predict(features)[0]
                fracture_x = fracture_reg_x.predict(features)[0]
                fracture_y = fracture_reg_y.predict(features)[0]
            except Exception as e:
                logger.error(f"ML prediction failed: {str(e)}")
                return None, None, 0, "None"
            fracture_center = (int(fracture_x), int(fracture_y))
            fracture_roi = (fracture_center[0]-50, fracture_center[1]-50, fracture_center[0]+50, fracture_center[1]+50)
            severity = 80 if fracture_type == "Comminuted" else 50
            # Validate ML prediction with contour analysis
            thresh = cv2.adaptiveThreshold(img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            edges = cv2.Canny(thresh, 20, 80)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.debug("ML predicted fracture, but no contours found. Assuming no fracture.")
                return None, None, 0, "None"
            return fracture_center, fracture_roi, severity, fracture_type
        
        # Contour-based detection with stricter thresholds
        thresh = cv2.adaptiveThreshold(img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
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
                logger.debug(f"Fracture at {center_x}, {center_y}, severity {severity}, area {area}")
        
        if not fracture_centers:
            logger.debug("No valid fractures found")
            return None, None, 0, "None"
        
        max_severity_idx = np.argmax(severities)
        fracture_type = "Comminuted" if severities[max_severity_idx] > 70 else "Simple"
        return fracture_centers[max_severity_idx], fracture_rois[max_severity_idx], severities[max_severity_idx], fracture_type
    except Exception as e:
        logger.error(f"Fracture detection failed: {str(e)}")
        return None, None, 0, "None"

def grade_severity(severity):
    """Grade the severity of the fracture."""
    if severity < 30:
        return "Mild"
    elif severity < 70:
        return "Moderate"
    else:
        return "Severe"

def calculate_bone_health(fracture_detected, severity, image_clarity):
    """Calculate bone health score."""
    base_score = 100
    if fracture_detected:
        base_score -= severity
    base_score = max(0, base_score - (100 - image_clarity))
    return base_score

def identify_fractured_bone(category, fracture_center, image_path=None):
    """Identify which bone is fractured based on fracture coordinates, ensuring it matches the category."""
    if not fracture_center:
        return "None"
    
    fx, fy = fracture_center
    bones = bone_info[category]['bones']
    
    # Check if fracture coordinates fall within any bone's ROI
    for bone in bones:
        x1, y1, x2, y2 = bone['roi']
        if x1 - 30 <= fx <= x2 + 30 and y1 - 30 <= fy <= y2 + 30:
            logger.debug(f"Fracture at {fx}, {fy} in {bone['name']}")
            return bone['name']
    
    # If ML model is trained, use it to predict the bone, but ensure it's in the category
    if fracture_model_trained and image_path:
        try:
            img_ml = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_ml is None:
                logger.warning(f"Failed to load image for ML bone prediction: {image_path}")
                return "None"
            img_ml_resized = cv2.resize(img_ml, (32, 32))
            features = img_ml_resized.ravel() / 255
            features = features.reshape(1, -1)  # Ensure correct shape
            predicted_bone = fracture_clf_bone.predict(features)[0]
            # Verify the predicted bone is in the category's bone list
            if any(bone['name'] == predicted_bone for bone in bones):
                return predicted_bone
        except Exception as e:
            logger.error(f"ML bone prediction failed: {str(e)}")
            return "None"
    
    # Fallback: Find the nearest bone, but only if it's in the category
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
    
    logger.debug(f"Fracture at {fx}, {fy} assigned to {nearest_bone} (distance: {min_distance})")
    return nearest_bone

def generate_heatmap(image, fracture_center, severity):
    """Generate a heatmap around the fracture location."""
    try:
        if severity <= 0:
            return np.zeros_like(image)
        heatmap = np.zeros_like(image, dtype=np.float32)
        if fracture_center:
            fx, fy = fracture_center
            fx = max(0, min(fx, image.shape[1] - 1))
            fy = max(0, min(fy, image.shape[0] - 1))
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    distance = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
                    heatmap[y, x] = np.exp(-distance / (severity + 1)) * 255
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    except Exception as e:
        logger.error(f"Heatmap generation failed: {str(e)}")
        return np.zeros_like(image)

def mark_fracture(image_path, fracture_center, output_path, severity, category, fracture_type):
    """Mark the fracture on the image and save the annotated version."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        img = cv2.resize(img, (400, 400))
        
        heatmap = generate_heatmap(img, fracture_center, severity)
        alpha = 0.4
        img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0.0)
        
        if fracture_center:
            fx, fy = fracture_center
            fx = max(0, min(fx, img.shape[1] - 1))
            fy = max(0, min(fy, img.shape[0] - 1))
            cv2.circle(img, (fx, fy), 10, (0, 0, 255), 2)
            cv2.rectangle(img, (fx - 50, fy - 50), (fx + 50, fy + 50), (0, 255, 0), 2)
            cv2.putText(img, f"Fracture: {fracture_type}", (fx - 50, fy - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"({fx}, {fy})", (fx - 50, fy + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        bones = bone_info[category]['bones']
        for bone in bones:
            x1, y1, x2, y2 = bone['roi']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            label_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
            cv2.putText(img, bone['name'], label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite(output_path, img)
        logger.debug(f"Fracture marked, saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Fracture marking failed: {str(e)}")
        return False

def generate_image_description(category, fracture_detected, fractured_bone, severity_grade, bone_health, fracture_type, bones_present, fracture_center):
    """Generate a textual description of the X-ray analysis."""
    description = f"This X-ray is a {category} radiograph. "
    description += f"Bones present: {', '.join(bones_present)}. "
    if fracture_detected:
        fx, fy = fracture_center
        description += f"A {fracture_type} fracture is detected in the {fractured_bone} at coordinates ({fx}, {fy}), with {severity_grade.lower()} severity. "
    else:
        description += "No fracture detected. "
    description += f"Bone health score: {bone_health:.1f}/100."
    return description

def assess_healing(initial_severity, current_severity):
    """Assess the healing progress between two X-rays."""
    if initial_severity is None or current_severity is None or initial_severity == 0:
        return 0, "Unknown"
    
    improvement_percentage = max(0, min(100, ((initial_severity - current_severity) / initial_severity) * 100))
    
    if current_severity == 0:
        healing_stage = "Healed"
    elif current_severity < 30:
        healing_stage = "Advanced"
    elif current_severity < 70:
        healing_stage = "Intermediate"
    else:
        healing_stage = "Early"
    
    return improvement_percentage, healing_stage

# Initialize dataset annotations and models
load_dataset_annotations()
success, init_log = load_model()
load_history()
load_improvement_data()
load_feedback_data()

@app.route('/')
def index():
    """Render the index page."""
    logger.debug("Serving index.html")
    try:
        return render_template('index.html', init_log=init_log, model_trained=model_trained)
    except Exception as e:
        logger.error(f"Failed to render index.html: {str(e)}")
        return jsonify({'error': f'Failed to load index page: {str(e)}', 'status': 'error'}), 500

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    """Handle single X-ray analysis with improved error handling."""
    logger.debug("Serving analysis page")
    if request.method == 'POST':
        if not model_trained:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded. Ensure model.pkl exists.', 'status': 'error'}), 500
        
        if 'image' not in request.files:
            logger.error("No image uploaded")
            return jsonify({'error': 'No image uploaded. Select a PNG/JPEG X-ray.', 'status': 'error'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected.', 'status': 'error'}), 400
        
        timestamp = int(time.time())
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.error("Unsupported image format")
            return jsonify({'error': 'Only PNG/JPEG supported.', 'status': 'error'}), 400
        
        file_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        try:
            file.save(file_path)
            logger.debug(f"Image saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return jsonify({'error': f'Failed to save image: {str(e)}', 'status': 'error'}), 500
        
        try:
            if not is_xray_image(file_path):
                logger.error("Not a valid X-ray")
                return jsonify({'error': 'Not a valid X-ray. Upload a bone X-ray.', 'status': 'error'}), 400
            
            # Detect fracture
            fracture_center, fracture_roi, severity, fracture_type = detect_fracture(file_path)
            severity_grade = grade_severity(severity) if fracture_center else "N/A"
            
            # Calculate image clarity for bone health
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error("Failed to load image for clarity")
                raise ValueError("Failed to load image for clarity calculation")
            img = cv2.resize(img, (400, 400))
            edges = cv2.Canny(img, 30, 100)
            image_clarity = np.sum(edges) / (400 * 400) * 100
            image_clarity = min(image_clarity, 100)
            
            bone_health = calculate_bone_health(fracture_center is not None, severity, image_clarity)
            
            # Classify bone category
            img = cv2.imread(file_path)
            if img is None:
                logger.error("Failed to load image for classification")
                raise ValueError("Failed to load image for classification")
            
            img_resized = cv2.resize(img, (32, 32))
            test = np.array(img_resized, dtype='float32') / 255
            test = test.ravel()
            test_data = np.array([test])
            logger.debug(f"Feature vector shape: {test_data.shape}")
            
            try:
                prediction = classifier.predict(test_data)[0]
                prediction_proba = classifier.predict_proba(test_data)[0]
            except Exception as e:
                logger.error(f"Classifier prediction failed: {str(e)}")
                raise ValueError(f"Model prediction failed: {str(e)}")
            confidence = float(max(prediction_proba) * 100)
            logger.debug(f"Prediction probabilities: {prediction_proba}")
            
            if confidence < 30.0:
                logger.warning(f"Confidence {confidence}% too low")
                raise ValueError("Image unlikely to be a valid X-ray")
            
            low_confidence_warning = confidence < CONFIDENCE_THRESHOLD
            if low_confidence_warning:
                logger.warning(f"Confidence {confidence}% below {CONFIDENCE_THRESHOLD}%")
            
            category = labels[prediction]
            # Adjust misclassification for Elbow vs Wrist
            if category == 'Elbow':
                img_for_bone_check = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img_for_bone_check is not None:
                    img_for_bone_check = cv2.resize(img_for_bone_check, (400, 400))
                    edges = cv2.Canny(img_for_bone_check, 30, 100)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    wrist_features = 0
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h != 0 else 0
                        if 0.8 < aspect_ratio < 1.2 and w < 50 and h < 50:
                            wrist_features += 1
                    if wrist_features > 5:
                        logger.warning("Elbow misclassified, adjusting to Wrist")
                        category = 'Wrist'
            
            logger.debug(f"Predicted category: {category}, Confidence: {confidence}")
            
            bones = bone_info[category]['bones']
            bones_present = [bone['name'] for bone in bones]
            bone_details = [{'name': bone['name'], 'description': bone['description'], 'roi': bone['roi']} for bone in bones]
            
            # Identify fractured bone, passing the file path directly
            fractured_bone = identify_fractured_bone(category, fracture_center, image_path=file_path) if fracture_center else "None"
            
            fracture_detected = fracture_center is not None
            fracture_coordinates = fracture_center if fracture_center else (0, 0)
            
            image_description = generate_image_description(category, fracture_detected, fractured_bone, severity_grade, bone_health, fracture_type, bones_present, fracture_coordinates)
            
            # Annotate the image
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename))
            if not mark_fracture(file_path, fracture_center, annotated_path, severity, category, fracture_type):
                logger.error("Failed to annotate image")
                raise ValueError("Failed to annotate image")
            
            # Store in history
            history_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'category': category,
                'bones_present': bones_present,
                'fractured_bone': fractured_bone,
                'fracture_detected': fracture_detected,
                'fracture_type': fracture_type,
                'fracture_coordinates': fracture_coordinates,
                'severity': float(severity),
                'severity_grade': severity_grade,
                'bone_health': float(bone_health),
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'image_path': f'/static/uploads/{filename}',
                'annotated_image_path': f'/static/uploads/{annotated_filename}',
                'image_description': image_description
            }
            history.append(history_entry)
            save_history()
            
            # Update improvement data if this is the initial fracture
            if fracture_detected and 'initial_fracture' not in improvement_data:
                improvement_data['initial_fracture'] = {
                    'category': category,
                    'bones_present': bones_present,
                    'fractured_bone': fractured_bone,
                    'fracture_type': fracture_type,
                    'fracture_coordinates': fracture_coordinates,
                    'severity': float(severity),
                    'timestamp': history_entry['timestamp'],
                    'image_path': history_entry['image_path'],
                    'annotated_image_path': history_entry['annotated_image_path']
                }
                save_improvement_data()
            
            response = {
                'message': f"Predicted: {category}, Fracture: {fracture_detected}, Bone: {fractured_bone}, Type: {fracture_type}",
                'category': category,
                'bones_present': bones_present,
                'bones': bone_details,
                'fractured_bone': fractured_bone,
                'fracture_detected': fracture_detected,
                'fracture_type': fracture_type,
                'fracture_coordinates': fracture_coordinates,
                'severity': float(severity),
                'severity_grade': severity_grade,
                'bone_health': float(bone_health),
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'image_path': f'/static/uploads/{filename}',
                'annotated_image_path': f'/static/uploads/{annotated_filename}',
                'image_description': image_description,
                'status': 'success'
            }
            logger.debug(f"Returning response: {response}")
            return jsonify(response)
        
        except ValueError as ve:
            logger.error(f"ValueError during analysis: {str(ve)}")
            return jsonify({'error': str(ve), 'status': 'error'}), 400
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}', 'status': 'error'}), 500
        finally:
            # Only delete the file after all processing is complete
            safe_remove_file(file_path)
    
    try:
        return render_template('analysis.html', init_log=init_log, model_trained=model_trained)
    except Exception as e:
        logger.error(f"Failed to render analysis.html: {str(e)}")
        return jsonify({'error': f'Failed to load analysis page: {str(e)}', 'status': 'error'}), 500

@app.route('/difference', methods=['GET', 'POST'])
def difference():
    """Handle comparison between two X-rays with improved error handling."""
    logger.debug("Serving difference page")
    if request.method == 'POST':
        if not model_trained:
            return jsonify({'error': 'Model not loaded.', 'status': 'error'}), 500

        if 'before_image' not in request.files or 'after_image' not in request.files:
            return jsonify({'error': 'Both images required.', 'status': 'error'}), 400
        
        before_file = request.files['before_image']
        after_file = request.files['after_image']
        
        if before_file.filename == '' or after_file.filename == '':
            return jsonify({'error': 'No files selected.', 'status': 'error'}), 400
        
        timestamp = int(time.time())
        before_original_filename = secure_filename(before_file.filename)
        after_original_filename = secure_filename(after_file.filename)
        before_filename = f"before_{timestamp}_{before_original_filename}"
        after_filename = f"after_{timestamp}_{after_original_filename}"
        if not (before_filename.lower().endswith(('.png', '.jpg', '.jpeg')) and after_filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            return jsonify({'error': 'Only PNG/JPEG supported.', 'status': 'error'}), 400
        
        before_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], before_filename))
        after_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], after_filename))
        try:
            before_file.save(before_path)
            after_file.save(after_path)
            logger.debug(f"Images saved: {before_path}, {after_path}")
        except Exception as e:
            return jsonify({'error': f'Failed to save images: {str(e)}', 'status': 'error'}), 500
        
        try:
            if not is_xray_image(before_path):
                return jsonify({'error': 'Before image not a valid X-ray.', 'status': 'error'}), 400
            
            if not is_xray_image(after_path):
                return jsonify({'error': 'After image not a valid X-ray.', 'status': 'error'}), 400
            
            # Detect fractures in both images
            before_fracture_center, before_fracture_roi, initial_severity, before_fracture_type = detect_fracture(before_path)
            initial_severity_grade = grade_severity(initial_severity) if before_fracture_center else "N/A"
            
            after_fracture_center, after_fracture_roi, current_severity, after_fracture_type = detect_fracture(after_path)
            current_severity_grade = grade_severity(current_severity) if after_fracture_center else "N/A"
            
            # Calculate image clarity for bone health
            before_img = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
            after_img = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
            if before_img is None or after_img is None:
                raise ValueError("Failed to load images for clarity")
            before_img = cv2.resize(before_img, (400, 400))
            after_img = cv2.resize(after_img, (400, 400))
            before_edges = cv2.Canny(before_img, 30, 100)
            after_edges = cv2.Canny(after_img, 30, 100)
            before_clarity = min(np.sum(before_edges) / (400 * 400) * 100, 100)
            after_clarity = min(np.sum(after_edges) / (400 * 400) * 100, 100)
            
            initial_bone_health = calculate_bone_health(before_fracture_center is not None, initial_severity, before_clarity)
            current_bone_health = calculate_bone_health(after_fracture_center is not None, current_severity, after_clarity)
            
            # Classify category using the after image
            after_img = cv2.imread(after_path)
            if after_img is None:
                raise ValueError("Failed to load after image")
            after_img_resized = cv2.resize(after_img, (32, 32))
            test = np.array(after_img_resized, dtype='float32') / 255
            test = test.ravel()
            test_data = np.array([test])
            
            try:
                prediction = classifier.predict(test_data)[0]
                prediction_proba = classifier.predict_proba(test_data)[0]
            except Exception as e:
                logger.error(f"Classifier prediction failed: {str(e)}")
                raise ValueError(f"Model prediction failed: {str(e)}")
            confidence = float(max(prediction_proba) * 100)
            
            if confidence < 30.0:
                raise ValueError("After image unlikely a valid X-ray")
            
            low_confidence_warning = confidence < CONFIDENCE_THRESHOLD
            
            category = labels[prediction]
            if category == 'Elbow':
                after_img_check = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
                if after_img_check is not None:
                    after_img_check = cv2.resize(after_img_check, (400, 400))
                    edges = cv2.Canny(after_img_check, 30, 100)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    wrist_features = 0
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h != 0 else 0
                        if 0.8 < aspect_ratio < 1.2 and w < 50 and h < 50:
                            wrist_features += 1
                    if wrist_features > 5:
                        category = 'Wrist'
            
            bones = bone_info[category]['bones']
            bones_present = [bone['name'] for bone in bones]
            bone_details = [{'name': bone['name'], 'description': bone['description'], 'roi': bone['roi']} for bone in bones]
            
            # Identify fractured bones
            initial_fractured_bone = identify_fractured_bone(category, before_fracture_center, image_path=before_path) if before_fracture_center else "None"
            current_fractured_bone = identify_fractured_bone(category, after_fracture_center, image_path=after_path) if after_fracture_center else "None"
            
            fracture_detected = after_fracture_center is not None
            initial_fracture_coordinates = before_fracture_center if before_fracture_center else (0, 0)
            current_fracture_coordinates = after_fracture_center if after_fracture_center else (0, 0)
            
            initial_image_description = generate_image_description(category, before_fracture_center is not None, initial_fractured_bone, initial_severity_grade, initial_bone_health, before_fracture_type, bones_present, initial_fracture_coordinates)
            current_image_description = generate_image_description(category, after_fracture_center is not None, current_fractured_bone, current_severity_grade, current_bone_health, after_fracture_type, bones_present, current_fracture_coordinates)
            
            # Annotate both images
            before_annotated_filename = f"annotated_before_{timestamp}_{before_original_filename}"
            after_annotated_filename = f"annotated_after_{timestamp}_{after_original_filename}"
            before_annotated_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], before_annotated_filename))
            after_annotated_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], after_annotated_filename))
            if not (mark_fracture(before_path, before_fracture_center, before_annotated_path, initial_severity, category, before_fracture_type) and
                    mark_fracture(after_path, after_fracture_center, after_annotated_path, current_severity, category, after_fracture_type)):
                raise ValueError("Failed to annotate images")
            
            # Assess healing progress
            improvement_percentage, healing_stage = assess_healing(initial_severity, current_severity)
            
            # Store in history
            history_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'category': category,
                'bones_present': bones_present,
                'initial_fractured_bone': initial_fractured_bone,
                'current_fractured_bone': current_fractured_bone,
                'fracture_detected': fracture_detected,
                'initial_fracture_type': before_fracture_type,
                'current_fracture_type': after_fracture_type,
                'initial_fracture_coordinates': initial_fracture_coordinates,
                'current_fracture_coordinates': current_fracture_coordinates,
                'initial_severity': float(initial_severity),
                'initial_severity_grade': initial_severity_grade,
                'current_severity': float(current_severity),
                'current_severity_grade': current_severity_grade,
                'initial_bone_health': float(initial_bone_health),
                'current_bone_health': float(current_bone_health),
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'initial_image_path': f'/static/uploads/{before_filename}',
                'initial_annotated_image_path': f'/static/uploads/{before_annotated_filename}',
                'current_image_path': f'/static/uploads/{after_filename}',
                'current_annotated_image_path': f'/static/uploads/{after_annotated_filename}',
                'improvement_percentage': float(improvement_percentage),
                'healing_stage': healing_stage,
                'initial_image_description': initial_image_description,
                'current_image_description': current_image_description
            }
            history.append(history_entry)
            save_history()
            
            # Update improvement data
            if 'initial_fracture' not in improvement_data:
                improvement_data['initial_fracture'] = {
                    'category': category,
                    'bones_present': bones_present,
                    'fractured_bone': initial_fractured_bone,
                    'fracture_type': before_fracture_type,
                    'fracture_coordinates': initial_fracture_coordinates,
                    'severity': float(initial_severity),
                    'timestamp': history_entry['timestamp'],
                    'image_path': history_entry['initial_image_path'],
                    'annotated_image_path': history_entry['initial_annotated_image_path']
                }
                save_improvement_data()
            
            response = {
                'message': f"Initial Severity: {initial_severity}%, Current: {current_severity}%, Healing: {healing_stage}",
                'category': category,
                'bones_present': bones_present,
                'bones': bone_details,
                'initial_fractured_bone': initial_fractured_bone,
                'current_fractured_bone': current_fractured_bone,
                'fracture_detected': fracture_detected,
                'initial_fracture_type': before_fracture_type,
                'current_fracture_type': after_fracture_type,
                'initial_fracture_coordinates': initial_fracture_coordinates,
                'current_fracture_coordinates': current_fracture_coordinates,
                'initial_severity': float(initial_severity),
                'initial_severity_grade': initial_severity_grade,
                'current_severity': float(current_severity),
                'current_severity_grade': current_severity_grade,
                'initial_bone_health': float(initial_bone_health),
                'current_bone_health': float(current_bone_health),
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'initial_image_path': f'/static/uploads/{before_filename}',
                'initial_annotated_image_path': f'/static/uploads/{before_annotated_filename}',
                'current_image_path': f'/static/uploads/{after_filename}',
                'current_annotated_image_path': f'/static/uploads/{after_annotated_filename}',
                'improvement_percentage': float(improvement_percentage),
                'healing_stage': healing_stage,
                'initial_image_description': initial_image_description,
                'current_image_description': current_image_description,
                'status': 'success'
            }
            return jsonify(response)
        except ValueError as ve:
            logger.error(f"ValueError during difference analysis: {str(ve)}")
            return jsonify({'error': str(ve), 'status': 'error'}), 400
        except Exception as e:
            logger.error(f"Unexpected error during difference analysis: {str(e)}")
            return jsonify({'error': f'Difference analysis failed: {str(e)}', 'status': 'error'}), 500
        finally:
            safe_remove_file(before_path)
            safe_remove_file(after_path)
    
    try:
        return render_template('difference.html', init_log=init_log, model_trained=model_trained)
    except Exception as e:
        logger.error(f"Failed to render difference.html: {str(e)}")
        return jsonify({'error': f'Failed to load difference page: {str(e)}', 'status': 'error'}), 500

@app.route('/history', methods=['GET'])
def history_page():
    """Render the history page."""
    try:
        return render_template('history.html', init_log=init_log, model_trained=model_trained)
    except Exception as e:
        logger.error(f"Failed to render history.html: {str(e)}")
        return jsonify({'error': f'Failed to load history page: {str(e)}', 'status': 'error'}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    """Retrieve the analysis history."""
    try:
        return jsonify({'history': history, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        return jsonify({'error': f'Failed to retrieve history: {str(e)}', 'status': 'error'}), 500

@app.route('/export_history', methods=['GET'])
def export_history():
    """Export history as a CSV file."""
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        headers = [
            'ID', 'Timestamp', 'Category', 'Bones Present', 'Fractured Bone', 'Fracture Detected', 'Fracture Type',
            'Fracture Coordinates', 'Severity', 'Severity Grade', 'Bone Health', 'Confidence', 'Low Confidence Warning',
            'Image Path', 'Annotated Image Path', 'Image Description', 'Initial Fractured Bone', 'Current Fractured Bone',
            'Initial Fracture Type', 'Current Fracture Type', 'Initial Fracture Coordinates', 'Current Fracture Coordinates',
            'Initial Severity', 'Initial Severity Grade', 'Current Severity', 'Current Severity Grade', 'Initial Bone Health',
            'Current Bone Health', 'Improvement Percentage', 'Healing Stage', 'Initial Image Path', 'Initial Annotated Image Path',
            'Current Image Path', 'Current Annotated Image Path', 'Initial Image Description', 'Current Image Description'
        ]
        writer.writerow(headers)
        
        for entry in history:
            row = [
                entry.get('id', ''),
                entry.get('timestamp', ''),
                entry.get('category', ''),
                ', '.join(entry.get('bones_present', [])),
                entry.get('fractured_bone', ''),
                str(entry.get('fracture_detected', '')),
                entry.get('fracture_type', ''),
                str(entry.get('fracture_coordinates', '')),
                str(entry.get('severity', '')),
                entry.get('severity_grade', ''),
                str(entry.get('bone_health', '')),
                str(entry.get('confidence', '')),
                str(entry.get('low_confidence_warning', '')),
                entry.get('image_path', ''),
                entry.get('annotated_image_path', ''),
                entry.get('image_description', ''),
                entry.get('initial_fractured_bone', ''),
                entry.get('current_fractured_bone', ''),
                entry.get('initial_fracture_type', ''),
                entry.get('current_fracture_type', ''),
                str(entry.get('initial_fracture_coordinates', '')),
                str(entry.get('current_fracture_coordinates', '')),
                str(entry.get('initial_severity', '')),
                entry.get('initial_severity_grade', ''),
                str(entry.get('current_severity', '')),
                str(entry.get('current_severity_grade', '')),
                str(entry.get('initial_bone_health', '')),
                str(entry.get('current_bone_health', '')),
                str(entry.get('improvement_percentage', '')),
                entry.get('healing_stage', ''),
                entry.get('initial_image_path', ''),
                entry.get('initial_annotated_image_path', ''),
                entry.get('current_image_path', ''),
                entry.get('current_annotated_image_path', ''),
                entry.get('initial_image_description', ''),
                entry.get('current_image_description', '')
            ]
            writer.writerow(row)
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'analysis_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logger.error(f"Failed to export history: {str(e)}")
        return jsonify({'error': f'Failed to export history: {str(e)}', 'status': 'error'}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided.', 'status': 'error'}), 400
        
        history_id = data('history_id')
        rating = data.get('rating')
        comments = data.get('comments', '')
        
        if not history_id or rating is None:
            return jsonify({'error': 'Missing history_id or rating.', 'status': 'error'}), 400
        
        if not any(entry['id'] == history_id for entry in history):
            return jsonify({'error': f'Invalid history_id: {history_id}.', 'status': 'error'}), 400
        
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be 1-5.', 'status': 'error'}), 400
        
        feedback_entry = {
            'id': str(uuid.uuid4()),
            'history_id': history_id,
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        feedback_data.append(feedback_entry)
        save_feedback_data()
        
        return jsonify({'message': 'Feedback submitted.', 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        return jsonify({'error': f'Failed to submit feedback: {str(e)}', 'status': 'error'}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain the models using the dataset."""
    global model_trained, fracture_model_trained
    try:
        if not load_dataset_annotations():
            return jsonify({'error': 'Failed to load dataset annotations for retraining.', 'status': 'error'}), 500
        
        model_trained = train_bone_classifier()
        fracture_model_trained = train_fracture_model()
        
        if model_trained and fracture_model_trained:
            return jsonify({'message': 'Models retrained successfully.', 'status': 'success'})
        else:
            return jsonify({'error': 'Model retraining failed.', 'status': 'error'}), 500
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        return jsonify({'error': f'Model retraining failed: {str(e)}', 'status': 'error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True, host='0.0.0.0', port=5000)  # Match previous run configuration