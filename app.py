import asyncio
import os
import sys
import logging
import json
import uuid
import datetime
import time
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify
from flask_compress import Compress
import importlib.metadata
import aiofiles

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Log dependency versions
logger.info(f"Python version: {sys.version}")
try:
    for pkg in ['flask', 'scikit-learn', 'opencv-python', 'numpy']:
        logger.info(f"{pkg} version: {importlib.metadata.version(pkg)}")
except ImportError as e:
    logger.error(f"Missing dependency: {str(e)}")
    sys.exit(1)

# Flask app setup
app = Flask(__name__)
Compress(app)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
MODEL_FOLDER = BASE_DIR / 'model'
DATASET_FOLDER = BASE_DIR / 'Dataset' / 'datasets'
HISTORY_FILE = BASE_DIR / 'history.json'

# Ensure directories
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, DATASET_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(folder, 0o755)
        logger.info(f"Ensured directory: {folder}")
    except Exception as e:
        logger.error(f"Failed to create directory {folder}: {str(e)}")
        sys.exit(1)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Bone information
labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']
bone_info = {
    'Chest': {'bones': [
        {'name': 'Ribs', 'description': 'Protects heart and lungs.'},
        {'name': 'Sternum', 'description': 'Breastbone in chest center.'}
    ]},
    'Elbow': {'bones': [
        {'name': 'Humerus', 'description': 'Upper arm bone.'},
        {'name': 'Radius', 'description': 'Forearm bone (thumb side).'},
        {'name': 'Ulna', 'description': 'Forearm bone (pinky side).'}
    ]},
    'Finger': {'bones': [
        {'name': 'Phalanges', 'description': 'Finger bones.'}
    ]},
    'Hand': {'bones': [
        {'name': 'Metacarpals', 'description': 'Palm bones.'},
        {'name': 'Carpals', 'description': 'Wrist bones.'}
    ]},
    'Head': {'bones': [
        {'name': 'Skull', 'description': 'Protects the brain.'},
        {'name': 'Mandible', 'description': 'Lower jawbone.'}
    ]},
    'Shoulder': {'bones': [
        {'name': 'Clavicle', 'description': 'Collarbone.'},
        {'name': 'Scapula', 'description': 'Shoulder blade.'}
    ]},
    'Wrist': {'bones': [
        {'name': 'Carpals', 'description': 'Wrist bones.'},
        {'name': 'Distal Radius', 'description': 'Radius end near wrist.'},
        {'name': 'Distal Ulna', 'description': 'Ulna end near wrist.'}
    ]}
}

# Bounding box coordinates (percentages of image dimensions)
BONE_BBOXES = {
    'Chest': {
        'Ribs': [
            (0.15, 0.2, 0.25, 0.6),  # Left ribs
            (0.60, 0.2, 0.25, 0.6)   # Right ribs
        ],
        'Sternum': [(0.45, 0.2, 0.10, 0.6)]  # Center of the chest
    },
    'Elbow': {
        'Humerus': [(0.35, 0.05, 0.30, 0.40)],
        'Radius': [(0.35, 0.45, 0.15, 0.45)],
        'Ulna': [(0.50, 0.45, 0.15, 0.45)]
    },
    'Finger': {
        'Phalanges': [(0.4, 0.3, 0.2, 0.4)]
    },
    'Hand': {
        'Metacarpals': [(0.4, 0.3, 0.2, 0.3)],
        'Carpals': [(0.4, 0.6, 0.2, 0.2)]
    },
    'Head': {
        'Skull': [(0.3, 0.1, 0.4, 0.5)],
        'Mandible': [(0.3, 0.6, 0.4, 0.2)]
    },
    'Shoulder': {
        'Clavicle': [(0.3, 0.1, 0.4, 0.2)],
        'Scapula': [(0.3, 0.3, 0.4, 0.4)]
    },
    'Wrist': {
        'Carpals': [(0.30, 0.65, 0.40, 0.20)],
        'Distal Radius': [(0.30, 0.45, 0.20, 0.20)],
        'Distal Ulna': [(0.50, 0.45, 0.20, 0.20)]
    }
}

# Global variables
classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model_trained = False
history = []
CONFIDENCE_THRESHOLD = 60.0
X_features = None
y_labels = None
init_log = []

async def load_dataset_and_extract_features() -> Tuple[np.ndarray, np.ndarray]:
    global X_features, y_labels
    X, y = [], []
    
    for idx, label in enumerate(labels):
        folder_path = DATASET_FOLDER / label
        if not folder_path.exists():
            logger.warning(f"Dataset folder {folder_path} does not exist")
            continue
            
        for img_path in folder_path.glob('*.png') or folder_path.glob('*.jpg') or folder_path.glob('*.jpeg'):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                img_resized = cv2.resize(img, (32, 32))
                img_normalized = img_resized.astype('float32') / 255
                features = img_normalized.ravel()
                X.append(features)
                y.append(idx)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0 or len(y) == 0:
        logger.error("No valid images found in dataset for training")
        X_features = np.array([])
        y_labels = np.array([])
        return X_features, y_labels
    
    try:
        np.save(MODEL_FOLDER / 'X.txt.npy', X)
        np.save(MODEL_FOLDER / 'Y.txt.npy', y)
        logger.info("Saved extracted features to model directory")
    except Exception as e:
        logger.error(f"Failed to save features: {str(e)}")
    
    X_features = X
    y_labels = y
    logger.info(f"Loaded {len(X)} images with {X.shape[1]} features each")
    return X_features, y_labels

async def safe_remove_file(file_path: Path) -> None:
    if file_path.exists():
        try:
            file_path.unlink()
            logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {str(e)}")

async def load_json_file(file_path: Path, default: any) -> any:
    if file_path.exists():
        try:
            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
                return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
    return default

async def save_json_file(file_path: Path, data: any) -> None:
    try:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {str(e)}")

async def load_history() -> List[Dict]:
    global history
    history = await load_json_file(HISTORY_FILE, [])
    if not isinstance(history, list):
        logger.warning("Invalid history.json format, initializing empty")
        history = []
    return history

async def train_bone_classifier() -> bool:
    global classifier, X_features, y_labels
    model_path = MODEL_FOLDER / 'model.txt'

    if os.path.exists(MODEL_FOLDER / 'X.txt.npy') and os.path.exists(MODEL_FOLDER / 'Y.txt.npy'):
        try:
            X_features = np.load(MODEL_FOLDER / 'X.txt.npy')
            y_labels = np.load(MODEL_FOLDER / 'Y.txt.npy')
            logger.info("Loaded precomputed features")
        except Exception as e:
            logger.error(f"Failed to load precomputed features: {str(e)}")
            X_features, y_labels = await load_dataset_and_extract_features()
    else:
        X_features, y_labels = await load_dataset_and_extract_features()

    if X_features.size == 0 or y_labels.size == 0:
        logger.error("No valid data for training bone classifier")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
    
    if model_path.exists() and model_path.stat().st_size > 0:
        try:
            async with aiofiles.open(model_path, 'rb') as f:
                classifier = pickle.loads(await f.read())
            logger.info("Loaded existing bone classifier")
            return True
        except Exception as e:
            logger.error(f"Failed to load classifier: {str(e)}")

    try:
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_test, y_test) * 100
        logger.info(f"Trained bone classifier with accuracy: {accuracy:.2f}%")
        
        async with aiofiles.open(model_path, 'wb') as f:
            await f.write(pickle.dumps(classifier))
        logger.info("Saved bone classifier to model.txt")
        return True
    except Exception as e:
        logger.error(f"Failed to train classifier: {str(e)}")
        return False

async def load_model() -> Tuple[bool, List[str]]:
    global model_trained, init_log
    log = []

    try:
        model_trained = await train_bone_classifier()
        if model_trained:
            log.append("Bone classifier loaded/trained successfully")
        else:
            log.append("Failed to load/train bone classifier")
        return model_trained, log
    except Exception as e:
        logger.error(f"Unexpected error in load_model: {str(e)}")
        log.append(f"Unexpected error: {str(e)}")
        return False, log

async def is_xray_image(image_path: Path) -> bool:
    try:
        # Load the image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.debug(f"Failed to load image: {image_path}")
            return False

        # Check image size
        if img.size < 10 * 10:
            logger.debug(f"Image too small: {image_path}")
            return False

        # Convert to grayscale for analysis
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Check if the image is effectively grayscale by calculating mean absolute difference
            b, g, r = cv2.split(img)
            diff_bg = np.mean(np.abs(b - g))
            diff_gr = np.mean(np.abs(g - r))
            diff_threshold = 5  # Stricter threshold to reject color images
            if diff_bg > diff_threshold or diff_gr > diff_threshold:
                logger.debug(f"Image is not effectively grayscale (diff_bg={diff_bg}, diff_gr={diff_gr}): {image_path}")
                return False
        else:
            gray = img  # Already grayscale

        # Normalize the grayscale image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Check intensity distribution (X-rays typically have a significant portion of bright pixels for bones)
        bright_pixel_ratio = np.sum(gray > 150) / gray.size
        bright_pixel_threshold = 0.1  # At least 10% of pixels should be bright (bones)
        if bright_pixel_ratio < bright_pixel_threshold:
            logger.debug(f"Too few bright pixels (ratio={bright_pixel_ratio}): {image_path}")
            return False

        # Check contrast (standard deviation of pixel intensities)
        contrast = np.std(gray)
        contrast_threshold = 15  # Lowered to accept low-contrast X-rays
        if contrast < contrast_threshold:
            logger.debug(f"Contrast too low ({contrast}): {image_path}")
            return False

        # Bone detection using adaptive thresholding and contour analysis
        # Use adaptive thresholding to better handle varying brightness
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find bone-like structures
        bone_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Lowered minimum area to accept smaller bones
                continue

            # Compute aspect ratio (bones are typically elongated)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.2 < aspect_ratio < 5.0):  # Bones are not too square or too thin
                continue

            # Compute solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.5:  # Bones typically have high solidity
                continue

            bone_contours.append(cnt)

        if len(bone_contours) < 1:  # Expect at least one bone-like structure
            logger.debug(f"No bone-like structures found (contours={len(bone_contours)}): {image_path}")
            return False

        logger.debug(f"Image passed X-ray checks: {image_path}")
        return True
    except Exception as e:
        logger.error(f"X-ray validation failed: {str(e)}")
        return False

async def init_app():
    global model_trained, init_log
    model_trained, init_log = await load_model()
    await load_history()

@app.route('/')
async def index():
    logger.debug("Serving index.html")
    try:
        return render_template('index.html', model_trained=model_trained, init_log=init_log)
    except Exception as e:
        logger.error(f"Failed to render index.html: {str(e)}")
        return jsonify({'error': f'Failed to load index page: {str(e)}', 'status': 'error'}), 500

@app.route('/analysis', methods=['GET', 'POST'])
async def analysis():
    logger.debug("Serving analysis page")
    if request.method == 'POST':
        if not model_trained:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded.', 'status': 'error'}), 500

        if 'image' not in request.files:
            logger.error("No image uploaded")
            return jsonify({'error': 'No image uploaded.', 'status': 'error'}), 400

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

        file_path = app.config['UPLOAD_FOLDER'] / filename
        try:
            file.save(file_path)
            logger.debug(f"Image saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return jsonify({'error': f'Failed to save image: {str(e)}', 'status': 'error'}), 500

        try:
            if not await is_xray_image(file_path):
                logger.error("Not a valid X-ray")
                await safe_remove_file(file_path)
                return jsonify({'error': 'The uploaded image is not a valid X-ray. Please upload an X-ray image of bones.', 'status': 'error'}), 400

            img = cv2.imread(str(file_path))
            if img is None:
                logger.error("Failed to load image for classification")
                await safe_remove_file(file_path)
                raise ValueError("Failed to load image for classification")
            
            img_resized = cv2.resize(img, (32, 32))
            img_normalized = img_resized.astype('float32') / 255
            test_features = img_normalized.ravel().reshape(1, -1)
            
            try:
                prediction = classifier.predict(test_features)[0]
                prediction_proba = classifier.predict_proba(test_features)[0]
            except Exception as e:
                logger.error(f"Classifier prediction failed: {str(e)}")
                await safe_remove_file(file_path)
                raise ValueError(f"Model prediction failed: {str(e)}")
            
            confidence = float(max(prediction_proba) * 100)
            if confidence < 30.0:
                logger.warning(f"Confidence {confidence}% too low")
                await safe_remove_file(file_path)
                raise ValueError("Image unlikely to be a valid X-ray")

            low_confidence_warning = confidence < CONFIDENCE_THRESHOLD
            category = labels[prediction]

            # Get image dimensions
            img_height, img_width = img.shape[:2]

            # Annotate the image with bounding boxes and labels for each bone
            bone_bboxes = BONE_BBOXES[category]
            for bone in bone_info[category]['bones']:
                bone_name = bone['name']
                bboxes = bone_bboxes.get(bone_name, [])
                for idx, bbox in enumerate(bboxes):
                    # Convert percentage coordinates to pixel coordinates
                    x = int(bbox[0] * img_width)
                    y = int(bbox[1] * img_height)
                    w = int(bbox[2] * img_width)
                    h = int(bbox[3] * img_height)
                    # Ensure bounding box stays within image bounds
                    x = min(max(x, 0), img_width - w)
                    y = min(max(y, 0), img_height - h)
                    # Draw bounding box
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add bone name label above the box
                    label = f"{bone_name}{'' if len(bboxes) == 1 else f' {idx + 1}'}"
                    cv2.putText(img, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the annotated image
            annotated_filename = f"annotated_{filename}"
            annotated_file_path = app.config['UPLOAD_FOLDER'] / annotated_filename
            cv2.imwrite(str(annotated_file_path), img)
            logger.debug(f"Annotated image saved to {annotated_file_path}")

            bones = bone_info[category]['bones']
            bones_present = [bone['name'] for bone in bones]
            bone_details = [{'name': bone['name'], 'description': bone['description']} for bone in bones]
            description = f"This X-ray is a {category} radiograph. Bones present: {', '.join(bones_present)}."

            history_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'category': category,
                'bones_present': bones_present,
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'image_path': f'/static/uploads/{filename}',
                'annotated_image_path': f'/static/uploads/{annotated_filename}',
                'description': description
            }
            history.append(history_entry)
            await save_json_file(HISTORY_FILE, history)

            response = {
                'message': f"Predicted Bone: {category}",
                'category': category,
                'bones_present': bones_present,
                'bones': bone_details,
                'confidence': float(confidence),
                'low_confidence_warning': low_confidence_warning,
                'image_path': f'/static/uploads/{filename}',
                'annotated_image_path': f'/static/uploads/{annotated_filename}',
                'description': description,
                'status': 'success'
            }
            return jsonify(response)

        except ValueError as ve:
            logger.error(f"ValueError during analysis: {str(ve)}")
            await safe_remove_file(file_path)
            return jsonify({'error': str(ve), 'status': 'error'}), 400
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            await safe_remove_file(file_path)
            return jsonify({'error': f'Analysis failed: {str(e)}', 'status': 'error'}), 500

    try:
        return render_template('analysis.html', model_trained=model_trained, init_log=init_log)
    except Exception as e:
        logger.error(f"Failed to render analysis.html: {str(e)}")
        return jsonify({'error': f'Failed to load analysis page: {str(e)}', 'status': 'error'}), 500

@app.route('/history', methods=['GET'])
async def history_page():
    try:
        return render_template('history.html', model_trained=model_trained, init_log=init_log)
    except Exception as e:
        logger.error(f"Failed to render history.html: {str(e)}")
        return jsonify({'error': f'Failed to load history page: {str(e)}', 'status': 'error'}), 500

@app.route('/get_history', methods=['GET'])
async def get_history():
    try:
        return jsonify({'history': history, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        return jsonify({'error': f'Failed to retrieve history: {str(e)}', 'status': 'error'}), 500

@app.route('/retrain_model', methods=['POST'])
async def retrain_model():
    global model_trained
    try:
        model_trained = await train_bone_classifier()
        if model_trained:
            return jsonify({'message': 'Model retrained successfully.', 'status': 'success'})
        return jsonify({'error': 'Model retraining failed.', 'status': 'error'}), 500
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        return jsonify({'error': f'Model retraining failed: {str(e)}', 'status': 'error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server")
    asyncio.run(init_app())
    app.run(debug=True, host='0.0.0.0', port=5000)