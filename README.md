Bone Fracture Detection and Analysis
This is a Flask-based web application for analyzing X-ray images to detect bone fractures, classify bone categories, and assess fracture severity and healing progress. The application uses machine learning models (Random Forest) and image processing techniques (OpenCV) to identify fractures, annotate X-ray images, and provide detailed analysis reports.
Features

Single X-ray Analysis: Upload an X-ray image to detect fractures, classify the bone category (e.g., Wrist, Elbow), identify the fractured bone, and assess severity and bone health.
Difference Analysis: Compare two X-ray images (before and after) to evaluate healing progress and changes in fracture severity.
History Tracking: Store and view analysis results, with the option to export history as a CSV file.
Feedback System: Submit feedback on analysis results for continuous improvement.
Model Retraining: Retrain machine learning models using a dataset of annotated X-ray images.
Image Enhancement: Preprocess X-ray images to remove text, enhance clarity, and generate heatmaps for fracture visualization.
Responsive UI: User-friendly interface for uploading images, viewing results, and accessing history.

Technologies

Backend: Flask (Python), Flask-Compress
Machine Learning: Scikit-learn (RandomForestClassifier, RandomForestRegressor)
Image Processing: OpenCV, Pillow
Data Handling: Pandas, NumPy
Frontend: HTML, JavaScript, CSS (Bootstrap assumed for styling)
Logging: Python logging module for debugging and monitoring
File Management: JSON for history and feedback storage, CSV for annotations and exports

Prerequisites

Python 3.8+
pip (Python package manager)
Dataset of X-ray images (e.g., organized in Simple Bone Fracture and Comminuted Bone Fracture folders)
Optional: Pre-trained models (model.pkl, fracture_model.pkl)

Installation

Clone the Repository:
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

If requirements.txt is not provided, install the following packages:
pip install flask flask-compress opencv-python numpy scikit-learn pillow pandas


Prepare the Dataset:

Place your X-ray dataset in the directory specified by DATASET_FOLDER (default: C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal).
Ensure the dataset has folders named Simple Bone Fracture and Comminuted Bone Fracture containing PNG/JPEG images.
Optionally, provide an annotations.csv file in the dataset folder with columns: image_id, category, bone, fracture_type, fracture_x, fracture_y, severity. If not provided, the application generates mock annotations.


Directory Structure:Ensure the following folders exist:

static/uploads: For storing uploaded and annotated images.
static/outputs: For storing processed outputs (if needed).
model: For storing trained models (model.pkl, fracture_model.pkl).Create these folders manually or let the application create them on startup.


Optional: Pre-trained Models:

Place pre-trained models in the model folder to skip initial training.
If models are not provided, the application trains new models using the dataset.



Usage

Run the Application:
python app.py

The server starts on http://0.0.0.0:5000 in debug mode.

Access the Web Interface:

Open a browser and navigate to http://localhost:5000.
Available pages:
Home (/): Project overview and initialization logs.
Analysis (/analysis): Upload a single X-ray for fracture detection.
Difference (/difference): Upload two X-rays to compare healing progress.
History (/history): View past analyses and export as CSV.
Feedback: Submit feedback via the history page (POST request to /submit_feedback).
Retrain Model: Retrain models via a POST request to /retrain_model.




API Endpoints:

POST /analysis: Upload an X-ray image for analysis. Returns JSON with category, fracture details, severity, and annotated image path.
POST /difference: Upload two X-ray images for comparison. Returns JSON with healing progress and annotated image paths.
GET /get_history: Retrieve analysis history as JSON.
GET /export_history: Download analysis history as a CSV file.
POST /submit_feedback: Submit feedback with a history ID, rating (1-5), and comments.
POST /retrain_model: Retrain machine learning models using the dataset.


File Management:

Uploaded images are temporarily stored in static/uploads and deleted after processing.
Annotated images are saved in static/uploads with prefixes like annotated_.
History is stored in history.json, feedback in feedback.json, and improvement data in improvement.json.



Configuration

Folders:
UPLOAD_FOLDER: static/uploads
OUTPUT_FOLDER: static/outputs
MODEL_FOLDER: model
DATASET_FOLDER: C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal (update in code if needed)
ANNOTATION_FILE: annotations.csv in the dataset folder


Model Parameters:
Random Forest: 100 estimators, random state 42
Confidence Threshold: 60% (adjust CONFIDENCE_THRESHOLD in code)


Image Requirements:
Format: PNG or JPEG
Size: Minimum 10x10 pixels
Type: Grayscale or low-color X-ray images



Dataset

The application expects a dataset with:
Simple Bone Fracture: Folder with simple fracture X-rays.
Comminuted Bone Fracture: Folder with comminuted fracture X-rays.


Annotations (annotations.csv) should include:
image_id: Filename of the X-ray image.
category: Bone category (e.g., Wrist).
bone: Specific bone (e.g., Distal Radius).
fracture_type: Simple or Comminuted.
fracture_x, fracture_y: Fracture coordinates.
severity: Severity score (0-100).


If no annotations are provided, the application generates mock annotations based on image analysis.

Models

Bone Classifier: RandomForestClassifier to predict bone category (e.g., Wrist, Elbow).
Fracture Models:
fracture_clf_bone: RandomForestClassifier to predict the fractured bone.
fracture_clf_type: RandomForestClassifier to predict fracture type (Simple/Comminuted).
fracture_reg_x, fracture_reg_y: RandomForestRegressor to predict fracture coordinates.


Models are saved as model.pkl (bone classifier) and fracture_model.pkl (fracture models).

Logging

Logs are written to app.log and the console.
Includes:
Dependency versions (Python, Flask, OpenCV, etc.).
Directory creation and file operations.
Model loading and training.
Image processing and analysis details.


Log levels: INFO, DEBUG, WARNING, ERROR.

Troubleshooting

"Model not loaded" Error:
Ensure model.pkl and fracture_model.pkl exist in the model folder or a valid dataset is available for training.


"Not a valid X-ray" Error:
Upload grayscale PNG/JPEG images with clear bone structures.
Check image size (minimum 10x10 pixels).


"Failed to load image" Error:
Verify the image path and file format.
Ensure sufficient disk space in static/uploads.


Low Confidence Warning:
Confidence below 60% indicates possible misclassification.
Try a clearer X-ray image or retrain the model with more data.


Permission Errors:
Ensure write permissions for static/uploads, static/outputs, and model folders (set to 755).



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m "Add new feature").
Push to the branch (git push origin feature/new-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

OpenCV for image processing.
Scikit-learn for machine learning models.
Flask for web framework.
Bootstrap for frontend styling (assumed).

