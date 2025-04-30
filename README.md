# 🦴 Bone Fracture Detection and Analysis

A **Flask-based web application** for analyzing X-ray images to detect bone fractures, classify bone categories, and assess fracture severity and healing progress using **machine learning (Random Forest)** and **image processing (OpenCV)**.

---

## 🚀 Features

- **Single X-ray Analysis**: Upload an X-ray image to detect fractures, classify the bone (e.g., Wrist, Elbow), identify fractured bones, and assess severity.
- **Difference Analysis**: Compare two X-rays (before and after) to evaluate healing progress.
- **History Tracking**: View and export past analysis results as CSV.
- **Feedback System**: Submit feedback on analysis results for improvement.
- **Model Retraining**: Retrain models using a dataset of annotated X-rays.
- **Image Enhancement**: Remove text, enhance clarity, and generate heatmaps.
- **Responsive UI**: Intuitive interface for uploading, viewing results, and accessing history.

---

## 🛠 Technologies

**Backend**: Flask (Python), Flask-Compress  
**Machine Learning**: Scikit-learn (`RandomForestClassifier`, `RandomForestRegressor`)  
**Image Processing**: OpenCV, Pillow  
**Data Handling**: Pandas, NumPy  
**Frontend**: HTML, JavaScript, CSS (Bootstrap assumed)  
**Logging**: Python `logging` module  
**File Management**: JSON for local storage, CSV for data exports  

---

## 📋 Prerequisites

- Python 3.8+
- `pip` (Python package manager)
- Dataset of X-ray images (e.g., Simple and Comminuted fractures)
- Optional: Pre-trained models (`model.pkl`, `fracture_model.pkl`)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection
2. Set Up a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available:

bash
Copy
Edit
pip install flask flask-compress opencv-python numpy scikit-learn pillow pandas
4. Prepare Dataset
Place your X-ray images in the dataset folder (default:
C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal)

Ensure it contains:

Simple Bone Fracture/

Comminuted Bone Fracture/

Optional: Add annotations.csv with columns: image_id, category, bone, fracture_type, fracture_x, fracture_y, severity

5. Ensure Directory Structure
Create if not auto-generated:

bash
Copy
Edit
static/uploads/       # Uploaded and annotated images
static/outputs/       # Processed outputs
model/                # Stores models
🧠 Models
model.pkl: Classifies bone category (e.g., Wrist, Elbow)

fracture_model.pkl: Includes:

fracture_clf_bone: Predicts fractured bone

fracture_clf_type: Simple/Comminuted

fracture_reg_x, fracture_reg_y: Predict fracture coordinates

🚦 Usage
Run the App
bash
Copy
Edit
python app.py
Visit: http://localhost:5000

📄 Available Pages
/: Home (Project Overview)

/analysis: Single X-ray fracture detection

/difference: Compare two X-rays for healing progress

/history: View/export analysis history

/submit_feedback: Submit feedback

/retrain_model: Retrain ML models

🔌 API Endpoints
POST /analysis: Upload X-ray, returns JSON with detection

POST /difference: Upload two X-rays for healing comparison

GET /get_history: Fetch history as JSON

GET /export_history: Export history as CSV

POST /submit_feedback: Submit feedback

POST /retrain_model: Retrain ML models

🗂 File Management
Uploaded images → static/uploads/ (deleted post-processing)

Annotated images → static/uploads/ (prefix: annotated_)

History → history.json

Feedback → feedback.json

Improvement data → improvement.json

🛠 Configuration

Variable	Path/Value
UPLOAD_FOLDER	static/uploads
OUTPUT_FOLDER	static/outputs
MODEL_FOLDER	model
DATASET_FOLDER	C:/Users/Dell/Downloads/Bone -Fracture/Bone Fracture/Orginal
ANNOTATION_FILE	annotations.csv in dataset folder
CONFIDENCE_THRESHOLD	60%
🧾 Dataset Format
Folder Structure:
Simple Bone Fracture/

Comminuted Bone Fracture/

Annotation File (annotations.csv):

Column	Description
image_id	Filename of the image
category	Bone category (e.g., Wrist)
bone	Specific bone (e.g., Distal Radius)
fracture_type	Simple or Comminuted
fracture_x/y	Coordinates of fracture
severity	Score from 0 to 100
🧪 Troubleshooting
Model not loaded: Ensure .pkl files exist or a dataset is available.

Invalid X-ray: Use grayscale PNG/JPEG with clear bone visibility.

Failed to load image: Check path/format, and available disk space.

Low confidence: Try clearer images or retrain the model.

Permission errors: Ensure read/write access for /static/ and /model/ folders.

📄 Logging
Logs stored in app.log and shown in the console.

Log levels: INFO, DEBUG, WARNING, ERROR

🤝 Contributing
Fork the repository

Create a new branch

bash
Copy
Edit
git checkout -b feature/new-feature
Commit your changes

bash
Copy
Edit
git commit -m "Add new feature"
Push and open a Pull Request

bash
Copy
Edit
git push origin feature/new-feature
📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
OpenCV - Image processing

Scikit-learn - Machine learning models

Flask - Web framework

Bootstrap - Frontend styling (assumed)

yaml
Copy
Edit

---

Let me know if you'd like a downloadable `README.md` file or if you want a customized logo/banner for your GitHub page.







