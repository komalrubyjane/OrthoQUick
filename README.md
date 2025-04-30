

---

```markdown
# Bone Fracture Detection and Analysis

A Flask-based web application for analyzing X-ray images to detect bone fractures, classify bone categories, and assess fracture severity and healing progress. It leverages machine learning (Random Forest) and image processing (OpenCV) to generate detailed reports and visual annotations.

---

## 🚀 Features

- **Single X-ray Analysis**: Upload an X-ray to detect fractures, identify bone category (e.g., Wrist, Elbow), fractured bone, and severity.
- **Difference Analysis**: Compare two X-rays (before and after) to assess healing progress.
- **History Tracking**: Stores analysis results and allows CSV export.
- **Feedback System**: Users can submit feedback to improve the system.
- **Model Retraining**: Retrain ML models using annotated datasets.
- **Image Enhancement**: Clean text, enhance clarity, and generate heatmaps.
- **Responsive UI**: Simple web interface for interaction.

---

## 🧠 Technologies Used

- **Backend**: Flask (Python), Flask-Compress
- **Machine Learning**: Scikit-learn (RandomForestClassifier, RandomForestRegressor)
- **Image Processing**: OpenCV, Pillow
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML, JavaScript, CSS (Bootstrap for styling)
- **Logging**: Python's `logging` module
- **Storage**: JSON for history/feedback, CSV for annotations/exports

---

## 🛠️ Prerequisites

- Python 3.8+
- pip (Python package manager)
- X-ray image dataset:
  - Folder structure: `Simple Bone Fracture/` and `Comminuted Bone Fracture/`
- Optional:
  - Pre-trained models: `model.pkl`, `fracture_model.pkl`

---

## 📦 Installation

```bash
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection
```

### Set up virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install flask flask-compress opencv-python numpy scikit-learn pillow pandas
```

---

## 📂 Dataset Setup

- Place dataset at the path specified by `DATASET_FOLDER` in the code.
  - Default: `C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal`
- Ensure the following folders exist:
  - `static/uploads/` — uploaded/annotated images
  - `static/outputs/` — processed output images
  - `model/` — trained models
- Optional file: `annotations.csv` with the following columns:

```
image_id, category, bone, fracture_type, fracture_x, fracture_y, severity
```

If not present, mock annotations will be generated.

---

## 🧪 Usage

### Run the server:

```bash
python app.py
```

Access the app at: [http://localhost:5000](http://localhost:5000)

### UI Pages:

- `/` — Home & initialization logs
- `/analysis` — Single X-ray analysis
- `/difference` — Compare two X-rays
- `/history` — Analysis history with CSV export
- Feedback form and retrain model options included

---

## 📡 API Endpoints

| Method | Endpoint              | Description                                       |
|--------|-----------------------|---------------------------------------------------|
| POST   | `/analysis`           | Upload X-ray and return analysis in JSON         |
| POST   | `/difference`         | Upload two X-rays to assess healing              |
| GET    | `/get_history`        | Fetch history as JSON                            |
| GET    | `/export_history`     | Download history as CSV                          |
| POST   | `/submit_feedback`    | Submit feedback (ID, rating, comments)           |
| POST   | `/retrain_model`      | Retrain models using dataset                     |

---

## ⚙️ Configuration

| Key               | Default Value                                                        |
|------------------|----------------------------------------------------------------------|
| `UPLOAD_FOLDER`   | `static/uploads/`                                                    |
| `OUTPUT_FOLDER`   | `static/outputs/`                                                    |
| `MODEL_FOLDER`    | `model/`                                                             |
| `DATASET_FOLDER`  | `C:\Users\Dell\Downloads\Bone -Fracture\Bone Fracture\Orginal`       |
| `ANNOTATION_FILE` | `annotations.csv` in dataset folder                                  |

### Model Parameters:

- Random Forest: 100 estimators, `random_state=42`
- Confidence threshold: 60%

### Image Requirements:

- Format: PNG or JPEG
- Size: Minimum 10×10 px
- Type: Grayscale or low-color X-rays

---

## 🧠 Models

- **Bone Classifier**: Predicts bone category (`model.pkl`)
- **Fracture Models** (`fracture_model.pkl`):
  - `fracture_clf_bone`: Classifies fractured bone
  - `fracture_clf_type`: Simple or Comminuted
  - `fracture_reg_x`, `fracture_reg_y`: Fracture coordinates

---

## 📝 Logging

- Logs written to `app.log` and console
- Levels: INFO, DEBUG, WARNING, ERROR
- Logs include:
  - Dependency versions
  - Directory creation, file ops
  - Model training/load status
  - Image processing/debug info

---

## ❗ Troubleshooting

- **Model not loaded**:
  - Check for `model.pkl`, `fracture_model.pkl` or valid dataset.
- **Invalid X-ray**:
  - Ensure format is grayscale PNG/JPEG, clear structure, size > 10×10.
- **Load errors**:
  - Confirm correct paths and disk space.
- **Low confidence (<60%)**:
  - Use better image or retrain model.
- **Permission issues**:
  - Ensure read/write access to `static/`, `model/`.

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch:  
   `git checkout -b feature/your-feature`
3. Commit changes:  
   `git commit -m "Add feature"`
4. Push branch:  
   `git push origin feature/your-feature`
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **OpenCV** – Image processing  
- **Scikit-learn** – Machine learning  
- **Flask** – Backend framework  
- **Bootstrap** – UI styling  
```

---

Let me know if you want this as a downloadable file or if you'd like help generating a `requirements.txt`.
