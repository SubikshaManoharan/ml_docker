# Breast Cancer Prediction - MLOps Project

A complete end-to-end machine learning pipeline to predict breast cancer (benign or malignant) using the WDBC dataset, with MLOps best practices including MLflow tracking, Docker containerization, and Railway deployment.

---

## Features

- Full EDA and preprocessing
- Feature selection to reduce multicollinearity
- Multiple ML models benchmarked (Random Forest, SVM, etc.)
- **MLflow** for experiment tracking and model versioning
- **Docker** containerization for consistent deployments
- **Railway** cloud deployment
- Streamlit web app for real-time prediction

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML/Data | Python, scikit-learn, pandas, numpy |
| MLOps | MLflow |
| Containerization | Docker |
| Deployment | Railway, Streamlit |
| Visualization | seaborn, matplotlib |

---

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd Breast-Cancer-Prediction-Using-ML

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Train model with MLflow tracking
python train_and_save_model.py

# Run the Streamlit app
streamlit run app.py
```

### View MLflow Dashboard

```bash
mlflow ui
# Open http://localhost:5000 in browser
```

---

## Docker

### Build and Run Locally

```bash
# Build Docker image
docker build -t breast-cancer-prediction .

# Run container
docker run -p 8501:8501 breast-cancer-prediction

# Open http://localhost:8501
```

---

## Deploy to Railway

### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add MLOps: MLflow, Docker, Railway config"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" -> "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the Dockerfile and deploy
   - Get your public URL from Railway dashboard

---

## MLflow Tracking

The training script logs:
- **Parameters**: n_estimators, random_state, test_size, model_type
- **Metrics**: accuracy, precision, recall, f1_score
- **Artifacts**: trained model (pickle + MLflow format)

Run `mlflow ui` after training to view experiment results.

---

## Project Structure

```
├── app.py                      # Streamlit web application
├── train_and_save_model.py     # Training script with MLflow
├── breast_cancer.csv           # Dataset
├── brest_cancer.pkl            # Trained model
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── railway.json                # Railway deployment config
├── Procfile                    # Process file for deployment
├── .dockerignore               # Docker ignore file
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~96% |
| Precision | ~95% |
| Recall | ~95% |
| F1 Score | ~95% |

---

## Links

- **Deployed App**: [Your Railway URL]
- **GitHub Repo**: [Your GitHub URL]

---

## Author

[Your Name]

