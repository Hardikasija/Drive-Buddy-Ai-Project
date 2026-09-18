# DrivebuddyAI / Roadzen Technologies — Pav Bhaji Classification Challenge

## Overview

This project is a submission for the DrivebuddyAI / Roadzen Technologies Data Scientist campus placement challenge: classify whether an Instagram post depicts **Pav Bhaji** using **text and metadata only** — no computer vision, no CNNs, no image embeddings, and no OCR.

The supplied image folders are used only to establish labels (`images/1` = Pav Bhaji, `images/0` = Not Pav Bhaji).

## Directory Structure

```text
DrivebuddyAI_PavBhaji/
├── models/
│   └── final_model.joblib
├── notebooks/
│   └── DrivebuddyAI_PavBhaji_Challenge.ipynb
├── src/
│   └── DrivebuddyAI_PavBhaji_Challenge.py
├── reports/
│   ├── DrivebuddyAI_Data_Analysis_Report.md
│   └── DrivebuddyAI_Data_Analysis_Report.pdf
├── outputs/
│   ├── figures/
│   ├── model_results.csv
│   ├── cv_model_comparison.csv
│   ├── predictions.csv
│   ├── feature_importance.csv
│   └── error_analysis.csv
├── .gitignore
├── README.md
├── app.py
├── Hardik.ipynb
└── requirements.txt
```
Note: The raw challenge dataset is not included in this public repository.

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## Run the Python Pipeline

```bash
cd src
python DrivebuddyAI_PavBhaji_Challenge.py --data-dir ../dataset --output-dir ../outputs
```

## Run the Notebook

```bash
cd notebooks
jupyter notebook DrivebuddyAI_PavBhaji_Challenge.ipynb
```

Then run all cells top-to-bottom.

## Methodology

1. Deterministic JSON-to-image filename mapping creates the labels.
2. Captions + hashtags are cleaned and transformed with TF-IDF word 1–2 grams.
3. Structured metadata features are added through a single sklearn `ColumnTransformer`/`Pipeline`.
4. The dataset contains direct target vocabulary in almost all posts of both classes, so the final model uses **strict target-term removal** rather than a target placeholder.
5. Composite groups prevent both owner-level leakage and exact normalized-caption duplication across train/test and cross-validation folds.
6. Logistic Regression, Linear SVM, and Multinomial Naive Bayes are compared with grouped stratified 5-fold CV.
7. The model with the highest grouped CV F1 is selected and evaluated once on the untouched grouped holdout test set.

## Current Corrected Run

- Labeled samples: **452**
- Classes: **269 Not Pav Bhaji / 183 Pav Bhaji**
- Grouped CV winner: **Logistic Regression**
- Final holdout F1: **0.519**
- Final holdout ROC-AUC: **0.630**

The exact values are stored in `outputs/cv_model_comparison.csv` and `outputs/model_results.csv` and documented in the PDF report.


## Streamlit Demo (Optional)

The project also includes a local Streamlit demo that loads the **same fitted pipeline** saved by the training script. It does not retrain the model and does not use images or computer vision.

## Streamlit Demo

The project includes a local Streamlit demo that uses the saved trained pipeline.

Run:

streamlit run app.py

The app loads:
- models/final_model.joblib
- outputs/model_metadata.json

The raw dataset is not required to run the Streamlit demo.

Then launch the app:

```bash
streamlit run app.py
```

The app expects `models/final_model.joblib` and `outputs/model_metadata.json`.

## Important

The challenge does **not** require Streamlit or a web application. The submission package therefore focuses on the requested ML source code and data-analysis report. A demo application can be added later if useful for an interview, but it is not necessary for the placement submission.
