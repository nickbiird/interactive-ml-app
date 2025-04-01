# Interactive ML Model Trainer

This is a simple Streamlit application created for the "Interactive Application Development I" assignment.

## Features

* Select from Seaborn datasets (Tips, Penguins, Titanic, Iris) or upload a CSV.
* Choose target variable (y) and features (X - quantitative & qualitative).
* Select between Random Forest and Extra Trees classification models.
* Configure model hyperparameters (n_estimators, max_depth) and train/test split.
* View classification metrics (Accuracy, Report).
* Visualize results: Confusion Matrix, ROC Curve (for binary classification), Feature Importances.
* Download the trained model pipeline as a pickle file.

## Deployed App

[https://nick-ml-app.streamlit.app](nick-ml-app.streamlit.app)

## How to Run Locally

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

