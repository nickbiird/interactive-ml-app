import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
import base64 # To create download link for model

# --- Page Configuration ---
st.set_page_config(
    page_title="Simple ML Trainer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_data(dataset_name):
    """Loads a dataset from Seaborn or an uploaded CSV."""
    if isinstance(dataset_name, str): # Seaborn dataset
        try:
            # Limiting to specific datasets known to be good for classification
            if dataset_name == 'tips':
                df = sns.load_dataset('tips')
            elif dataset_name == 'penguins':
                df = sns.load_dataset('penguins')
            elif dataset_name == 'titanic':
                df = sns.load_dataset('titanic')
                # Basic preprocessing for titanic often needed
                df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
                df['age'].fillna(df['age'].median(), inplace=True)
                df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
            else: # Default or fallback
                 df = sns.load_dataset('iris') # Iris is simple
            st.success(f"Loaded '{dataset_name}' dataset.")
            return df.dropna() # Simple NaN handling for this example
        except Exception as e:
            st.error(f"Error loading Seaborn dataset '{dataset_name}': {e}")
            return None
    elif dataset_name is not None: # Uploaded file
        try:
            df = pd.read_csv(dataset_name)
            st.success("Loaded uploaded CSV file.")
            return df.dropna() # Simple NaN handling
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    return None

def get_model_and_params(model_name):
    """Returns model instance and default hyperparameters based on name."""
    params = {}
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        params['n_estimators'] = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
        params['max_depth'] = st.sidebar.slider("Max Tree Depth (max_depth)", 1, 50, 10, 1)
        params['criterion'] = st.sidebar.selectbox("Criterion", ["gini", "entropy"], index=0)
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42)
        params['n_estimators'] = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100, 50)
        params['max_depth'] = st.sidebar.slider("Max Tree Depth (max_depth)", 1, 50, 10, 1)
        params['criterion'] = st.sidebar.selectbox("Criterion", ["gini", "entropy"], index=0)
    else:
        st.sidebar.error("Select a valid model")
        return None, None
    return model, params

def create_download_link(model_pkl, filename="trained_model.pkl"):
    """Generates a link to download the pickled model file."""
    b64 = base64.b64encode(model_pkl).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">Download Trained Model (pickle)</a>'
    return href

# --- Sidebar Configuration ---
st.sidebar.header("📊 Data & Model Setup")

# 1. Dataset Selection
st.sidebar.subheader("1. Choose Data")
data_source = st.sidebar.radio("Select data source:", ("Seaborn Datasets", "Upload CSV"))

selected_dataset = None
uploaded_file = None

if data_source == "Seaborn Datasets":
    # Limit dataset choices
    available_datasets = ['tips', 'penguins', 'titanic', 'iris']
    selected_dataset = st.sidebar.selectbox("Select a Seaborn dataset:", available_datasets, index=0)
    df = load_data(selected_dataset)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = None # No data loaded yet

# --- Main Application Area ---
st.title("🧪 Simple ML Model Trainer App")
st.markdown("Train simple classification models interactively!")

if df is not None:
    st.markdown("### Data Preview")
    st.dataframe(df.head())

    # --- Column Setup for UI ---
    col1, col2 = st.columns([1, 2]) # Sidebar-like column on left, results on right

    with col1:
        st.subheader("⚙️ Configuration")

        # Using a form for batch input
        with st.form("ml_setup_form"):
            st.write("**Select Features and Target**")
            # 2. Feature and Target Selection
            potential_targets = [col for col in df.columns if df[col].nunique() < 15 and df[col].dtype in ['object', 'category', 'int64']] # Heuristic for classification target
            target_variable = st.selectbox("Select Target Variable (y):", potential_targets, index=0)

            # Infer quantitative and qualitative features (excluding target)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Remove target from feature lists
            if target_variable in numeric_cols: numeric_cols.remove(target_variable)
            if target_variable in categorical_cols: categorical_cols.remove(target_variable)

            selected_numeric = st.multiselect("Select Quantitative Features (X):", numeric_cols, default=numeric_cols[:min(len(numeric_cols), 3)]) # Default to first 3
            selected_categorical = st.multiselect("Select Qualitative Features (X):", categorical_cols, default=categorical_cols[:min(len(categorical_cols), 2)]) # Default to first 2

            st.write("**Model & Training Settings**")
            # 3. Model Selection
            model_options = ["Random Forest", "Extra Trees"]
            selected_model_name = st.selectbox("Select Model:", model_options, index=0)

            # 4. Basic Training Parameters
            test_set_size = st.slider("Test Set Size (%)", 10, 50, 25, 5)
            random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)

            # 5. Model Specific Hyperparameters (Inside Sidebar for this example)
            st.sidebar.subheader("2. Configure Hyperparameters")
            if selected_model_name:
                st.sidebar.write(f"**Parameters for {selected_model_name}:**")
                model_instance, model_params = get_model_and_params(selected_model_name)

            # Submit button for the form
            submitted = st.form_submit_button("🚀 Train Model")

    # --- Training and Results Logic (Runs after form submission) ---
    if submitted:
        if not target_variable:
            st.warning("Please select a target variable.")
        elif not selected_numeric and not selected_categorical:
            st.warning("Please select at least one feature (Quantitative or Qualitative).")
        elif model_instance is None:
             st.warning("Please select a valid model.")
        else:
            with col2: # Display results in the second column
                st.subheader("📊 Results & Evaluation")
                try:
                    # Prepare features (X) and target (y)
                    features = selected_numeric + selected_categorical
                    X = df[features]
                    y = df[target_variable]

                    # Preprocessing Steps
                    # 1. Encode Target Variable if it's categorical/object
                    le = LabelEncoder()
                    if y.dtype == 'object' or y.dtype.name == 'category':
                        y = le.fit_transform(y)
                        target_classes = le.classes_ # Store class names for plotting later
                    else:
                        target_classes = sorted(y.unique()) # For numerical targets treated as classes

                    # 2. Preprocess Features: OneHotEncode categorical, pass through numerical
                    numeric_transformer = Pipeline(steps=[('passthrough', 'passthrough')]) # No scaling for Trees needed here
                    categorical_transformer = Pipeline(steps=[
                        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handle potential unknown categories in test set
                    ])

                    # Create preprocessor with ColumnTransformer
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, selected_numeric),
                            ('cat', categorical_transformer, selected_categorical)
                        ])

                    # Create the full pipeline with preprocessing and model
                    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                   ('classifier', model_instance.set_params(**model_params, random_state=random_state))])

                    # Split Data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_set_size/100.0, random_state=random_state, stratify=(y if len(target_classes) > 1 else None)
                    )

                    # Train Model
                    with st.spinner("Training model... please wait."):
                        model_pipeline.fit(X_train, y_train)
                    st.success("✅ Model Training Complete!")

                    # Make Predictions
                    y_pred = model_pipeline.predict(X_test)
                    y_pred_proba = model_pipeline.predict_proba(X_test) # For ROC curve

                    # --- Display Metrics ---
                    st.markdown("#### Performance Metrics")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric(label="Accuracy Score", value=f"{accuracy:.3f}")

                    st.text("Classification Report:")
                    report = classification_report(y_test, y_pred, target_names=[str(tc) for tc in target_classes], output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())


                    # --- Display Plots ---
                    st.markdown("#### Visualizations")
                    fig_col1, fig_col2 = st.columns(2)

                    # Plot 1: Confusion Matrix
                    with fig_col1:
                        st.write("**Confusion Matrix**")
                        cm = confusion_matrix(y_test, y_pred, labels=model_pipeline.classes_)
                        disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(tc) for tc in target_classes])
                        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                        disp_cm.plot(ax=ax_cm, cmap='Blues')
                        st.pyplot(fig_cm)
                        plt.close(fig_cm) # Close plot to free memory

                    # Plot 2: ROC Curve (only for binary or handle multi-class)
                    with fig_col2:
                        st.write("**ROC Curve & AUC**")
                        # Check if binary or multi-class and plot accordingly
                        if len(target_classes) == 2:
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=model_pipeline.classes_[1])
                            roc_auc = auc(fpr, tpr)
                            disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=selected_model_name)
                            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                            disp_roc.plot(ax=ax_roc)
                            ax_roc.set_title("ROC Curve")
                            st.pyplot(fig_roc)
                            plt.close(fig_roc)
                            st.write(f"AUC: {roc_auc:.3f}")
                        elif len(target_classes) > 2:
                             st.info("ROC Curve plotting for multi-class is more complex and not shown in this simple version. AUC can be calculated.")
                             # You could calculate macro/micro average AUC here if needed
                        else:
                             st.warning("Only one class detected in target. Cannot plot ROC curve.")


                    # Plot 3: Feature Importance (if available)
                    st.markdown("#### Feature Importances")
                    if hasattr(model_pipeline.named_steps['classifier'], 'feature_importances_'):
                        try:
                            # Get feature names AFTER preprocessing (OneHotEncoder creates new names)
                            feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
                            importances = model_pipeline.named_steps['classifier'].feature_importances_
                            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

                            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax_imp) # Show top 15
                            ax_imp.set_title(f"Feature Importances ({selected_model_name})")
                            st.pyplot(fig_imp)
                            plt.close(fig_imp)
                        except Exception as e:
                            st.warning(f"Could not plot feature importances: {e}")
                    else:
                        st.info(f"The selected model ({selected_model_name}) does not provide standard feature importances.")

                    # --- Model Saving ---
                    st.markdown("#### Model Download")
                    try:
                        pickled_model = pickle.dumps(model_pipeline)
                        st.markdown(create_download_link(pickled_model, f"{selected_dataset or 'custom_data'}_{selected_model_name.replace(' ','_')}_pipeline.pkl"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error pickling the model: {e}")

                except Exception as e:
                    st.error(f"An error occurred during training or evaluation: {e}")
                    st.exception(e) # Show full traceback for debugging

elif data_source == "Upload CSV" and uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
elif data_source == "Seaborn Datasets" and df is None:
     st.error("Failed to load the selected Seaborn dataset.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed with Streamlit 💡")