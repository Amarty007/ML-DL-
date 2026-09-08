"""
ML + DL Playground
-------------------
Ek interactive Streamlit app jisme:
- CSV data upload karo
- Machine Learning models train/test karo (Logistic Regression, Decision Tree,
  Random Forest, KNN, SVM, Naive Bayes)
- Deep Learning model (Artificial Neural Network) train karo with LIVE
  epoch-by-epoch training curve
- Sab models ke test results compare karo (Accuracy, Confusion Matrix, Report)

Run karne ke liye:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# Optional boosters are disabled at startup to keep the Streamlit process light.
# The core model set below does not depend on either package.
XGBClassifier = None
LGBMClassifier = None

# ---------------------------------------------------------------------------
# Page config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML + DL Playground", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### ML + DL Playground\nInteractive machine learning and deep learning training platform."
    }
)

# Custom CSS for clean UI
st.markdown("""
<style>
    /* Main container */
    .main { padding-top: 2rem; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: white; }
    
    /* Headers */
    h1 { 
        color: #1f3a93; 
        font-weight: 700; 
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    h2 { 
        color: #2d5aa3; 
        font-weight: 600; 
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    h3 { color: #2d5aa3; font-weight: 600; }
    
    /* Cards and boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f4ff 0%, #f0e8ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4f4dd 0%, #e0ffe0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e7ff;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #e0e7ff;
        border-radius: 8px;
    }
    
    /* Text styling */
    .caption { color: #666; font-size: 0.9rem; }
    
    /* Divider */
    hr { border-color: #e0e7ff; }
</style>
""", unsafe_allow_html=True)

# Title with gradient effect
col_title_1, col_title_2 = st.columns([3, 1])
with col_title_1:
    st.markdown("# 🤖 ML + DL Playground")
    st.markdown("**Interactive Machine Learning & Deep Learning Platform**", 
                help="Train ML and DL models on your data with real-time visualization")
with col_title_2:
    st.info("📚 Advanced AI Training", icon="ℹ️")

# Session state to store results across model runs
if "results" not in st.session_state:
    st.session_state.results = {}  # {model_name: {metrics...}}

if "model_artifacts" not in st.session_state:
    st.session_state.model_artifacts = {}


# ---------------------------------------------------------------------------
# STEP 1: Upload Data
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Step 1: Upload Your Dataset")

col_upload_1, col_upload_2 = st.columns([2, 1])
with col_upload_1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a classification dataset in CSV format"
    )
with col_upload_2:
    st.markdown("### Supported Formats")
    st.markdown("• **CSV** files\n• Tabular data")

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Getting Started</h4>
        <p>Upload a CSV file to begin. Supported datasets include:</p>
        <ul>
            <li>Iris Dataset</li>
            <li>Titanic Dataset</li>
            <li>Your custom classification data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 50000

if uploaded_file.size > MAX_UPLOAD_BYTES:
    st.error(
        f"Uploaded CSV is too large ({uploaded_file.size / (1024 * 1024):.1f} MB). "
        f"Please upload a file smaller than {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB."
    )
    st.stop()

df = pd.read_csv(uploaded_file, nrows=MAX_UPLOAD_ROWS)
if len(df) == MAX_UPLOAD_ROWS:
    st.warning(
        f"Only the first {MAX_UPLOAD_ROWS:,} rows were loaded to keep the app responsive."
    )
st.markdown(f"""
<div class="success-box">
    <strong>✅ Dataset Loaded Successfully!</strong><br>
    📈 Shape: <strong>{df.shape[0]} rows × {df.shape[1]} columns</strong>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 **View Dataset Preview & Info**", expanded=False):
    tab1, tab2 = st.tabs(["Data Preview", "Column Information"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True)


# ---------------------------------------------------------------------------
# STEP 1.4: Select Target Column FIRST
# ---------------------------------------------------------------------------
# FIX (Bug 3): Target column must be chosen BEFORE ID-column auto-detection.
# Previously, ID detection ran first and could silently drop the target column
# itself (e.g. a target named "diagnosis_id" or "class_code", or one that is
# >95% unique) before the user ever got to pick it in the dropdown.
st.markdown("---")
st.markdown("## 🎯 Step 1.4: Select Target Column")
target_col = st.selectbox(
    "Select Target Column",
    df.columns,
    index=len(df.columns) - 1,
    help="Choose the column to predict. Do this before ID auto-detection so your target is never accidentally excluded."
)


# ---------------------------------------------------------------------------
# Auto-detect ID Columns
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🔍 Step 1.5: Auto-Detect & Exclude ID Columns")

def detect_id_columns(df_input, exclude_col=None):
    """
    Auto-detect ID-like columns:
    - Column name contains 'id', 'index', 'pk', 'serial'
    - All values are unique or nearly unique (>95% unique)
    - Integer or string type

    `exclude_col` (e.g. the chosen target column) is never flagged as an ID
    column, even if its name or uniqueness would otherwise match.
    """
    id_columns = []
    
    for col in df_input.columns:
        if col == exclude_col:
            continue

        col_lower = col.lower()
        unique_ratio = df_input[col].nunique() / len(df_input)
        
        # Check by name
        is_id_by_name = any(keyword in col_lower for keyword in ['id', 'index', 'pk', 'serial', 'code'])
        
        # Check by uniqueness (>95%)
        is_id_by_uniqueness = unique_ratio > 0.95 and df_input[col].dtype in ['int64', 'int32', 'object']
        
        if is_id_by_name or is_id_by_uniqueness:
            id_columns.append(col)
    
    return id_columns

# FIX (Bug 3): pass target_col as exclude_col so it's never dropped here.
detected_ids = detect_id_columns(df, exclude_col=target_col)

if detected_ids:
    st.info(f"🤖 **Auto-detected potential ID columns:** {', '.join(detected_ids)}")
    
    col_auto1, col_auto2 = st.columns([2, 1])
    with col_auto1:
        exclude_ids = st.checkbox(
            "✅ Exclude these ID columns automatically",
            value=True,
            help="ID columns won't be used as features"
        )
    with col_auto2:
        st.metric("ID Columns Found", len(detected_ids))
    
    if exclude_ids:
        df = df.drop(columns=detected_ids)
        st.success(f"✅ Excluded {len(detected_ids)} ID column(s) from features")
else:
    st.success("✅ No ID columns detected. All columns retained.")


# ---------------------------------------------------------------------------
# STEP 2: Preprocessing Configuration
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ⚙️ Step 2: Configure Preprocessing")

config_col2, config_col3 = st.columns([1, 1])
with config_col2:
    test_size = st.slider(
        "Test Data Percentage",
        10, 50, 20,
        step=5,
        help="Percentage of data to use for testing"
    ) / 100
with config_col3:
    random_state = st.number_input(
        "Random Seed",
        value=42,
        help="For reproducibility"
    )

st.caption(f"🎯 Target column selected in Step 1.4: **{target_col}**")

# Feature Count Confirmation - BEFORE preprocessing
available_features_before = [col for col in df.columns if col != target_col]
st.markdown("### 📊 Feature Count Confirmation")
feature_confirm_col1, feature_confirm_col2, feature_confirm_col3 = st.columns(3)
with feature_confirm_col1:
    st.metric("📋 Total Columns", len(df.columns))
with feature_confirm_col2:
    st.metric("🎯 Target Column", 1)
with feature_confirm_col3:
    st.metric("✨ Available Features", len(available_features_before))

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]
original_feature_data = X.copy()

# Handle missing values in features (simple fill)
numeric_fill_values = {}
categorical_fill_values = {}
for c in X.columns:
    # Check if column is numeric using proper type checking
    if pd.api.types.is_numeric_dtype(X[c]):
        numeric_fill_values[c] = X[c].mean()
        X[c] = X[c].fillna(numeric_fill_values[c])
    else:
        # For non-numeric (string, object, etc.), fill with mode or default
        categorical_fill_values[c] = (
            X[c].mode()[0] if not X[c].mode().empty else "missing"
        )
        X[c] = X[c].fillna(categorical_fill_values[c])

# Encode categorical features
# FIX (OOM bug): pd.get_dummies() on a high-cardinality text column (e.g. a
# "Name"/"Address"/"Comments" column with thousands of unique values) creates
# one new column PER unique value. On a dataset of tens of thousands of rows
# this can balloon to gigabytes of RAM instantly and get the process OOM-killed
# by the OS (visible as Streamlit silently printing "Stopping...").
#
# Safeguard: any categorical column above ONEHOT_CARDINALITY_LIMIT unique
# values is label-encoded (a single integer column) instead of one-hot
# encoded. Columns below the limit keep the original one-hot behavior.
ONEHOT_CARDINALITY_LIMIT = 50

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
high_card_cols = [c for c in cat_cols if X[c].nunique() > ONEHOT_CARDINALITY_LIMIT]
low_card_cols = [c for c in cat_cols if c not in high_card_cols]
high_card_maps = {}

if high_card_cols:
    st.warning(
        f"⚠️ **High-cardinality columns label-encoded instead of one-hot** "
        f"(too many unique values for safe one-hot encoding, would exhaust memory): "
        f"{', '.join(high_card_cols)}"
    )
    for c in high_card_cols:
        encoder = LabelEncoder()
        X[c] = encoder.fit_transform(X[c].astype(str))
        high_card_maps[c] = {
            str(value): int(index)
            for index, value in enumerate(encoder.classes_)
        }

if low_card_cols:
    st.info(f"🔄 **Encoding categorical columns:** {', '.join(low_card_cols)}")
    X = pd.get_dummies(X, columns=low_card_cols, drop_first=True, dtype=np.float64)

cat_cols = low_card_cols  # for the info message shown further below stays accurate

# Force every remaining column into numeric form. This keeps one-hot dummy columns
# like Sex_male / Ticket_1234 as 0/1 floats, while dropping any leftover text columns.
X = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))
remaining_non_numeric = X.columns[X.isna().all()].tolist()
if remaining_non_numeric:
    st.warning(f"⚠️ **Dropping non-numeric columns:** {', '.join(remaining_non_numeric)}")
    X = X.drop(columns=remaining_non_numeric)

# Convert all columns to float64 to ensure compatibility
X = X.astype("float64")
training_feature_defaults = X.median(numeric_only=True).to_dict()
preprocessed_feature_columns = list(X.columns)

# Encode target if it's categorical/text
label_encoder = None
y_is_numeric = pd.api.types.is_numeric_dtype(y)

if not y_is_numeric:
    # Non-numeric target: must encode with LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    st.info(f"🏷️ **Target classes:** {', '.join(map(str, label_encoder.classes_))}")
else:
    # Numeric target: try to convert to int, but if it fails, encode anyway
    try:
        y = y.astype("int64")
    except (ValueError, TypeError):
        # Conversion failed, treat as categorical
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.info(f"🏷️ **Target classes:** {', '.join(map(str, label_encoder.classes_))}")

# Ensure target is numeric and keep pandas semantics for later statistical checks
# (nunique/value_counts are pandas methods, so we should keep a Series here)
y = pd.Series(y, dtype="int64")
y_series = y
num_classes = y_series.nunique()
class_counts = y_series.value_counts()
min_class_count = int(class_counts.min()) if not class_counts.empty else 0
y_unique_ratio = y_series.nunique() / max(len(y_series), 1)
looks_continuous_target = pd.api.types.is_numeric_dtype(y_series) and y_unique_ratio > 0.2

# Stats in columns
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    st.metric("Total Classes", num_classes)
with stat_col2:
    st.metric("Min Class Count", min_class_count)
with stat_col3:
    st.metric("Total Samples", len(y_series))
with stat_col4:
    st.metric("Feature Count", X.shape[1])

if num_classes < 2:
    st.error("❌ Target column must have at least 2 classes for classification.", icon="⚠️")
    st.stop()

if looks_continuous_target:
    st.error(
        "❌ Target column appears to be continuous. Please select a categorical target "
        "with fewer unique classes for classification.",
        icon="⚠️"
    )
    st.stop()

if min_class_count < 2:
    st.error(
        "❌ Some target classes have too few samples. Each class needs at least 2 examples.",
        icon="⚠️"
    )
    st.stop()

can_stratify = num_classes > 1 and min_class_count >= 2 and not looks_continuous_target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y if can_stratify else None,
)

# Ensure y_train and y_test are numpy arrays for model training
# while keeping y as a pandas Series for earlier target validation steps.
y_train = np.array(y_train, dtype="int64")
y_test = np.array(y_test, dtype="int64")

split_col1, split_col2 = st.columns(2)
with split_col1:
    st.metric("🏋️ Training Samples", X_train.shape[0])
with split_col2:
    st.metric("🧪 Testing Samples", X_test.shape[0])


# ---------------------------------------------------------------------------
# STEP 2.5: Feature Selection
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🎯 Step 2.5: Feature Selection & Confirmation")

st.markdown("### Select Features to Use in Training")

# Show all available features
feature_count_before = X_train.shape[1]
all_features = list(X_train.columns)

col_feat1, col_feat2 = st.columns([3, 1])
with col_feat1:
    st.info(f"📊 Total features after preprocessing: **{feature_count_before}**")
with col_feat2:
    feature_selection_mode = st.radio(
        "Selection Mode",
        ["All Features", "Manual Select"],
        horizontal=True,
        help="Choose how to select features"
    )

if feature_selection_mode == "Manual Select":
    selected_features = st.multiselect(
        "Choose features for model training",
        all_features,
        default=all_features,
        help="At least 1 feature must be selected"
    )
    
    if not selected_features:
        st.error("❌ Please select at least 1 feature to continue.", icon="⚠️")
        st.stop()
    
    # Filter X_train and X_test based on selection
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    feature_count_after = len(selected_features)
else:
    selected_features = all_features
    feature_count_after = feature_count_before

# Feature Count Confirmation - AFTER selection
feat_confirm_col1, feat_confirm_col2, feat_confirm_col3 = st.columns(3)
with feat_confirm_col1:
    st.metric("📊 Initial Features", feature_count_before)
with feat_confirm_col2:
    st.metric("✨ Selected Features", feature_count_after)
with feat_confirm_col3:
    if feature_count_after < feature_count_before:
        reduction_pct = ((feature_count_before - feature_count_after) / feature_count_before) * 100
        st.metric("🔽 Reduction", f"{reduction_pct:.1f}%")
    else:
        st.metric("✅ All Features", "Used")

# Show selected features
with st.expander("📋 **View Selected Features**"):
    X_selected = X_train[selected_features]
    
    # Calculate mean and std only for numeric columns
    mean_values = []
    std_values = []
    for feat in selected_features:
        if pd.api.types.is_numeric_dtype(X_selected[feat]):
            mean_values.append(f"{X_selected[feat].mean():.4f}")
            std_values.append(f"{X_selected[feat].std():.4f}")
        else:
            mean_values.append("N/A")
            std_values.append("N/A")
    
    feat_info = pd.DataFrame({
        "Feature Name": selected_features,
        "Data Type": X_selected.dtypes.astype(str),
        "Mean": mean_values,
        "Std Dev": std_values,
    })
    st.dataframe(feat_info, use_container_width=True)
    st.caption(f"✅ Ready to train models with **{feature_count_after}** feature(s)")

# Scale features AFTER feature selection (needed for KNN, SVM, Logistic Regression, and ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final validation: ensure no NaN or infinite values
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    st.error("❌ Invalid values in training data (NaN or Inf). Please check your data.", icon="⚠️")
    st.stop()

if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
    st.error("❌ Invalid values in test data (NaN or Inf). Please check your data.", icon="⚠️")
    st.stop()


# ---------------------------------------------------------------------------
# Helper function: evaluate + store results
# ---------------------------------------------------------------------------
def classify_model_fit(train_accuracy, test_accuracy):
    gap = train_accuracy - test_accuracy

    if train_accuracy < 0.75 and test_accuracy < 0.75:
        return "Underfitting"
    if gap > 0.12 and train_accuracy >= 0.9:
        return "Overfitting"
    return "Best Fitting"


def evaluate_and_store(name, y_true, y_pred, train_accuracy=None, model=None, model_type="sklearn"):
    test_accuracy = accuracy_score(y_true, y_pred)
    if train_accuracy is None:
        train_accuracy = test_accuracy

    fit_status = classify_model_fit(train_accuracy, test_accuracy)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    st.session_state.results[name] = {
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Accuracy": test_accuracy,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Fit Status": fit_status,
    }
    st.session_state.model_artifacts[name] = {
        "model": model,
        "model_type": model_type,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "test_features": original_feature_data.loc[X_test.index].copy(),
        "test_indices": X_test.index.to_numpy(),
        "feature_columns": list(X_train.columns),
        "target_classes": label_encoder.classes_.tolist() if label_encoder is not None else None,
        "scaler": scaler,
        "preprocessing": {
            "raw_feature_columns": list(original_feature_data.columns),
            "numeric_fill_values": numeric_fill_values,
            "categorical_fill_values": categorical_fill_values,
            "high_card_maps": high_card_maps,
            "low_card_columns": low_card_cols,
            "preprocessed_feature_columns": preprocessed_feature_columns,
            "training_feature_defaults": training_feature_defaults,
        },
    }

    # Metrics in nice columns
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    with metric_col1:
        st.metric("📈 Train Acc", f"{train_accuracy:.4f}")
    with metric_col2:
        st.metric("✅ Test Acc", f"{test_accuracy:.4f}")
    with metric_col3:
        st.metric("🎯 Precision", f"{prec:.4f}")
    with metric_col4:
        st.metric("🔍 Recall", f"{rec:.4f}")
    with metric_col5:
        st.metric("⚖️ F1 Score", f"{f1:.4f}")

    st.markdown("### 🧾 Result Understanding")
    st.caption("These metrics tell you how well the model worked on unseen data.")
    col_explain_1, col_explain_2, col_explain_3, col_explain_4 = st.columns(4)
    with col_explain_1:
        st.info("**Accuracy** = overall correct predictions over all samples.")
    with col_explain_2:
        st.info("**Precision** = of predicted positives, how many were actually correct.")
    with col_explain_3:
        st.info("**Recall** = of real positives, how many the model successfully found.")
    with col_explain_4:
        st.info("**F1 Score** = balance between precision and recall.")

    if fit_status == "Underfitting":
        st.warning(
            "⚠️ **Underfitting**: model is too simple or not trained well. "
            "Both training and testing accuracy are low, so it is not learning the pattern properly. "
            "Try a more complex model, tune hyperparameters, or improve feature quality."
        )
    elif fit_status == "Overfitting":
        st.warning(
            "⚠️ **Overfitting**: model is learning the train data too strongly and performing worse on test data. "
            "The gap between train and test accuracy is large. Use regularization, less complexity, more data, or cross-validation."
        )
    else:
        st.success(
            "✅ **Best Fitting**: model has a good balance between learning well and generalizing to new data. "
            "Train and test performance are close and healthy."
        )

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Classification Report
    with st.expander("📋 **Detailed Classification Report**"):
        report_text = classification_report(y_true, y_pred, zero_division=0)
        st.code(report_text, language="text")


def display_target_values(values, target_classes=None):
    values = np.asarray(values)
    if target_classes is not None:
        return np.asarray(target_classes, dtype=object)[values.astype(int)]
    return values


def build_prediction_table(model_name, artifact):
    test_features = artifact["test_features"].copy().reset_index(drop=True)
    actual = display_target_values(artifact["y_true"], artifact["target_classes"])
    predicted = display_target_values(artifact["y_pred"], artifact["target_classes"])
    table = test_features
    table.insert(0, "Test Row", artifact["test_indices"])
    table["Actual Label"] = actual
    table["Predicted Label"] = predicted
    table["Correct"] = table["Actual Label"].to_numpy() == table["Predicted Label"].to_numpy()
    table["Status"] = np.where(table["Correct"], "✅ Correct", "❌ Incorrect")
    return table


def preprocess_unseen_features(unseen_df, artifact):
    """Apply the exact feature preparation used by the selected model."""
    preprocessing = artifact["preprocessing"]
    features = unseen_df.drop(columns=[target_col], errors="ignore").copy()
    raw_columns = preprocessing["raw_feature_columns"]
    features = features.reindex(columns=raw_columns)

    for column, fill_value in preprocessing["numeric_fill_values"].items():
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            features[column] = features[column].fillna(fill_value)
    for column, fill_value in preprocessing["categorical_fill_values"].items():
        if column in features:
            features[column] = features[column].fillna(fill_value).astype(str)

    for column, value_map in preprocessing["high_card_maps"].items():
        if column in features:
            features[column] = features[column].astype(str).map(value_map).fillna(-1)

    if preprocessing["low_card_columns"]:
        features = pd.get_dummies(
            features,
            columns=preprocessing["low_card_columns"],
            drop_first=True,
            dtype=np.float64,
        )
    features = features.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    features = features.reindex(
        columns=preprocessing["preprocessed_feature_columns"],
        fill_value=np.nan,
    )
    features = features.fillna(
        pd.Series(preprocessing["training_feature_defaults"])
    ).fillna(0).astype("float64")
    return features


# ---------------------------------------------------------------------------
# STEP 3: Machine Learning Models
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🤖 Step 3: Machine Learning Models")

ml_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(loss="log_loss", random_state=42),
    "Passive Aggressive Classifier": PassiveAggressiveClassifier(random_state=42),
    "Linear SVM": LinearSVC(random_state=42, max_iter=5000),
    "Kernel SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Extra Tree": ExtraTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

if XGBClassifier is not None:
    ml_models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=200)
if LGBMClassifier is not None:
    ml_models["LightGBM"] = LGBMClassifier(random_state=42, n_estimators=200, verbosity=-1)

col_model_1, col_model_2 = st.columns([2, 1])
with col_model_1:
    selected_ml = st.multiselect(
        "Select ML Models to Train",
        list(ml_models.keys()),
        default=["Logistic Regression", "Random Forest"],
        help="Choose one or more models to train"
    )
with col_model_2:
    st.markdown("### Available Models")
    st.markdown("• Logistic Regression\n• Decision Tree\n• Random Forest\n• KNN\n• SVM\n• Naive Bayes")

if st.button("🚀 **Train ML Models**", key="train_ml", use_container_width=True):
    if selected_ml:
        for idx, name in enumerate(selected_ml):
            with st.container():
                st.markdown(f"### 📌 {idx + 1}. {name}")
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with status_placeholder.container():
                    with st.spinner(f"⏳ Training {name}..."):
                        model = ml_models[name]
                        model.fit(X_train_scaled, y_train)
                        train_accuracy = model.score(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                
                st.success(f"✅ {name} training complete!")
                evaluate_and_store(
                    name, y_test, y_pred, train_accuracy=train_accuracy,
                    model=model, model_type="sklearn"
                )
                st.markdown("---")
    else:
        st.warning("⚠️ Please select at least one model to train.", icon="⚠️")


# ---------------------------------------------------------------------------
# Helper function: draw Neural Network architecture diagram
# ---------------------------------------------------------------------------
def draw_nn_architecture(input_dim, hidden_layers_list, output_dim, output_activation):
    """
    input_dim: number of input features
    hidden_layers_list: list of neuron-counts, one per hidden layer
    output_dim: number of output neurons
    """
    # Cap how many neurons we actually draw per layer, so huge layers
    # (e.g. 128 neurons) don't make an unreadable diagram. We show a
    # representative sample and label the true count.
    MAX_DRAW = 10

    layer_sizes = [input_dim] + hidden_layers_list + [output_dim]
    layer_names = (
        ["Input Layer"]
        + [f"Hidden Layer {i + 1}" for i in range(len(hidden_layers_list))]
        + ["Output Layer"]
    )

    n_layers = len(layer_sizes)
    fig_width = max(8, n_layers * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    v_spacing = 1.0
    h_spacing = 2.5

    layer_positions = []  # list of list of (x, y) per layer

    for i, size in enumerate(layer_sizes):
        drawn_count = min(size, MAX_DRAW)
        x = i * h_spacing
        top = (drawn_count - 1) * v_spacing / 2
        positions = [(x, top - j * v_spacing) for j in range(drawn_count)]
        layer_positions.append(positions)

        # Node colors: input=green, hidden=blue, output=orange
        if i == 0:
            color = "#4CAF50"
        elif i == n_layers - 1:
            color = "#FF9800"
        else:
            color = "#2196F3"

        for (nx, ny) in positions:
            circle = plt.Circle((nx, ny), 0.28, color=color, ec="black", zorder=3)
            ax.add_patch(circle)

        # If we truncated the drawing, show "..." below the node stack
        if size > MAX_DRAW:
            ax.text(x, top - drawn_count * v_spacing, "⋮", fontsize=18, ha="center")

        # Label under each layer
        label = f"{layer_names[i]}\n({size} neurons)"
        ax.text(x, top + v_spacing * 1.3, label, ha="center", fontsize=10, fontweight="bold")

    # Draw connections between consecutive layers (only among drawn nodes)
    for i in range(n_layers - 1):
        for (x1, y1) in layer_positions[i]:
            for (x2, y2) in layer_positions[i + 1]:
                ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.4, zorder=1, alpha=0.6)

    ax.set_xlim(-1, (n_layers - 1) * h_spacing + 1)
    ax.axis("off")
    ax.set_title("Neural Network Architecture", fontsize=13, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# STEP 4: Deep Learning Model (ANN) with LIVE training visualization
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧠 Step 4: Deep Learning Model (Neural Network)")

with st.expander("⚙️ **Configure Deep Learning Architecture**", expanded=True):
    dl_type = st.selectbox(
        "Neural Network Type",
        ["Dense ANN", "Deep Dense ANN", "Regularized ANN", "1D CNN"],
        index=0,
    )

    nn_col1, nn_col2, nn_col3 = st.columns(3)
    with nn_col1:
        n_layers = st.slider(
            "Hidden Layers",
            1, 5, 2,
            help="Number of hidden layers in the network"
        )
    with nn_col2:
        n_neurons = st.slider(
            "Neurons per Layer",
            8, 256, 32,
            step=8,
            help="Number of neurons in each hidden layer"
        )
    with nn_col3:
        epochs = st.slider(
            "Training Epochs",
            5, 100, 20,
            help="Number of complete passes through training data"
        )

    batch_size = st.select_slider(
        "Batch Size",
        options=[8, 16, 32, 64, 128],
        value=32,
        help="Number of samples per gradient update"
    )

# Show architecture diagram live, based on current settings
output_neurons = 1 if num_classes == 2 else num_classes
output_activation = "sigmoid" if num_classes == 2 else "softmax"

st.markdown("### 🏗️ Network Architecture Diagram")
arch_fig = draw_nn_architecture(
    input_dim=X_train_scaled.shape[1],
    hidden_layers_list=[n_neurons] * n_layers,
    output_dim=output_neurons,
    output_activation=output_activation,
)
st.pyplot(arch_fig, use_container_width=True)

arch_col1, arch_col2, arch_col3 = st.columns(3)
with arch_col1:
    st.metric("Input Neurons", X_train_scaled.shape[1])
with arch_col2:
    st.metric("Hidden Layers", n_layers)
with arch_col3:
    st.metric("Output Neurons", output_neurons)

if st.button("🚀 **Train Deep Learning Model**", key="train_dl", use_container_width=True):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        st.error(
            "TensorFlow is not available. Install it with `pip install tensorflow-cpu` "
            "to use Deep Learning training."
        )
        st.stop()

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

    if dl_type == "Dense ANN":
        for _ in range(n_layers):
            model.add(layers.Dense(n_neurons, activation="relu"))
    elif dl_type == "Deep Dense ANN":
        for i in range(n_layers):
            model.add(layers.Dense(n_neurons + i * 16, activation="relu"))
    elif dl_type == "Regularized ANN":
        for _ in range(n_layers):
            model.add(layers.Dense(n_neurons, activation="relu"))
            model.add(layers.Dropout(0.3))
            model.add(layers.BatchNormalization())
    else:
        # 1D CNN on flattened feature vector
        model.add(layers.Reshape((X_train_scaled.shape[1], 1)))
        model.add(layers.Conv1D(filters=max(8, n_neurons // 2), kernel_size=3, activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        for _ in range(max(1, n_layers - 1)):
            model.add(layers.Dense(n_neurons, activation="relu"))

    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss_fn = "binary_crossentropy"
        y_train_dl = y_train.astype("float32") if isinstance(y_train, np.ndarray) else np.array(y_train, dtype="float32")
        y_test_dl = y_test.astype("float32") if isinstance(y_test, np.ndarray) else np.array(y_test, dtype="float32")
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss_fn = "sparse_categorical_crossentropy"
        y_train_dl = y_train.astype("int64") if isinstance(y_train, np.ndarray) else np.array(y_train, dtype="int64")
        y_test_dl = y_test.astype("int64") if isinstance(y_test, np.ndarray) else np.array(y_test, dtype="int64")

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    st.markdown("### 📈 Real-Time Training Progress")

    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    history_data = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            history_data["epoch"].append(epoch + 1)
            history_data["loss"].append(logs.get("loss"))
            history_data["accuracy"].append(logs.get("accuracy"))
            history_data["val_loss"].append(logs.get("val_loss"))
            history_data["val_accuracy"].append(logs.get("val_accuracy"))

            chart_df = pd.DataFrame(history_data).set_index("epoch")

            with chart_placeholder.container():
                chart_tabs = st.tabs(["Accuracy", "Loss"])
                with chart_tabs[0]:
                    st.line_chart(chart_df[["accuracy", "val_accuracy"]])
                with chart_tabs[1]:
                    st.line_chart(chart_df[["loss", "val_loss"]])

            progress_bar.progress(min((epoch + 1) / epochs, 1.0))
            status_text.info(
                f"**Epoch {epoch + 1}/{epochs}** — "
                f"Loss: {logs.get('loss', 0):.4f} | "
                f"Accuracy: {logs.get('accuracy', 0):.4f} | "
                f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
            )

    with st.spinner(f"🔄 Training {dl_type}..."):
        model.fit(
            X_train_scaled,
            y_train_dl,
            validation_data=(X_test_scaled, y_test_dl),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[StreamlitCallback()],
            verbose=0,
        )

    st.success(f"✅ {dl_type} training complete!")

    st.markdown("### 🧪 Model Test Results")
    if num_classes == 2:
        y_pred_probs = model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    else:
        y_pred_probs = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

    train_accuracy = model.evaluate(X_train_scaled, y_train_dl, verbose=0)[1]
    evaluate_and_store(
        f"Deep Learning ({dl_type})", y_test, y_pred,
        train_accuracy=train_accuracy, model=model, model_type="keras"
    )


# ---------------------------------------------------------------------------
# STEP 5: Compare All Models
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Step 5: Model Comparison & Analysis")

if st.session_state.results:
    # Results table with styling
    results_df = pd.DataFrame(st.session_state.results).T
    results_df = results_df.sort_values("Test Accuracy", ascending=False)

    st.markdown("### 📋 Performance Metrics Table")
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    styled_df = results_df.style.highlight_max(axis=0, color="lightgreen").highlight_min(axis=0, color="lightcoral")
    if numeric_cols:
        styled_df = styled_df.format({col: "{:.4f}" for col in numeric_cols})
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("### 🧠 Fit Diagnosis Summary")
    fit_summary = results_df[["Train Accuracy", "Test Accuracy", "Fit Status"]].copy()
    st.dataframe(fit_summary, use_container_width=True)

    st.markdown("### � Metric Explanation Guide")
    st.markdown(
        "**🎯 Accuracy** = (Correct predictions) / (Total predictions) — How many did the model get right overall?\n\n"
        "**🎪 Precision** = (Correct positives) / (All predicted positives) — Of the positives it found, how many were actually correct?\n\n"
        "**🔍 Recall** = (Correct positives) / (All actual positives) — Of all the actual positives, how many did it find?\n\n"
        "**⚖️ F1 Score** = Balance between precision and recall — Best when you care about both false positives and false negatives.\n\n"
        "**📊 Train vs Test** = If train >> test, the model is overfitting. If both are low, the model is underfitting."
    )

    st.markdown("---")
    st.markdown("### 🏆 Best Model Analysis")
    best_model = results_df.index[0]
    best_scores = results_df.iloc[0]
    
    col_best_1, col_best_2 = st.columns([2, 1])
    with col_best_1:
        st.markdown(f"#### 🥇 Winner: **{best_model}**")
        st.markdown(f"""
        - **Test Accuracy**: {best_scores['Accuracy']:.2%} — This model got **{best_scores['Accuracy']:.2%}** of the test predictions correct.
        - **Precision**: {best_scores['Precision']:.4f} — When it said 'positive', it was correct **{best_scores['Precision']:.2%}** of the time.
        - **Recall**: {best_scores['Recall']:.4f} — It found **{best_scores['Recall']:.2%}** of all the actual positives.
        - **F1 Score**: {best_scores['F1 Score']:.4f} — Overall balance score (closer to 1.0 is better).
        - **Fit Status**: {best_scores['Fit Status']} — This model has a healthy learning curve.
        """)
    with col_best_2:
        st.metric("🥇 Best Test Accuracy", f"{best_scores['Accuracy']:.4f}")
        st.metric("✅ F1 Score", f"{best_scores['F1 Score']:.4f}")

    best_artifact = st.session_state.model_artifacts.get(best_model)
    if best_artifact and best_artifact["model"] is not None:
        st.markdown("### 💾 Export Best Model")
        if best_artifact["model_type"] == "sklearn":
            export_buffer = io.BytesIO()
            joblib.dump({
                "model": best_artifact["model"],
                "scaler": best_artifact["scaler"],
                "feature_columns": best_artifact["feature_columns"],
                "target_classes": best_artifact["target_classes"],
                "preprocessing": best_artifact["preprocessing"],
            }, export_buffer)
            st.download_button(
                "⬇️ Download Best Model (.pkl)",
                data=export_buffer.getvalue(),
                file_name="best_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as model_file:
                model_path = model_file.name
            best_artifact["model"].save(model_path)
            with open(model_path, "rb") as model_file:
                keras_bytes = model_file.read()
            os.remove(model_path)
            st.download_button(
                "⬇️ Download Best Model (.keras)",
                data=keras_bytes,
                file_name="best_model.keras",
                mime="application/octet-stream",
                use_container_width=True,
            )

    st.markdown("### 🔮 Predict New / Unseen Data")
    prediction_model = st.selectbox(
        "Model for new predictions",
        list(results_df.index),
        key="prediction_model",
    )
    # FIX: previously only .csv was accepted here. If the person uploaded an
    # Excel file (like a Netflix dataset .xlsx), Streamlit silently rejected
    # it, unseen_file stayed None, and the whole block below never ran —
    # producing no result and no visible error.
    unseen_file = st.file_uploader(
        "Upload a CSV or Excel file without the target column",
        type=["csv", "xlsx", "xls"],
        key="unseen_data_upload",
        help="Use the same feature columns used during training.",
    )
    if unseen_file is not None:
        try:
            if unseen_file.name.lower().endswith((".xlsx", ".xls")):
                unseen_df = pd.read_excel(unseen_file)
            else:
                unseen_df = pd.read_csv(unseen_file)
        except Exception as read_err:
            st.error(f"❌ Could not read the uploaded file: {read_err}")
            st.stop()

        artifact = st.session_state.model_artifacts[prediction_model]
        unseen_features = preprocess_unseen_features(unseen_df, artifact)
        selected_for_prediction = artifact["feature_columns"]
        expected_raw_columns = set(artifact["preprocessing"]["raw_feature_columns"])
        uploaded_raw_columns = set(unseen_df.columns)

        # FIX: warn (instead of silently predicting garbage) when the uploaded
        # file's columns don't actually match what the model was trained on.
        overlap = uploaded_raw_columns & expected_raw_columns
        if len(overlap) == 0:
            st.error(
                "❌ None of the uploaded file's columns match the features this model "
                f"was trained on. Expected columns: {', '.join(sorted(expected_raw_columns)[:15])}"
                + (" ..." if len(expected_raw_columns) > 15 else "") +
                ". Please upload a file with the same column structure as your training data."
            )
            st.stop()
        elif len(overlap) < len(expected_raw_columns):
            missing = expected_raw_columns - uploaded_raw_columns
            st.warning(
                f"⚠️ {len(missing)} expected feature column(s) not found in the uploaded file "
                f"and will be filled with training-data averages: {', '.join(list(missing)[:10])}"
                + (" ..." if len(missing) > 10 else "")
            )

        unseen_features = unseen_features.reindex(columns=selected_for_prediction)
        if st.button("🔮 Generate Predictions", key="predict_unseen"):
            if artifact["model_type"] == "keras":
                probabilities = artifact["model"].predict(
                    artifact["scaler"].transform(unseen_features), verbose=0
                )
                unseen_predictions = (
                    (probabilities > 0.5).astype(int).flatten()
                    if num_classes == 2 else np.argmax(probabilities, axis=1)
                )
            else:
                unseen_predictions = artifact["model"].predict(
                    artifact["scaler"].transform(unseen_features)
                )
            prediction_table = unseen_df.copy()
            prediction_table["Predicted Label"] = display_target_values(
                unseen_predictions, artifact["target_classes"]
            )
            st.dataframe(prediction_table, use_container_width=True)
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=prediction_table.to_csv(index=False).encode("utf-8"),
                file_name="unseen_data_predictions.csv",
                mime="text/csv",
                key="download_unseen_predictions",
            )

    # FIX (Bug 1): results_df.to_string() returns a single string. Using
    # list.extend() on a string iterates it character-by-character, corrupting
    # the report. Use append() to add it as one block instead.
    report_lines = ["ML + DL Playground Result Report", "", "Model Comparison"]
    report_lines.append(results_df.to_string())
    report_lines.extend(["", f"Best Model: {best_model}"])
    st.download_button(
        "⬇️ Download Result Report (.txt)",
        data="\n".join(report_lines).encode("utf-8"),
        file_name="model_result_report.txt",
        mime="text/plain",
    )

    st.markdown("---")
    st.markdown("### 📊 Individual Model Results")
    
    for idx, (model_name, row) in enumerate(results_df.iterrows(), 1):
        with st.expander(f"{idx}. **{model_name}** — Test Acc: {row['Accuracy']:.4f} | Fit: {row['Fit Status']}"):
            col_model_a, col_model_b = st.columns([2, 1])
            with col_model_a:
                st.markdown(f"""
                **Performance Summary:**
                - Trained on {X_train.shape[0]} samples, tested on {X_test.shape[0]} samples
                - Achieved **{row['Accuracy']:.2%}** accuracy on unseen test data
                - Precision: {row['Precision']:.4f} | Recall: {row['Recall']:.4f}
                - **Status**: {row['Fit Status']}
                
                **What This Means:**
                """)
                
                # Interpretation
                if row['Fit Status'] == 'Overfitting':
                    st.info(
                        f"⚠️ The model learned training data too well (Train: {row['Train Accuracy']:.2%}). "
                        f"On new data it performs worse (Test: {row['Accuracy']:.2%}). "
                        "Try simpler models or add regularization."
                    )
                elif row['Fit Status'] == 'Underfitting':
                    st.warning(
                        f"❌ The model is too weak. Both train ({row['Train Accuracy']:.2%}) and test ({row['Accuracy']:.2%}) are low. "
                        "Try a more complex model or better features."
                    )
                else:
                    st.success(
                        f"✅ Perfect balance! Train ({row['Train Accuracy']:.2%}) ≈ Test ({row['Accuracy']:.2%}). "
                        "This model generalizes well to new data."
                    )
                    
            with col_model_b:
                st.metric("Test Accuracy", f"{row['Accuracy']:.4f}")
                st.metric("Train Accuracy", f"{row['Train Accuracy']:.4f}")
                gap = row['Train Accuracy'] - row['Accuracy']
                st.metric("Train-Test Gap", f"{gap:.4f}")

            artifact = st.session_state.model_artifacts.get(model_name)
            if artifact:
                st.markdown("#### Actual vs Predicted Test Rows")
                prediction_table = build_prediction_table(model_name, artifact)
                show_wrong_only = st.checkbox(
                    "Show only incorrect predictions",
                    key=f"wrong_only_{model_name}",
                )
                visible_table = prediction_table[
                    ~prediction_table["Correct"]
                ] if show_wrong_only else prediction_table
                st.dataframe(visible_table, use_container_width=True)
                st.download_button(
                    "⬇️ Download Prediction Table CSV",
                    data=prediction_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv",
                    mime="text/csv",
                    key=f"download_predictions_{model_name}",
                )

    st.markdown("---")
    # FIX (Bug 2): points 3, 4 and 5 were duplicated (adjacent string literals
    # implicitly concatenate in Python), causing repeated/garbled text in this
    # expander. Rewritten as a single clean block with each point appearing once.
    with st.expander("💡 **Click here for: How to choose the best model?**", expanded=False):
        st.markdown(
            "**1️⃣ Highest Accuracy**\n"
            "> Choose the model with the best test accuracy if all are well-fit.\n\n"
            "**2️⃣ Balanced Metrics**\n"
            "> If F1 score and recall matter, don't just look at accuracy alone.\n\n"
            "**3️⃣ Avoid Overfitting**\n"
            "> Even if accuracy is high, a heavily overfitting model might fail on real data.\n\n"
            "**4️⃣ Simplicity**\n"
            "> Sometimes a simpler model (e.g., Logistic Regression) is better and faster than complex ones.\n\n"
            "**5️⃣ Business Cost**\n"
            "> False positives and false negatives have different costs in the real world; choose wisely."
        )

    # Comparison chart
    tab1, tab2, tab3 = st.tabs(["Accuracy Comparison", "All Metrics", "Train vs Test Gap"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        bars = ax.barh(results_df.index, results_df["Accuracy"], color=colors)
        ax.set_xlabel("Test Accuracy Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Test Accuracy Comparison", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        numeric_results = results_df[["Accuracy", "Precision", "Recall", "F1 Score"]].copy()
        numeric_results.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Complete Metrics Comparison Across All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with tab3:
        results_gap_df = results_df[["Train Accuracy", "Test Accuracy"]].copy()
        results_gap_df["Overfitting Gap"] = results_gap_df["Train Accuracy"] - results_gap_df["Test Accuracy"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(results_gap_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, results_gap_df["Train Accuracy"], width, label="Train Accuracy", color="green", alpha=0.7)
        bars2 = ax.bar(x + width/2, results_gap_df["Test Accuracy"], width, label="Test Accuracy", color="blue", alpha=0.7)
        
        ax.set_xlabel("Models", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title("Train vs Test Accuracy (Smaller Gap = Better Generalization)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(results_gap_df.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        # Add gap annotations
        for i, (idx, row) in enumerate(results_gap_df.iterrows()):
            gap = row["Overfitting Gap"]
            ax.text(i, max(row["Train Accuracy"], row["Test Accuracy"]) + 0.05, 
                   f'Gap: {gap:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Reset button
    col_reset_1, col_reset_2 = st.columns([3, 1])
    with col_reset_2:
        if st.button("🔄 Reset Results", use_container_width=True):
            st.session_state.results = {}
            st.rerun()
else:
    st.info(
        "📌 **No models trained yet!**\n\n"
        "Train some ML or Deep Learning models from the sections above to see comparison results here.",
        icon="ℹ️"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><small>🔬 ML + DL Playground | Interactive Machine Learning Platform</small></p>
    <p><small>Powered by Streamlit, TensorFlow, and Scikit-learn</small></p>
</div>
""", unsafe_allow_html=True)
