
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)


# ------------------------------------------------------------
# Required artifact paths
# ------------------------------------------------------------
MODEL_PATH = "artifacts/final_churn_ann_model.keras"
PREPROCESSOR_PATH = "artifacts/final_churn_preprocessor.pkl"
METADATA_PATH = "artifacts/final_churn_metadata.json"

DATASET_PATH_CANDIDATES = [
    "Churn Modeling.csv",
    "churn_modeling.csv",
    "data/Churn Modeling.csv",
    "data/churn_modeling.csv",
]

BANNER_PATH_CANDIDATES = [
    "assets/customer_churn_banner.png",
    "assets/churn_banner.png",
    "customer_churn_banner.png",
]


# ------------------------------------------------------------
# Project constants
# ------------------------------------------------------------
DEFAULT_RAW_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

DEFAULT_NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]

DEFAULT_CATEGORICAL_FEATURES = ["Geography", "Gender"]
DEFAULT_BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

TARGET_COLUMN = "Exited"
EXPECTED_FINAL_MODEL_NAME = "Improved ANN with tuned threshold"
EXPECTED_FINAL_THRESHOLD = 0.30
EXPECTED_PROCESSED_FEATURES = 13

BALANCE_MIN = 0.00
BALANCE_MAX = 250898.09
SALARY_MIN = 11.58
SALARY_MAX = 199992.48

BLUE = "#0B3D91"
DARK_BLUE = "#062B63"
RED = "#D71920"
GREY = "#6B7280"
LIGHT_GREY = "#F3F6FA"
WHITE = "#FFFFFF"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #F7F9FC;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #062B63;
    }

    .subtle-note {
        color: #4B5563;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .metric-card {
        background: linear-gradient(135deg, #FFFFFF, #F1F5F9);
        border: 1px solid #D7DEE9;
        border-radius: 16px;
        padding: 1.0rem 1.1rem;
        box-shadow: 0 4px 14px rgba(11, 61, 145, 0.08);
        min-height: 105px;
    }

    .metric-label {
    color: #374151;
    font-size: 1.08rem;
    font-weight: 900;
    margin-bottom: 0.45rem;
    letter-spacing: 0.01em;
    }

    .metric-value {
    color: #062B63;
    font-size: 2rem;
    font-weight: 900;
    line-height: 1.1;
    }   

    .risk-card-low {
        background: #F0F9FF;
        border-left: 6px solid #0B3D91;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #0B1F3A;
        margin-top: 0.7rem;
    }

    .risk-card-medium {
        background: #F8FAFC;
        border-left: 6px solid #6B7280;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #0B1F3A;
        margin-top: 0.7rem;
    }

    .risk-card-high {
        background: #FEF2F2;
        border-left: 6px solid #D71920;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: #0B1F3A;
        margin-top: 0.7rem;
    }

    .info-box {
        background: #FFFFFF;
        border: 1px solid #D7DEE9;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #6B7280;
        font-size: 0.88rem;
    }

    div.stButton > button:first-child {
        background: #D71920;
        color: #FFFFFF;
        border: 0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 850;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        background: #B9151B;
        color: #FFFFFF;
        border: 0;
    }

    [data-testid="stMetricValue"] {
        color: #062B63;
    }

    /* Bigger and bolder Streamlit tab headings */
    div[data-testid="stTabs"] div[role="tablist"] button[role="tab"] {
    padding: 0.85rem 1.25rem !important;
    }

    div[data-testid="stTabs"] div[role="tablist"] button[role="tab"] p {
    font-size: 1.35rem !important;
    font-weight: 900 !important;
    line-height: 1.4 !important;
    }

    div[data-testid="stTabs"] div[role="tablist"] button[role="tab"][aria-selected="true"] p {
    color: #D71920 !important;
    font-weight: 900 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #D7DEE9;
        margin: 1.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Artifact loading and compatibility validation
# ------------------------------------------------------------
def load_json(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with open(path_obj, "r", encoding="utf-8") as file:
        return json.load(file)


def load_preprocessor(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Missing preprocessor file: {path}")
    return joblib.load(path_obj)


def to_dense_float32(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def get_model_input_feature_count(model) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("The saved ANN must have exactly one input tensor.")
        input_shape = input_shape[0]

    if not input_shape or input_shape[-1] is None:
        raise ValueError("The saved ANN input feature count could not be determined.")

    return int(input_shape[-1])


def validate_loaded_artifacts(model, preprocessor, metadata: dict) -> dict:
    required_metadata_keys = [
        "final_model_name",
        "final_threshold",
        "raw_input_features",
        "numerical_features",
        "categorical_features",
        "binary_features",
        "processed_feature_names",
        "expected_processed_features",
        "categorical_options",
        "target_column",
        "model_artifact",
        "preprocessor_artifact",
    ]

    missing_keys = [key for key in required_metadata_keys if key not in metadata]
    if missing_keys:
        raise ValueError(f"Metadata is missing required keys: {missing_keys}")

    raw_features = list(metadata["raw_input_features"])
    numerical_features = list(metadata["numerical_features"])
    categorical_features = list(metadata["categorical_features"])
    binary_features = list(metadata["binary_features"])
    processed_feature_names = list(metadata["processed_feature_names"])
    categorical_options = metadata["categorical_options"]

    if raw_features != DEFAULT_RAW_FEATURES:
        raise ValueError(
            "Raw feature names/order do not match the approved model input order. "
            f"Expected {DEFAULT_RAW_FEATURES}, received {raw_features}."
        )

    if numerical_features != DEFAULT_NUMERICAL_FEATURES:
        raise ValueError("Numerical feature metadata does not match the approved project configuration.")

    if categorical_features != DEFAULT_CATEGORICAL_FEATURES:
        raise ValueError("Categorical feature metadata does not match the approved project configuration.")

    if binary_features != DEFAULT_BINARY_FEATURES:
        raise ValueError("Binary feature metadata does not match the approved project configuration.")

    if metadata["target_column"] != TARGET_COLUMN:
        raise ValueError(
            f"Metadata target column must be {TARGET_COLUMN!r}, received {metadata['target_column']!r}."
        )

    final_model_name = str(metadata["final_model_name"])
    if final_model_name != EXPECTED_FINAL_MODEL_NAME:
        raise ValueError(
            "The loaded metadata does not describe the approved final model. "
            f"Expected {EXPECTED_FINAL_MODEL_NAME!r}, received {final_model_name!r}."
        )

    final_threshold = float(metadata["final_threshold"])
    if not np.isclose(final_threshold, EXPECTED_FINAL_THRESHOLD, rtol=0.0, atol=1e-12):
        raise ValueError(
            "The saved classification threshold does not match the approved threshold. "
            f"Expected {EXPECTED_FINAL_THRESHOLD:.2f}, received {final_threshold:.12g}."
        )

    if metadata["model_artifact"] != Path(MODEL_PATH).name:
        raise ValueError("Metadata model filename does not match the model file loaded by the app.")

    if metadata["preprocessor_artifact"] != Path(PREPROCESSOR_PATH).name:
        raise ValueError("Metadata preprocessor filename does not match the preprocessor loaded by the app.")

    expected_processed_features = int(metadata["expected_processed_features"])
    if expected_processed_features != EXPECTED_PROCESSED_FEATURES:
        raise ValueError(
            "Metadata processed feature count does not match the approved model. "
            f"Expected {EXPECTED_PROCESSED_FEATURES}, received {expected_processed_features}."
        )

    preprocessor_feature_names = preprocessor.get_feature_names_out().tolist()
    if preprocessor_feature_names != processed_feature_names:
        raise ValueError("Saved processed feature names do not match the fitted preprocessor output.")

    if len(preprocessor_feature_names) != expected_processed_features:
        raise ValueError("The fitted preprocessor feature count does not match metadata.")

    model_input_features = get_model_input_feature_count(model)
    if model_input_features != expected_processed_features:
        raise ValueError(
            "The ANN input dimension does not match the fitted preprocessor output. "
            f"Model expects {model_input_features}; metadata expects {expected_processed_features}."
        )

    if not isinstance(categorical_options, dict):
        raise ValueError("Metadata categorical_options must be a dictionary.")

    geography_options = [str(value) for value in categorical_options.get("Geography", [])]
    gender_options = [str(value) for value in categorical_options.get("Gender", [])]

    if set(geography_options) != {"France", "Germany", "Spain"}:
        raise ValueError("Geography options in metadata do not match the fitted preprocessor categories.")

    if set(gender_options) != {"Female", "Male"}:
        raise ValueError("Gender options in metadata do not match the fitted preprocessor categories.")

    validation_probe = pd.DataFrame(
        [{
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 40,
            "Tenure": 5,
            "Balance": 100000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 100000.0,
        }],
        columns=raw_features,
    )

    probe_processed = to_dense_float32(preprocessor.transform(validation_probe))
    if probe_processed.ndim != 2 or probe_processed.shape[1] != expected_processed_features:
        raise ValueError(
            "The fitted preprocessor did not reproduce the approved 13-feature model input."
        )

    return {
        "raw_features": raw_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "final_threshold": final_threshold,
        "final_model_name": final_model_name,
        "expected_processed_features": expected_processed_features,
        "model_input_features": model_input_features,
        "geography_options": geography_options,
        "gender_options": gender_options,
    }


@st.cache_resource
def load_artifacts():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model = tf.keras.models.load_model(model_path, compile=False)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    metadata = load_json(METADATA_PATH)
    settings = validate_loaded_artifacts(model, preprocessor, metadata)

    return model, preprocessor, metadata, settings


try:
    model, preprocessor, metadata, settings = load_artifacts()
except Exception as error:
    st.error("The app could not load or validate the saved model artifacts.")
    st.exception(error)
    st.info(
        "Check that the final notebook artifacts are present and were created together: "
        "`artifacts/final_churn_ann_model.keras`, "
        "`artifacts/final_churn_preprocessor.pkl`, and "
        "`artifacts/final_churn_metadata.json`."
    )
    st.stop()


RAW_FEATURES = settings["raw_features"]
NUMERICAL_FEATURES = settings["numerical_features"]
CATEGORICAL_FEATURES = settings["categorical_features"]
BINARY_FEATURES = settings["binary_features"]
FINAL_THRESHOLD = settings["final_threshold"]
FINAL_MODEL_NAME = settings["final_model_name"]
EXPECTED_INPUT_FEATURES = settings["expected_processed_features"]
MODEL_INPUT_FEATURES = settings["model_input_features"]
GEOGRAPHY_OPTIONS = settings["geography_options"]
GENDER_OPTIONS = settings["gender_options"]


# ------------------------------------------------------------
# Data loading for dashboard
# ------------------------------------------------------------
@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_existing_dataset_path() -> str | None:
    for path in DATASET_PATH_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def clean_dashboard_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        TARGET_COLUMN,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df["Churn label"] = df[TARGET_COLUMN].map({0: "Not churned", 1: "Churned"})
    df["Activity status"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    df["Credit card status"] = df["HasCrCard"].map({0: "No credit card", 1: "Has credit card"})

    age_bins = [17, 30, 40, 50, 60, 100]
    age_labels = ["18-30", "31-40", "41-50", "51-60", "60+"]
    df["Age group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

    return df


# ------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------
def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def build_input_dataframe(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    estimated_salary,
) -> pd.DataFrame:
    row = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": yes_no_to_int(has_card),
        "IsActiveMember": yes_no_to_int(is_active),
        "EstimatedSalary": float(estimated_salary),
    }

    missing_features = [feature for feature in RAW_FEATURES if feature not in row]
    unexpected_features = [feature for feature in row if feature not in RAW_FEATURES]

    if missing_features or unexpected_features:
        raise ValueError(
            "Input feature mismatch. "
            f"Missing: {missing_features}; unexpected: {unexpected_features}."
        )

    input_df = pd.DataFrame([row], columns=RAW_FEATURES)

    if input_df.isna().any().any():
        null_columns = input_df.columns[input_df.isna().any()].tolist()
        raise ValueError(f"Input contains missing values in: {null_columns}")

    return input_df


def transform_for_model(input_df: pd.DataFrame) -> np.ndarray:
    if input_df.columns.tolist() != RAW_FEATURES:
        raise ValueError("Input feature order does not match the approved model metadata.")

    processed_input = to_dense_float32(preprocessor.transform(input_df))

    if processed_input.ndim != 2:
        raise ValueError("Preprocessed model input must be a two-dimensional array.")

    if processed_input.shape[1] != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            "Preprocessed feature count does not match metadata. "
            f"Expected {EXPECTED_INPUT_FEATURES}, received {processed_input.shape[1]}."
        )

    if processed_input.shape[1] != MODEL_INPUT_FEATURES:
        raise ValueError(
            "Preprocessed feature count does not match the saved ANN input dimension."
        )

    return processed_input


def predict_churn(input_df: pd.DataFrame) -> tuple[float, int]:
    processed_input = transform_for_model(input_df)

    prediction_proba = float(model.predict(processed_input, verbose=0).ravel()[0])
    if not 0.0 <= prediction_proba <= 1.0:
        raise ValueError("The ANN returned a score outside the valid [0, 1] range.")

    prediction_label = int(prediction_proba >= FINAL_THRESHOLD)
    return prediction_proba, prediction_label


def get_risk_display(probability: float, threshold: float) -> dict:
    probability_gap = probability - threshold

    if probability >= 0.75:
        return {
            "card_class": "risk-card-high",
            "risk_level": "Critical churn risk",
            "business_priority": "Critical priority",
            "threshold_message": (
                f"The model-estimated churn score is {probability:.2%}. "
                f"This is {probability_gap:.2%} above the saved decision threshold of {threshold:.2f}."
            ),
            "management_action": (
                "Assign immediate retention follow-up. Review customer experience, product fit, "
                "relationship value, and a suitable retention intervention."
            ),
        }

    if probability >= 0.50:
        return {
            "card_class": "risk-card-high",
            "risk_level": "High churn risk",
            "business_priority": "High priority",
            "threshold_message": (
                f"The model-estimated churn score is {probability:.2%}. "
                f"This is {probability_gap:.2%} above the saved decision threshold of {threshold:.2f}."
            ),
            "management_action": (
                "Prioritize proactive retention outreach and review engagement, product fit, "
                "service experience, and relationship-manager follow-up."
            ),
        }

    if probability >= threshold:
        return {
            "card_class": "risk-card-high",
            "risk_level": "Elevated churn risk",
            "business_priority": "Retention follow-up priority",
            "threshold_message": (
                f"The model-estimated churn score is {probability:.2%}. "
                f"This is {probability_gap:.2%} above the saved decision threshold of {threshold:.2f}."
            ),
            "management_action": (
                "The customer is classified as likely to churn. Use proportionate proactive outreach "
                "and review the profile context before choosing an intervention."
            ),
        }

    watchlist_floor = max(0.0, threshold - 0.10)
    if probability >= watchlist_floor:
        return {
            "card_class": "risk-card-medium",
            "risk_level": "Below-threshold watchlist",
            "business_priority": "Standard monitoring with attention",
            "threshold_message": (
                f"The model-estimated churn score is {probability:.2%}. "
                f"This is {abs(probability_gap):.2%} below the saved decision threshold of {threshold:.2f}; "
                "the final classification remains likely to stay."
            ),
            "management_action": (
                "No immediate escalation is required. Continue monitoring and consider light-touch "
                "preventive engagement when supported by customer context."
            ),
        }

    return {
        "card_class": "risk-card-low",
        "risk_level": "Low churn risk",
        "business_priority": "Low priority",
        "threshold_message": (
            f"The model-estimated churn score is {probability:.2%}. "
            f"This is well below the saved decision threshold of {threshold:.2f}."
        ),
        "management_action": (
            "Continue normal relationship management. No immediate retention action is required."
        ),
    }


# ------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------
def add_bar_labels(ax, bars, suffix="%"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def churn_rate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(group_col, observed=False)[TARGET_COLUMN]
        .agg(total_customers="count", churned_customers="sum", churn_rate="mean")
        .reset_index()
    )
    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    return summary


def plot_churn_rate_bar(summary: pd.DataFrame, x_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(summary[x_col].astype(str), summary["churn_rate_pct"])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(35, summary["churn_rate_pct"].max() * 1.25))
    ax.grid(axis="y", alpha=0.25)
    add_bar_labels(ax, bars)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
banner_path = next((path for path in BANNER_PATH_CANDIDATES if Path(path).exists()), None)

if banner_path:
    st.image(banner_path, use_container_width=True)
else:
    st.title("Customer Churn Prediction")
    st.markdown(
        "<div class='subtle-note'>ANN-based customer retention analytics for bank customer churn prediction.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
prediction_tab, dashboard_tab = st.tabs(["🔮 Churn prediction", "📊 Churn dashboard"])


# ------------------------------------------------------------
# Tab 1: Prediction
# ------------------------------------------------------------
with prediction_tab:
    st.subheader("Predict customer churn risk")

    st.markdown(
        f"""
        <div class="info-box">
            <h3>How you should use this churn predictor</h3>
            <p>
                This tool produces a model-estimated churn score for a bank customer based on the profile entered below.
                The prediction should be used as a <b>decision-support signal</b>, not as an automatic final decision.
            </p>
            <p>
                A customer is classified as <b>likely to churn</b> only when the model-estimated churn score is greater than or equal
                to the saved model threshold of <b>{FINAL_THRESHOLD:.2f}</b>. Customers below the threshold can still be monitored
                if their model-estimated score is meaningfully high.
            </p>
            <p class="subtle-note">
                Recommended business use: prioritize retention follow-up, review descriptive profile context, and support proactive customer
                engagement. Do not use this output alone for customer rejection, penalty, or unfair treatment decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Customer profile input")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit score", min_value=350, max_value=850, value=650, step=1)
        geography = st.selectbox("Geography", GEOGRAPHY_OPTIONS)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        age = st.slider("Age", min_value=18, max_value=92, value=40, step=1)

    with col2:
        tenure = st.slider("Tenure", min_value=0, max_value=10, value=5, step=1)
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            max_value=BALANCE_MAX,
            value=100000.0,
            step=1000.0,
            format="%.2f",
        )
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)

    with col3:
        has_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Is active member?", ["Yes", "No"], horizontal=True)
        estimated_salary = st.number_input(
            "Estimated salary",
            min_value=SALARY_MIN,
            max_value=SALARY_MAX,
            value=100000.0,
            step=1000.0,
            format="%.2f",
        )

    input_df = build_input_dataframe(
        credit_score=credit_score,
        geography=geography,
        gender=gender,
        age=age,
        tenure=tenure,
        balance=balance,
        num_products=num_products,
        has_card=has_card,
        is_active=is_active,
        estimated_salary=estimated_salary,
    )

    with st.expander("Input preview", expanded=False):
        st.dataframe(input_df, use_container_width=True)

    predict_clicked = st.button("Predict churn risk", type="primary")

    if predict_clicked:
        try:
            churn_probability, predicted_label = predict_churn(input_df)
            prediction_text = "Likely to churn" if predicted_label == 1 else "Likely to stay"

            row = input_df.iloc[0]
            risk_display = get_risk_display(churn_probability, FINAL_THRESHOLD)

            card_class = risk_display["card_class"]
            risk_level = risk_display["risk_level"]
            business_priority = risk_display["business_priority"]
            threshold_message = risk_display["threshold_message"]
            management_action = risk_display["management_action"]

            profile_context = []

            if row["IsActiveMember"] == 0:
                profile_context.append(
                    "Inactive members had higher observed churn in the historical dataset. "
                    "This is descriptive cohort context, not a causal explanation of this individual score."
                )
            else:
                profile_context.append(
                    "Active members had lower observed churn than inactive members in the historical dataset. "
                    "This association does not prove that activity status caused this prediction."
                )

            if row["Geography"] == "Germany":
                profile_context.append(
                    "Germany had the highest historical churn rate in the dataset (32.44%)."
                )
            else:
                profile_context.append(
                    f"{row['Geography']} had a lower historical churn rate than Germany in the dataset."
                )

            if row["NumOfProducts"] in [3, 4]:
                profile_context.append(
                    "Customers with 3 or 4 products showed high observed churn in the historical data. "
                    "The 4-product groups are small, so this pattern should be interpreted cautiously."
                )
            elif row["NumOfProducts"] == 1:
                profile_context.append(
                    "One-product customers had higher observed churn than two-product customers. "
                    "This may justify reviewing relationship depth, but it is not an individual model attribution."
                )
            else:
                profile_context.append(
                    "Two-product customers showed the strongest historical retention pattern in the dashboard."
                )

            if row["Age"] >= 51:
                profile_context.append(
                    "The customer is in an older age band that showed higher historical churn in the dataset."
                )
            elif row["Age"] >= 41:
                profile_context.append(
                    "The customer is in an age band where historical churn began to rise."
                )
            else:
                profile_context.append(
                    "The customer is in a younger-to-mid age band, not the highest-churn age segment in the dashboard."
                )

            if row["Balance"] > 100000:
                profile_context.append(
                    "The customer has a comparatively high balance, which may increase business exposure if churn occurs."
                )
            elif row["Balance"] == 0:
                profile_context.append(
                    "The customer has a zero balance; relationship depth may be worth reviewing, without assuming causation."
                )
            else:
                profile_context.append("The customer has a moderate balance level within the training-data range.")

            retention_actions = []

            if predicted_label == 1:
                retention_actions.append(
                    "Use proportionate proactive retention follow-up because the saved threshold classifies this customer as likely to churn."
                )
            else:
                retention_actions.append(
                    "The customer is classified as likely to stay; continue routine monitoring of the model-estimated churn score."
                )

            if row["IsActiveMember"] == 0:
                retention_actions.append(
                    "Consider a re-engagement action such as personal outreach, a service check-in, or relevant loyalty communication."
                )

            if row["Geography"] == "Germany":
                retention_actions.append(
                    "Investigate Germany-specific service, pricing, or customer-experience patterns before deciding an intervention."
                )

            if row["NumOfProducts"] >= 3:
                retention_actions.append(
                    "Investigate whether product complexity or product fit requires review; the dataset does not establish the cause of churn."
                )
            elif row["NumOfProducts"] == 1:
                retention_actions.append(
                    "Consider whether relevant relationship-deepening options are appropriate; do not assume that cross-sell will reduce churn."
                )
            elif row["NumOfProducts"] == 2:
                retention_actions.append(
                    "Maintain appropriate relationship support; two-product customers showed stronger historical retention."
                )

            if row["Age"] >= 51:
                retention_actions.append(
                    "Consider assisted support or relationship-manager outreach where appropriate and fair."
                )

            kpi1, kpi2, kpi3 = st.columns(3)

            with kpi1:
                st.metric("Model-estimated churn score", f"{churn_probability:.2%}")

            with kpi2:
                st.metric("Final threshold", f"{FINAL_THRESHOLD:.2f}")

            with kpi3:
                st.metric("Prediction", prediction_text)

            st.progress(min(max(churn_probability, 0.0), 1.0))

            st.markdown(
                f"""
                <div class="{card_class}">
                    <h4>{risk_level}</h4>
                    <p><b>Probability meaning:</b> {threshold_message}</p>
                    <p><b>Business risk level:</b> {business_priority}</p>
                    <p><b>Management action:</b> {management_action}</p>
                    <p class="small-muted">
                        The final classification is based on whether the model-estimated churn score is greater than or equal to the saved threshold.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Profile context from historical EDA")
            st.caption(
                "These are descriptive cohort patterns from the historical dataset. They are not causal findings, "
                "ANN feature contributions, or proof of why this individual received the displayed score."
            )
            for context_item in profile_context:
                st.write(f"- {context_item}")

            st.markdown("### Retention action suggestions")
            for action in retention_actions:
                st.write(f"- {action}")

        except Exception as error:
            st.error("Prediction failed. Please check the saved artifact compatibility and input feature names.")
            st.exception(error)


# ------------------------------------------------------------
# Tab 2: Dashboard
# ------------------------------------------------------------
with dashboard_tab:
    st.subheader("Bank customer churn and retention dashboard")

    dataset_path = find_existing_dataset_path()

    uploaded_dataset = None
    if dataset_path is None:
        st.warning(
            "Dataset file was not found automatically. Place `Churn Modeling.csv` in the project root, "
            "or upload it here for dashboard charts."
        )
        uploaded_dataset = st.file_uploader("Upload Churn Modeling.csv", type=["csv"])

    try:
        if dataset_path is not None:
            raw_df = load_dataset_from_path(dataset_path)
        elif uploaded_dataset is not None:
            raw_df = pd.read_csv(uploaded_dataset)
        else:
            raw_df = None

        if raw_df is not None:
            dashboard_df = clean_dashboard_dataset(raw_df)

            total_customers = len(dashboard_df)
            churned_customers = int(dashboard_df[TARGET_COLUMN].sum())
            retained_customers = total_customers - churned_customers
            churn_rate = churned_customers / total_customers if total_customers > 0 else 0

            geo_summary = churn_rate_summary(dashboard_df, "Geography").sort_values(
                "churn_rate_pct", ascending=False
            )

            age_summary = churn_rate_summary(
                dashboard_df.dropna(subset=["Age group"]), "Age group"
            )

            highest_geo = geo_summary.iloc[0]
            highest_age = age_summary.sort_values("churn_rate_pct", ascending=False).iloc[0]


            st.markdown(
                f"""
                <div style="
                    background: #EFF6FF;
                    border-left: 7px solid #0B3D91;
                    border-radius: 14px;
                    padding: 22px 24px;
                    margin-top: 16px;
                    margin-bottom: 28px;
                    box-shadow: 0 4px 14px rgba(11, 61, 145, 0.08);
                ">
                    <b>Key readout:</b>
                    The overall churn rate is <b>{churn_rate:.2%}</b>.
                    The highest churn geography in this dataset is <b>{highest_geo['Geography']}</b>
                    with a churn rate of <b>{highest_geo['churn_rate_pct']:.2f}%</b>.
                    The highest churn age group is <b>{highest_age['Age group']}</b>
                    with a churn rate of <b>{highest_age['churn_rate_pct']:.2f}%</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #FFFFFF, #F3F6FA);
                        border: 1px solid #D7DEE9;
                        border-top: 6px solid #0B3D91;
                        border-radius: 18px;
                        padding: 22px 20px;
                        min-height: 135px;
                        box-shadow: 0 8px 24px rgba(11, 61, 145, 0.14);
                    ">
                        <div style="
                            color: #374151;
                            font-size: 20px;
                            font-weight: 900;
                            margin-bottom: 12px;
                            line-height: 1.2;
                        ">
                            Total customers
                        </div>
                        <div style="
                            color: #062B63;
                            font-size: 34px;
                            font-weight: 900;
                            line-height: 1.1;
                        ">
                            {total_customers:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col2:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #FFFFFF, #FFF5F5);
                        border: 1px solid #D7DEE9;
                        border-top: 6px solid #D71920;
                        border-radius: 18px;
                        padding: 22px 20px;
                        min-height: 135px;
                        box-shadow: 0 8px 24px rgba(215, 25, 32, 0.14);
                    ">
                        <div style="
                            color: #374151;
                            font-size: 20px;
                            font-weight: 900;
                            margin-bottom: 12px;
                            line-height: 1.2;
                        ">
                            Churned customers
                        </div>
                        <div style="
                            color: #D71920;
                            font-size: 34px;
                            font-weight: 900;
                            line-height: 1.1;
                        ">
                            {churned_customers:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col3:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #FFFFFF, #F3F6FA);
                        border: 1px solid #D7DEE9;
                        border-top: 6px solid #0B3D91;
                        border-radius: 18px;
                        padding: 22px 20px;
                        min-height: 135px;
                        box-shadow: 0 8px 24px rgba(11, 61, 145, 0.14);
                    ">
                        <div style="
                            color: #374151;
                            font-size: 20px;
                            font-weight: 900;
                            margin-bottom: 12px;
                            line-height: 1.2;
                        ">
                            Retained customers
                        </div>
                        <div style="
                            color: #062B63;
                            font-size: 34px;
                            font-weight: 900;
                            line-height: 1.1;
                        ">
                            {retained_customers:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col4:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #FFFFFF, #F3F6FA);
                        border: 1px solid #D7DEE9;
                        border-top: 6px solid #6B7280;
                        border-radius: 18px;
                        padding: 22px 20px;
                        min-height: 135px;
                        box-shadow: 0 8px 24px rgba(107, 114, 128, 0.14);
                    ">
                        <div style="
                            color: #374151;
                            font-size: 20px;
                            font-weight: 900;
                            margin-bottom: 12px;
                            line-height: 1.2;
                        ">
                            Overall churn rate
                        </div>
                        <div style="
                            color: #062B63;
                            font-size: 34px;
                            font-weight: 900;
                            line-height: 1.1;
                        ">
                            {churn_rate:.2%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### Churn Risk Patterns and Retention Priorities")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_geo = plot_churn_rate_bar(
                    geo_summary,
                    "Geography",
                    "Churn rate by geography",
                )
                st.pyplot(fig_geo, clear_figure=True)

                highest_geo = geo_summary.iloc[0]
                lowest_geo = geo_summary.iloc[-1]

                st.markdown(
                    f"""
                    <div style="
                        background: #EFF6FF;
                        border-left: 7px solid #0B3D91;
                        border-radius: 14px;
                        padding: 22px 24px;
                        margin-top: 16px;
                        margin-bottom: 18px;
                        box-shadow: 0 4px 14px rgba(11, 61, 145, 0.08);
                    ">
                        <p style="margin-top: 0;">
                            <b>What the chart says:</b>
                            {highest_geo['Geography']} has the highest churn rate at
                            <b>{highest_geo['churn_rate_pct']:.2f}%</b>, while {lowest_geo['Geography']}
                            has the lowest churn rate at <b>{lowest_geo['churn_rate_pct']:.2f}%</b>.
                        </p>
                        <p style="margin-bottom: 0;">
                            <b>Management takeaway:</b>
                            Prioritize geography-level churn investigation for {highest_geo['Geography']}.
                            Review service experience, pricing, relationship management, or product-fit issues in this market.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with chart_col2:
                fig_age = plot_churn_rate_bar(
                    age_summary,
                    "Age group",
                    "Churn rate by age group",
                )
                st.pyplot(fig_age, clear_figure=True)

                highest_age = age_summary.sort_values("churn_rate_pct", ascending=False).iloc[0]
                lowest_age = age_summary.sort_values("churn_rate_pct", ascending=True).iloc[0]

                st.markdown(
                    f"""
                    <div style="
                        background: #EFF6FF;
                        border-left: 7px solid #0B3D91;
                        border-radius: 14px;
                        padding: 22px 24px;
                        margin-top: 16px;
                        margin-bottom: 18px;
                        box-shadow: 0 4px 14px rgba(11, 61, 145, 0.08);
                    ">
                        <p style="margin-top: 0;">
                            <b>What the chart says:</b>
                            The <b>{highest_age['Age group']}</b> age group has the highest churn rate at
                            <b>{highest_age['churn_rate_pct']:.2f}%</b>. The lowest churn rate is in the
                            <b>{lowest_age['Age group']}</b> group at <b>{lowest_age['churn_rate_pct']:.2f}%</b>.
                        </p>
                        <p style="margin-bottom: 0;">
                            <b>Management takeaway:</b>
                            Design age-sensitive retention actions for higher-risk age groups, especially proactive service support,
                            relationship-manager outreach, and personalized engagement.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div style="height: 70px;"></div>
                """,
                unsafe_allow_html=True,
            )

            product_activity_summary = (
                dashboard_df.groupby(["NumOfProducts", "Activity status"], observed=False)[TARGET_COLUMN]
                .agg(total_customers="count", churned_customers="sum", churn_rate="mean")
                .reset_index()
            )
            product_activity_summary["churn_rate_pct"] = product_activity_summary["churn_rate"] * 100

            pivot_summary = product_activity_summary.pivot(
                index="NumOfProducts",
                columns="Activity status",
                values="churn_rate_pct",
            ).fillna(0)

            fig_product, ax = plt.subplots(figsize=(10, 4.5))

            chart_colors = []
            for column in pivot_summary.columns:
                if column == "Active":
                    chart_colors.append("#0B3D91")
                elif column == "Inactive":
                    chart_colors.append("#D71920")
                else:
                    chart_colors.append("#6B7280")

            pivot_summary.plot(kind="bar", ax=ax, color=chart_colors)

            ax.set_title("Churn rate by product count and activity status", fontsize=13, fontweight="bold")
            ax.set_xlabel("Number of products")
            ax.set_ylabel("Churn rate (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(title="Activity status")
            plt.xticks(rotation=0)
            plt.tight_layout()

            st.pyplot(fig_product, clear_figure=True)

            highest_product_activity_rate = product_activity_summary["churn_rate_pct"].max()
            highest_product_activity_rows = product_activity_summary[
                np.isclose(
                    product_activity_summary["churn_rate_pct"],
                    highest_product_activity_rate,
                    rtol=0.0,
                    atol=1e-12,
                )
            ].copy()

            highest_group_descriptions = []
            for _, highest_row in highest_product_activity_rows.iterrows():
                highest_group_descriptions.append(
                    f"{int(highest_row['NumOfProducts'])} products / {highest_row['Activity status']}: "
                    f"{int(highest_row['churned_customers'])} of {int(highest_row['total_customers'])} churned "
                    f"({highest_row['churn_rate_pct']:.2f}%)"
                )

            highest_group_text = "; ".join(highest_group_descriptions)
            highest_group_customer_count = int(highest_product_activity_rows["total_customers"].sum())

            if len(highest_product_activity_rows) > 1:
                highest_group_statement = (
                    f"The highest observed churn rate is tied across <b>{highest_group_text}</b>."
                )
            else:
                highest_group_statement = (
                    f"The highest observed churn rate appears in <b>{highest_group_text}</b>."
                )

            st.markdown(
                f"""
                <div style="
                    background: #EFF6FF;
                    border-left: 7px solid #0B3D91;
                    border-radius: 14px;
                    padding: 22px 24px;
                    margin-top: 16px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 14px rgba(11, 61, 145, 0.08);
                ">
                    <p style="margin-top: 0;">
                        <b>What the chart says:</b>
                        {highest_group_statement}
                        The highest-rate group(s) contain only <b>{highest_group_customer_count}</b> customers in total, so the
                        <b>{highest_product_activity_rate:.2f}%</b> observed rate should be interpreted cautiously. Product counts 3
                        and 4 show higher observed churn than 1 or 2 products in this dataset.
                    </p>
                    <p style="margin-bottom: 0;">
                        <b>Management takeaway:</b>
                        Do not assume that more products automatically strengthen retention. Treat product count as a historical
                        cohort signal and investigate product complexity, fit, service burden, and engagement without assuming causation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.info("Upload the dataset or place it in the project folder to view the dashboard charts.")

    except Exception as error:
        st.error("Dashboard could not be created from the dataset.")
        st.exception(error)