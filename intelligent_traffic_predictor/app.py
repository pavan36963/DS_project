import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from datetime import datetime
import os

st.set_page_config(page_title="Intelligent Traffic Predictor", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.main {background: #f5f7fb;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#d9edf5; margin:.4rem 0 0 0;}
.small-note {color:#64748b; font-size:.88rem;}
div[data-testid="stMetric"] {background:white; padding:14px; border-radius:14px; border:1px solid #e8edf3;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_traffic_data.csv")

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)

def congestion_label(value):
    if value < 300: return "Low"
    if value < 650: return "Moderate"
    return "High"

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_model(df, target, task):
    X = df.drop(columns=[target])
    y = df[target]
    # The sample congestion label is calculated from traffic_volume; never use one
    # as an input when predicting the other.
    leakage_pairs = {"traffic_volume": ["congestion_level"], "congestion_level": ["traffic_volume"]}
    X = X.drop(columns=leakage_pairs.get(target, []), errors="ignore")
    # Remove identifiers and free-text columns that are unlikely to be useful.
    drop_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable input columns remain after removing constant columns.")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    pre = ColumnTransformer(transformers)
    if task == "Regression":
        estimator = RandomForestRegressor(n_estimators=220, random_state=42, min_samples_leaf=2, n_jobs=-1)
    else:
        estimator = RandomForestClassifier(n_estimators=220, random_state=42, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("preprocess", pre), ("model", estimator)])
    stratify = y if task == "Classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, X.columns.tolist(), X_test, y_test, pred

def make_recommendation(level, hour):
    if level == "High":
        return "High congestion expected. Consider leaving earlier, choosing an alternate route, or delaying non-essential travel."
    if level == "Moderate":
        return "Moderate traffic expected. Allow extra travel time and monitor conditions before departure."
    return "Traffic is expected to be relatively smooth. Normal travel planning should be sufficient."

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_features = []
    st.session_state.target = None
    st.session_state.task = None
    st.session_state.metrics = None
    st.session_state.dataset = None

st.markdown("""
<div class="hero">
  <h1>🚦 Intelligent Traffic Predictor</h1>
  <p>Machine Learning-powered traffic forecasting • Congestion insights • Smarter travel decisions</p>
</div>
""", unsafe_allow_html=True)

sample = load_sample()
if st.session_state.dataset is None:
    st.session_state.dataset = sample.copy()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/交通/交通.png" if False else "https://cdn-icons-png.flaticon.com/512/854/854878.png", width=62)
    st.title("Navigation")
    page = st.radio("Go to", ["Overview", "Explore Traffic Data", "Train ML Model", "Predict Traffic", "Insights & Reports", "About Project"])
    st.divider()
    st.caption("Demo mode uses the included sample dataset. Upload your own compatible CSV to train on your data.")
    if st.button("↻ Reset to sample dataset", use_container_width=True):
        st.session_state.dataset = sample.copy()
        st.session_state.model = None
        st.session_state.metrics = None
        st.rerun()

df = st.session_state.dataset

if page == "Overview":
    st.subheader("Traffic intelligence at a glance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Traffic records", f"{len(df):,}")
    c2.metric("Average traffic volume", f"{df['traffic_volume'].mean():.0f} vehicles/hr")
    c3.metric("High congestion records", f"{(df['congestion_level']=='High').mean()*100:.1f}%")
    c4.metric("Model status", "Trained" if st.session_state.model else "Not trained")
    left,right = st.columns([1.5,1])
    with left:
        hourly = df.groupby("hour", as_index=False)["traffic_volume"].mean()
        fig = px.line(hourly, x="hour", y="traffic_volume", markers=True, title="Average traffic volume by hour")
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Vehicles per hour")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = df["congestion_level"].value_counts().rename_axis("Congestion").reset_index(name="Records")
        fig = px.pie(counts, names="Congestion", values="Records", hole=.55, title="Congestion distribution")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Start with **Explore Traffic Data**, then train a model and use **Predict Traffic** to estimate future congestion.")
    st.markdown("**Important:** This project currently demonstrates predictions from historical/sample data. It does not connect to live traffic cameras, GPS feeds or government traffic APIs.")

elif page == "Explore Traffic Data":
    st.subheader("Explore and upload traffic data")
    uploaded = st.file_uploader("Upload your traffic dataset (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            new_df = prepare_data(pd.read_csv(uploaded))
            if new_df.empty:
                st.error("The uploaded CSV has no records.")
            else:
                st.session_state.dataset = new_df
                st.session_state.model = None
                st.session_state.metrics = None
                df = new_df
                st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns. Train a new model for this dataset.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    st.write("**Dataset preview**")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("**Column summary**")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(summary, use_container_width=True)
    if "traffic_volume" in df.columns:
        a,b = st.columns(2)
        with a:
            fig = px.histogram(df, x="traffic_volume", nbins=30, title="Traffic volume distribution")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            if "weather" in df.columns:
                weather = df.groupby("weather", as_index=False)["traffic_volume"].mean()
                fig = px.bar(weather, x="weather", y="traffic_volume", title="Average traffic by weather")
                st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download current dataset", df.to_csv(index=False).encode("utf-8"), "traffic_dataset.csv", "text/csv")

elif page == "Train ML Model":
    st.subheader("Train and evaluate a Machine Learning model")
    st.write("Choose the column the model should predict. Regression predicts a numeric value; classification predicts a category.")
    target_default = "traffic_volume" if "traffic_volume" in df.columns else df.columns[-1]
    target = st.selectbox("Target column", df.columns, index=list(df.columns).index(target_default))
    task_guess = "Regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10 else "Classification"
    task = st.radio("Prediction type", ["Regression", "Classification"], index=0 if task_guess=="Regression" else 1, horizontal=True)
    st.caption("The model uses all other columns as input features. Ensure the target does not leak the answer into the inputs.")
    if st.button("🚀 Train Random Forest model", type="primary"):
        with st.spinner("Training model and evaluating held-out data..."):
            try:
                work = df.dropna(subset=[target]).copy()
                if len(work) < 30:
                    raise ValueError("Please provide at least 30 valid rows for a meaningful demonstration.")
                if task == "Classification" and work[target].nunique() < 2:
                    raise ValueError("Classification requires at least two target classes.")
                model, features, X_test, y_test, pred = build_model(work, target, task)
                st.session_state.model = model
                st.session_state.model_features = features
                st.session_state.target = target
                st.session_state.task = task
                if task == "Regression":
                    metrics = {
                        "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                        "R²": r2_score(y_test, pred)
                    }
                else:
                    metrics = {
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                        "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                        "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0)
                    }
                st.session_state.metrics = metrics
                st.session_state.test_results = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
                st.success("Model trained successfully.")
            except Exception as e:
                st.error(str(e))
    if st.session_state.model is not None:
        st.markdown("#### Evaluation results")
        cols = st.columns(len(st.session_state.metrics))
        for col, (name, value) in zip(cols, st.session_state.metrics.items()):
            col.metric(name, f"{value:.3f}")
        st.dataframe(st.session_state.test_results.head(30), use_container_width=True)
        if st.session_state.task == "Classification":
            labels = list(pd.Series(st.session_state.test_results["Actual"]).unique())
            cm = confusion_matrix(st.session_state.test_results["Actual"], st.session_state.test_results["Predicted"], labels=labels)
            fig = px.imshow(cm, x=labels, y=labels, text_auto=True, title="Confusion matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download evaluation results", st.session_state.test_results.to_csv(index=False).encode("utf-8"), "model_evaluation.csv", "text/csv")

elif page == "Predict Traffic":
    st.subheader("Predict traffic for a planned journey")
    if st.session_state.model is None:
        st.warning("Train a model first in **Train ML Model**.")
        st.markdown("For a quick demonstration, train the included sample dataset using `traffic_volume` as the target.")
    else:
        target = st.session_state.target
        features = st.session_state.model_features
        input_values = {}
        st.write("Enter conditions for the time and road you want to evaluate.")
        cols = st.columns(2)
        for i, feature in enumerate(features):
            series = df[feature]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = float(series.median()) if series.notna().any() else 0.0
                    lo = float(series.min()) if series.notna().any() else median-1
                    hi = float(series.max()) if series.notna().any() else median+1
                    if lo == hi: hi = lo + 1
                    input_values[feature] = st.number_input(feature.replace("_"," ").title(), value=median, min_value=lo, max_value=hi)
                else:
                    opts = sorted(series.dropna().astype(str).unique().tolist())
                    if not opts: opts = ["Unknown"]
                    input_values[feature] = st.selectbox(feature.replace("_"," ").title(), opts)
        if st.button("Predict traffic now", type="primary"):
            row = pd.DataFrame([input_values], columns=features)
            result = st.session_state.model.predict(row)[0]
            st.markdown("### Prediction result")
            if st.session_state.task == "Regression":
                st.metric("Estimated traffic volume", f"{float(result):,.0f} vehicles/hour")
                level = congestion_label(float(result)) if target == "traffic_volume" or "volume" in target.lower() else "Estimated"
            else:
                level = str(result)
                st.metric("Predicted category", level)
            if level.lower() in ["high", "heavy", "severe"]:
                st.error("🔴 High congestion likely")
            elif level.lower() in ["moderate", "medium"]:
                st.warning("🟠 Moderate congestion likely")
            else:
                st.success(f"🟢 Predicted result: {level}")
            st.info(make_recommendation(level, input_values.get("hour", 12)))
            st.caption("Prediction quality depends on the dataset, selected target, and similarity between training data and the entered conditions.")

elif page == "Insights & Reports":
    st.subheader("Traffic patterns and planning insights")
    if "hour" in df.columns and "traffic_volume" in df.columns:
        hourly = df.groupby("hour", as_index=False)["traffic_volume"].mean()
        peak = hourly.loc[hourly["traffic_volume"].idxmax()]
        low = hourly.loc[hourly["traffic_volume"].idxmin()]
        c1,c2 = st.columns(2)
        c1.metric("Peak traffic hour", f"{int(peak['hour']):02d}:00", f"{peak['traffic_volume']:.0f} vehicles/hr")
        c2.metric("Lowest traffic hour", f"{int(low['hour']):02d}:00", f"{low['traffic_volume']:.0f} vehicles/hr")
        fig = px.area(hourly, x="hour", y="traffic_volume", title="Daily traffic profile")
        st.plotly_chart(fig, use_container_width=True)
    if "day" in df.columns and "traffic_volume" in df.columns:
        daily = df.groupby("day", as_index=False)["traffic_volume"].mean()
        fig = px.bar(daily, x="day", y="traffic_volume", title="Average traffic by day")
        st.plotly_chart(fig, use_container_width=True)
    if "weather" in df.columns and "traffic_volume" in df.columns:
        weather = df.groupby("weather", as_index=False)["traffic_volume"].mean().sort_values("traffic_volume", ascending=False)
        st.write("**Weather and traffic relationship**")
        st.dataframe(weather, use_container_width=True)
    st.download_button("Export traffic report data", df.to_csv(index=False).encode("utf-8"), "traffic_report_data.csv", "text/csv")

else:
    st.subheader("About the project")
    st.markdown("""
    **Project title:** Intelligent Traffic Predictor

    **Domain:** Machine Learning • Intelligent Transportation Systems • Data Analytics

    **Problem:** Urban congestion increases travel time, fuel consumption and pollution. Reactive traffic management often responds only after congestion has developed.

    **Proposed solution:** Use historical traffic patterns and contextual features such as time, day and weather to estimate traffic volume or congestion category in advance.

    **Application modules**
    1. Data collection and CSV upload
    2. Data cleaning and exploration
    3. Feature preparation and Machine Learning
    4. Model evaluation using MAE, RMSE and R² for regression, or accuracy, precision, recall and F1 for classification
    5. Traffic prediction and travel recommendations
    6. Dashboard and report export

    **Current implementation scope:** This is a local demonstration web application using a sample dataset or a CSV supplied by the user. Real-time traffic feeds, GPS routing, map integration, and automatic traffic-signal control are future enhancements and are not currently connected.
    """)
    st.info("Use this application as a project prototype. For a real deployment, validate it with reliable local traffic data and monitor model performance over time.")
