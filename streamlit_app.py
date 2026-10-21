"""
Streamlit UI for the XGBoost pipeline.
Consumes the same FastAPI backend endpoints.
Run with: streamlit run streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st


def get_default_api_base() -> str:
    env_candidates = []
    for env_name in ("API_BASE_URL", "FASTAPI_URL", "API_URL"):
        value = os.getenv(env_name)
        if value:
            env_candidates.append(value.strip().rstrip("/"))

    # Prefer the explicit environment override, then common local/Docker values.
    candidates = env_candidates + [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://host.docker.internal:8000",
    ]

    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            return candidate
    return "http://localhost:8000"


def api_url(path: str) -> str:
    base = st.session_state.get("api_base", get_default_api_base()).rstrip("/")
    return f"{base}/{path.lstrip('/')}"


st.set_page_config(page_title="XGBoost Incremental Pipeline", layout="wide")
if "api_base" not in st.session_state:
    st.session_state["api_base"] = get_default_api_base()

st.title("🚀 XGBoost Incremental Training Pipeline")
with st.sidebar:
    st.caption("FastAPI backend")
    st.text_input("API base URL", key="api_base", value=st.session_state["api_base"])

tab1, tab2, tab3 = st.tabs(["📤 Train Model", "🔮 Predict", "📊 Model Evolution"])

# ---------------- TRAIN TAB ----------------
with tab1:
    st.header("Upload New Training Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        st.write("Preview:", preview_df.head())
        uploaded_file.seek(0)  # reset pointer after preview

    if st.button("Train / Continue Training", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
        else:
            with st.spinner("Training in progress..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    response = requests.post(api_url("/train"), files=files, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Training completed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Training failed: {response.json().get('detail')}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not connect to API at {st.session_state['api_base']}. "
                        "Start the FastAPI service or set API_BASE_URL before launching Streamlit."
                    )

# ---------------- PREDICT TAB ----------------
with tab2:
    st.header("Make a Prediction")

    if st.button("Load Required Input Fields"):
        try:
            schema_resp = requests.get(api_url("/model/schema"), timeout=30)
            if schema_resp.status_code == 200:
                st.session_state["features"] = schema_resp.json()["required_features"]
            else:
                st.error(schema_resp.json().get("detail"))
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to API at {st.session_state['api_base']}. "
                "Start the FastAPI service or set API_BASE_URL before launching Streamlit."
            )

    if "features" in st.session_state:
        st.subheader("Enter Feature Values")
        input_values = {}
        cols = st.columns(3)
        for idx, feature in enumerate(st.session_state["features"]):
            with cols[idx % 3]:
                input_values[feature] = st.number_input(feature, value=0.0, format="%.4f")

        if st.button("Predict", type="primary"):
            try:
                pred_resp = requests.post(
                    api_url("/predict"),
                    json={"features": input_values},
                    timeout=30,
                )
                if pred_resp.status_code == 200:
                    prediction = pred_resp.json()["prediction"]
                    st.success(f"### Prediction Result: `{prediction:.4f}`")
                else:
                    st.error(pred_resp.json().get("detail"))
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to API at {st.session_state['api_base']}. "
                    "Start the FastAPI service or set API_BASE_URL before launching Streamlit."
                )
    else:
        st.info("Click 'Load Required Input Fields' to begin.")

# ---------------- METRICS TAB ----------------
with tab3:
    st.header("Model Evolution / Metrics History")

    if st.button("Refresh Metrics History"):
        try:
            hist_resp = requests.get(api_url("/metrics/history"), timeout=30)
            if hist_resp.status_code == 200:
                history = hist_resp.json()["history"]
                if history:
                    df_hist = pd.DataFrame(history)
                    df_hist["r2_score"] = df_hist["r2_score"].astype(float)
                    df_hist["mae"] = df_hist["mae"].astype(float)
                    df_hist["mse"] = df_hist["mse"].astype(float)

                    st.dataframe(df_hist, use_container_width=True)

                    st.subheader("R² Score Trend")
                    st.line_chart(df_hist.set_index("version")["r2_score"])

                    st.subheader("MAE / MSE Trend")
                    st.line_chart(df_hist.set_index("version")[["mae", "mse"]])
                else:
                    st.info("No training history yet.")
            else:
                st.error("Could not fetch metrics history.")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to API at {st.session_state['api_base']}. "
                "Start the FastAPI service or set API_BASE_URL before launching Streamlit."
            )