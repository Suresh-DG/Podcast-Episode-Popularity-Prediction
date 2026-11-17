"""
Podcast Listening Time Prediction — simple Streamlit app (single file)
Run:  streamlit run app.py
Flow: load data -> Train -> Predict. No other files needed.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET = "Listening_Time_minutes"
SEED = 42
RAW_COLS = ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment",
            "Episode_Length_minutes", "Host_Popularity_percentage",
            "Guest_Popularity_percentage", "Number_of_Ads"]

st.set_page_config(page_title="Podcast Listening Time Prediction", page_icon="🎧")


# ----------------------------------------------------------------------
# Data + features
# ----------------------------------------------------------------------
def demo_data(n=5000):
    rng = np.random.default_rng(SEED)
    length = np.clip(rng.normal(45, 20, n), 5, 180)
    host = np.clip(rng.normal(55, 20, n), 0, 100)
    guest = np.clip(rng.normal(50, 25, n), 0, 100)
    ads = rng.integers(0, 6, n)
    sent = rng.choice(["Positive", "Neutral", "Negative"], n, p=[0.45, 0.35, 0.20])
    boost = np.select([sent == "Positive", sent == "Neutral", sent == "Negative"], [8, 0, -6])
    y = np.clip(0.55 * length + 0.15 * host + 0.08 * guest - 1.8 * ads + boost
                + rng.normal(0, 8, n), 1, None)
    return pd.DataFrame({
        "Podcast_Name": rng.choice([f"Podcast_{i}" for i in range(1, 51)], n),
        "Genre": rng.choice(["Comedy", "Technology", "News", "True Crime", "Education",
                             "Health", "Business", "Sports", "Music", "Lifestyle"], n),
        "Publication_Day": rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday",
                                       "Friday", "Saturday", "Sunday"], n),
        "Publication_Time": rng.choice(["Morning", "Afternoon", "Evening", "Night"], n),
        "Episode_Sentiment": sent,
        "Episode_Length_minutes": length.round(1),
        "Host_Popularity_percentage": host.round(1),
        "Guest_Popularity_percentage": guest.round(1),
        "Number_of_Ads": ads,
        TARGET: y.round(1),
    })


def add_features(df):
    """Derived columns. Built automatically for training AND for predictions."""
    df = df.copy()
    df["Ads_per_minute"] = (df["Number_of_Ads"] /
                            df["Episode_Length_minutes"].replace(0, np.nan)).fillna(0)
    df["Is_Weekend"] = df["Publication_Day"].isin(["Saturday", "Sunday"]).astype(int)
    df["Popularity_Gap"] = df["Host_Popularity_percentage"] - df["Guest_Popularity_percentage"]
    return df


def train_all(raw, max_rows):
    """Clean -> features -> encode -> scale -> train 3 models. Returns one dict."""
    missing = [c for c in RAW_COLS + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = raw[RAW_COLS + [TARGET]].drop_duplicates().dropna(subset=[TARGET])
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED)

    # fill gaps with the median (numbers) / most common value (text)
    fills = {}
    for c in RAW_COLS:
        fills[c] = df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]
        df[c] = df[c].fillna(fills[c])

    X = add_features(df[RAW_COLS])
    y = df[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)

    encoders = {}
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            encoders[c] = LabelEncoder().fit(X[c].astype(str))
            X[c] = encoders[c].transform(X[c].astype(str))

    cols = list(X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14,
                                               random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, max_depth=8,
                                                           learning_rate=0.08, random_state=SEED),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        p = m.predict(X_te_s)
        rows.append({"Model": name,
                     "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
                     "MAE": float(mean_absolute_error(y_te, p)),
                     "R²": float(r2_score(y_te, p))})
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    return {"models": models, "metrics": metrics, "best": metrics.loc[0, "Model"],
            "encoders": encoders, "scaler": scaler, "cols": cols, "fills": fills,
            "options": {c: sorted(df[c].astype(str).unique()) for c in RAW_COLS
                        if not pd.api.types.is_numeric_dtype(df[c])},
            "n_rows": len(df), "X_te_s": X_te_s, "y_te": y_te}


def predict_one(b, raw_inputs, model_name):
    """Same steps as training: derived features -> encode -> scale -> predict."""
    row = add_features(pd.DataFrame([raw_inputs]))
    for c, enc in b["encoders"].items():
        row[c] = enc.transform(row[c].astype(str))
    scaled = b["scaler"].transform(row[b["cols"]])
    pred = float(b["models"][model_name].predict(scaled)[0])
    # listening time can't be negative or longer than the episode itself
    return float(np.clip(pred, 0, raw_inputs["Episode_Length_minutes"]))


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🎧 Podcast Listening Time Prediction")
st.caption("Predict how many minutes listeners will stay with an episode.")

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Demo data", "Upload CSV"])
    raw = None
    if source == "Demo data":
        raw = demo_data()
    else:
        f = st.file_uploader("Upload train.csv", type=["csv"])
        if f is not None:
            raw = pd.read_csv(f)
    max_rows = st.number_input("Max rows to train on", 5000, 750000, 50000, step=5000,
                               help="Use a smaller number for faster training.")
    train_clicked = st.button("🚀 Train models", type="primary", width="stretch")

if raw is None:
    st.info("Upload your CSV in the sidebar (it must contain **Listening_Time_minutes**), "
            "or choose **Demo data**.")
    st.stop()

if train_clicked:
    try:
        with st.spinner("Training 3 models..."):
            # saved in session_state so it is still there after every later click
            st.session_state["bundle"] = train_all(raw, int(max_rows))
    except Exception as e:
        st.error(str(e))
        st.stop()

b = st.session_state.get("bundle")
if b is None:
    st.info("👈 Click **Train models** in the sidebar first.")
    st.stop()

# ---- results ----
st.subheader("Model comparison")
st.caption(f"Trained on {b['n_rows']:,} rows · best model: **{b['best']}**")
st.dataframe(b["metrics"].style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"}),
             hide_index=True, width="stretch")

# ---- prediction ----
st.subheader("Predict a new episode")
model_name = st.selectbox("Model", list(b["models"]),
                          index=list(b["models"]).index(b["best"]), key="in_model")

c1, c2, c3 = st.columns(3)
podcast = c1.selectbox("Podcast name", b["options"]["Podcast_Name"], key="in_podcast")
genre = c1.selectbox("Genre", b["options"]["Genre"], key="in_genre")
sentiment = c1.selectbox("Episode sentiment", b["options"]["Episode_Sentiment"], key="in_sent")
length = c2.number_input("Episode length (minutes)", 1.0, 400.0,
                         float(round(b["fills"]["Episode_Length_minutes"], 1)),
                         step=1.0, key="in_length")
host = c2.number_input("Host popularity (%)", 0.0, 100.0,
                       float(round(b["fills"]["Host_Popularity_percentage"], 1)),
                       step=1.0, key="in_host")
guest = c2.number_input("Guest popularity (%)", 0.0, 100.0,
                        float(round(b["fills"]["Guest_Popularity_percentage"], 1)),
                        step=1.0, key="in_guest")
ads = c3.number_input("Number of ads", 0.0, 20.0, float(round(b["fills"]["Number_of_Ads"])),
                      step=1.0, key="in_ads")
day = c3.selectbox("Publication day", b["options"]["Publication_Day"], key="in_day")
time_ = c3.selectbox("Publication time", b["options"]["Publication_Time"], key="in_time")

if st.button("Predict", type="primary"):
    inputs = {"Podcast_Name": podcast, "Genre": genre, "Publication_Day": day,
              "Publication_Time": time_, "Episode_Sentiment": sentiment,
              "Episode_Length_minutes": length, "Host_Popularity_percentage": host,
              "Guest_Popularity_percentage": guest, "Number_of_Ads": ads}
    pred = predict_one(b, inputs, model_name)
    st.metric("Predicted listening time", f"{pred:.1f} minutes")
    st.progress(min(pred / length, 1.0), text=f"{pred / length:.0%} of the episode is listened to")

    with st.expander("See all three models' predictions"):
        st.dataframe(pd.DataFrame({
            "Model": list(b["models"]),
            "Predicted minutes": [round(predict_one(b, inputs, m), 1) for m in b["models"]],
        }), hide_index=True, width="stretch")
