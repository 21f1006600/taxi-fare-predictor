import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import time
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,205,66,0.2);
    }
    .hero h1 { font-size: 2rem; font-weight: 700; color: #fff; margin: 0; }
    .hero p  { color: rgba(255,255,255,0.65); margin: 0.4rem 0 0; font-size: 0.95rem; }

    .badge {
        display: inline-block;
        background: rgba(255,205,66,0.15);
        border: 1px solid rgba(255,205,66,0.5);
        color: #ffcd42;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }

    .result-box {
        background: linear-gradient(135deg, #1a2a1a, #0d1f0d);
        border: 2px solid #4caf50;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box .label { color: rgba(255,255,255,0.6); font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .result-box .amount { color: #4caf50; font-size: 3rem; font-weight: 700; line-height: 1.2; }
    .result-box .sub { color: rgba(255,255,255,0.45); font-size: 0.8rem; margin-top: 0.3rem; }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .mval { color: #ffcd42; font-size: 1.4rem; font-weight: 700; }
    .metric-card .mlabel { color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.2rem; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #ffcd42, #ffa500);
        color: #000;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.85rem !important;
    }
    hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ── Model training (cached so it only runs once) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_csv("train.csv")

    # Feature engineering — travel time in minutes
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['travel_time'] = (
        (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])
        .dt.total_seconds() / 60
    ).abs()

    # Drop datetime cols
    df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    # Fill missing
    df['store_and_fwd_flag'].fillna('N', inplace=True)
    df['payment_type'].fillna('Credit Card', inplace=True)
    df['payment_type'].replace('unknown', 'Credit Card', inplace=True)

    # One-hot encode
    catcols = ['store_and_fwd_flag', 'payment_type']
    df_enc = pd.get_dummies(df, columns=catcols, prefix=['onehot', 'onehot'], dtype=int)

    # Ensure all expected columns exist
    expected_onehot = ['onehot_N', 'onehot_Y', 'onehot_Cash',
                       'onehot_Credit Card', 'onehot_UPI', 'onehot_Wallet']
    for col in expected_onehot:
        if col not in df_enc.columns:
            df_enc[col] = 0

    feature_cols = [
        'VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
        'PULocationID', 'DOLocationID', 'extra', 'tip_amount',
        'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
        'Airport_fee', 'travel_time',
        'onehot_N', 'onehot_Y',
        'onehot_Cash', 'onehot_Credit Card', 'onehot_UPI', 'onehot_Wallet'
    ]

    X = df_enc[feature_cols]
    y = df_enc['total_amount']

    # Impute remaining nulls
    si = SimpleImputer(strategy='mean')
    X_imp = si.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, model.predict(X_test))

    return model, si, feature_cols, r2


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">🚕 NYC TAXI · XGBoost · Kaggle Top 12%</div>
  <h1>Taxi Fare Predictor</h1>
  <p>Predict NYC taxi total fare using a trained XGBoost model — R² score 0.94, ranked 82 out of 700+</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model... (first load takes ~30s)"):
    model, imputer, feature_cols, r2 = load_model()

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="mval">0.94</div><div class="mlabel">R² Score</div></div>
  <div class="metric-card"><div class="mval">82 / 700+</div><div class="mlabel">Kaggle Rank</div></div>
  <div class="metric-card"><div class="mval">175K</div><div class="mlabel">Training Rows</div></div>
  <div class="metric-card"><div class="mval">XGBoost</div><div class="mlabel">Model</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔢 Enter Trip Details")

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    trip_distance      = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    travel_time        = st.number_input("Travel Time (minutes)", min_value=1.0, max_value=300.0, value=15.0, step=1.0)
    passenger_count    = st.slider("Passenger Count", 1, 6, 1)
    payment_type       = st.selectbox("Payment Type", ["Credit Card", "Cash", "Wallet", "UPI"])

with col2:
    extra              = st.number_input("Extra Charges ($)", min_value=0.0, max_value=10.0, value=2.5, step=0.5)
    tip_amount         = st.number_input("Tip Amount ($)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    tolls_amount       = st.number_input("Tolls Amount ($)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
    store_and_fwd_flag = st.selectbox("Store & Forward Flag", ["N", "Y"])

with st.expander("⚙️ Advanced / Surcharge Fields"):
    c1, c2, c3 = st.columns(3)
    with c1:
        vendor_id             = st.selectbox("Vendor ID", [0, 1], index=1)
        ratecode_id           = st.selectbox("Rate Code", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=0,
                                             help="1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated, 6=Group")
    with c2:
        pu_location_id        = st.number_input("Pickup Location ID", 1, 265, 120)
        do_location_id        = st.number_input("Dropoff Location ID", 1, 265, 90)
    with c3:
        improvement_surcharge = st.number_input("Improvement Surcharge ($)", 0.0, 1.0, 1.0, 0.5)
        congestion_surcharge  = st.number_input("Congestion Surcharge ($)", 0.0, 3.0, 2.5, 0.5)
        airport_fee           = st.number_input("Airport Fee ($)", 0.0, 10.0, 0.0, 1.25)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🚕 Predict Fare"):
    # Build one-hot flags
    oh_n     = 1 if store_and_fwd_flag == "N" else 0
    oh_y     = 1 if store_and_fwd_flag == "Y" else 0
    oh_cash  = 1 if payment_type == "Cash" else 0
    oh_cc    = 1 if payment_type == "Credit Card" else 0
    oh_upi   = 1 if payment_type == "UPI" else 0
    oh_wallet= 1 if payment_type == "Wallet" else 0

    input_data = pd.DataFrame([[
        vendor_id, passenger_count, trip_distance, ratecode_id,
        pu_location_id, do_location_id, extra, tip_amount,
        tolls_amount, improvement_surcharge, congestion_surcharge,
        airport_fee, travel_time,
        oh_n, oh_y, oh_cash, oh_cc, oh_upi, oh_wallet
    ]], columns=feature_cols)

    input_imp = imputer.transform(input_data)

    with st.spinner("Calculating..."):
        time.sleep(0.4)
        prediction = model.predict(input_imp)[0]

    st.markdown(f"""
    <div class="result-box">
      <div class="label">Predicted Total Fare</div>
      <div class="amount">$ {prediction:.2f}</div>
      <div class="sub">Based on {trip_distance} miles · {travel_time:.0f} min · {passenger_count} passenger(s)</div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown estimate
    base_est  = prediction - tip_amount - tolls_amount - extra - improvement_surcharge - congestion_surcharge - airport_fee
    st.markdown("**Fare Breakdown (estimate)**")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Base Fare",    f"${max(base_est,0):.2f}")
    bc2.metric("Tip + Tolls",  f"${tip_amount + tolls_amount:.2f}")
    bc3.metric("Surcharges",   f"${extra + improvement_surcharge + congestion_surcharge + airport_fee:.2f}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.78rem'>"
    "Built by Vinayak Kumar · "
    "<a href='https://github.com/21f1006600/taxi-fare-predictor' style='color:#ffcd42'>GitHub</a> · "
    "<a href='https://www.kaggle.com/code/vinayakkumar23/21f1006600-notebook-t32023/notebook' style='color:#ffcd42'>Kaggle Notebook</a> · "
    "<a href='https://www.linkedin.com/in/vinayak-kumar-48378753/' style='color:#ffcd42'>LinkedIn</a>"
    "</p>",
    unsafe_allow_html=True
)