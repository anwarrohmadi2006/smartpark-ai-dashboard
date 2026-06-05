import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="User Portal | SmartPark", page_icon="🚗", layout="centered")

# ==========================================
# 1. Load Data & AI Assets (Silent)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1, name='score')

    def call(self, x):
        e = self.score(x)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

@st.cache_resource
def load_ml_assets():
    base_dir = Path("models")
    with open(base_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(base_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open(base_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
        
    model = tf.keras.models.load_model(str(base_dir / "best_model.keras"), custom_objects={
        'TemporalAttention': TemporalAttention,
        'Orthogonal': tf.keras.initializers.Orthogonal,
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'Zeros': tf.keras.initializers.Zeros,
        'Ones': tf.keras.initializers.Ones
    }, compile=False)
    return model, scaler_X, scaler_y, feature_cols

@st.cache_data
def load_raw_data():
    df = pd.read_csv("dashboard_dataset.csv")
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values('datetime').reset_index(drop=True)
    if 'occupied' in df.columns and 'total_slot' in df.columns:
        df['occupancy_rate'] = df['occupied'] / df['total_slot']
    elif 'occupancy_rate' in df.columns and df['occupancy_rate'].max() > 1:
        df['occupancy_rate'] = df['occupancy_rate'] / 100.0
    return df

def build_features_for_inference(df_window, feature_cols):
    df = df_window.copy()
    weather_map = {'S': 0, 'C': 1, 'R': 2, 'SUNNY': 0, 'OVERCAST': 1, 'RAINY': 2, 'O': 1}
    if 'weather' in df.columns:
        df['weather_encoded'] = df['weather'].map(weather_map).fillna(0)
    else:
        df['weather_encoded'] = 0
        
    if 'datetime' in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
    elif 'day_of_week' in df.columns:
        df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0)
        
    if 'hour' not in df.columns and 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['is_morning_peak'] = df['hour'].between(8, 11).astype(int)
    df['is_evening_peak'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour']    = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)

    if 'occupancy_rate' in df.columns and df['occupancy_rate'].max() > 1:
        df['occupancy_rate'] = df['occupancy_rate'] / 100.0
    elif 'occupancy' in df.columns:
        df['occupancy_rate'] = df['occupancy']
        
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df['occupancy_rate'].shift(lag).fillna(0)
        
    for w in [3, 6, 12, 24, 48]:
        df[f'roll_mean_{w}'] = df['occupancy_rate'].rolling(w, min_periods=1).mean().fillna(0)
        df[f'roll_std_{w}']  = df['occupancy_rate'].rolling(w, min_periods=1).std().fillna(0)
        
    df['momentum']     = df['occupancy_rate'].diff().fillna(0)
    df['acceleration'] = df['momentum'].diff().fillna(0)
    df['ema_01']       = df['occupancy_rate'].ewm(alpha=0.1).mean().fillna(0)
    df['ema_03']       = df['occupancy_rate'].ewm(alpha=0.3).mean().fillna(0)
    
    for c in feature_cols:
        if c not in df.columns: 
            df[c] = 0.0
            
    df[feature_cols] = df[feature_cols].bfill().ffill().fillna(0)
    return df

# Coba Load Model
try:
    model, scaler_X, scaler_y, feature_cols = load_ml_assets()
    assets_loaded = True
except:
    assets_loaded = False

df_raw = load_raw_data()
total_slots_cap = 164 # Berdasarkan info Next.js app

# Asumsikan kita ambil waktu paling akhir sebagai "Current Time" di simulasi ini
current_idx = len(df_raw) - 1
df_window = df_raw.iloc[current_idx - 17 : current_idx + 1].copy()
current_occ_rate = df_window['occupancy_rate'].iloc[-1]
current_occupied = int(current_occ_rate * total_slots_cap)
current_available = total_slots_cap - current_occupied

# ==========================================
# 2. TAMPILAN PENGGUNA (SANGAT SIMPEL)
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🚗 Cari Parkir</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Cek ketersediaan lahan parkir SmartPark secara <i>real-time</i> sebelum Anda tiba.</p>", unsafe_allow_html=True)

st.write("")

# Membuat Gauge/Donut Chart yang sangat intuitif
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = current_available,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Slot Parkir Tersedia", 'font': {'size': 24}},
    number = {'suffix': f" / {total_slots_cap}", 'font': {'size': 50, 'color': '#10B981' if current_available > 20 else '#EF4444'}},
    gauge = {
        'axis': {'range': [0, total_slots_cap]},
        'bar': {'color': "#10B981" if current_available > 20 else "#EF4444"},
        'steps': [
            {'range': [0, 20], 'color': "#FEE2E2"},
            {'range': [20, total_slots_cap], 'color': "#D1FAE5"}
        ],
    }
))

fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

# Prediksi AI untuk End-User
if assets_loaded:
    with st.spinner("AI sedang menerawang kepadatan 30 menit ke depan..."):
        df_features = build_features_for_inference(df_window, feature_cols)
        df_scaled = df_features.copy()
        df_scaled[feature_cols] = scaler_X.transform(df_features[feature_cols])
        seq = df_scaled[feature_cols].values.reshape(1, 18, len(feature_cols))
        
        pred_scaled = model.predict(seq, verbose=0)
        pred_occ_rate = float(np.clip(scaler_y.inverse_transform(pred_scaled).flatten()[0], 0, 1))
        
    pred_available = int(total_slots_cap - (pred_occ_rate * total_slots_cap))
    
    st.markdown("### 🤖 Saran AI Assistant")
    
    if pred_available <= 10:
        st.error(f"**🚨 PERHATIAN!** Dalam 30 menit ke depan, parkiran diprediksi **HAMPIR PENUH** (hanya tersisa sekitar {pred_available} slot). Sebaiknya berangkat sekarang atau cari alternatif parkir lain.")
    elif pred_available < current_available:
        st.warning(f"**⚠️ Tren Semakin Ramai.** Dalam 30 menit ke depan, slot kosong akan berkurang menjadi sekitar **{pred_available} slot**. Segera amankan tempat Anda.")
    else:
        st.success(f"**✅ Parkiran Aman.** Dalam 30 menit ke depan, diprediksi masih ada sekitar **{pred_available} slot** yang kosong. Anda bisa berkendara dengan santai.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Aplikasi simulasi ini dibangun oleh Tim CC26-PRU436 untuk kenyamanan mobilitas Anda.</p>", unsafe_allow_html=True)
