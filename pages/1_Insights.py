import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")

st.title("💡 Eksekutif & Operational Insights")
st.markdown("Halaman ini menyajikan temuan bisnis utama (*Business Insights*) yang ditarik dari data historis SmartPark. Wawasan ini menjadi landasan utama mengapa model AI Prediktif (Deep Learning) sangat dibutuhkan untuk efisiensi manajemen lahan parkir.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dashboard_dataset.csv")
    except:
        df = pd.read_csv("data/dashboard_dataset.csv")
        
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
    if 'occupied' in df.columns and 'total_slot' in df.columns:
        df['occupancy_rate'] = df['occupied'] / df['total_slot']
    elif 'occupancy_rate' in df.columns and df['occupancy_rate'].max() > 1:
        df['occupancy_rate'] = df['occupancy_rate'] / 100.0
        
    if 'datetime' in df.columns:
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour'] = df['datetime'].dt.hour
    return df

df = load_data()

# ==================================
# Metrics
# ==================================
st.markdown("### 📌 *Key Performance Indicators* (KPI)")
col1, col2, col3, col4 = st.columns(4)

if 'occupancy_rate' in df.columns and 'hour' in df.columns:
    avg_occ = df['occupancy_rate'].mean() * 100
    max_occ = df['occupancy_rate'].max() * 100
    rush_hours_df = df[df['hour'].isin([8,9,17,18])]
    rush_avg = rush_hours_df['occupancy_rate'].mean() * 100
    
    col1.metric("Rata-Rata Okupansi", f"{avg_occ:.1f}%", "Harian Keseluruhan")
    col2.metric("Puncak Okupansi (Max)", f"{max_occ:.1f}%", "Kapasitas Penuh", delta_color="inverse")
    col3.metric("Rata-Rata Jam Sibuk", f"{rush_avg:.1f}%", "08:00 & 17:00", delta_color="off")
    col4.metric("Total Observasi", f"{len(df):,}", "Data Poin (Interval)")

st.markdown("---")

# ==================================
# Insight 1: Pola Harian & Weekend
# ==================================
st.subheader("1. Perilaku Mobilitas: Hari Kerja vs Akhir Pekan")

if 'day_of_week' in df.columns:
    # Mengurutkan hari
    cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_order = pd.CategoricalDtype(categories=cats, ordered=True)
    df['day_of_week'] = df['day_of_week'].astype(day_order)

    peak_day = df.groupby('day_of_week', observed=False)['occupancy_rate'].mean().reset_index()

    fig_day = px.bar(
        peak_day, 
        x="day_of_week", 
        y="occupancy_rate",
        color="occupancy_rate",
        color_continuous_scale="Viridis",
        labels={"day_of_week": "Hari", "occupancy_rate": "Rata-Rata Okupansi"},
        title="Rata-Rata Tingkat Keterisian per Hari"
    )
    fig_day.update_layout(yaxis_tickformat='.0%', showlegend=False)

    col_chart1, col_text1 = st.columns([2, 1])
    with col_chart1:
        st.plotly_chart(fig_day, use_container_width=True)
    with col_text1:
        st.info("""
        **Analisis & Implikasi:**
        - Terlihat pola **anomali pada akhir pekan (Weekend)**, di mana tingkat okupansi bisa sangat berbeda (lebih fluktuatif atau menurun) dibandingkan hari kerja (Senin-Jumat).
        - **Solusi AI:** Model Deep Learning kita menggunakan fitur waktu siklik (*Cyclical Time Encoding* Sin/Cos) sehingga AI secara otomatis mampu membedakan perilaku dan pola antara hari kerja dan akhir pekan tanpa harus di-hardcode.
        """)

# ==================================
# Insight 2: Kemacetan Jam Sibuk (Heatmap Plotly)
# ==================================
st.markdown("---")
st.subheader("2. Deteksi Titik Kritis Jam Sibuk (Rush Hour)")

if 'day_of_week' in df.columns and 'hour' in df.columns:
    heatmap_data = df.groupby(['day_of_week', 'hour'], observed=False)['occupancy_rate'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='occupancy_rate')

    fig_heat = px.imshow(
        heatmap_pivot,
        labels=dict(x="Jam Operasional", y="Hari", color="Okupansi"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Peta Panas (Heatmap) Okupansi Berdasarkan Waktu"
    )

    col_text2, col_chart2 = st.columns([1, 2])
    with col_text2:
        st.warning("""
        **Analisis & Implikasi:**
        - Warna merah gelap menunjukkan **zona kritis (okupansi membeludak, mendekati 100%)**. 
        - Kemacetan paling konsisten terjadi pada jam masuk kerja (Pagi) dan jam pulang kerja (Sore).
        - **Solusi AI:** Berbekal data ini, model LSTM kita dapat memprediksi *rush hour* 30 menit sebelum benar-benar terjadi, memberikan ruang bagi manajemen untuk mengarahkan pengguna ke fasilitas parkir alternatif.
        """)
    with col_chart2:
        st.plotly_chart(fig_heat, use_container_width=True)

# ==================================
# Insight 3: Utilisasi Kamera/Zona
# ==================================
if 'camera_id' in df.columns:
    st.markdown("---")
    st.subheader("3. Distribusi Beban per Zona/Kamera")
    cam_data = df.groupby('camera_id')['occupancy_rate'].mean().sort_values(ascending=False).reset_index()
    
    fig_cam = px.pie(
        cam_data, 
        names='camera_id', 
        values='occupancy_rate', 
        hole=0.4,
        title="Distribusi Rata-Rata Beban per Area",
        color_discrete_sequence=px.colors.sequential.Teal
    )
    
    c1, c2 = st.columns([1,1])
    with c1:
        st.plotly_chart(fig_cam, use_container_width=True)
    with c2:
        st.success("""
        **Analisis & Implikasi:**
        - Utilisasi ruang parkir **tidak merata**. Beberapa area selalu menjadi favorit pengunjung, menyebabkan penumpukan, sementara area lain dibiarkan kosong.
        - **Solusi Strategis:** Melalui prediksi AI yang akurat per zona/kamera, *SmartPark* kelak bisa mendistribusikan *flow* kendaraan langsung ke area yang terprediksi kosong.
        """)

st.markdown("---")
st.markdown("*(Wawasan bisnis di atas merupakan pondasi fundamental dalam merancang fitur-fitur pada arsitektur Artificial Intelligence yang didemonstrasikan di halaman A/B Testing & Live Prediction).*")
