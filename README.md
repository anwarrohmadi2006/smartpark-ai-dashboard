# 🚗 SmartPark AI: Intelligent Parking Occupancy Prediction

![SmartPark AI Header](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange.svg)

**SmartPark AI** is an advanced *Exploratory Data Analysis (EDA)* and *Live Prediction Dashboard* built as part of the DBS Foundation Coding Camp 2026 Capstone Project. It focuses on solving real-world parking congestion and mobility issues using state-of-the-art Deep Learning techniques.

## 🌟 Key Features

1. **Comprehensive EDA Dashboard (`app.py`)**
   - Professional analytics of historical parking occupancy data.
   - Interactive visualizations of operational hours, weather impact, and day-type distributions.
   - Correlation heatmap to validate variables driving occupancy rates.

2. **Offline A/B Testing Validation**
   - Empirical comparison between our **Bidirectional LSTM (BiDir)** model and the traditional **Naive Persistence Baseline**.
   - Demonstrates a statistically significant improvement, reducing the Margin Error (MAE) from 4.2% to **1.4%**.
   - Includes KDE distributions derived from 1000x bootstrapping.

3. **Live AI Inference Engine**
   - Real-time simulation of the deployed predictive model.
   - Automatically extracts *Temporal Attention* and sequence patterns from 18 historical intervals (lags, rolling averages, momentum).
   - Forecasts the exact parking occupancy probability **30 minutes into the future**.

## 🛠️ Architecture & Tech Stack

This repository forms the frontend Analytics layer of the overall architecture:
- **Frontend & Analytics:** Streamlit, Pandas, Matplotlib, Seaborn
- **AI/ML Engine:** TensorFlow / Keras (Custom `TemporalAttention` layer)
- **Statistical Testing:** SciPy

*(Note: The main production API is deployed serverless via Modal.com and consumed by a Next.js web application).*

## 🚀 Running the Dashboard Locally

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Access the dashboard in your browser via `http://localhost:8501`.

## 👥 Development Team
**ID Tim Capstone Project:** CC26-PRU436  
**Tema:** Inclusive & Resilient Communities

- Anwar Rohmadi (Data Science)
- Audie Quisha Jerome Tampubolon (Data Science)
- Salwa Sayyidati Azkia (Artificial Intelligence)
- Gerardus Jeremy Hendrawan (Artificial Intelligence)
- Ryan Fajar Ramadhani (Full Stack Developer)
- M. Faiz Septian (Full Stack Developer)

---
*Built with ❤️ for the DBS Foundation Coding Camp 2026.*
