# 🔋 BatteryLAB Prototype

**Recipe in → Performance out.**  
BatteryLAB Prototype is the first step toward an AI-powered tool that connects electrode **recipes** with battery **performance predictions**.  

This prototype combines:
- ✅ **MATLAB hybrid physics–ML models** for degradation prediction  
- ✅ **Streamlit interface** for interactive testing  
- ✅ A vision of fast, data-driven battery design  

---

## 🚀 Features
- Upload or define **battery parameters / recipes**
- Predict **early degradation trends** and state-of-health
- Hybrid modeling: physics-based + machine learning

- **Data Analytics Tab**:
  - Upload CSV or MAT files for instant analysis.
  - Robust cycle counting: Automatically handles time-series data (multiple rows per cycle) and invalid entries.
  - Automatic feature extraction: Capacity fade, IR growth, and temperature trends.
  - Interactive visualizations: Voltage curves, dQ/dV analysis, and cycle life plots.

- **Cleaning Module**:
  - Pre-process raw battery data before analysis.
  - Anomaly detection (Z-score, IQR) and correction (Savitzky-Golay smoothing).
  - Advanced missing data imputation (Linear, Cubic, Forward-fill).
  - Resampling options: Uniform capacity/voltage axis, or uniform time frequency (1Hz).
  - Automatic harmonization of column names and units.

- Prototype Streamlit app :

- **[Try the Live Prototype](https://batterylab.streamlit.app/)**

- **Disclaimer:**
- “This is an illustrative model, not lab-calibrated predictions.”

