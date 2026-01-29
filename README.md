# 🔋 BatteryLAB

<div align="center">

**Recipe in → Performance out.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://batterylab.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

*An AI-powered tool connecting electrode recipes with battery performance predictions*

[**Try the Live Demo →**](https://batterylab.streamlit.app/)

</div>

---

## ✨ Overview

BatteryLAB is a comprehensive battery data analysis platform that combines **hybrid physics-ML models** with an intuitive **Streamlit interface**. Designed for researchers and engineers, it streamlines the workflow from raw experimental data to actionable insights.

## 🚀 Key Features

### 📊 Data Analytics
- **Multi-format Support** — Upload CSV, MAT (MATLAB), or Excel files for instant analysis
- **Robust Cycle Counting** — Automatically handles time-series data with multiple rows per cycle and invalid entries
- **Auto Feature Extraction** — Capacity fade, internal resistance (IR) growth, and temperature trend analysis
- **Interactive Visualizations** — Voltage curves, dQ/dV (differential capacity analysis), and cycle life plots
- **PDF Report Generation** — Export comprehensive analysis reports with all visualizations

### 🧹 Data Cleaning Module
- **Anomaly Detection** — Z-score and IQR-based outlier identification
- **Data Correction** — Savitzky-Golay smoothing for noise reduction
- **Advanced Imputation** — Linear, Cubic, and Forward-fill methods for missing data
- **Resampling Options** — Uniform capacity/voltage axis or time-based frequency (1Hz)
- **Column Harmonization** — Automatic standardization of column names and units

### 🔬 Recipe Engine
- **Electrode Specifications** — Define active material, binder, and conductive additive compositions
- **Cell Design** — Pouch cell configuration with customizable parameters
- **Temperature-Aware Modeling** — Predictions account for thermal effects on performance
- **Early Degradation Prediction** — Forecast state-of-health and capacity fade trends

### 🤖 AI Copilot
- **Contextual Assistance** — Chat-based interface aware of your current analysis
- **Data-Driven Insights** — Interpretations based on actual uploaded data
- **Guided Workflow** — Step-by-step help for complex analyses

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ajaynair710/BatteryLab.git
cd BatteryLab

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
BatteryLab/
├── app.py                      # Main Streamlit application
├── cleaning_module.py          # Data preprocessing & cleaning logic
├── batterylab_recipe_engine.py # Physics-ML hybrid prediction engine
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version specification
├── LICENSE                     # Apache 2.0 license
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

---

## 📖 Usage Guide

### Data Analytics Workflow
1. Navigate to the **Data Analytics** tab
2. Upload your battery cycling data (CSV, MAT, or Excel)
3. Review auto-detected columns and cycle count
4. Explore interactive visualizations:
   - **Voltage Curves** — Compare charge/discharge profiles
   - **dQ/dV Analysis** — Identify degradation mechanisms
   - **Cycle Life Plots** — Track capacity fade over time
5. Download the generated PDF report

### Data Cleaning Workflow
1. Navigate to the **Cleaning Module** tab
2. Upload raw experimental data
3. Configure cleaning parameters:
   - Select anomaly detection method (Z-score/IQR)
   - Choose smoothing and imputation options
   - Set resampling preferences
4. Preview cleaned data and download processed file

---

## 🧪 Supported File Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Standard comma-separated values |
| MATLAB | `.mat` | v7.3 HDF5 format supported via h5py |
| Excel | `.xlsx`, `.xls` | Multi-sheet selection available |

---

## 📚 Documentation

- **[README_RAG.md](README_RAG.md)** — RAG Copilot setup (OpenAI / local LLM), ingestion, usage  
- **[RAG_FREE.md](RAG_FREE.md)** — Run RAG **free** (no OpenAI, no S3): Ollama or llama-cpp  
- **[RAG_PRODUCTION.md](RAG_PRODUCTION.md)** — Run RAG in production (Streamlit Cloud, Docker, VPS)  
- **[PRODUCTION_USAGE.md](PRODUCTION_USAGE.md)** — **How to use BatteryLab in production** (run, access, configure, Copilot)  
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Full deployment guide (Nginx, SSL, secrets, monitoring)

---

## ⚠️ Disclaimer

> **Note:** This is an illustrative prototype with demonstration models, not lab-calibrated predictions. Results should be validated against experimental data before use in production environments.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the battery research community**

</div>
