# ==============================================================
# BatteryLab Prototype — Full app with in-tab Copilot + data-aware replies
# ==============================================================

import io
from io import BytesIO
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import random, time  # for Copilot typing effect
import os  # for temp file handling

# Suppress numpy divide/nan warnings for gradient calculations
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice.*")

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Matplotlib for plots we embed in the PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional (MAT/ICA features)
try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.io import loadmat, savemat
    import h5py
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Your temperature-aware engine module
from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

# Data cleaning module
from cleaning_module import clean_dataframe, get_dataframe_info, to_csv_download, get_excel_sheets, read_excel_sheets, filter_channel_sheets, validate_analysis_compatibility, to_mat_download, export_to_cell11_mat_format, read_file_universal

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="BatteryLab Prototype", page_icon=None, layout="wide")
st.title("BatteryLab — Prototype")
st.caption(
    "Recipe -> Performance (with temperature advisories) and Data Analytics: "
    "Upload -> analysis first (richness and next steps) -> button to visualize -> plots, features, interpretations, PDF export."
)

# ---- Copilot chat memory ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "text": str}

def _copilot_add(role: str, text: str):
    st.session_state.chat_history.append({"role": role, "text": text})

# ---- Copilot data caches (latest outputs) ----
if "latest_design" not in st.session_state:
    st.session_state.latest_design = None   # {"spec_summary": {...}, "result": {...}}
if "latest_analytics" not in st.session_state:
    st.session_state.latest_analytics = None  # {"features_by_group": {...}, "vc_all": DataFrame, "ica_all": DataFrame}
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_applied" not in st.session_state:
    st.session_state.cleaning_applied = False

# =========================
# Helpers (shared)
# =========================
def _downsample_series(df: pd.DataFrame, max_points: int = 10000) -> pd.DataFrame:
    """Downsample a DataFrame to a maximum number of points while preserving shape."""
    if len(df) <= max_points:
        return df
    return df.iloc[::len(df) // max_points].copy()

def _standardize_columns(df: pd.DataFrame):
    cols_l = [c.lower() for c in df.columns]
    vcol = None
    for i, c in enumerate(cols_l):
        if c in ["v", "volt", "voltage", "voltage_v"]:
            vcol = df.columns[i]; break
    qcol = None
    for i, c in enumerate(cols_l):
        if "capacity" in c or c in ["q", "ah", "mah", "capacity_ah", "capacity_mah"]:
            qcol = df.columns[i]; break
    cyc = None
    for i, c in enumerate(cols_l):
        if c in ["cycle", "label", "group"]:
            cyc = df.columns[i]; break
    return vcol, qcol, cyc

def _detect_cycle_level_data(df: pd.DataFrame):
    """Detect if data is in cycle-level format (one row per cycle)"""
    cols_l = [c.lower() for c in df.columns]
    # Normalize column names (remove parentheses, spaces, underscores for matching)
    cols_normalized = [c.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_') for c in cols_l]
    
    # Check for cycle column - expanded to include cycle_index
    has_cycle = any(
        c in ["cycle", "cycle_number", "cycle_idx", "cycle_index"] or 
        c_norm in ["cycle", "cycle_number", "cycle_idx", "cycle_index"]
        for c, c_norm in zip(cols_l, cols_normalized)
    )
    
    # Accept QDischarge, discharge_capacity, discharge_cap, capacity, cap, q
    # Also handle Discharge_Capacity(Ah) format
    has_discharge_cap = any(
        "discharge_capacity" in c or "discharge_cap" in c or "qdischarge" in c or
        (c in ["capacity", "cap", "q"] and "discharge" in " ".join(cols_l)) or c == "qdischarge" or
        "dischargecapacity" in c_norm or "discharge_capacity" in c_norm
        for c, c_norm in zip(cols_l, cols_normalized)
    )
    
    # Accept IR, internal_resistance, resistance - handle Internal_Resistance(Ohm) format
    has_ir = any(
        "internal_resistance" in c or c == "ir" or "resistance" in c or 
        "internalresistance" in c_norm or "internal_resistance" in c_norm
        for c, c_norm in zip(cols_l, cols_normalized)
    )
    
    # Accept temperature, temp, tmax, tavg, tmin
    has_temp = any("temperature" in c or c == "temp" or c in ["tmax", "tavg", "tmin"] for c in cols_l)
    
    return has_cycle and (has_discharge_cap or has_ir or has_temp)

def _parse_cycle_level_data(df: pd.DataFrame):
    """Parse cycle-level data and extract cycle metrics"""
    cols_l = [c.lower() for c in df.columns]
    # Normalize column names for matching
    cols_normalized = [c.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_') for c in cols_l]
    
    # Find cycle column - expanded to include cycle_index
    cycle_col = None
    for i, (c, c_norm) in enumerate(zip(cols_l, cols_normalized)):
        if c in ["cycle", "cycle_number", "cycle_idx", "cycle_index"] or c_norm in ["cycle", "cycle_number", "cycle_idx", "cycle_index"]:
            cycle_col = df.columns[i]
            break
    
    # Find discharge capacity - handle Discharge_Capacity(Ah) format
    cap_col = None
    for i, (c, c_norm) in enumerate(zip(cols_l, cols_normalized)):
        if ("discharge_capacity" in c or "discharge_cap" in c or "qdischarge" in c or 
            "dischargecapacity" in c_norm or "discharge_capacity" in c_norm or
            ("capacity" in c and "discharge" in " ".join(cols_l[:i+1]))):
            cap_col = df.columns[i]
            break
    if cap_col is None:
        for i, (c, c_norm) in enumerate(zip(cols_l, cols_normalized)):
            if c in ["capacity", "cap", "q", "discharge_cap", "qdischarge"] or "dischargecapacity" in c_norm:
                cap_col = df.columns[i]
                break
    
    # Find internal resistance - handle Internal_Resistance(Ohm) format
    ir_col = None
    for i, (c, c_norm) in enumerate(zip(cols_l, cols_normalized)):
        if ("internal_resistance" in c or c == "ir" or "resistance" in c or
            "internalresistance" in c_norm or "internal_resistance" in c_norm):
            ir_col = df.columns[i]
            break
    
    # Find temperature (accept temperature, temp, tmax, tavg, tmin; prefer tavg, then tmax, then tmin)
    temp_col = None
    for pref in ["tavg", "temperature", "temp", "tmax", "tmin"]:
        for i, c in enumerate(cols_l):
            if c == pref or pref in c:
                temp_col = df.columns[i]
                break
        if temp_col:
            break
    
    if cycle_col is None or cap_col is None:
        return None
    
    # Extract cycle-level data
    cycle_data = df[[cycle_col, cap_col]].copy()
    cycle_data.columns = ['cycle', 'discharge_capacity']
    
    if ir_col:
        cycle_data['internal_resistance'] = pd.to_numeric(df[ir_col], errors='coerce')
    else:
        cycle_data['internal_resistance'] = np.nan
    
    if temp_col:
        cycle_data['temperature'] = pd.to_numeric(df[temp_col], errors='coerce')
    else:
        cycle_data['temperature'] = np.nan
    
    # Convert cycle to numeric
    cycle_data['cycle'] = pd.to_numeric(cycle_data['cycle'], errors='coerce')
    cycle_data['discharge_capacity'] = pd.to_numeric(cycle_data['discharge_capacity'], errors='coerce')
    
    # Convert mAh to Ah if needed
    if cycle_data['discharge_capacity'].max() > 100:
        cycle_data['discharge_capacity'] = cycle_data['discharge_capacity'] / 1000.0
    
    cycle_data = cycle_data.sort_values('cycle').reset_index(drop=True)
    return cycle_data

def _parse_time_series_columns(df: pd.DataFrame):
    """Parse optional time-series columns: time_<cycle>, current_<cycle>, voltage_<cycle>"""
    time_series = {}
    cols = df.columns
    
    # Find all time-series columns
    for col in cols:
        col_l = col.lower()
        if col_l.startswith('time_'):
            cycle_num = col_l.replace('time_', '')
            if cycle_num.isdigit():
                cycle_num = int(cycle_num)
                if cycle_num not in time_series:
                    time_series[cycle_num] = {}
                time_series[cycle_num]['time'] = pd.to_numeric(df[col], errors='coerce').values
        elif col_l.startswith('current_'):
            cycle_num = col_l.replace('current_', '')
            if cycle_num.isdigit():
                cycle_num = int(cycle_num)
                if cycle_num not in time_series:
                    time_series[cycle_num] = {}
                time_series[cycle_num]['current'] = pd.to_numeric(df[col], errors='coerce').values
        elif col_l.startswith('voltage_'):
            cycle_num = col_l.replace('voltage_', '')
            if cycle_num.isdigit():
                cycle_num = int(cycle_num)
                if cycle_num not in time_series:
                    time_series[cycle_num] = {}
                time_series[cycle_num]['voltage'] = pd.to_numeric(df[col], errors='coerce').values
    
    return time_series

def _prep_series(voltage, capacity):
    V = pd.to_numeric(voltage, errors="coerce").to_numpy()
    Qraw = pd.to_numeric(capacity, errors="coerce").to_numpy()
    Q = Qraw / 1000.0 if np.nanmax(Qraw) > 100 else Qraw
    mask = np.isfinite(V) & np.isfinite(Q)
    V = V[mask]; Q = Q[mask]
    order = np.argsort(V)
    return V[order], Q[order]

def _compute_robust_dqdv(voltage, capacity, window_length=15, polyorder=2):
    """
    Compute a robust dQ/dV curve using sorting, linear interpolation, and smoothing.
    """
    # 1. Basic Cleaning
    V = pd.to_numeric(voltage, errors='coerce').to_numpy()
    Q = pd.to_numeric(capacity, errors='coerce').to_numpy()
    
    mask = np.isfinite(V) & np.isfinite(Q)
    V = V[mask]
    Q = Q[mask]
    
    if len(V) < 5:
        return V, np.zeros_like(V)
        
    # 2. Sort by Voltage
    sorted_idx = np.argsort(V)
    V_sorted = V[sorted_idx]
    Q_sorted = Q[sorted_idx]
    
    # 3. Drop duplicates in V to allow interpolation
    _, unique_idx = np.unique(V_sorted, return_index=True)
    V_unique = V_sorted[unique_idx]
    Q_unique = Q_sorted[unique_idx]
    
    if len(V_unique) < 5:
        return V_unique, np.zeros_like(V_unique)
        
    # 4. Interpolate to a uniform grid (optional but helps with stability)
    # create a grid with roughly same number of points, but capped to avoid excessive noise/size
    grid_size = min(len(V_unique), 2000)
    V_grid = np.linspace(V_unique.min(), V_unique.max(), grid_size)
    Q_grid = np.interp(V_grid, V_unique, Q_unique)
    
    # 5. Smooth Capacity (Savitzky-Golay)
    # Ensure window_length is odd and <= len(Q_grid)
    # Adaptive window length: ~1-2% of grid size
    target_wl = max(15, int(grid_size * 0.02))
    if target_wl % 2 == 0: target_wl += 1
    
    wl = target_wl
    if wl > len(Q_grid): wl = len(Q_grid) if len(Q_grid) % 2 != 0 else len(Q_grid) - 1
    
    if SCIPY_OK and wl >= 5:
        try:
            Q_smooth = savgol_filter(Q_grid, wl, polyorder)
        except:
            Q_smooth = Q_grid
    else:
        Q_smooth = Q_grid
        
    # 6. Calculate Gradient dQ/dV
    dQ = np.gradient(Q_smooth)
    dV = np.gradient(V_grid)
    
    # Avoid div by zero
    mask_dv = np.abs(dV) > 1e-8
    dQdV = np.zeros_like(V_grid)
    dQdV[mask_dv] = dQ[mask_dv] / dV[mask_dv]
    
    return V_grid, dQdV

def _extract_features(V, Q):
    # Use robust dQ/dV calculation
    V_robust, dQdV = _compute_robust_dqdv(V, Q)
    
    peak_info = {"n_peaks": 0, "voltages": [], "widths_V": []}
    if SCIPY_OK and np.isfinite(dQdV).any():
        try:
            prom = np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0
            peaks, _ = find_peaks(dQdV, prominence=prom)
            peak_info["n_peaks"] = int(len(peaks))
            peak_info["voltages"] = [float(V_robust[p]) for p in peaks]
            if len(peaks) > 0:
                widths, _, _, _ = peak_widths(dQdV, peaks, rel_height=0.5)
                if len(V_robust) > 1:
                    dv = np.mean(np.diff(V_robust))
                    peak_info["widths_V"] = [float(w*dv) for w in widths]
        except Exception:
            pass
    
    # Estimate noise level (median |dV/dQ|)
    try:
        # Approximate dV/dQ as 1/(dQ/dV)
        valid = (np.abs(dQdV) > 1e-6) & np.isfinite(dQdV)
        if np.any(valid):
            dVdQ_med = float(np.nanmedian(np.abs(1.0 / dQdV[valid])))
        else:
            dVdQ_med = float("nan")
    except Exception:
        dVdQ_med = float("nan")
        
    return V_robust, dQdV, peak_info, dVdQ_med

def _compare_two_sets(name_a, feat_a, name_b, feat_b):
    interp = []
    cap_a = feat_a.get("cap_range_Ah", [np.nan, np.nan])[1]
    cap_b = feat_b.get("cap_range_Ah", [np.nan, np.nan])[1]
    if np.isfinite(cap_a) and np.isfinite(cap_b) and cap_a > 0:
        fade_pct = 100.0 * (cap_a - cap_b) / cap_a
        if abs(fade_pct) >= 3:
            interp.append(f"Capacity change from {name_a} to {name_b}: {fade_pct:.1f}% (negative = fade).")
    Va = feat_a.get("ica_peak_voltages_V", [])
    Vb = feat_b.get("ica_peak_voltages_V", [])
    if Va and Vb:
        n = min(len(Va), len(Vb))
        if n >= 1:
            mean_shift_mV = 1000.0 * float(np.nanmean(np.array(Vb[:n]) - np.array(Va[:n])))
            if abs(mean_shift_mV) >= 5:
                direction = "up" if mean_shift_mV > 0 else "down"
                interp.append(f"Average ICA peak shift {direction} ~{abs(mean_shift_mV):.0f} mV ({name_b} vs {name_a}).")
    Wa = feat_a.get("ica_peak_widths_V", [])
    Wb = feat_b.get("ica_peak_widths_V", [])
    if Wa and Wb:
        n = min(len(Wa), len(Wb))
        if n >= 1:
            mean_broad_mV = 1000.0 * float(np.nanmean(np.array(Wb[:n]) - np.array(Wa[:n])))
            if mean_broad_mV > 2:
               interp.append(f"ICA peak broadening ~{mean_broad_mV:.0f} mV ({name_b} vs {name_a}).")

    iva = feat_a.get("dVdQ_median_abs", np.nan)
    ivb = feat_b.get("dVdQ_median_abs", np.nan)
    if np.isfinite(iva) and np.isfinite(ivb) and ivb > iva * 1.05:
        interp.append(f"Median |dV/dQ| increased ({name_b} vs {name_a}).")
    if not interp:
        interp.append(f"No strong differences detected between {name_a} and {name_b} within prototype sensitivity.")
    return interp

def _extract_cycle_features(cycle_data: pd.DataFrame):
    """Extract key features from cycle-level data.
    
    Returns:
        tuple: (features_dict, cycle_data_agg) where cycle_data_agg is the 
               aggregated/corrected DataFrame ready for plotting.
    """
    features = {}
    
    if len(cycle_data) == 0:
        return features, cycle_data
    
    # Ensure cycle column is numeric and drop invalid cycles
    cycle_data = cycle_data.copy()
    cycle_data['cycle'] = pd.to_numeric(cycle_data['cycle'], errors='coerce')
    cycle_data = cycle_data.dropna(subset=['cycle'])
    
    if len(cycle_data) == 0:
        return features, cycle_data

    # Check if we need to aggregate (multiple rows per cycle)
    # We suspect this if the number of rows > number of unique cycles
    n_unique = cycle_data['cycle'].nunique()
    
    if len(cycle_data) > n_unique:
        # Aggregate by cycle for summary statistics
        # Capacity: take MAX (battery cycler data accumulates within each cycle, starting from 0)
        # The MAX value represents the total capacity discharged in that cycle
        agg_rules = {
            'discharge_capacity': 'max',  # Use max since capacity accumulates from 0 within each cycle
        }
        if 'internal_resistance' in cycle_data.columns:
            agg_rules['internal_resistance'] = 'mean'
        if 'temperature' in cycle_data.columns:
            agg_rules['temperature'] = 'mean'
            
        # Group and aggregate
        cycle_data_agg = cycle_data.groupby('cycle').agg(agg_rules).reset_index()
        cycle_data_agg = cycle_data_agg.sort_values('cycle')
        
        # Check if the aggregated capacity is CUMULATIVE ACROSS CYCLES
        # (i.e., capacity keeps increasing with cycle number, not resetting per cycle)
        # This happens when the data logger records total throughput, not per-cycle capacity
        cap_values = cycle_data_agg['discharge_capacity'].values
        if len(cap_values) >= 3:
            # Check if capacity is monotonically increasing (within tolerance for noise)
            diffs = np.diff(cap_values)
            pct_positive = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else 0
            
            # If >90% of diffs are positive and final >> initial, it's cumulative
            if pct_positive > 0.9 and cap_values[-1] > cap_values[0] * 2:
                # This is cumulative throughput - convert to per-cycle by differencing
                per_cycle_cap = np.diff(cap_values, prepend=0)
                # First value should use the actual first MAX (since 0 was prepended)
                per_cycle_cap[0] = cap_values[0]
                cycle_data_agg['discharge_capacity'] = per_cycle_cap
    else:
        cycle_data_agg = cycle_data.copy()
        cycle_data_agg = cycle_data_agg.sort_values('cycle')

    # Basic stats
    features['n_cycles'] = int(len(cycle_data_agg))
    features['cycle_range'] = [int(cycle_data_agg['cycle'].min()), int(cycle_data_agg['cycle'].max())]
    
    # Capacity features
    cap = cycle_data_agg['discharge_capacity'].dropna()
    if len(cap) > 0:
        features['capacity_max_Ah'] = float(cap.max())
        features['capacity_min_Ah'] = float(cap.min())
        features['capacity_mean_Ah'] = float(cap.mean())
        features['capacity_std_Ah'] = float(cap.std()) if len(cap) > 1 else 0.0
        features['capacity_initial_Ah'] = float(cap.iloc[0]) if len(cap) > 0 else np.nan
        features['capacity_final_Ah'] = float(cap.iloc[-1]) if len(cap) > 0 else np.nan
        
        # Fade rate calculations
        if len(cap) >= 2:
            # Linear fade rate (% per cycle)
            cycles = cycle_data_agg['cycle'].dropna().values
            if len(cycles) == len(cap):
                valid_mask = np.isfinite(cap.values) & np.isfinite(cycles)
                if np.sum(valid_mask) >= 2:
                    coeffs = np.polyfit(cycles[valid_mask], cap.values[valid_mask], 1)
                    fade_rate_per_cycle = -coeffs[0] / cap.values[valid_mask][0] * 100.0 if cap.values[valid_mask][0] > 0 else 0.0
                    features['fade_rate_percent_per_cycle'] = float(fade_rate_per_cycle)
            
            # Total fade percentage
            if features['capacity_initial_Ah'] > 0:
                total_fade_pct = 100.0 * (features['capacity_initial_Ah'] - features['capacity_final_Ah']) / features['capacity_initial_Ah']
                features['total_fade_percent'] = float(total_fade_pct)
    
    # Internal resistance features
    if 'internal_resistance' in cycle_data_agg.columns:
        ir = cycle_data_agg['internal_resistance'].dropna()
        if len(ir) > 0:
            features['ir_max_Ohm'] = float(ir.max())
            features['ir_min_Ohm'] = float(ir.min())
            features['ir_mean_Ohm'] = float(ir.mean())
            features['ir_std_Ohm'] = float(ir.std()) if len(ir) > 1 else 0.0
            features['ir_initial_Ohm'] = float(ir.iloc[0]) if len(ir) > 0 else np.nan
            features['ir_final_Ohm'] = float(ir.iloc[-1]) if len(ir) > 0 else np.nan
            if features['ir_initial_Ohm'] > 0:
                ir_growth_pct = 100.0 * (features['ir_final_Ohm'] - features['ir_initial_Ohm']) / features['ir_initial_Ohm']
                features['ir_growth_percent'] = float(ir_growth_pct)
    
    # Temperature features
    if 'temperature' in cycle_data_agg.columns:
        temp = cycle_data_agg['temperature'].dropna()
        if len(temp) > 0:
            features['temp_max_C'] = float(temp.max())
            features['temp_min_C'] = float(temp.min())
            features['temp_mean_C'] = float(temp.mean())
            features['temp_std_C'] = float(temp.std()) if len(temp) > 1 else 0.0
    
    return features, cycle_data_agg

def _assess_data_quality_and_richness(cycle_data: pd.DataFrame, time_series: dict, cycle_features: dict):
    """Assess data quality and richness for cycle-level battery data"""
    quality_notes = []
    richness_notes = []
    
    # Assess cycle count
    n_cycles = cycle_features.get('n_cycles', 0)
    if n_cycles < 10:
        quality_notes.append("**Low cycle count** (<10 cycles): Limited degradation trend visibility")
    elif n_cycles < 50:
        quality_notes.append("**Moderate cycle count** (10-50 cycles): Good for initial degradation analysis")
    else:
        quality_notes.append("**High cycle count** (>50 cycles): Excellent for comprehensive lifetime analysis")
    
    # Assess capacity data completeness
    cap_valid = cycle_data['discharge_capacity'].notna().sum() if 'discharge_capacity' in cycle_data.columns else 0
    if cap_valid > 0:
        cap_completeness = (cap_valid / n_cycles * 100) if n_cycles > 0 else 0
        if cap_completeness < 50:
            quality_notes.append(f"**Capacity data incomplete** ({cap_completeness:.1f}% coverage)")
        else:
            quality_notes.append(f"**Capacity data available** ({cap_valid}/{n_cycles} cycles, {cap_completeness:.1f}% coverage)")
    
    # Assess internal resistance data
    ir_valid = cycle_data['internal_resistance'].notna().sum() if 'internal_resistance' in cycle_data.columns else 0
    if ir_valid > 0:
        ir_completeness = (ir_valid / n_cycles * 100) if n_cycles > 0 else 0
        quality_notes.append(f"**Internal resistance data available** ({ir_valid}/{n_cycles} cycles, {ir_completeness:.1f}% coverage)")
        richness_notes.append("Internal resistance enables power fade analysis and impedance growth tracking")
    else:
        quality_notes.append("**Internal resistance data missing**: Power fade analysis not possible")
    
    # Assess temperature data
    temp_valid = cycle_data['temperature'].notna().sum() if 'temperature' in cycle_data.columns else 0
    if temp_valid > 0:
        temp_completeness = (temp_valid / n_cycles * 100) if n_cycles > 0 else 0
        quality_notes.append(f"**Temperature data available** ({temp_valid}/{n_cycles} cycles, {temp_completeness:.1f}% coverage)")
        richness_notes.append("Temperature data enables thermal analysis and correlation with performance degradation")
    else:
        quality_notes.append("**Temperature data missing**: Thermal-electrochemical coupling analysis not possible")
    
    # Assess time-series data
    if time_series and len(time_series) > 0:
        ts_cycles = len(time_series)
        quality_notes.append(f"**Time-series data available** ({ts_cycles} cycles with current/voltage profiles)")
        richness_notes.append(f"Time-series profiles enable voltage-capacity curve analysis, dQ/dV, and detailed cycle behavior")
        if ts_cycles < n_cycles * 0.1:
            richness_notes.append(f"Only {ts_cycles} cycles have time-series data - consider extracting more for richer analysis")
    else:
        quality_notes.append("**Time-series data missing**: Voltage-capacity curve and dQ/dV analysis not possible")
    
    # Overall richness assessment
    param_count = sum([
        1 if cap_valid > 0 else 0,
        1 if ir_valid > 0 else 0,
        1 if temp_valid > 0 else 0,
        1 if time_series and len(time_series) > 0 else 0
    ])
    
    if param_count >= 4:
        richness_notes.insert(0, "**Rich dataset**: Contains all 4 key parameters (capacity, IR, temperature, time-series)")
    elif param_count >= 3:
        richness_notes.insert(0, "**Good dataset**: Contains 3 out of 4 key parameters")
    elif param_count >= 2:
        richness_notes.insert(0, "**Basic dataset**: Contains 2 out of 4 key parameters")
    else:
        richness_notes.insert(0, "**Limited dataset**: Contains only 1 key parameter")
    
    return quality_notes, richness_notes

def _generate_degradation_interpretations(cycle_features: dict, cycle_data: pd.DataFrame):
    """Generate AI-style interpretations of degradation trends"""
    interps = []
    
    if not cycle_features:
        return ["Insufficient cycle data for degradation analysis."]
    
    # Capacity fade analysis
    if 'total_fade_percent' in cycle_features:
        fade_pct = cycle_features['total_fade_percent']
        if fade_pct > 20:
            interps.append(f"Severe capacity fade detected: {fade_pct:.1f}% loss over {cycle_features.get('n_cycles', 'N')} cycles. This suggests accelerated degradation, possibly due to high C-rates, temperature extremes, or material instability.")
        elif fade_pct > 10:
            interps.append(f"Moderate capacity fade: {fade_pct:.1f}% loss. Typical for standard cycling conditions. Monitor for acceleration in later cycles.")
        elif fade_pct > 5:
            interps.append(f"Mild capacity fade: {fade_pct:.1f}% loss. Performance is relatively stable.")
        elif fade_pct > 0:
            interps.append(f"Minimal capacity fade: {fade_pct:.1f}% loss. Excellent cycle life performance.")
        else:
            interps.append("No capacity fade detected. Capacity may have increased slightly (measurement variation or conditioning).")
    
    # Fade rate analysis
    if 'fade_rate_percent_per_cycle' in cycle_features:
        rate = cycle_features['fade_rate_percent_per_cycle']
        if rate > 0.1:
            interps.append(f"High fade rate: {rate:.3f}% per cycle. Projected end-of-life (80% SOH) in ~{int(20/rate)} cycles if trend continues.")
        elif rate > 0.05:
            interps.append(f"Moderate fade rate: {rate:.3f}% per cycle. Projected 80% SOH in ~{int(20/rate)} cycles.")
        elif rate > 0:
            interps.append(f"Low fade rate: {rate:.3f}% per cycle. Good cycle life expected.")
    
    # Internal resistance growth
    if 'ir_growth_percent' in cycle_features:
        ir_growth = cycle_features['ir_growth_percent']
        if ir_growth > 50:
            interps.append(f"Significant IR growth: {ir_growth:.1f}% increase. This indicates severe impedance rise, likely from SEI growth, contact loss, or electrolyte degradation.")
        elif ir_growth > 20:
            interps.append(f"Moderate IR growth: {ir_growth:.1f}% increase. Typical for extended cycling. May correlate with capacity fade.")
        elif ir_growth > 0:
            interps.append(f"Mild IR growth: {ir_growth:.1f}% increase. Resistance evolution is within normal range.")
    
    # Temperature analysis
    if 'temp_mean_C' in cycle_features:
        temp_mean = cycle_features['temp_mean_C']
        if temp_mean > 45:
            interps.append(f"High average temperature: {temp_mean:.1f}°C. Elevated temperatures accelerate degradation and may explain observed fade.")
        elif temp_mean < 10:
            interps.append(f"Low average temperature: {temp_mean:.1f}°C. Cold conditions can increase impedance and reduce capacity utilization.")
        else:
            interps.append(f"Temperature in normal range: {temp_mean:.1f}°C average. Thermal conditions appear favorable.")
    
    # Correlation insights
    if 'ir_growth_percent' in cycle_features and 'total_fade_percent' in cycle_features:
        ir_growth = cycle_features['ir_growth_percent']
        fade = cycle_features['total_fade_percent']
        if ir_growth > 20 and fade > 10:
            interps.append("Both capacity fade and IR growth are significant. This pattern suggests combined loss of active material (LAM) and loss of lithium inventory (LLI), possibly from SEI growth or electrode degradation.")
        elif ir_growth > fade * 2:
            interps.append("IR growth exceeds capacity fade proportionally. This may indicate impedance-dominated degradation, possibly from contact issues or electrolyte depletion.")
    
    if not interps:
        interps.append("Cycle data available but degradation trends are minimal or within measurement uncertainty.")
    
    return interps

# Plot -> PNG bytes for embedding in PDF
def _plot_to_bytes(df_all, ycol, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    for c in df_all["Cycle"].unique():
        sub = df_all[df_all["Cycle"] == c]
        ax.plot(sub["Voltage"], sub[ycol], label=str(c))
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _plot_cycle_life(cycle_data: pd.DataFrame, y_col: str, y_label: str, title: str):
    """Plot cycle-life data (capacity, IR, or temperature vs cycle)"""
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    valid = cycle_data.dropna(subset=['cycle', y_col])
    if len(valid) > 0:
        ax.plot(valid['cycle'], valid[y_col], 'o-', markersize=4, linewidth=1.5)
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _plot_time_series(time_series: dict, cycle_nums: list, title_prefix: str):
    """Plot current/voltage time-series for selected cycles"""
    if not time_series or not cycle_nums:
        return None
    
    fig, axes = plt.subplots(len(cycle_nums), 2, figsize=(10, 3*len(cycle_nums)), dpi=150)
    if len(cycle_nums) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, cycle_num in enumerate(cycle_nums):
        if cycle_num not in time_series:
            continue
        
        ts = time_series[cycle_num]
        time = ts.get('time', None)
        current = ts.get('current', None)
        voltage = ts.get('voltage', None)
        
        # Ensure time is a numpy array
        if time is not None:
            if isinstance(time, (list, tuple)):
                time = np.array(time)
        
        if time is not None and current is not None:
            # Ensure current is a numpy array and match length with time
            if isinstance(current, (list, tuple)):
                current = np.array(current)
            # Match lengths
            min_len = min(len(time), len(current))
            time_aligned = time[:min_len]
            current_aligned = current[:min_len]
            valid = np.isfinite(time_aligned) & np.isfinite(current_aligned)
            if np.any(valid):
                axes[idx, 0].plot(time_aligned[valid], current_aligned[valid], 'b-', linewidth=1.5)
                axes[idx, 0].set_xlabel("Time (s)")
                axes[idx, 0].set_ylabel("Current (A)")
                axes[idx, 0].set_title(f"Cycle {cycle_num} - Current")
                axes[idx, 0].grid(True, alpha=0.3)
        
        if time is not None and voltage is not None:
            # Ensure voltage is a numpy array and match length with time
            if isinstance(voltage, (list, tuple)):
                voltage = np.array(voltage)
            # Match lengths
            min_len = min(len(time), len(voltage))
            time_aligned = time[:min_len]
            voltage_aligned = voltage[:min_len]
            valid = np.isfinite(time_aligned) & np.isfinite(voltage_aligned)
            if np.any(valid):
                axes[idx, 1].plot(time_aligned[valid], voltage_aligned[valid], 'r-', linewidth=1.5)
                axes[idx, 1].set_xlabel("Time (s)")
                axes[idx, 1].set_ylabel("Voltage (V)")
                axes[idx, 1].set_title(f"Cycle {cycle_num} - Voltage")
                axes[idx, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_pdf_report(features_by_group=None, richness_notes=None, suggestions=None, interps=None, 
                       vc_all=None, ica_all=None, cycle_data=None, cycle_features=None, 
                       degradation_interps=None, time_series=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("BatteryLab Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 8))

    # Cycle-life analysis section (if available)
    if cycle_data is not None and len(cycle_data) > 0:
        elements.append(Paragraph("Cycle-Life Analysis", styles["Heading2"]))

        # Cycle features table
        if cycle_features:
            elements.append(Paragraph("Cycle-Level Statistics", styles["Heading3"]))
            feat_table = [["Metric", "Value"]]
            if 'n_cycles' in cycle_features:
                feat_table.append(["Number of Cycles", str(cycle_features['n_cycles'])])
            if 'capacity_initial_Ah' in cycle_features and np.isfinite(cycle_features['capacity_initial_Ah']):
                feat_table.append(["Initial Capacity (Ah)", f"{cycle_features['capacity_initial_Ah']:.3f}"])
            if 'capacity_final_Ah' in cycle_features and np.isfinite(cycle_features['capacity_final_Ah']):
                feat_table.append(["Final Capacity (Ah)", f"{cycle_features['capacity_final_Ah']:.3f}"])
            if 'total_fade_percent' in cycle_features and np.isfinite(cycle_features['total_fade_percent']):
                feat_table.append(["Total Fade (%)", f"{cycle_features['total_fade_percent']:.2f}"])
            if 'fade_rate_percent_per_cycle' in cycle_features and np.isfinite(cycle_features['fade_rate_percent_per_cycle']):
                feat_table.append(["Fade Rate (%/cycle)", f"{cycle_features['fade_rate_percent_per_cycle']:.4f}"])
            if 'ir_initial_Ohm' in cycle_features and np.isfinite(cycle_features['ir_initial_Ohm']):
                feat_table.append(["Initial IR (Ohm)", f"{cycle_features['ir_initial_Ohm']:.4f}"])
            if 'ir_final_Ohm' in cycle_features and np.isfinite(cycle_features['ir_final_Ohm']):
                feat_table.append(["Final IR (Ohm)", f"{cycle_features['ir_final_Ohm']:.4f}"])
            if 'ir_growth_percent' in cycle_features and np.isfinite(cycle_features['ir_growth_percent']):
                feat_table.append(["IR Growth (%)", f"{cycle_features['ir_growth_percent']:.2f}"])

            if len(feat_table) > 1:
                tbl_feat = Table(feat_table, repeatRows=1)
                tbl_feat.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 0.25, colors.black),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                ]))
                elements.append(tbl_feat)
                elements.append(Spacer(1, 6))

        # Cycle-life plots
        elements.append(Paragraph("Cycle-Life Plots", styles["Heading3"]))
        if 'discharge_capacity' in cycle_data.columns:
            img_bytes = _plot_cycle_life(cycle_data, 'discharge_capacity', 'Discharge Capacity (Ah)',
                                         'Discharge Capacity vs Cycle Life')
            elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
            elements.append(Spacer(1, 6))

        if 'internal_resistance' in cycle_data.columns and cycle_data['internal_resistance'].notna().any():
            img_bytes = _plot_cycle_life(cycle_data, 'internal_resistance', 'Internal Resistance (Ohm)',
                                         'Internal Resistance vs Cycle Life')
            elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
            elements.append(Spacer(1, 6))

        if 'temperature' in cycle_data.columns and cycle_data['temperature'].notna().any():
            img_bytes = _plot_cycle_life(cycle_data, 'temperature', 'Temperature (°C)',
                                         'Temperature vs Cycle Life')
            elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
            elements.append(Spacer(1, 6))

        # Time-series plots
        if time_series:
            elements.append(Paragraph("Time-Series Plots (First/Middle/Last Cycles)", styles["Heading3"]))
            cycle_nums = sorted(time_series.keys())
            if len(cycle_nums) >= 3:
                selected = [cycle_nums[0], cycle_nums[len(cycle_nums)//2], cycle_nums[-1]]
            elif len(cycle_nums) >= 1:
                selected = cycle_nums[:min(3, len(cycle_nums))]
            else:
                selected = []

            if selected:
                img_bytes = _plot_time_series(time_series, selected, "Current and Voltage Time-Series")
                if img_bytes:
                    elements.append(Image(io.BytesIO(img_bytes), width=500, height=300))
                    elements.append(Spacer(1, 6))

        # Degradation interpretations
        if degradation_interps:
            elements.append(Paragraph("Degradation Trend Analysis", styles["Heading3"]))
            for di in degradation_interps:
                elements.append(Paragraph("- " + di, styles["Normal"]))
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 12))

    # Voltage-capacity / ICA analysis (if available)
    if features_by_group is not None:
        elements.append(Paragraph("Dataset Quality & Richness", styles["Heading2"]))
        if richness_notes:
            for r in richness_notes:
                elements.append(Paragraph("- " + r, styles["Normal"]))
        else:
            elements.append(Paragraph("No specific richness notes.", styles["Normal"]))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Next-Step Suggestions", styles["Heading2"]))
        if suggestions:
            for s in suggestions:
                elements.append(Paragraph("- " + s, styles["Normal"]))
        else:
            elements.append(Paragraph("No suggestions generated.", styles["Normal"]))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Key Features by Curve", styles["Heading2"]))
        table_data = [["Curve", "n_samples", "V Range (V)", "Capacity Range (Ah)",
                       "ICA Peaks", "ICA Voltages (V)", "ICA Widths (V)", "Median |dV/dQ|"]]
        for g, f in features_by_group.items():
            table_data.append([
                str(g),
                f['n_samples'],
                f"{f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f}",
                f"{f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f}",
                f['ica_peaks_count'],
                ", ".join([f"{v:.3f}" for v in f['ica_peak_voltages_V']]) if f['ica_peak_voltages_V'] else "—",
                ", ".join([f"{w:.3f}" for w in f['ica_peak_widths_V']]) if f['ica_peak_widths_V'] else "—",
                f"{f['dVdQ_median_abs']:.4f}" if np.isfinite(f['dVdQ_median_abs']) else "—"
            ])
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('ALIGN', (1,1), (-1,-1), 'CENTER')
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("AI-style Interpretations", styles["Heading2"]))
        if interps:
            for i in interps:
                elements.append(Paragraph("- " + i, styles["Normal"]))
        else:
            elements.append(Paragraph("No interpretations generated.", styles["Normal"]))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("Voltage-Capacity Visualizations", styles["Heading2"]))
        if vc_all is not None and len(vc_all) > 0:
            elements.append(Paragraph("Voltage vs Capacity", styles["Heading3"]))
            img_bytes = _plot_to_bytes(vc_all, "Capacity_Ah", "Voltage vs Capacity")
            elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
        if ica_all is not None and len(ica_all) > 0:
            elements.append(Paragraph("ICA: dQ/dV vs Voltage", styles["Heading3"]))
            img_bytes = _plot_to_bytes(ica_all, "dQdV", "ICA: dQ/dV vs Voltage")
            elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def generate_recipe_pdf(spec_summary, result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("BatteryLab Recipe Report", styles["Title"]))
    elements.append(Spacer(1, 8))

    # Inputs
    elements.append(Paragraph("Design Inputs", styles["Heading2"]))
    geom_txt = ""
    if spec_summary.get("area_cm2") is not None:
        geom_txt = f"Direct area: {spec_summary.get('area_cm2')} cm2"
    else:
        geom_txt = f"W x H: {spec_summary.get('width_mm')} x {spec_summary.get('height_mm')} mm"

    rows = [
        ["Geometry", geom_txt],
        ["Layers", f"{spec_summary.get('n_layers')}"],
        ["N/P ratio", f"{spec_summary.get('n_p_ratio'):.2f}"],
        ["Cathode", f"{spec_summary.get('cathode_material')} — thickness {spec_summary.get('cathode_thk_um')} um, porosity {spec_summary.get('cathode_por'):.2f}"],
        ["Anode", f"{spec_summary.get('anode_material')} (Si {spec_summary.get('anode_si_frac'):.2f}) — thickness {spec_summary.get('anode_thk_um')} um, porosity {spec_summary.get('anode_por'):.2f}"],
        ["Separator", f"{spec_summary.get('sep_thk_um')} um, porosity {spec_summary.get('sep_por'):.2f}"],
        ["Foils", f"Al {spec_summary.get('foil_al_um')} um, Cu {spec_summary.get('foil_cu_um')} um"],
        ["Electrolyte", f"{spec_summary.get('electrolyte')}"],
        ["Ambient", f"{spec_summary.get('ambient_C')} C"],
    ]

    table = Table([["Parameter", "Value"]] + rows, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 8))

    # Results
    elements.append(Paragraph("Computed Performance", styles["Heading2"]))
    perf_rows = [
        ["Capacity (Ah)", f"{result['electrochem']['capacity_Ah']:.2f}"],
        ["Nominal Voltage (V)", f"{result['electrochem']['V_nom']:.2f}"],
        ["Wh/kg", f"{result['electrochem']['Wh_per_kg']:.0f}"],
        ["Wh/L", f"{result['electrochem']['Wh_per_L']:.0f}"],
        ["DeltaT @1C (C)", f"{result['thermal']['deltaT_1C_C']:.2f}"],
        ["DeltaT @3C (C)", f"{result['thermal']['deltaT_3C_C']:.2f}"],
    ]
    t2 = Table([["Metric", "Value"]] + perf_rows, repeatRows=1)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 8))

    # Feasibility
    elements.append(Paragraph("Mechanical & Feasibility", styles["Heading2"]))
    fz = result.get("feasibility", {})
    mech = result.get("mechanical", {})
    feas_points = [
        f"Swelling flag: {fz.get('swelling_flag')}",
        f"Thermal @3C: {fz.get('thermal_flag_3C')}",
        f"Swelling % @ 100% SOC: {round(mech.get('swelling_pct_100SOC', float('nan')), 2)}%",
    ]
    for p in feas_points:
        elements.append(Paragraph("- " + str(p), styles["Normal"]))
    elements.append(Spacer(1, 6))

    # Temperature advisories
    elements.append(Paragraph("Temperature Advisories", styles["Heading2"]))
    tg = result.get("temperature_guidance", {})
    ea = result.get("electrochem_temp_adjusted", {})
    t_lines = [
        f"Ambient: {tg.get('ambient_C', '—')} C",
        f"Ideal window: {tg.get('ideal_low_C','—')}–{tg.get('ideal_high_C','—')} C",
        f"Effective Capacity @ ambient: {ea.get('effective_capacity_Ah_at_ambient', '—')}",
        f"Relative Power vs 25 C: {ea.get('relative_power_vs_25C', '—')}x",
    ]
    if tg.get("cold_temp_risk", False):
        t_lines.append("Cold-condition risk (<= 0 C): expect higher impedance/lower power; pre-heat or derate C-rate.")
    if tg.get("high_temp_risk", False):
        t_lines.append("High ambient (>= 45 C): accelerated side reactions; consider high-temp electrolyte, charge derating, better cooling.")
    for p in t_lines:
        elements.append(Paragraph("- " + str(p), styles["Normal"]))
    elements.append(Spacer(1, 6))

    # AI Suggestions
    elements.append(Paragraph("AI Suggestions", styles["Heading2"]))
    for s in result.get("ai_suggestions", []):
        elements.append(Paragraph("- " + str(s), styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Data-aware utilities for Copilot ----------
def _compute_fade_pct(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves (e.g., Fresh vs Aged) to compute fade."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    cap_a = features_by_group[a]["cap_range_Ah"][1]
    cap_b = features_by_group[b]["cap_range_Ah"][1]
    if not np.isfinite(cap_a) or not np.isfinite(cap_b) or cap_a <= 0:
        return "Couldn’t compute fade from the current features."
    fade = 100.0 * (cap_a - cap_b) / cap_a
    sign = "fade" if fade >= 0 else "gain"
    return f"Capacity {sign}: {abs(fade):.1f}% (comparing '{a}' → '{b}')."

def _compute_peak_shift(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves to estimate ICA peak shifts."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    Va = np.array(features_by_group[a]["ica_peak_voltages_V"] or [], dtype=float)
    Vb = np.array(features_by_group[b]["ica_peak_voltages_V"] or [], dtype=float)
    if len(Va) == 0 or len(Vb) == 0:
        return "No ICA peaks detected (or insufficient resolution) to compute peak shifts."
    n = min(len(Va), len(Vb))
    shift_mV = 1000.0 * float(np.nanmean(Vb[:n] - Va[:n]))
    direction = "up" if shift_mV > 0 else "down"
    return f"Mean ICA peak shift: {abs(shift_mV):.0f} mV {direction} ('{b}' vs '{a}')."

def _compute_broadening(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves to estimate peak broadening."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    Wa = np.array(features_by_group[a]["ica_peak_widths_V"] or [], dtype=float)
    Wb = np.array(features_by_group[b]["ica_peak_widths_V"] or [], dtype=float)
    if len(Wa) == 0 or len(Wb) == 0:
        return "Not enough peak width info to estimate broadening."
    n = min(len(Wa), len(Wb))
    broad_mV = 1000.0 * float(np.nanmean(Wb[:n] - Wa[:n]))
    tag = "broadened" if broad_mV >= 0 else "narrowed"
    return f"ICA peaks {tag} by ~{abs(broad_mV):.0f} mV ('{b}' vs '{a}')."

def _summarize_design_temperature(result: dict) -> str:
    if not result:
        return "Run a design first — I don’t have a computed pouch result yet."
    tg = result.get("temperature_guidance", {}) or {}
    ea = result.get("electrochem_temp_adjusted", {}) or {}
    parts = []
    parts.append(f"Ambient {tg.get('ambient_C','?')}°C; ideal window {tg.get('ideal_low_C','?')}–{tg.get('ideal_high_C','?')}°C.")
    if "effective_capacity_Ah_at_ambient" in ea:
        parts.append(f"Effective capacity @ ambient ~ {ea['effective_capacity_Ah_at_ambient']:.2f} Ah.")
    if "relative_power_vs_25C" in ea:
        parts.append(f"Relative power vs 25°C ~ {ea['relative_power_vs_25C']:.2f}×.")
    if tg.get("cold_temp_risk"): parts.append("Cold-risk flagged → pre-heat or derate C-rate.")
    if tg.get("high_temp_risk"): parts.append("High-temp risk → consider cooling, high-temp electrolyte, charge derating.")
    return " ".join(parts) or "No temperature guidance available."

# =========================
# Data-aware Copilot
# =========================
def _copilot_reply(user_text: str, context: str = "general") -> str:
    """
    Data-aware Copilot:
      - understands a few intents
      - pulls from st.session_state.latest_design / latest_analytics when available
    """
    text = (user_text or "").strip().lower()

    # quick intent detection
    wants_help   = any(k in text for k in ["/help", "help", "what can you do", "commands"])
    wants_fade   = any(k in text for k in ["fade", "capacity drop", "soh"])
    wants_shift  = any(k in text for k in ["peak shift", "ica shift", "peak position"])
    wants_broad  = any(k in text for k in ["broadening", "width", "fwhm"])
    wants_temp   = any(k in text for k in ["temperature", "thermal", "ambient", "cooling", "heating"])
    wants_reco   = any(k in text for k in ["recommend", "suggest", "next step", "what next"])
    wants_best   = any(k in text for k in ["best curve", "which curve", "highest capacity", "lowest fade"])

    # HELP
    if wants_help:
        return (
            "Here’s what I can do now:\n"
            "• Analytics: fade %, ICA peak shifts, broadening, quick insights (/help, 'fade', 'peak shift', 'broadening').\n"
            "• Design: summarize temperature risks & adjustments ('temperature summary').\n"
            "• Recommendations: next steps for analysis or design ('recommend next step').\n"
            "Tip: upload or analyze data first (Analytics tab), or run Compute Performance (Design tab)."
        )

    # pull caches
    la = st.session_state.latest_analytics
    ld = st.session_state.latest_design

    # ANALYTICS INTENTS
    if wants_fade and la and la.get("features_by_group"):
        return _compute_fade_pct(la["features_by_group"])

    if wants_shift and la and la.get("features_by_group"):
        return _compute_peak_shift(la["features_by_group"])

    if wants_broad and la and la.get("features_by_group"):
        return _compute_broadening(la["features_by_group"])

    if wants_best and la and la.get("features_by_group"):
        fbg = la["features_by_group"]
        best = max(fbg.items(), key=lambda kv: kv[1]["cap_range_Ah"][1])
        return f"Best curve by capacity: '{best[0]}' (max ~{best[1]['cap_range_Ah'][1]:.2f} Ah)."

    # DESIGN INTENTS
    if wants_temp and ld and ld.get("result"):
        return _summarize_design_temperature(ld["result"])

    if wants_reco:
        if context == "design" and ld and ld.get("result"):
            tg = ld["result"].get("temperature_guidance", {}) or {}
            recos = []
            if tg.get("high_temp_risk"): recos.append("Improve cooling or reduce charge C-rate; consider high-temp electrolyte/additives.")
            if tg.get("cold_temp_risk"): recos.append("Pre-heat or lower discharge C-rate; increase porosity for low-T power.")
            recos.append("Try ±5–10 µm cathode thickness and ±0.02 porosity sweep, compare Wh/L vs ΔT@3C.")
            return " • ".join(recos) if recos else "Looks stable—try small thickness/porosity sweeps to optimize Wh/L vs ΔT."
        if context == "analytics" and la and la.get("features_by_group"):
            return (
                "1) Quantify fade and peak shifts (‘fade’, ‘peak shift’). "
                "2) If IR/temperature available, correlate with |dV/dQ| median. "
                "3) Export MAT + repro script, then fit a simple regressor for EOL prediction."
            )

    # fallbacks by context
    if context == "design":
        return "Design Copilot here. Try: 'temperature summary', 'recommend next step'."
    if context == "analytics":
        return "Analytics Copilot here. Try: 'fade', 'peak shift', 'broadening', or 'recommend next step'."

    return "Got it! Use /help to see data-aware commands."

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Recipe -> Performance", "Data Analytics (CSV/MAT)", "Cleaning Module"])

# ==========
# In-tab Copilot (sticky right column) – shared memory via st.session_state.chat_history
# ==========
def render_copilot(context_key: str, default_context: str = "Design"):
    radio_key = f"ctx_{context_key}"
    input_key = f"msg_{context_key}"
    send_key  = f"send_{context_key}"

    st.markdown(
        """
        <style>
          .blab-box { position: sticky; top: 70px; max-height: 78vh; overflow-y: auto; 
                      background: #f7f7f9; border: 1px solid #ddd; padding: 12px 14px; border-radius: 8px; }
          .blab-bubble { padding: 8px 10px; border-radius: 10px; margin-bottom: 8px; line-height: 1.45; }
          .blab-user { background:#DCF8C6; text-align:right; }
          .blab-bot  { background:#fff; border:1px solid #e6e6e6; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### BatteryLAB Copilot 💬")
    st.caption("Always-on chat for this tab. Brainstorm designs, request plots, or ask next steps.")

    ctx = st.radio(
        "Context",
        ["Design", "Analytics"],
        index=(0 if default_context.lower() == "design" else 1),
        horizontal=True,
        key=radio_key
    )

    with st.container():
        st.markdown('<div class="blab-box">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history[-12:]:
            cls = "blab-user" if msg["role"] == "user" else "blab-bot"
            who = "You" if msg["role"] == "user" else "Copilot"
            st.markdown(f'<div class="blab-bubble {cls}"><b>{who}:</b> {msg["text"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    user_text = st.text_input(
        "Type a message…",
        key=input_key,
        placeholder="e.g. Try Si=0.05 and lower anode porosity by 0.02"
    )
    if st.button("Send", key=send_key, use_container_width=True):
        if user_text.strip():
            _copilot_add("user", user_text.strip())
            reply = _copilot_reply(user_text, context=("design" if ctx == "Design" else "analytics"))
            _copilot_add("assistant", reply)

# =========================
# TAB 1: Recipe -> Performance (temperature-aware) WITH in-tab Copilot
# =========================
with tab1:
    col_main, col_chat = st.columns([0.68, 0.32], gap="large")

    with col_main:
        PRESETS = {
            "LFP 2.5Ah (demo)": {
                "cathode": {"material": "LFP", "thk": 70, "por": 0.35},
                "anode":   {"material": "Graphite", "thk": 85, "por": 0.40, "si": 0.00},
                "geom":    {"mode": "area", "area_cm2": 100.0, "n_layers": 36},
                "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
                "np": 1.10, "elyte": "1M LiPF6 in EC:EMC 3:7", "amb": 25
            },
            "NMC811 3Ah (demo)": {
                "cathode": {"material": "NMC811", "thk": 75, "por": 0.33},
                "anode":   {"material": "Graphite", "thk": 90, "por": 0.40, "si": 0.05},
                "geom":    {"mode": "area", "area_cm2": 110.0, "n_layers": 32},
                "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
                "np": 1.08, "elyte": "1M LiPF6 + 2% VC in EC:DEC", "amb": 25
            }
        }

        with st.sidebar:
            st.header("Quick Start")
            preset = st.selectbox("Preset", ["— none —"] + list(PRESETS.keys()))

            defaults = {
                "geom_mode": "area",
                "area_cm2": 100.0, "width_mm": 70.0, "height_mm": 100.0,
                "n_layers": 36, "n_p_ratio": 1.10,
                "electrolyte": "1M LiPF6 in EC:EMC 3:7", "ambient_C": 25,
                "cath_mat": "LFP", "cath_thk": 70, "cath_por": 0.35,
                "anode_mat": "Graphite", "anode_thk": 85, "anode_por": 0.40, "anode_si": 0.00,
                "sep_thk": 20, "sep_por": 0.45, "foil_al": 15, "foil_cu": 8,
            }

            if preset != "— none —":
                p = PRESETS[preset]
                defaults.update({
                    "geom_mode": p["geom"]["mode"],
                    "area_cm2": p["geom"]["area_cm2"], "n_layers": p["geom"]["n_layers"],
                    "n_p_ratio": p["np"], "electrolyte": p["elyte"], "ambient_C": p["amb"],
                    "cath_mat": p["cathode"]["material"], "cath_thk": p["cathode"]["thk"], "cath_por": p["cathode"]["por"],
                    "anode_mat": p["anode"]["material"], "anode_thk": p["anode"]["thk"],
                    "anode_por": p["anode"]["por"], "anode_si": p["anode"]["si"],
                    "sep_thk": p["sep"]["thk"], "sep_por": p["sep"]["por"],
                    "foil_al": p["foil"]["al"], "foil_cu": p["foil"]["cu"],
                })

            st.header("Cell Geometry")
            area_mode = st.radio("Area Input Mode", ["Direct area (cm2)", "Width x Height (mm)"],
                                 index=0 if defaults["geom_mode"] == "area" else 1)

            if area_mode == "Direct area (cm2)":
                area_cm2 = st.number_input("Layer area (cm2)", min_value=10.0, value=float(defaults["area_cm2"]), step=5.0)
                dims = {"area_cm2": area_cm2}
            else:
                width_mm = st.number_input("Width (mm)", min_value=10.0, value=float(defaults["width_mm"]), step=1.0)
                height_mm = st.number_input("Height (mm)", min_value=10.0, value=float(defaults["height_mm"]), step=1.0)
                dims = {"width_mm": width_mm, "height_mm": height_mm}

            n_layers = st.number_input("# Layers", min_value=2, value=int(defaults["n_layers"]), step=2)
            n_p_ratio = st.slider("N/P ratio", 1.00, 1.30, float(defaults["n_p_ratio"]), 0.01)
            electrolyte = st.text_input("Electrolyte (free text)", defaults["electrolyte"])
            ambient_C = st.slider("Ambient Temp (C)", -20, 60, int(defaults["ambient_C"]), 1)

        st.subheader("Cathode")
        cathode_material = st.selectbox("Material (Cathode)", ["LFP", "NMC811"],
                                        index=(0 if defaults["cath_mat"] == "LFP" else 1))
        cathode_thk = st.slider("Cathode thickness (um)", 20, 140, int(defaults["cath_thk"]), 1)
        cathode_por = st.slider("Cathode porosity", 0.20, 0.60, float(defaults["cath_por"]), 0.01)

        st.subheader("Anode")
        anode_material = st.selectbox("Material (Anode)", ["Graphite"], index=0)
        anode_thk = st.slider("Anode thickness (um)", 20, 140, int(defaults["anode_thk"]), 1)
        anode_por = st.slider("Anode porosity", 0.20, 0.60, float(defaults["anode_por"]), 0.01)
        anode_si = st.slider("Anode silicon fraction (0..1)", 0.0, 0.20, float(defaults["anode_si"]), 0.01)

        st.subheader("Separator & Foils")
        sep_thk = st.slider("Separator thickness (um)", 10, 40, int(defaults["sep_thk"]), 1)
        sep_por = st.slider("Separator porosity", 0.20, 0.70, float(defaults["sep_por"]), 0.01)
        foil_al = st.slider("Cathode Al foil (um)", 8, 20, int(defaults["foil_al"]), 1)
        foil_cu = st.slider("Anode Cu foil (um)", 4, 15, int(defaults["foil_cu"]), 1)

        if cathode_thk < 20 or anode_thk < 20:
            st.warning("Very thin coatings may make predictions less reliable.")
        if not (0.2 <= cathode_por <= 0.6 and 0.2 <= anode_por <= 0.6):
            st.warning("Porosity outside typical ranges may reduce accuracy.")

        run = st.button("Compute Performance", type="primary")

        if run:
            with st.spinner("Computing physics + AI suggestions..."):
                cathode = ElectrodeSpec(material=cathode_material, thickness_um=cathode_thk, porosity=cathode_por, active_frac=0.96)
                anode   = ElectrodeSpec(material=anode_material, thickness_um=anode_thk, porosity=anode_por, active_frac=0.96, silicon_frac=anode_si)
                spec = CellDesignInput(
                    cathode=cathode, anode=anode, n_layers=int(n_layers),
                    separator_thickness_um=sep_thk, separator_porosity=sep_por, n_p_ratio=float(n_p_ratio),
                    cathode_foil_um=foil_al, anode_foil_um=foil_cu, electrolyte=electrolyte, ambient_C=float(ambient_C),
                    **dims
                )
                result = design_pouch(spec)

            st.success("Computed successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Capacity (Ah)", f"{result['electrochem']['capacity_Ah']:.2f}")
                st.metric("Nominal Voltage (V)", f"{result['electrochem']['V_nom']:.2f}")
            with col2:
                st.metric("Wh/kg", f"{result['electrochem']['Wh_per_kg']:.0f}")
                st.metric("Wh/L", f"{result['electrochem']['Wh_per_L']:.0f}")
            with col3:
                st.metric("DeltaT @1C (C)", f"{result['thermal']['deltaT_1C_C']:.2f}")
                st.metric("DeltaT @3C (C)", f"{result['thermal']['deltaT_3C_C']:.2f}")

            st.markdown("### Mechanical & Feasibility")
            fz = result["feasibility"]
            cols = st.columns(3)
            cols[0].markdown(f"Swelling flag: {fz['swelling_flag']}")
            cols[1].markdown(f"Thermal @3C: {fz['thermal_flag_3C']}")
            cols[2].markdown(f"Swelling % @100% SOC: {round(result['mechanical']['swelling_pct_100SOC'],2)}%")

            # Temperature advisories
            st.markdown("### Temperature Advisories")
            tg = result.get("temperature_guidance", {})
            ea = result.get("electrochem_temp_adjusted", {})

            cols_t = st.columns(4)
            cols_t[0].markdown(f"Ambient: {tg.get('ambient_C', '—')} C")
            cols_t[1].markdown(f"Ideal window: {tg.get('ideal_low_C','—')}–{tg.get('ideal_high_C','—')} C")
            try:
                eff_cap = float(ea.get('effective_capacity_Ah_at_ambient', float('nan')))
                cols_t[2].markdown(f"Effective Capacity @ ambient: {eff_cap:.2f} Ah")
            except Exception:
                cols_t[2].markdown("Effective Capacity @ ambient: —")
            try:
                rel_pow = float(ea.get('relative_power_vs_25C', float('nan')))
                cols_t[3].markdown(f"Relative Power vs 25 C: {rel_pow:.2f}x")
            except Exception:
                cols_t[3].markdown("Relative Power vs 25 C: —")

            risk_msgs = []
            if tg.get("cold_temp_risk", False):
                risk_msgs.append("Cold-condition risk (<= 0 C): expect higher impedance/lower power; pre-heat or derate C-rate.")
            if tg.get("high_temp_risk", False):
                risk_msgs.append("High ambient (>= 45 C): accelerated side reactions; consider high-temp electrolyte, charge derating, better cooling.")
            if not risk_msgs:
                risk_msgs.append("Ambient in acceptable range for typical operation.")
            for m in risk_msgs:
                st.write("- " + m)

            st.markdown("### AI Suggestions")
            for s in result["ai_suggestions"]:
                st.write("- " + s)

            # Build a compact design-specs summary for the PDF
            spec_summary = {
                "area_cm2": (dims.get("area_cm2") if "area_cm2" in dims else None),
                "width_mm": (dims.get("width_mm") if "width_mm" in dims else None),
                "height_mm": (dims.get("height_mm") if "height_mm" in dims else None),
                "n_layers": int(n_layers),
                "n_p_ratio": float(n_p_ratio),
                "cathode_material": cathode_material,
                "cathode_thk_um": int(cathode_thk),
                "cathode_por": float(cathode_por),
                "anode_material": anode_material,
                "anode_thk_um": int(anode_thk),
                "anode_por": float(anode_por),
                "anode_si_frac": float(anode_si),
                "sep_thk_um": int(sep_thk),
                "sep_por": float(sep_por),
                "foil_al_um": int(foil_al),
                "foil_cu_um": int(foil_cu),
                "electrolyte": electrolyte,
                "ambient_C": float(ambient_C),
            }

            pdf_recipe = generate_recipe_pdf(spec_summary, result)
            st.download_button(
                "Download Recipe Report (PDF)",
                data=pdf_recipe,
                file_name="BatteryLab_recipe_report.pdf",
                mime="application/pdf"
            )

            # Store for Copilot (design cache)
            st.session_state.latest_design = {"spec_summary": spec_summary, "result": result}

        else:
            st.info("Pick a preset for a 1-click demo, or set your recipe parameters, then press Compute Performance.")

    with col_chat:
        render_copilot(context_key="tab1", default_context="Design")

# =========================
# TAB 2: Data Analytics WITH in-tab Copilot + exports (.mat, .py)
# =========================
with tab2:
    col_main, col_chat = st.columns([0.68, 0.32], gap="large")

    with col_main:
        st.subheader("Upload a dataset")
        st.write(
            "Accepted: .csv (recommended) or .mat (if SciPy is available). "
            "Two formats supported:\n"
            "1. **Cycle-level data**: Columns: cycle, discharge_capacity, internal_resistance (optional), temperature (optional). "
            "Optional time-series: time_<cycle>, current_<cycle>, voltage_<cycle>.\n"
            "2. **Voltage-Capacity curves**: Columns: Voltage, Capacity_Ah (or Capacity_mAh). Optional: Cycle (e.g., Fresh/Aged)."
        )

        up = st.file_uploader("Upload CSV, MAT or Excel file", type=["csv", "mat", "xlsx", "xls"])
        if up is not None:
            # Check file size before processing
            file_size_mb = len(up.getvalue()) / (1024 * 1024)
            
            # Configure memory limits based on file size
            if file_size_mb > 100:
                max_depth = 3
                max_arrays = 500
                max_array_size = 10_000_000  # Skip arrays with more than 10M elements
            else:
                max_depth = 8
                max_arrays = 1000
                max_array_size = 10_000_000

            def _extract_from_hdf5(group, path="", depth=0, extracted_count=None):
                """Recursively extract arrays from HDF5 group (MATLAB v7.3) with memory limits"""
                if extracted_count is None:
                    extracted_count = {'count': 0}
                
                results = {}
                
                # Prevent infinite recursion and limit depth
                if depth > max_depth:
                    return results
                
                # Early termination if too many arrays extracted
                if extracted_count['count'] >= max_arrays:
                    return results
                    
                try:
                    # Iterate over items in the group
                    for key, item in group.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        if isinstance(item, h5py.Dataset):
                            # Check size before loading
                            if item.size > max_array_size:
                                continue
                                
                            # HDF5 in MATLAB is often transposed (column-major vs row-major)
                            # We need to handle object references and basic types
                            try:
                                # Check for object references first
                                if item.dtype.kind == 'O': 
                                    pass
                                else:
                                    # Numeric data
                                    val = item[()]
                                    # Transpose to match MATLAB behavior (if it's 2D+)
                                    if val.ndim >= 2:
                                        val = val.T
                                        
                                    # Handle 1x1 arrays as scalars or squeezed
                                    val = np.squeeze(val)
                                    
                                    # Only add if we have valid data
                                    if val.size > 0:
                                        results[current_path] = val
                                        extracted_count['count'] += 1
                            except:
                                pass
                        elif isinstance(item, h5py.Group):
                            # Recursive call
                            results.update(_extract_from_hdf5(item, current_path, depth+1, extracted_count))
                            
                            # Check limit after recursion
                            if extracted_count['count'] >= max_arrays:
                                return results
                except Exception:
                    pass
                    
                return results
            
            # Show warnings based on file size, but allow loading
            if file_size_mb > 100:
                st.warning(f"⚠️ Very large file detected ({file_size_mb:.1f} MB). Loading will take longer and use significant memory.")
                st.info("**Performance Tips:**\n"
                        "- Loading may take several minutes\n"
                        "- Memory-efficient mode is automatically enabled\n"
                        "- Consider exporting to CSV format for faster loading\n"
                        "- Close other applications to free up memory")
            elif file_size_mb > 30:
                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Loading may take longer and use significant memory.")
            
            with st.spinner(f"Parsing and extracting features... ({file_size_mb:.1f} MB)"):
                df = None
                name = up.name.lower()
                if name.endswith(".csv"):
                    df = pd.read_csv(up)
                elif name.endswith(".xlsx") or name.endswith(".xls"):
                    try:
                        # Extract metadata if available
                        metadata = {}
                        try:
                            from cleaning_module import extract_excel_metadata
                            up.seek(0)
                            metadata = extract_excel_metadata(up)
                            up.seek(0)
                        except: pass
                        
                        all_sheets = get_excel_sheets(up)
                        channel_sheets = filter_channel_sheets(all_sheets)
                        
                        if len(all_sheets) > 1:
                            st.info(f"Found {len(all_sheets)} sheet(s): {', '.join(all_sheets)}")
                            sheets_to_show = channel_sheets if channel_sheets else [s for s in all_sheets if s.lower() != 'info']
                            
                            selected_sheets = st.multiselect(
                                "Select sheets to analyze",
                                options=sheets_to_show,
                                default=sheets_to_show[:1],
                                help="Select which sheets to combine for analysis"
                            )
                            
                            if not selected_sheets:
                                st.warning("Please select at least one sheet.")
                                st.stop()
                                
                            up.seek(0)
                            sheets_data = read_excel_sheets(up, selected_sheets)
                            dfs = []
                            for s in selected_sheets:
                                if s in sheets_data:
                                    temp_df = sheets_data[s].copy()
                                    temp_df.insert(0, 'Sheet_Source', s)
                                    dfs.append(temp_df)
                            
                            if dfs:
                                df = pd.concat(dfs, ignore_index=True)
                            else:
                                st.error("No data found in selected sheets.")
                                st.stop()
                        else:
                            # Single sheet
                            up.seek(0)
                            df = pd.read_excel(up)
                            
                    except Exception as e:
                        st.error(f"Error loading Excel file: {e}")
                        st.stop()
                elif name.endswith(".mat"):
                    if not SCIPY_OK:
                        st.error("SciPy not available. Please upload a CSV for now.")
                    else:
                        # Try loading with different options
                        mat = None
                        try:
                            # First try with struct_as_record=True (default) to handle structures as dicts
                            mat = loadmat(io.BytesIO(up.getvalue()), struct_as_record=True, squeeze_me=True)
                        except MemoryError:
                            st.error(f"❌ Out of memory while loading MAT file ({file_size_mb:.1f} MB)")
                            st.info("**Solutions:**\n"
                                    "- Export your data to CSV format (more memory-efficient)\n"
                                    "- Reduce the file size by downsampling\n"
                                    "- Close other applications to free up memory\n"
                                    "- Use a machine with more RAM")
                            st.stop()
                        except Exception as e:
                            try:
                                # Fallback to False
                                mat = loadmat(io.BytesIO(up.getvalue()), struct_as_record=False, squeeze_me=True)
                            except MemoryError:
                                st.error(f"❌ Out of memory while loading MAT file ({file_size_mb:.1f} MB)")
                                st.info("**Solutions:**\n"
                                        "- Export your data to CSV format (more memory-efficient)\n"
                                        "- Reduce the file size by downsampling\n"
                                        "- Close other applications to free up memory\n"
                                        "- Use a machine with more RAM")
                                st.stop()
                        except Exception as e2:
                            # Try HDF5 (MATLAB v7.3)
                            try:
                                import h5py
                                f_bytes = io.BytesIO(up.getvalue())
                                mat = h5py.File(f_bytes, 'r')
                                # Extract data using HDF5 walker
                                st.info("ℹ️ Detected HDF5-based MAT file (v7.3). Using specialized loader.")
                                
                                # Simple wrapper to match expected structure
                                # We'll extract everything into a flat dictionary, similar to how we process structs
                                extracted_data = _extract_from_hdf5(mat)
                                all_arrays = extracted_data
                                mat_keys = list(extracted_data.keys())
                                
                                # Create a pseudo-mat dictionary for compatibility with downstream logic
                                # This is a bit hacky but avoids rewriting the entire downstream processing
                                mat = extracted_data
                                
                            except ImportError:
                                st.error("❌ HDF5 support requires 'h5py'. Please install it.")
                                st.stop()
                            except Exception as e3:
                                st.error(f"❌ Error loading .mat file: {str(e2)}\n\nAlso failed HDF5 load: {str(e3)}")
                                st.info("**Troubleshooting:**\n"
                                        "- Verify the file is a valid MATLAB .mat file\n"
                                        "- Try exporting to CSV format instead\n"
                                        "- Check if the file is corrupted")
                                st.stop()
                        
                        
                        
                        # Helper function to extract data from MATLAB structures
                        # Adjust limits based on file size - more aggressive for very large files
                        if file_size_mb > 100:
                            max_depth = 4  # Very conservative for huge files
                            max_arrays = 300  # Strict limit
                            max_array_size = 5_000_000  # Skip arrays with more than 5M elements
                        elif file_size_mb > 30:
                            max_depth = 6  # Reduce depth for large files
                            max_arrays = 500  # Limit total arrays extracted
                            max_array_size = 10_000_000  # Skip arrays with more than 10M elements
                        else:
                            max_depth = 8  # Normal depth
                            max_arrays = 1000  # More arrays allowed
                            max_array_size = 10_000_000  # Skip arrays with more than 10M elements
                        
                        def extract_from_struct(obj, path="", depth=0, extracted_count=None):
                            """Recursively extract arrays from MATLAB structures with memory limits"""
                            if extracted_count is None:
                                extracted_count = {'count': 0}
                            
                            results = {}
                            
                            # Prevent infinite recursion and limit depth for large files
                            if depth > max_depth:
                                return results
                            
                            # Early termination if too many arrays extracted
                            if extracted_count['count'] >= max_arrays:
                                return results
                            
                            try:
                                if isinstance(obj, np.ndarray):
                                    # Skip extremely large arrays to prevent memory issues
                                    if obj.size > max_array_size:
                                        return results
                                    
                                    if obj.dtype.names:  # Structured array (like MatlabOpaque)
                                        # First, try to extract from known MATLAB opaque fields
                                        for fname in ['arr', 's0', 's1', 's2']:
                                            if fname in obj.dtype.names:
                                                field_data = obj[fname]
                                                if isinstance(field_data, np.ndarray):
                                                    if field_data.dtype == object and field_data.size > 0:
                                                        # Object array - drill into each element (limit iterations)
                                                        for idx, item in enumerate(field_data.flat):
                                                            if item is not None and idx < 100:  # Limit iterations
                                                                results.update(extract_from_struct(item, f"{path}.{fname}_{idx}" if path else f"{fname}_{idx}", depth+1, extracted_count))
                                                    elif field_data.size > 1 and field_data.dtype != object and field_data.size <= max_array_size:
                                                        # Numeric array
                                                        results[f"{path}.{fname}" if path else fname] = np.squeeze(field_data)
                                                        extracted_count['count'] += 1
                                        
                                        # Also extract all other fields recursively
                                        for name in obj.dtype.names:
                                            if name not in ['arr', 's0', 's1', 's2']:  # Skip already processed
                                                field_data = obj[name]
                                                if isinstance(field_data, np.ndarray):
                                                    if field_data.dtype == object and field_data.size > 0:
                                                        for idx, item in enumerate(field_data.flat):
                                                            if item is not None and idx < 100:  # Limit iterations
                                                                results.update(extract_from_struct(item, f"{path}.{name}_{idx}" if path else f"{name}_{idx}", depth+1, extracted_count))
                                                    elif field_data.size > 1 and field_data.dtype != object and field_data.size <= max_array_size:
                                                        results[f"{path}.{name}" if path else name] = np.squeeze(field_data)
                                                        extracted_count['count'] += 1
                                    elif obj.dtype == object and obj.size > 0:
                                        # Object array - extract each element (limit iterations)
                                        for idx, item in enumerate(obj.flat):
                                            if item is not None and idx < 100:  # Limit iterations
                                                results.update(extract_from_struct(item, f"{path}_{idx}" if path else f"item_{idx}", depth+1, extracted_count))
                                    elif obj.size > 1 and obj.dtype != object and obj.size <= max_array_size:
                                        # Regular numeric array
                                        if path:
                                            results[path] = np.squeeze(obj)
                                            extracted_count['count'] += 1
                                elif isinstance(obj, dict):
                                    for k, v in obj.items():
                                        if not k.startswith('__'):
                                            new_path = f"{path}.{k}" if path else k
                                            results.update(extract_from_struct(v, new_path, depth+1, extracted_count))
                                elif hasattr(obj, '_fieldnames'):  # MATLAB struct object
                                    for field in obj._fieldnames:
                                        field_data = getattr(obj, field, None)
                                        if field_data is not None:
                                            new_path = f"{path}.{field}" if path else field
                                            results.update(extract_from_struct(field_data, new_path, depth+1, extracted_count))
                                elif isinstance(obj, (list, tuple)):
                                    for idx, item in enumerate(obj):
                                        if item is not None and idx < 100:  # Limit iterations
                                            results.update(extract_from_struct(item, f"{path}_{idx}" if path else f"item_{idx}", depth+1, extracted_count))
                            except Exception as e:
                                pass
                            return results
                        

                        
                        
                        # Get all keys (including metadata for debugging)
                        all_keys = list(mat.keys())
                        non_meta_keys = [k for k in all_keys if not k.startswith('__')]
                        
                        # Extract all data from structures, including MatlabOpaque
                        extracted_data = {}
                        extraction_counter = {'count': 0}
                        
                        for key in non_meta_keys:
                            if key in mat:
                                extracted = extract_from_struct(mat[key], key, extracted_count=extraction_counter)
                                # Fix keys that start with "None" - replace with actual key name
                                fixed_extracted = {}
                                for ek, ev in extracted.items():
                                    if ek.startswith("None"):
                                        # Replace "None" with the actual key
                                        new_key = ek.replace("None", key, 1)
                                        fixed_extracted[new_key] = ev
                                    else:
                                        fixed_extracted[ek] = ev
                                extracted_data.update(fixed_extracted)
                        
                        
                        # Show extraction progress
                        if extraction_counter['count'] > 0:
                            st.info(f"✓ Extracted {extraction_counter['count']} arrays from MAT file structure")
                            if extraction_counter['count'] >= max_arrays:
                                st.warning(f"⚠️ Extraction limited to {max_arrays} arrays due to file size. Some data may not be loaded.")
                        
                        # Iterate over a static snapshot to avoid modifying the dict during iteration
                        for key, value in list(extracted_data.items()):
                            if isinstance(value, np.ndarray) and value.size > 1:
                                # Store in a simplified key format (only add if new)
                                simple_key = key.split('.')[-1] if '.' in key else key
                                if simple_key not in extracted_data:
                                    extracted_data[simple_key] = value
                        
                        # Also try to extract from any top-level opaque objects
                        for key, value in mat.items():
                            if not key.startswith('__'):
                                # Check if it's a MatlabOpaque-like structure with s0, s1, s2, arr fields
                                if isinstance(value, np.ndarray) and value.dtype.names:
                                    opaque_fields = ['arr', 's0', 's1', 's2']
                                    for opaque_field in opaque_fields:
                                        if opaque_field in value.dtype.names:
                                            field_data = value[opaque_field]
                                            if isinstance(field_data, np.ndarray):
                                                if field_data.dtype == object and field_data.size > 0:
                                                    # It's an object array - drill into each element
                                                    for idx, item in enumerate(field_data.flat):
                                                        if item is not None:
                                                            item_extracted = extract_from_struct(item, f"{key}_{opaque_field}_{idx}")
                                                            extracted_data.update(item_extracted)
                                                elif field_data.size > 1 and field_data.dtype != object:
                                                    # Numeric array - store it
                                                    extracted_data[f"{key}_{opaque_field}"] = np.squeeze(field_data)
                        
                        # Add extracted data to mat dictionary for easier access
                        if extracted_data:
                            mat.update(extracted_data)
                        
                        # Collect ALL arrays from the .mat file for comprehensive DataFrame
                        all_arrays = {}
                        
                        # Start with extracted_data (from MatlabOpaque structures)
                        if extracted_data:
                            for key, value in extracted_data.items():
                                if isinstance(value, np.ndarray) and value.size > 1:
                                    all_arrays[key] = np.squeeze(value)
                                # Also handle keys that contain "arr" - these might be the data
                                if "arr" in key.lower() and isinstance(value, np.ndarray):
                                    # Try to extract nested arrays from object arrays
                                    if value.dtype == object and value.size > 0:
                                        for idx, item in enumerate(value.flat):
                                            if isinstance(item, np.ndarray) and item.size > 1:
                                                all_arrays[f"{key}_item_{idx}"] = np.squeeze(item)
                        
                        # Also extract from mat dictionary
                        for key, value in mat.items():
                            if not key.startswith('__'):
                                try:
                                    if isinstance(value, np.ndarray):
                                        if value.dtype.names:  # Structured array
                                            extracted = extract_from_struct(value, key)
                                            all_arrays.update(extracted)
                                        elif value.size > 1:
                                            all_arrays[key] = np.squeeze(value)
                                    elif isinstance(value, (list, tuple)):
                                        for i, item in enumerate(value):
                                            if isinstance(item, np.ndarray) and item.size > 1:
                                                all_arrays[f"{key}_{i}"] = np.squeeze(item)
                                except:
                                    pass
                        
                        # Show available keys for debugging
                        available_keys = list(all_arrays.keys())
                        # Store in session state for access in display section
                        st.session_state.mat_all_arrays = all_arrays
                        st.session_state.mat_available_keys = available_keys

                        # Try to reconstruct DataFrame from .mat
                        # First, check for cell11dataset_for_python structure
                        df = None
                        cycles_data = None
                        
                        # Check if this is a cell11dataset_for_python structure
                        # Top-level variables: policy, policy_readable, cycle_life, cycles, summary, Vdlin
                        has_summary = 'summary' in mat or 'summary' in all_arrays
                        has_cycles = 'cycles' in mat or 'cycles' in all_arrays
                        
                        if has_summary and has_cycles:
                            try:
                                # Extract summary struct - check multiple possible locations
                                summary_data = None
                                # Try direct access first
                                if 'summary' in mat:
                                    summary_data = mat['summary']
                                elif 'Summary' in mat:
                                    summary_data = mat['Summary']
                                
                                # If not found, try from all_arrays
                                if summary_data is None:
                                    # First try direct keys
                                    for key in ['summary', 'Summary']:
                                        if key in all_arrays:
                                            summary_data = all_arrays[key]
                                            break
                                    
                                    # If still not found, try to reconstruct from extracted fields (summary.*)
                                    if summary_data is None:
                                        summary_data = {}
                                        summary_field_names = ['cycle', 'QDischarge', 'QCharge', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']
                                        found_fields = False
                                        
                                        # Look for summary.fieldname or Summary.fieldname patterns
                                        for arr_key in all_arrays.keys():
                                            # Check for summary.fieldname pattern
                                            if arr_key.startswith('summary.') or arr_key.startswith('Summary.'):
                                                field_name = arr_key.split('.', 1)[1]
                                                # Also check simplified name (without summary prefix)
                                                if field_name in summary_field_names:
                                                    summary_data[field_name] = all_arrays[arr_key]
                                                    found_fields = True
                                            # Also check for direct field names (might have been extracted without prefix)
                                            elif arr_key in summary_field_names:
                                                summary_data[arr_key] = all_arrays[arr_key]
                                                found_fields = True
                                        
                                        # If we found individual fields, use them; otherwise set to None
                                        if not found_fields or len(summary_data) == 0:
                                            summary_data = None
                                        
                                        # Additional fallback: try to find summary fields that might have been extracted differently
                                        if summary_data is None or (isinstance(summary_data, dict) and len(summary_data) < 3):
                                            # Look for any arrays that match summary field names (case-insensitive)
                                            potential_summary = {}
                                            expected_len_fallback = None
                                            for arr_key, arr_value in all_arrays.items():
                                                arr_key_lower = arr_key.lower()
                                                # Check if key contains summary field names
                                                if isinstance(arr_value, np.ndarray):
                                                    arr_clean = np.squeeze(arr_value)
                                                    if arr_clean.ndim == 1 and len(arr_clean) > 0:
                                                        # Check for cycle first to set expected length
                                                        if 'cycle' in arr_key_lower and 'cycle' not in potential_summary:
                                                            if len(arr_clean) == 1077 or expected_len_fallback is None:  # Expected length or first field
                                                                potential_summary['cycle'] = arr_clean
                                                                expected_len_fallback = len(arr_clean)
                                                        # Check for QDischarge
                                                        elif ('qdischarge' in arr_key_lower or ('discharge' in arr_key_lower and 'q' in arr_key_lower)) and 'cycle' not in arr_key_lower:
                                                            if 'QDischarge' not in potential_summary and (expected_len_fallback is None or len(arr_clean) == expected_len_fallback):
                                                                potential_summary['QDischarge'] = arr_clean
                                                                if expected_len_fallback is None:
                                                                    expected_len_fallback = len(arr_clean)
                                                        # Check for IR (make sure it's not part of another word like "circuit")
                                                        elif (arr_key_lower == 'ir' or arr_key_lower.endswith('.ir') or arr_key_lower.endswith('_ir')) and expected_len_fallback is not None:
                                                            if 'IR' not in potential_summary and len(arr_clean) == expected_len_fallback:
                                                                potential_summary['IR'] = arr_clean
                                                        # Check for Tmax
                                                        elif ('tmax' in arr_key_lower or arr_key_lower.endswith('.tmax')) and expected_len_fallback is not None:
                                                            if 'Tmax' not in potential_summary and len(arr_clean) == expected_len_fallback:
                                                                potential_summary['Tmax'] = arr_clean
                                                        # Check for Tavg
                                                        elif ('tavg' in arr_key_lower or arr_key_lower.endswith('.tavg')) and expected_len_fallback is not None:
                                                            if 'Tavg' not in potential_summary and len(arr_clean) == expected_len_fallback:
                                                                potential_summary['Tavg'] = arr_clean
                                                        # Check for Tmin
                                                        elif ('tmin' in arr_key_lower or arr_key_lower.endswith('.tmin')) and expected_len_fallback is not None:
                                                            if 'Tmin' not in potential_summary and len(arr_clean) == expected_len_fallback:
                                                                potential_summary['Tmin'] = arr_clean
                                            
                                            # If we found enough fields, use them
                                            if len(potential_summary) >= 2 and 'cycle' in potential_summary:
                                                summary_data = potential_summary
                                
                                # Extract cycles array
                                cycles_data_raw = None
                                if 'cycles' in mat:
                                    cycles_data_raw = mat['cycles']
                                elif 'Cycles' in mat:
                                    cycles_data_raw = mat['Cycles']
                                if cycles_data_raw is None:
                                    for key in ['cycles', 'Cycles']:
                                        if key in all_arrays:
                                            cycles_data_raw = all_arrays[key]
                                            break
                                
                                # Process summary struct if found
                                if summary_data is not None:
                                    summary_df = None
                                    
                                    # Collect all fields with their data
                                    field_data_dict = {}
                                    expected_length = None
                                    
                                    # Handle different MATLAB struct formats
                                    if isinstance(summary_data, dict) and not any(k.startswith('_') for k in summary_data.keys()):
                                        # Dict format (struct_as_record=True) or reconstructed from all_arrays
                                        # Check if it has the required fields
                                        if 'cycle' in summary_data or 'QDischarge' in summary_data:
                                            for field, value in summary_data.items():
                                                if field.startswith('_'):
                                                    continue
                                                # Handle numpy arrays
                                                if isinstance(value, np.ndarray):
                                                    value = np.squeeze(value)
                                                    if value.ndim > 1:
                                                        value = value.flatten()
                                                    # Only keep 1D arrays with valid length
                                                    if value.ndim == 1 and len(value) > 0:
                                                        # Set expected length from cycle field first
                                                        if expected_length is None and field == 'cycle':
                                                            expected_length = len(value)
                                                        # Only add if length matches expected (or set expected if first)
                                                        if expected_length is None or len(value) == expected_length:
                                                            if expected_length is None:
                                                                expected_length = len(value)
                                                            field_data_dict[field] = value
                                    elif isinstance(summary_data, np.ndarray):
                                        if summary_data.dtype.names:
                                            # Structured array format - could be 1x1 or Nx1
                                            summary_fields = summary_data.dtype.names
                                            if 'cycle' in summary_fields or 'QDischarge' in summary_fields:
                                                # Handle 1x1 structured array (common MATLAB format)
                                                if summary_data.size == 1:
                                                    # Extract from single element
                                                    for field in summary_fields:
                                                        field_data = summary_data[field].flat[0]
                                                        if isinstance(field_data, np.ndarray):
                                                            field_data = np.squeeze(field_data)
                                                            if field_data.ndim > 1:
                                                                field_data = field_data.flatten()
                                                            if field_data.ndim == 1 and len(field_data) > 0:
                                                                if expected_length is None and field == 'cycle':
                                                                    expected_length = len(field_data)
                                                                if expected_length is None or len(field_data) == expected_length:
                                                                    if expected_length is None:
                                                                        expected_length = len(field_data)
                                                                    field_data_dict[field] = field_data
                                                else:
                                                    # Multiple elements - extract first or combine
                                                    for field in summary_fields:
                                                        field_data = summary_data[field]
                                                        if isinstance(field_data, np.ndarray):
                                                            field_data = np.squeeze(field_data)
                                                            if field_data.ndim > 1:
                                                                field_data = field_data.flatten()
                                                            if field_data.ndim == 1 and len(field_data) > 0:
                                                                if expected_length is None and field == 'cycle':
                                                                    expected_length = len(field_data)
                                                                if expected_length is None or len(field_data) == expected_length:
                                                                    if expected_length is None:
                                                                        expected_length = len(field_data)
                                                                    field_data_dict[field] = field_data
                                    elif hasattr(summary_data, '_fieldnames'):
                                        # MATLAB struct object format
                                        for field in summary_data._fieldnames:
                                            value = getattr(summary_data, field, None)
                                            if value is not None and isinstance(value, np.ndarray):
                                                value = np.squeeze(value)
                                                if value.ndim > 1:
                                                    value = value.flatten()
                                                if value.ndim == 1 and len(value) > 0:
                                                    if expected_length is None and field == 'cycle':
                                                        expected_length = len(value)
                                                    if expected_length is None or len(value) == expected_length:
                                                        if expected_length is None:
                                                            expected_length = len(value)
                                                        field_data_dict[field] = value
                                    
                                    # Create DataFrame only if we have valid data with matching lengths
                                    if field_data_dict and expected_length is not None:
                                        # Verify all fields have the same length
                                        valid_fields = {}
                                        for field, data in field_data_dict.items():
                                            if len(data) == expected_length:
                                                valid_fields[field] = data
                                        
                                        if valid_fields and 'cycle' in valid_fields:
                                            # Debug: check what fields we found
                                            found_fields_list = list(valid_fields.keys())
                                            
                                            summary_df = pd.DataFrame(valid_fields)
                                            
                                            # Standardize column names
                                            column_mapping = {
                                                'QDischarge': 'discharge_capacity',
                                                'QCharge': 'charge_capacity',
                                                'IR': 'internal_resistance',
                                                'Tmax': 'temperature_max',
                                                'Tavg': 'temperature_avg',
                                                'Tmin': 'temperature_min',
                                                'chargetime': 'charge_time'
                                            }
                                            # Only rename columns that exist
                                            existing_mapping = {k: v for k, v in column_mapping.items() if k in summary_df.columns}
                                            summary_df = summary_df.rename(columns=existing_mapping)
                                            
                                            # Create a combined temperature column
                                            # Priority: Tavg > Tmax > Tmin
                                            if 'temperature_avg' in summary_df.columns:
                                                summary_df['temperature'] = summary_df['temperature_avg'].copy()
                                            elif 'temperature_max' in summary_df.columns:
                                                summary_df['temperature'] = summary_df['temperature_max'].copy()
                                            elif 'temperature_min' in summary_df.columns:
                                                summary_df['temperature'] = summary_df['temperature_min'].copy()
                                            
                                            # Ensure internal_resistance column exists (might be NaN if IR field had NaN values)
                                            if 'internal_resistance' not in summary_df.columns and 'IR' in found_fields_list:
                                                # This shouldn't happen, but just in case
                                                summary_df['internal_resistance'] = valid_fields['IR']
                                            
                                            # Ensure temperature column exists
                                            if 'temperature' not in summary_df.columns:
                                                # Create empty column if none of the temperature fields were found
                                                summary_df['temperature'] = np.nan
                                            
                                            df = summary_df
                                
                                # Process cycles array for time-series data (V, I, T, t, Qdlin)
                                if cycles_data_raw is not None:
                                    cycles_list = []
                                    
                                    # Handle different formats
                                    if isinstance(cycles_data_raw, np.ndarray):
                                        if cycles_data_raw.dtype == object:
                                            # Object array - each element is a struct
                                            for cycle_idx, cycle_struct in enumerate(cycles_data_raw.flat):
                                                if cycle_struct is not None:
                                                    cycle_dict = {}
                                                    
                                                    # Extract fields from struct
                                                    if isinstance(cycle_struct, np.ndarray) and cycle_struct.dtype.names:
                                                        # Structured array
                                                        for field in cycle_struct.dtype.names:
                                                            field_data = cycle_struct[field]
                                                            if isinstance(field_data, np.ndarray):
                                                                field_data = np.squeeze(field_data)
                                                                cycle_dict[field] = field_data
                                                    elif isinstance(cycle_struct, dict):
                                                        # Dict format
                                                        for field, value in cycle_struct.items():
                                                            if not field.startswith('_') and isinstance(value, np.ndarray):
                                                                cycle_dict[field] = np.squeeze(value)
                                                    elif hasattr(cycle_struct, '_fieldnames'):
                                                        # MATLAB struct object
                                                        for field in cycle_struct._fieldnames:
                                                            value = getattr(cycle_struct, field, None)
                                                            if value is not None and isinstance(value, np.ndarray):
                                                                cycle_dict[field] = np.squeeze(value)
                                                    
                                                    # Create DataFrame for this cycle
                                                    if cycle_dict:
                                                        cycle_df = pd.DataFrame(cycle_dict)
                                                        cycle_df['cycle_number'] = cycle_idx
                                                        cycles_list.append(cycle_df)
                                        elif cycles_data_raw.dtype.names:
                                            # Single structured array with multiple records
                                            for cycle_idx in range(len(cycles_data_raw)):
                                                cycle_dict = {}
                                                for field in cycles_data_raw.dtype.names:
                                                    field_data = cycles_data_raw[field][cycle_idx]
                                                    if isinstance(field_data, np.ndarray):
                                                        cycle_dict[field] = np.squeeze(field_data)
                                                if cycle_dict:
                                                    cycle_df = pd.DataFrame(cycle_dict)
                                                    cycle_df['cycle_number'] = cycle_idx
                                                    cycles_list.append(cycle_df)
                                    
                                    if cycles_list:
                                        # Combine all cycles into one DataFrame
                                        cycles_data = pd.concat(cycles_list, ignore_index=True)
                                        # Standardize column names
                                        column_mapping_cycles = {
                                            'V': 'voltage',
                                            'I': 'current',
                                            'T': 'temperature',
                                            't': 'time',
                                            'Qdlin': 'Qdlin'
                                        }
                                        cycles_data = cycles_data.rename(columns=column_mapping_cycles)
                                        # Store cycles data for later use (dQ/dV plots, curve plots)
                                        st.session_state.cycles_time_series = cycles_data
                                
                                # Extract other top-level variables if needed
                                if 'cycle_life' in mat:
                                    st.session_state.cycle_life = np.squeeze(mat['cycle_life'])
                                if 'policy' in mat:
                                    st.session_state.policy = mat.get('policy', None)
                                if 'policy_readable' in mat:
                                    st.session_state.policy_readable = mat.get('policy_readable', None)
                                if 'Vdlin' in mat:
                                    st.session_state.Vdlin = np.squeeze(mat['Vdlin'])
                                
                            except Exception as e:
                                # If processing fails, silently fall through to generic parser
                                # (Error message suppressed per user request)
                                pass
                        
                        # If cell11dataset_for_python format was successfully parsed, skip generic parser
                        if df is not None:
                            cycle_data_mat = df
                        else:
                            # Try to reconstruct DataFrame from .mat
                            # Look for cycle-level data first
                            cycle_data_mat = None
                        
                        # Check for direct cycle column
                        if 'cycle' in mat or 'cycle_number' in mat:
                            cycle_col = 'cycle' if 'cycle' in mat else 'cycle_number'
                            cycle_data_mat = pd.DataFrame()
                            cycle_data_mat['cycle'] = np.squeeze(mat[cycle_col])
                            for key in ['discharge_capacity', 'capacity', 'discharge_cap']:
                                if key in mat:
                                    cycle_data_mat['discharge_capacity'] = np.squeeze(mat[key])
                                    break
                            for key in ['internal_resistance', 'ir', 'resistance']:
                                if key in mat:
                                    cycle_data_mat['internal_resistance'] = np.squeeze(mat[key])
                                    break
                            for key in ['temperature', 'temp']:
                                if key in mat:
                                    cycle_data_mat['temperature'] = np.squeeze(mat[key])
                                    break
                            for key in ['voltage', 'v', 'volt']:
                                if key in mat:
                                    cycle_data_mat['voltage'] = np.squeeze(mat[key])
                                    break
                            for key in ['current', 'i', 'curr']:
                                if key in mat:
                                    cycle_data_mat['current'] = np.squeeze(mat[key])
                                    break
                            
                            # Add ALL other valid arrays to cycle_data_mat
                            for key, arr in all_arrays.items():
                                try:
                                    if not isinstance(arr, np.ndarray):
                                        continue
                                    arr_clean = np.squeeze(arr)
                                    # Skip if already in DataFrame or not a valid 1D array
                                    if key in cycle_data_mat.columns or arr_clean.ndim != 1:
                                        continue
                                    # Only add if length matches the cycle column
                                    if len(arr_clean) == len(cycle_data_mat):
                                        # Clean column name (remove dots and special chars)
                                        clean_key = key.split('.')[-1].replace(' ', '_').replace('-', '_')
                                        if clean_key not in cycle_data_mat.columns:
                                            cycle_data_mat[clean_key] = arr_clean
                                except:
                                    pass
                            
                            # If we have cycle-level data, use it
                            if len(cycle_data_mat) > 0:
                                df = cycle_data_mat
                        
                        # If no direct cycle column found, try to extract from array names (e.g., cycles.discharge_capacity_0, cycles.discharge_capacity_1)
                        if df is None:
                            import re
                            # Look for patterns like cycles.*_N or cycles.*[N] where N is cycle number
                            cycle_arrays = {}
                            capacity_arrays = {}
                            ir_arrays = {}
                            temp_arrays = {}
                            voltage_arrays = {}
                            current_arrays = {}
                            time_arrays = {}
                            
                            for key, arr in all_arrays.items():
                                try:
                                    if not isinstance(arr, np.ndarray):
                                        continue
                                    arr_clean = np.squeeze(arr)
                                    if arr_clean.ndim != 1 or arr_clean.size < 2:
                                        continue
                                    
                                    key_lower = key.lower()
                                    
                                    # Extract cycle number from key (look for _N or [N] at end)
                                    cycle_match = re.search(r'_(\d+)$|\[(\d+)\]$', key)
                                    if cycle_match:
                                        cycle_num = int(cycle_match.group(1) or cycle_match.group(2))
                                        
                                        # Categorize arrays by type
                                        if 'cycle' in key_lower and ('number' in key_lower or 'num' in key_lower or 'idx' in key_lower):
                                            cycle_arrays[cycle_num] = arr_clean
                                        elif 'capacity' in key_lower or ('discharge' in key_lower and 'q' in key_lower):
                                            capacity_arrays[cycle_num] = arr_clean
                                        elif 'resistance' in key_lower or key_lower.endswith('_ir') or key_lower == 'ir':
                                            ir_arrays[cycle_num] = arr_clean
                                        elif 'temperature' in key_lower or 'temp' in key_lower:
                                            temp_arrays[cycle_num] = arr_clean
                                        elif 'voltage' in key_lower or key_lower.startswith('v_') or key_lower.endswith('_v'):
                                            voltage_arrays[cycle_num] = arr_clean
                                        elif 'current' in key_lower or key_lower.startswith('i_') or key_lower.endswith('_i'):
                                            current_arrays[cycle_num] = arr_clean
                                        elif 'time' in key_lower or key_lower.startswith('t_'):
                                            time_arrays[cycle_num] = arr_clean
                                except:
                                    pass
                            
                            # If we found cycle-indexed arrays, try to build cycle-level DataFrame
                            all_cycle_nums = set()
                            if cycle_arrays:
                                all_cycle_nums.update(cycle_arrays.keys())
                            if capacity_arrays:
                                all_cycle_nums.update(capacity_arrays.keys())
                            if ir_arrays:
                                all_cycle_nums.update(ir_arrays.keys())
                            if temp_arrays:
                                all_cycle_nums.update(temp_arrays.keys())
                            if voltage_arrays:
                                all_cycle_nums.update(voltage_arrays.keys())
                            if current_arrays:
                                all_cycle_nums.update(current_arrays.keys())
                            
                            # Also look for summary arrays (not indexed by cycle)
                            # These might be per-cycle summary metrics
                            summary_capacity = None
                            summary_cycles = None
                            summary_ir = None
                            summary_temp = None
                            summary_voltage = None
                            summary_current = None
                            
                            for key, arr in all_arrays.items():
                                try:
                                    key_lower = key.lower()
                                    arr_clean = np.squeeze(arr)
                                    if not isinstance(arr_clean, np.ndarray) or arr_clean.ndim != 1:
                                        continue
                                    
                                    # Look for summary/aggregate arrays (no cycle index in name)
                                    if re.search(r'_\d+$|\[\d+\]$', key):
                                        continue  # Skip cycle-indexed arrays here
                                    
                                    if 'cycle' in key_lower and ('number' in key_lower or 'num' in key_lower or 'idx' in key_lower) and len(arr_clean) > 1:
                                        summary_cycles = arr_clean
                                    elif ('discharge' in key_lower or 'capacity' in key_lower) and len(arr_clean) > 1 and np.nanmax(arr_clean) < 1000:
                                        summary_capacity = arr_clean
                                    elif ('resistance' in key_lower or key_lower.endswith('_ir')) and len(arr_clean) > 1:
                                        summary_ir = arr_clean
                                    elif ('temperature' in key_lower or 'temp' in key_lower) and len(arr_clean) > 1:
                                        summary_temp = arr_clean
                                    elif ('voltage' in key_lower or key_lower.startswith('v_') or key_lower.endswith('_v')) and len(arr_clean) > 1:
                                        # Check if it's a reasonable voltage range (0.5-6.5V)
                                        if np.nanmin(arr_clean) >= 0.5 and np.nanmax(arr_clean) <= 6.5:
                                            summary_voltage = arr_clean
                                    elif ('current' in key_lower or key_lower.startswith('i_') or key_lower.endswith('_i')) and len(arr_clean) > 1:
                                        summary_current = arr_clean
                                except:
                                    pass
                            
                            # Build cycle-level DataFrame from summary arrays if they match in length
                            if summary_cycles is not None and summary_capacity is not None:
                                if len(summary_cycles) == len(summary_capacity):
                                    cycle_data_mat = pd.DataFrame()
                                    cycle_data_mat['cycle'] = summary_cycles
                                    cycle_data_mat['discharge_capacity'] = summary_capacity
                                    
                                    # Initialize with NaN, then fill if data exists
                                    cycle_data_mat['internal_resistance'] = np.nan
                                    cycle_data_mat['temperature'] = np.nan
                                    cycle_data_mat['voltage'] = np.nan
                                    cycle_data_mat['current'] = np.nan
                                    
                                    if summary_ir is not None and len(summary_ir) == len(summary_cycles):
                                        cycle_data_mat['internal_resistance'] = summary_ir
                                    if summary_temp is not None and len(summary_temp) == len(summary_cycles):
                                        cycle_data_mat['temperature'] = summary_temp
                                    if summary_voltage is not None and len(summary_voltage) == len(summary_cycles):
                                        cycle_data_mat['voltage'] = summary_voltage
                                    if summary_current is not None and len(summary_current) == len(summary_cycles):
                                        cycle_data_mat['current'] = summary_current
                                    
                                    df = cycle_data_mat
                            
                            # If we still don't have cycle data, try extracting from cycles.* structure
                            if df is None and all_cycle_nums:
                                # Sort cycles and extract per-cycle summary (e.g., mean capacity per cycle)
                                sorted_cycles = sorted(all_cycle_nums)
                                cycle_summaries = {}
                                
                                for cycle_num in sorted_cycles:
                                    cycle_summaries[cycle_num] = {}
                                    if cycle_num in capacity_arrays:
                                        cap_arr = capacity_arrays[cycle_num]
                                        # Use mean or last value as cycle summary
                                        cycle_summaries[cycle_num]['capacity'] = np.nanmean(cap_arr) if np.any(np.isfinite(cap_arr)) else np.nan
                                    if cycle_num in ir_arrays:
                                        ir_arr = ir_arrays[cycle_num]
                                        cycle_summaries[cycle_num]['ir'] = np.nanmean(ir_arr) if np.any(np.isfinite(ir_arr)) else np.nan
                                    if cycle_num in temp_arrays:
                                        temp_arr = temp_arrays[cycle_num]
                                        cycle_summaries[cycle_num]['temp'] = np.nanmean(temp_arr) if np.any(np.isfinite(temp_arr)) else np.nan
                                    if cycle_num in voltage_arrays:
                                        volt_arr = voltage_arrays[cycle_num]
                                        cycle_summaries[cycle_num]['voltage'] = np.nanmean(volt_arr) if np.any(np.isfinite(volt_arr)) else np.nan
                                    if cycle_num in current_arrays:
                                        curr_arr = current_arrays[cycle_num]
                                        cycle_summaries[cycle_num]['current'] = np.nanmean(curr_arr) if np.any(np.isfinite(curr_arr)) else np.nan
                                
                                # Build DataFrame from cycle summaries
                                if cycle_summaries and any('capacity' in s for s in cycle_summaries.values()):
                                    cycle_data_mat = pd.DataFrame({
                                        'cycle': sorted_cycles,
                                        'discharge_capacity': [cycle_summaries.get(c, {}).get('capacity', np.nan) for c in sorted_cycles],
                                        'internal_resistance': [cycle_summaries.get(c, {}).get('ir', np.nan) for c in sorted_cycles],
                                        'temperature': [cycle_summaries.get(c, {}).get('temp', np.nan) for c in sorted_cycles],
                                        'voltage': [cycle_summaries.get(c, {}).get('voltage', np.nan) for c in sorted_cycles],
                                        'current': [cycle_summaries.get(c, {}).get('current', np.nan) for c in sorted_cycles]
                                    })
                                    # Ensure internal_resistance, temperature, voltage, and current columns exist even if all NaN
                                    if 'internal_resistance' not in cycle_data_mat.columns:
                                        cycle_data_mat['internal_resistance'] = np.nan
                                    if 'temperature' not in cycle_data_mat.columns:
                                        cycle_data_mat['temperature'] = np.nan
                                    if 'voltage' not in cycle_data_mat.columns:
                                        cycle_data_mat['voltage'] = np.nan
                                    if 'current' not in cycle_data_mat.columns:
                                        cycle_data_mat['current'] = np.nan
                                    df = cycle_data_mat
                        
                        # Fallback to voltage-capacity format
                        if df is None:
                            V = None
                            Q = None

                            # Priority 1: Try extracted_data first (from MatlabOpaque structures)
                            if V is None or Q is None:
                                for key, arr in extracted_data.items():
                                    try:
                                        if not isinstance(arr, np.ndarray) or arr.size < 2:
                                            continue
                                        arr_clean = np.squeeze(arr)
                                        if arr_clean.size < 2 or not np.isfinite(arr_clean).any():
                                            continue
                                        min_val = np.nanmin(arr_clean)
                                        max_val = np.nanmax(arr_clean)
                                        lk = str(key).lower()

                                        # Voltage: 0.5-6.5V range
                                        if V is None and 0.5 <= min_val and max_val <= 6.5 and (max_val - min_val) > 0.1:
                                            V = arr_clean
                                        # Capacity: non-negative, has some range
                                        elif Q is None and min_val >= 0 and max_val > 0.01:
                                            Q = arr_clean
                                    except Exception:
                                        continue

                            # Priority 2: Look at all_arrays with keyword matching
                            for key, arr in all_arrays.items():
                                try:
                                    if not isinstance(arr, np.ndarray) or arr.size < 2:
                                        continue
                                    arr_clean = np.squeeze(arr)
                                    if arr_clean.size < 2:
                                        continue
                                    if not np.isfinite(arr_clean).any():
                                        continue

                                    lk = str(key).lower()
                                    min_val = np.nanmin(arr_clean)
                                    max_val = np.nanmax(arr_clean)

                                    # Voltage keyword matching
                                    if V is None and any(vk in lk for vk in ["volt", "v_", "_v", "voltage", "v("]):
                                        if 0.5 <= min_val and max_val <= 6.5:
                                            V = arr_clean
                                            continue

                                    # Capacity keyword matching
                                    if Q is None and any(ck in lk for ck in ["cap", "q_", "_q", "capacity", "mah", "ah", "discharge"]):
                                        if min_val >= 0 and max_val > 0.01:
                                            Q = arr_clean
                                            continue
                                except Exception:
                                    continue

                            # Priority 3: Brute-force heuristic - just use the two most different numeric ranges
                            if V is None or Q is None:
                                candidates = []
                                for key, arr in list(all_arrays.items()) + list(extracted_data.items()):
                                    try:
                                        if not isinstance(arr, np.ndarray) or arr.size < 2:
                                            continue
                                        arr_clean = np.squeeze(arr)
                                        if arr_clean.size < 2 or not np.isfinite(arr_clean).any():
                                            continue
                                        min_val = np.nanmin(arr_clean)
                                        max_val = np.nanmax(arr_clean)
                                        if max_val - min_val > 0.01:  # Has real range
                                            candidates.append((key, arr_clean, min_val, max_val))
                                    except Exception:
                                        continue
                                
                                if len(candidates) >= 2:
                                    # Sort by min value to separate voltage (higher min) from capacity (lower min)
                                    candidates_sorted = sorted(candidates, key=lambda x: x[2])
                                    if V is None:
                                        # Take the one with the higher minimum value (likely voltage)
                                        V = candidates_sorted[-1][1]
                                    if Q is None:
                                        # Take the one with the lower minimum value (likely capacity)
                                        Q = candidates_sorted[0][1]
                                elif len(candidates) == 1:
                                    # Only one candidate - use it as first available
                                    if V is None:
                                        V = candidates[0][1]
                                    elif Q is None:
                                        Q = candidates[0][1]

                            if V is None or Q is None:
                                error_msg = "Could not find voltage/capacity arrays in .mat file.\n\n"
                                if available_keys:
                                    error_msg += f"**Available arrays found:** {', '.join(available_keys[:30])}"
                                    if len(available_keys) > 30:
                                        error_msg += f" (and {len(available_keys) - 30} more...)"

                                error_msg += f"\n\n**Top-level keys in .mat file:** {', '.join(non_meta_keys[:30]) if non_meta_keys else 'None found'}"
                                if len(non_meta_keys) > 30:
                                    error_msg += f" (and {len(non_meta_keys) - 30} more...)"

                                # Show details about what we found
                                if non_meta_keys:
                                    error_msg += "\n\n**Details:**\n"
                                    for key in non_meta_keys[:10]:
                                        try:
                                            val = mat[key]
                                            val_type = type(val).__name__
                                            if isinstance(val, np.ndarray):
                                                val_type += f" (shape: {val.shape}, dtype: {val.dtype})"
                                                if getattr(val, 'dtype', None) is not None and getattr(val.dtype, 'names', None):
                                                    val_type += f" [structured array with fields: {', '.join(val.dtype.names[:5])}]"
                                            error_msg += f"- {key}: {val_type}\n"
                                        except Exception:
                                            error_msg += f"- {key}: <unreadable>\n"

                                error_msg += "\n\n**Please ensure your .mat file contains arrays named with 'voltage'/'volt'/'v' and 'capacity'/'cap'/'q', or use CSV format.**"
                                st.error(error_msg)
                                st.stop()
                            else:
                                # Ensure same length
                                min_len = min(len(V), len(Q))
                                V = V[:min_len]
                                Q = Q[:min_len]
                                # Convert mAh to Ah if needed
                                if np.nanmax(Q) > 100:
                                    Q = Q / 1000.0
                                
                                # Build DataFrame with ALL arrays that match the length
                                df_data = {"Voltage": V, "Capacity": Q}
                                for key, arr in all_arrays.items():
                                    try:
                                        if not isinstance(arr, np.ndarray):
                                            continue
                                        arr_clean = np.squeeze(arr)
                                        # Skip if not 1D or already added
                                        if arr_clean.ndim != 1 or key in df_data or key.lower() in ['voltage', 'capacity']:
                                            continue
                                        # Only add if length matches
                                        if len(arr_clean) == min_len:
                                            clean_key = key.split('.')[-1].replace(' ', '_').replace('-', '_')
                                            # Skip if already exists
                                            if clean_key not in df_data:
                                                # Check if it's numeric
                                                if np.issubdtype(arr_clean.dtype, np.number):
                                                    df_data[clean_key] = arr_clean
                                    except:
                                        pass
                                
                                df = pd.DataFrame(df_data)
                        
                        # Extract time-series data from all_arrays if available
                        # Look for patterns like cycles.*voltage*_N, cycles.*current*_N, cycles.*time*_N
                        if df is not None and 'all_arrays' in locals():
                            time_series_from_arrays = {}
                            import re
                            
                            for key, arr in all_arrays.items():
                                try:
                                    if not isinstance(arr, np.ndarray):
                                        continue
                                    arr_clean = np.squeeze(arr)
                                    if arr_clean.ndim != 1 or arr_clean.size < 2:
                                        continue
                                    
                                    key_lower = key.lower()
                                    
                                    # Extract cycle number from key
                                    cycle_match = re.search(r'_(\d+)$|\[(\d+)\]$', key)
                                    if cycle_match:
                                        cycle_num = int(cycle_match.group(1) or cycle_match.group(2))
                                        
                                        if cycle_num not in time_series_from_arrays:
                                            time_series_from_arrays[cycle_num] = {}
                                        
                                        # Categorize by type
                                        if 'time' in key_lower or key_lower.startswith('t_'):
                                            time_series_from_arrays[cycle_num]['time'] = arr_clean
                                        elif 'voltage' in key_lower or key_lower.startswith('v_') or key_lower.endswith('_v'):
                                            time_series_from_arrays[cycle_num]['voltage'] = arr_clean
                                        elif 'current' in key_lower or key_lower.startswith('i_') or key_lower.endswith('_i'):
                                            time_series_from_arrays[cycle_num]['current'] = arr_clean
                                except:
                                    pass
                            
                            # Store time-series in session state for later use
                            if time_series_from_arrays:
                                st.session_state.mat_time_series = time_series_from_arrays

                if df is not None:
                    # Show detected columns / arrays to the user for transparency
                    try:
                        st.markdown("### Detected Columns / Arrays")
                        cols = list(df.columns)
                        
                        # For .mat files, show comprehensive information about all arrays
                        if name.endswith(".mat") and 'mat_available_keys' in st.session_state:
                            available_keys = st.session_state.mat_available_keys
                            all_arrays = st.session_state.mat_all_arrays
                            
                            st.markdown(f"**Total columns in DataFrame:** {len(cols)}")
                            st.markdown(f"**Total arrays found in .mat file:** {len(available_keys)}")
                            
                            # Show all columns in DataFrame
                            st.markdown("#### All Columns in DataFrame")
                            col_types = [(c, str(df[c].dtype), len(df[c]), 
                                         f"{df[c].min():.4f}" if np.issubdtype(df[c].dtype, np.number) else "N/A",
                                         f"{df[c].max():.4f}" if np.issubdtype(df[c].dtype, np.number) else "N/A") 
                                        for c in cols]
                            cols_df = pd.DataFrame(col_types, columns=["Column", "Data Type", "Length", "Min", "Max"])
                            st.dataframe(cols_df, use_container_width=True, hide_index=True)
                            
                            # Show all arrays found in .mat file (even if not in DataFrame)
                            if available_keys:
                                st.markdown("#### All Arrays Found in .mat File")
                                all_arrays_info = []
                                for key in available_keys[:100]:  # Limit to first 100 to avoid overwhelming display
                                    try:
                                        if key in all_arrays:
                                            arr = all_arrays[key]
                                            arr_clean = np.squeeze(arr)
                                            dtype = arr_clean.dtype if isinstance(arr_clean, np.ndarray) else type(arr_clean).__name__
                                            shape = arr_clean.shape if isinstance(arr_clean, np.ndarray) else "N/A"
                                            in_df = "Yes" if key in cols or key.split('.')[-1].replace(' ', '_').replace('-', '_') in cols else "No"
                                            all_arrays_info.append({
                                                "Array Name": key,
                                                "In DataFrame": in_df,
                                                "Shape": str(shape),
                                                "Data Type": str(dtype)
                                            })
                                    except:
                                        pass
                                
                                if all_arrays_info:
                                    arrays_df = pd.DataFrame(all_arrays_info)
                                    st.dataframe(arrays_df, use_container_width=True, hide_index=True)
                                    if len(available_keys) > 100:
                                        st.info(f"Showing first 100 arrays. Total arrays found: {len(available_keys)}")
                        else:
                            # For CSV files or if available_keys not defined, show simple table
                            col_types = [(c, str(df[c].dtype)) for c in cols]
                            st.write(pd.DataFrame(col_types, columns=["column", "dtype"]))

                        # Map likely roles for cycle-level datasets
                        cols_l = [c.lower() for c in cols]
                        # Normalize column names for matching (remove parentheses, spaces, etc.)
                        cols_normalized = [c.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_') for c in cols_l]
                        role_map = {}
                        
                        # cycle - expanded to include cycle_index
                        for cand in ["cycle", "cycle_number", "cycle_idx", "cycle_index"]:
                            if cand in cols_l:
                                role_map['cycle'] = cols[cols_l.index(cand)]
                                break
                            # Also check normalized names
                            for i, c_norm in enumerate(cols_normalized):
                                if cand == c_norm:
                                    role_map['cycle'] = cols[i]
                                    break
                            if 'cycle' in role_map:
                                break
                        
                        # discharge capacity candidates - handle Discharge_Capacity(Ah) format
                        for cand in ["qdischarge", "discharge_capacity", "discharge_cap", "qdis", "q_discharge", "dischargecapacity"]:
                            if cand in cols_l:
                                role_map['discharge_capacity'] = cols[cols_l.index(cand)]
                                break
                            # Also check normalized names
                            for i, c_norm in enumerate(cols_normalized):
                                if cand in c_norm or "dischargecapacity" in c_norm:
                                    role_map['discharge_capacity'] = cols[i]
                                    break
                            if 'discharge_capacity' in role_map:
                                break
                        
                        # common capacity names
                        if 'discharge_capacity' not in role_map:
                            for cand in ['capacity', 'cap', 'q', 'q_ah', 'capacity_ah', 'qcharge', 'qcharge']:
                                if cand in cols_l:
                                    role_map['discharge_capacity'] = cols[cols_l.index(cand)]
                                    break
                                # Also check normalized names
                                for i, c_norm in enumerate(cols_normalized):
                                    if cand in c_norm and "discharge" in cols_l[i]:
                                        role_map['discharge_capacity'] = cols[i]
                                        break
                                if 'discharge_capacity' in role_map:
                                    break
                        
                        # IR - handle Internal_Resistance(Ohm) format
                        for cand in ['internal_resistance', 'ir', 'resistance', 'internalresistance']:
                            if cand in cols_l:
                                role_map['internal_resistance'] = cols[cols_l.index(cand)]
                                break
                            # Also check normalized names
                            for i, c_norm in enumerate(cols_normalized):
                                if cand in c_norm or "internalresistance" in c_norm:
                                    role_map['internal_resistance'] = cols[i]
                                    break
                            if 'internal_resistance' in role_map:
                                break
                        # Temperature
                        for cand in ['tavg', 'temperature', 'temp', 'tmax', 'tmin']:
                            if cand in cols_l:
                                role_map['temperature'] = cols[cols_l.index(cand)]
                                break

                        # Time-series columns (time_<cycle>, current_<cycle>, voltage_<cycle>)
                        ts_cols = [c for c in cols if c.lower().startswith('time_') or c.lower().startswith('current_') or c.lower().startswith('voltage_')]
                        if ts_cols:
                            role_map['time_series_example'] = ts_cols[:6]

                        # [NEW] Column Mapping UI
                        with st.expander("Select Columns for Analysis", expanded=True):
                            st.info("Please confirm the column selection below. This ensures the graphs and analysis are accurate.")
                            
                            col_options = ["— none —"] + list(cols)
                            
                            def _get_idx(key):
                                # Helper to find index of deduced column
                                if key in role_map and role_map[key] in col_options:
                                    return col_options.index(role_map[key])
                                return 0

                            # Try to deduce voltage/current for defaults if not in role_map
                            def _deduce_col(keywords):
                                for c in cols:
                                    c_norm = c.lower().replace('_', '').replace(' ', '')
                                    for k in keywords:
                                        if k in c_norm: return c
                                return None

                            # Defaults
                            default_volt = _deduce_col(['voltage', 'volt', 'v']) if 'voltage' not in role_map else role_map['voltage']
                            default_curr = _deduce_col(['current', 'curr', 'i', 'amps']) if 'current' not in role_map else role_map['current']
                            
                            # Interactive Selectboxes
                            c_temp = st.selectbox("Temperature", col_options, index=_get_idx('temperature'))
                            c_curr = st.selectbox("Current", col_options, index=(col_options.index(default_curr) if default_curr in col_options else 0))
                            c_volt = st.selectbox("Voltage", col_options, index=(col_options.index(default_volt) if default_volt in col_options else 0))
                            c_cycle = st.selectbox("Cycle", col_options, index=_get_idx('cycle'))
                            c_cap = st.selectbox("Discharge Capacity", col_options, index=_get_idx('discharge_capacity'))

                            # Apply mapping (renaming columns in the DataFrame)
                            rename_map = {}
                            # Use standard names that the rest of the app expects
                            if c_cycle != "— none —": rename_map[c_cycle] = "cycle"
                            if c_volt != "— none —": rename_map[c_volt] = "voltage"
                            if c_curr != "— none —": rename_map[c_curr] = "current"
                            if c_temp != "— none —": rename_map[c_temp] = "temperature"
                            if c_cap != "— none —": rename_map[c_cap] = "discharge_capacity"
                            
                            if rename_map:
                                df = df.rename(columns=rename_map)
                                st.caption(f"mapped: {list(rename_map.keys())} -> {list(rename_map.values())}")
                    except Exception as e:
                        # Fallback to simple display if detailed display fails
                        try:
                            st.markdown("### Detected Columns / Arrays")
                            cols = list(df.columns)
                            col_types = [(c, str(df[c].dtype)) for c in cols]
                            st.write(pd.DataFrame(col_types, columns=["column", "dtype"]))
                        except:
                            pass
                    # Check if this is cycle-level data
                    is_cycle_level = _detect_cycle_level_data(df)
                    
                    if is_cycle_level:
                        # Process cycle-level data
                        cycle_data = _parse_cycle_level_data(df)
                        time_series = _parse_time_series_columns(df)
                        
                        # [NEW] Handle flat time-series data (Cycle, Voltage, Current columns)
                        if 'voltage' in df.columns and 'cycle' in df.columns:
                            # If we have flat data (Voltage, Cycle), we can generate curves
                            # Check if we have enough data (e.g. > 10 points per cycle average) to justify plotting
                            if len(df) > len(cycle_data) * 10: 
                                st.session_state.cycles_time_series = df.rename(columns={'discharge_capacity': 'Qdlin', 'cycle': 'cycle_number'})
                                
                                # Also populate time_series dict for the "Time Series Plots"
                                if 'current' in df.columns:
                                    # Extract for a few cycles only to save memory/time
                                    cycles = sorted(df['cycle'].unique())
                                    if len(cycles) > 3:
                                        selected_cycles = [cycles[0], cycles[len(cycles)//2], cycles[-1]]
                                    else:
                                        selected_cycles = cycles
                                    
                                    for c in selected_cycles:
                                        sub = df[df['cycle'] == c]
                                        if c not in time_series:
                                            time_series[c] = {}
                                        time_series[c]['voltage'] = sub['voltage'].values
                                        time_series[c]['current'] = sub['current'].values if 'current' in sub.columns else None
                                        
                                        # Use 'time' if available, else synthetic
                                        if 'time' in sub.columns:
                                            time_series[c]['time'] = sub['time'].values
                                        else:
                                            time_series[c]['time'] = np.arange(len(sub))

                        # Merge with time-series data extracted from .mat arrays if available
                        if name.endswith(".mat") and 'mat_time_series' in st.session_state:
                            mat_ts = st.session_state.mat_time_series
                            # Merge into time_series dict
                            for cycle_num, ts_data in mat_ts.items():
                                if cycle_num not in time_series:
                                    time_series[cycle_num] = {}
                                # Update with mat time-series data (don't overwrite existing)
                                for key, value in ts_data.items():
                                    if key not in time_series[cycle_num]:
                                        time_series[cycle_num][key] = value
                        
                        # Handle cycles_time_series from cell11dataset_for_python structure
                        vc_all_cell11 = None
                        ica_all_cell11 = None
                        if 'cycles_time_series' in st.session_state:
                            cycles_ts_df = st.session_state.cycles_time_series
                            if cycles_ts_df is not None and len(cycles_ts_df) > 0:
                                # Generate voltage-capacity curves and dQ/dV plots from cycles data
                                vc_rows = []
                                ica_rows = []
                                
                                # Check if we have the required columns
                                has_voltage = 'voltage' in cycles_ts_df.columns or 'V' in cycles_ts_df.columns
                                has_qdlin = 'Qdlin' in cycles_ts_df.columns or 'qdlin' in cycles_ts_df.columns
                                
                                if has_voltage and has_qdlin:
                                    # Group by cycle_number
                                    if 'cycle_number' in cycles_ts_df.columns:
                                        for cycle_num in cycles_ts_df['cycle_number'].unique():
                                            cycle_data = cycles_ts_df[cycles_ts_df['cycle_number'] == cycle_num].copy()
                                            
                                            # Get voltage and capacity
                                            v_col = 'voltage' if 'voltage' in cycle_data.columns else 'V'
                                            q_col = 'Qdlin' if 'Qdlin' in cycle_data.columns else 'qdlin'
                                            
                                            V = cycle_data[v_col].values
                                            Q = cycle_data[q_col].values
                                            
                                            # Remove NaN and invalid values
                                            valid_mask = np.isfinite(V) & np.isfinite(Q)
                                            if np.sum(valid_mask) > 5:  # Need at least 5 points
                                                V_clean = V[valid_mask]
                                                Q_clean = Q[valid_mask]
                                                
                                                # Sort by voltage for proper plotting
                                                sort_idx = np.argsort(V_clean)
                                                V_clean = V_clean[sort_idx]
                                                Q_clean = Q_clean[sort_idx]
                                                
                                                # Add to voltage-capacity curves
                                                vc_rows.append(pd.DataFrame({
                                                    'Voltage': V_clean,
                                                    'Capacity_Ah': Q_clean,
                                                    'Cycle': f'Cycle {int(cycle_num)}'
                                                }))
                                                
                                                # Calculate dQ/dV for ICA plot
                                                if SCIPY_OK and len(V_clean) > 3:
                                                    try:
                                                        V_robust, dQdV_robust = _compute_robust_dqdv(V_clean, Q_clean)
                                                        ica_rows.append(pd.DataFrame({
                                                            'Voltage': V_robust,
                                                            'dQdV': dQdV_robust,
                                                            'Cycle': f'Cycle {int(cycle_num)}'
                                                        }))
                                                    except:
                                                        pass
                                
                                if vc_rows:
                                    vc_all_cell11 = pd.concat(vc_rows, ignore_index=True)
                                    st.session_state.vc_all_cell11 = vc_all_cell11
                                if ica_rows:
                                    ica_all_cell11 = pd.concat(ica_rows, ignore_index=True)
                                    st.session_state.ica_all_cell11 = ica_all_cell11
                        
                        if cycle_data is None or len(cycle_data) == 0:
                            st.error("Could not parse cycle-level data. Please check column names.")
                            st.stop()
                        
                        # Extract cycle features AND get corrected data for plotting
                        cycle_features, cycle_data_plot = _extract_cycle_features(cycle_data)
                        
                        # Use cycle_data_plot for visualizations (it has correct per-cycle capacity)
                        degradation_interps = _generate_degradation_interpretations(cycle_features, cycle_data_plot)
                        quality_notes, richness_notes = _assess_data_quality_and_richness(cycle_data_plot, time_series, cycle_features)
                        
                        # ============================================================
                        # DATA QUALITY & RICHNESS ASSESSMENT
                        # ============================================================
                        st.markdown("### Data Quality & Richness Assessment")
                        
                        col_qual1, col_qual2 = st.columns(2, gap="medium")
                        
                        with col_qual1:
                            st.markdown("#### Data Quality")
                            for note in quality_notes:
                                st.write("- " + note)
                        
                        with col_qual2:
                            st.markdown("#### Dataset Richness")
                            for note in richness_notes:
                                st.write("- " + note)
                        
                        st.divider()
                        
                        # ============================================================
                        # KEY FEATURES (Max, Min, Number of Cycles)
                        # ============================================================
                        st.markdown("### Key Features")
                        
                        # Create feature summary table
                        features_data = []
                        
                        # Capacity features
                        if 'discharge_capacity' in cycle_data_plot.columns:
                            cap_valid = cycle_data_plot['discharge_capacity'].dropna()
                            if len(cap_valid) > 0:
                                features_data.append({
                                    "Parameter": "Discharge Capacity",
                                    "Max": f"{cycle_features.get('capacity_max_Ah', np.nan):.4f} Ah" if 'capacity_max_Ah' in cycle_features else "N/A",
                                    "Min": f"{cycle_features.get('capacity_min_Ah', np.nan):.4f} Ah" if 'capacity_min_Ah' in cycle_features else "N/A",
                                    "Mean": f"{cycle_features.get('capacity_mean_Ah', np.nan):.4f} Ah" if 'capacity_mean_Ah' in cycle_features else "N/A",
                                    "No. of Cycles": f"{len(cap_valid)}",
                                    "Initial": f"{cycle_features.get('capacity_initial_Ah', np.nan):.4f} Ah" if 'capacity_initial_Ah' in cycle_features else "N/A",
                                    "Final": f"{cycle_features.get('capacity_final_Ah', np.nan):.4f} Ah" if 'capacity_final_Ah' in cycle_features else "N/A"
                                })
                        
                        # Internal Resistance features
                        if 'internal_resistance' in cycle_data_plot.columns:
                            ir_valid = cycle_data_plot['internal_resistance'].dropna()
                            if len(ir_valid) > 0:
                                features_data.append({
                                    "Parameter": "Internal Resistance",
                                    "Max": f"{cycle_features.get('ir_max_Ohm', np.nan):.4f} Ω" if 'ir_max_Ohm' in cycle_features else "N/A",
                                    "Min": f"{cycle_features.get('ir_min_Ohm', np.nan):.4f} Ω" if 'ir_min_Ohm' in cycle_features else "N/A",
                                    "Mean": f"{cycle_features.get('ir_mean_Ohm', np.nan):.4f} Ω" if 'ir_mean_Ohm' in cycle_features else "N/A",
                                    "No. of Cycles": f"{len(ir_valid)}",
                                    "Initial": f"{cycle_features.get('ir_initial_Ohm', np.nan):.4f} Ω" if 'ir_initial_Ohm' in cycle_features else "N/A",
                                    "Final": f"{cycle_features.get('ir_final_Ohm', np.nan):.4f} Ω" if 'ir_final_Ohm' in cycle_features else "N/A"
                                })
                        
                        # Temperature features
                        if 'temperature' in cycle_data_plot.columns:
                            temp_valid = cycle_data_plot['temperature'].dropna()
                            if len(temp_valid) > 0:
                                features_data.append({
                                    "Parameter": "Temperature",
                                    "Max": f"{cycle_features.get('temp_max_C', np.nan):.2f} °C" if 'temp_max_C' in cycle_features else "N/A",
                                    "Min": f"{cycle_features.get('temp_min_C', np.nan):.2f} °C" if 'temp_min_C' in cycle_features else "N/A",
                                    "Mean": f"{cycle_features.get('temp_mean_C', np.nan):.2f} °C" if 'temp_mean_C' in cycle_features else "N/A",
                                    "No. of Cycles": f"{len(temp_valid)}",
                                    "Initial": "N/A",
                                    "Final": "N/A"
                                })
                        
                        # Overall cycle info
                        if cycle_features.get('n_cycles', 0) > 0:
                            features_data.append({
                                "Parameter": "Cycle Life",
                                "Max": f"{cycle_features.get('cycle_range', [0, 0])[1]}" if 'cycle_range' in cycle_features else "N/A",
                                "Min": f"{cycle_features.get('cycle_range', [0, 0])[0]}" if 'cycle_range' in cycle_features else "N/A",
                                "Mean": "N/A",
                                "No. of Cycles": f"{cycle_features.get('n_cycles', 0)}",
                                "Initial": "N/A",
                                "Final": "N/A"
                            })
                        
                        if features_data:
                            features_df = pd.DataFrame(features_data)
                            st.dataframe(features_df, use_container_width=True, hide_index=True)
                        
                        # Additional summary metrics
                        st.markdown("#### Summary Statistics")
                        col_sum1, col_sum2, col_sum3 = st.columns(3, gap="medium")
                        
                        with col_sum1:
                            if 'total_fade_percent' in cycle_features and np.isfinite(cycle_features['total_fade_percent']):
                                st.metric("Total Capacity Fade", f"{cycle_features['total_fade_percent']:.2f}%")
                            if 'fade_rate_percent_per_cycle' in cycle_features and np.isfinite(cycle_features['fade_rate_percent_per_cycle']):
                                st.metric("Fade Rate", f"{cycle_features['fade_rate_percent_per_cycle']:.4f}% per cycle")
                        
                        with col_sum2:
                            if 'ir_growth_percent' in cycle_features and np.isfinite(cycle_features['ir_growth_percent']):
                                st.metric("IR Growth", f"{cycle_features['ir_growth_percent']:.2f}%")
                        
                        with col_sum3:
                            c_min, c_max = cycle_features.get('cycle_range', [0, 0])
                            st.metric("Total Cycles", f"{cycle_features.get('n_cycles', 0)} ({c_min}–{c_max})")
                        
                        st.divider()
                        
                        # ============================================================
                        # 4 REQUIRED PLOTS (Automatically Displayed)
                        # ============================================================
                        st.markdown("### Cycle-Life Visualizations")
                        
                        # Create 2x2 grid layout for the 4 plots
                        col1, col2 = st.columns(2, gap="medium")
                        
                        # Plot 1: Discharge Capacity vs Cycle Life
                        with col1:
                            if 'discharge_capacity' in cycle_data_plot.columns:
                                cap_data = cycle_data_plot.dropna(subset=['cycle', 'discharge_capacity'])
                                if len(cap_data) > 0:
                                    # Downsample for plotting to avoid browser crash
                                    plot_data = _downsample_series(cap_data, max_points=3000)
                                    cap_chart = alt.Chart(plot_data).mark_line(point={'size': 100}).encode(
                                        x=alt.X("cycle:Q", title="Cycle Number"),
                                        y=alt.Y("discharge_capacity:Q", title="Discharge Capacity (Ah)"),
                                        tooltip=['cycle:Q', 'discharge_capacity:Q']
                                    ).properties(title="1. Discharge Capacity vs Cycle Life", width=450, height=300)
                                    st.altair_chart(cap_chart, width='stretch')
                                else:
                                    st.info("No capacity data available")
                            else:
                                st.info("Discharge capacity column not found")
                        
                        # Plot 2: Internal Resistance vs Cycle Life
                        with col2:
                            if 'internal_resistance' in cycle_data_plot.columns and cycle_data_plot['internal_resistance'].notna().any():
                                ir_data = cycle_data_plot.dropna(subset=['cycle', 'internal_resistance'])
                                if len(ir_data) > 0:
                                    plot_data = _downsample_series(ir_data, max_points=3000)
                                    ir_chart = alt.Chart(plot_data).mark_line(point={'size': 100}, color='#FF6B6B').encode(
                                        x=alt.X("cycle:Q", title="Cycle Number"),
                                        y=alt.Y("internal_resistance:Q", title="Internal Resistance (Ohm)"),
                                        tooltip=['cycle:Q', 'internal_resistance:Q']
                                    ).properties(title="2. Internal Resistance vs Cycle Life", width=450, height=300)
                                    st.altair_chart(ir_chart, width='stretch')
                                else:
                                    st.info("No IR data available")
                            else:
                                st.info("Internal resistance column not found")
                        
                        # Plot 3: Temperature vs Cycle Life
                        col3, col4 = st.columns(2, gap="medium")
                        with col3:
                            if 'temperature' in cycle_data_plot.columns and cycle_data_plot['temperature'].notna().any():
                                temp_data = cycle_data_plot.dropna(subset=['cycle', 'temperature'])
                                if len(temp_data) > 0:
                                    plot_data = _downsample_series(temp_data, max_points=3000)
                                    temp_chart = alt.Chart(plot_data).mark_line(point={'size': 100}, color='#FFA500').encode(
                                        x=alt.X("cycle:Q", title="Cycle Number"),
                                        y=alt.Y("temperature:Q", title="Temperature (°C)"),
                                        tooltip=['cycle:Q', 'temperature:Q']
                                    ).properties(title="3. Temperature vs Cycle Life", width=450, height=300)
                                    st.altair_chart(temp_chart, width='stretch')
                                else:
                                    st.info("No temperature data available")
                            else:
                                st.info("Temperature column not found")
                            
                        # Plot 4: Current & Voltage Time-Series Profile
                        with col4:
                            if time_series:
                                cycle_nums = sorted(time_series.keys())
                                if len(cycle_nums) > 0:
                                    # Select cycles for time-series display
                                    if len(cycle_nums) >= 3:
                                        selected_cycles = [cycle_nums[0], cycle_nums[len(cycle_nums)//2], cycle_nums[-1]]
                                        selected_label = f"First, Middle, Last cycles: {', '.join(map(str, selected_cycles))}"
                                    elif len(cycle_nums) >= 1:
                                        selected_cycles = cycle_nums[:min(3, len(cycle_nums))]
                                        selected_label = f"Cycles: {', '.join(map(str, selected_cycles))}"
                                    else:
                                        selected_cycles = []
                                        selected_label = "No cycles available"
                                    
                                    ts_charts = []
                                    for cycle_num in selected_cycles:
                                        if cycle_num in time_series:
                                            ts = time_series[cycle_num]
                                            time = ts.get('time', None)
                                            current = ts.get('current', None)
                                            voltage = ts.get('voltage', None)
                                            
                                            # Combine current and voltage data
                                            if time is not None and (current is not None or voltage is not None):
                                                ts_data = []
                                                # Ensure time is a numpy array
                                                if isinstance(time, (list, tuple)):
                                                    time = np.array(time)
                                                
                                                if current is not None:
                                                    # Ensure current is a numpy array and same length as time
                                                    if isinstance(current, (list, tuple)):
                                                        current = np.array(current)
                                                    # Match lengths
                                                    min_len = min(len(time), len(current))
                                                    time_aligned = time[:min_len]
                                                    current_aligned = current[:min_len]
                                                    valid = np.isfinite(time_aligned) & np.isfinite(current_aligned)
                                                    if np.any(valid):
                                                        ts_data.append(pd.DataFrame({
                                                            'time': time_aligned[valid],
                                                            'value': current_aligned[valid],
                                                            'metric': 'Current (A)',
                                                            'cycle': cycle_num
                                                        }))
                                                if voltage is not None:
                                                    # Ensure voltage is a numpy array and same length as time
                                                    if isinstance(voltage, (list, tuple)):
                                                        voltage = np.array(voltage)
                                                    # Match lengths
                                                    min_len = min(len(time), len(voltage))
                                                    time_aligned = time[:min_len]
                                                    voltage_aligned = voltage[:min_len]
                                                    valid = np.isfinite(time_aligned) & np.isfinite(voltage_aligned)
                                                    if np.any(valid):
                                                        ts_data.append(pd.DataFrame({
                                                            'time': time_aligned[valid],
                                                            'value': voltage_aligned[valid],
                                                            'metric': 'Voltage (V)',
                                                            'cycle': cycle_num
                                                        }))
                                                if ts_data:
                                                    ts_charts.append(pd.concat(ts_data, ignore_index=True))
                                    
                                    if ts_charts:
                                        combined_ts = pd.concat(ts_charts, ignore_index=True)
                                        # Downsample aggressively for time-series as they can be huge
                                        plot_data = _downsample_series(combined_ts, max_points=5000)
                                        ts_chart = alt.Chart(plot_data).mark_line().encode(
                                            x=alt.X("time:Q", title="Time (s)"),
                                            y=alt.Y("value:Q", title="Value"),
                                            color=alt.Color("metric:N", title="Metric"),
                                            detail='cycle:N',
                                            tooltip=['time:Q', 'value:Q', 'metric:N', 'cycle:N']
                                        ).properties(title=f"4. Current & Voltage Profile\n({selected_label})", width=450, height=300)
                                        st.altair_chart(ts_chart, width='stretch')
                                    else:
                                        st.info("No valid time-series data found")
                                else:
                                    st.info("No time-series data available")
                            else:
                                st.info("Time-series data not provided in dataset")
                        
                        # Additional detailed time-series plots (if available)
                        if time_series:
                            st.markdown("### Detailed Time-Series Plots (First/Middle/Last Cycles)")
                            cycle_nums = sorted(time_series.keys())
                            if len(cycle_nums) >= 3:
                                selected = [cycle_nums[0], cycle_nums[len(cycle_nums)//2], cycle_nums[-1]]
                            elif len(cycle_nums) >= 1:
                                selected = cycle_nums[:min(3, len(cycle_nums))]
                            else:
                                selected = []
                            
                            for cycle_num in selected:
                                if cycle_num in time_series:
                                    ts = time_series[cycle_num]
                                    time = ts.get('time', None)
                                    current = ts.get('current', None)
                                    voltage = ts.get('voltage', None)
                                    
                                    if time is not None and (current is not None or voltage is not None):
                                        st.markdown(f"**Cycle {cycle_num}**")
                                        cols_ts = st.columns(2)
                                        
                                        # Ensure time is a numpy array
                                        if isinstance(time, (list, tuple)):
                                            time = np.array(time)
                                        
                                        if current is not None:
                                            # Ensure current is a numpy array and match length with time
                                            if isinstance(current, (list, tuple)):
                                                current = np.array(current)
                                            min_len = min(len(time), len(current))
                                            time_aligned = time[:min_len]
                                            current_aligned = current[:min_len]
                                            ts_df = pd.DataFrame({'time': time_aligned, 'current': current_aligned})
                                            ts_df = ts_df.dropna()
                                            if len(ts_df) > 0:
                                                with cols_ts[0]:
                                                    plot_data = _downsample_series(ts_df, max_points=2000)
                                                    curr_chart = alt.Chart(plot_data).mark_line().encode(
                                                        x=alt.X("time:Q", title="Time (s)"),
                                                        y=alt.Y("current:Q", title="Current (A)")
                                                    ).properties(title=f"Cycle {cycle_num} - Current", width=300, height=200)
                                                    st.altair_chart(curr_chart, width='stretch')
                                        if voltage is not None:
                                            # Ensure voltage is a numpy array and match length with time
                                            if isinstance(voltage, (list, tuple)):
                                                voltage = np.array(voltage)
                                            min_len = min(len(time), len(voltage))
                                            time_aligned = time[:min_len]
                                            voltage_aligned = voltage[:min_len]
                                            ts_df = pd.DataFrame({'time': time_aligned, 'voltage': voltage_aligned})
                                            ts_df = ts_df.dropna()
                                            if len(ts_df) > 0:
                                                with cols_ts[1]:
                                                    plot_data = _downsample_series(ts_df, max_points=2000)
                                                    volt_chart = alt.Chart(plot_data).mark_line().encode(
                                                        x=alt.X("time:Q", title="Time (s)"),
                                                        y=alt.Y("voltage:Q", title="Voltage (V)")
                                                    ).properties(title=f"Cycle {cycle_num} - Voltage", width=300, height=200)
                                                    st.altair_chart(volt_chart, width='stretch')
                        
                        st.divider()
                        
                        # ============================================================
                        # VOLTAGE-CAPACITY CURVES AND DQ/DV PLOTS (from cell11dataset_for_python)
                        # ============================================================
                        if 'vc_all_cell11' in st.session_state or 'ica_all_cell11' in st.session_state:
                            st.markdown("### Voltage-Capacity Curves and dQ/dV Analysis")
                            
                            col_vc, col_ica = st.columns(2, gap="medium")
                            
                            # Plot voltage-capacity curves
                            with col_vc:
                                if 'vc_all_cell11' in st.session_state:
                                    vc_data = st.session_state.vc_all_cell11
                                    if vc_data is not None and len(vc_data) > 0:
                                        plot_data = _downsample_series(vc_data, max_points=5000)
                                        vc_chart = alt.Chart(plot_data).mark_line(point={'size': 50}).encode(
                                            x=alt.X("Voltage:Q", title="Voltage (V)", scale=alt.Scale(nice=True)),
                                            y=alt.Y("Capacity_Ah:Q", title="Capacity (Ah)", scale=alt.Scale(nice=True)),
                                            color=alt.Color("Cycle:N", title="Cycle"),
                                            tooltip=['Voltage:Q', 'Capacity_Ah:Q', 'Cycle:N']
                                        ).properties(title="Voltage vs Capacity", width=450, height=350)
                                        st.altair_chart(vc_chart, width='stretch')
                                    else:
                                        st.info("No voltage-capacity data available")
                                else:
                                    st.info("Voltage-capacity curves not available")
                            
                            # Plot dQ/dV (ICA) curves
                            with col_ica:
                                if 'ica_all_cell11' in st.session_state:
                                    ica_data = st.session_state.ica_all_cell11
                                    if ica_data is not None and len(ica_data) > 0:
                                        plot_data = _downsample_series(ica_data, max_points=5000)
                                        ica_chart = alt.Chart(plot_data).mark_line(point={'size': 50}).encode(
                                            x=alt.X("Voltage:Q", title="Voltage (V)", scale=alt.Scale(nice=True)),
                                            y=alt.Y("dQdV:Q", title="dQ/dV (Ah/V)", scale=alt.Scale(nice=True)),
                                            color=alt.Color("Cycle:N", title="Cycle"),
                                            tooltip=['Voltage:Q', 'dQdV:Q', 'Cycle:N']
                                        ).properties(title="ICA: dQ/dV vs Voltage", width=450, height=350)
                                        st.altair_chart(ica_chart, width='stretch')
                                    else:
                                        st.info("No dQ/dV data available")
                                else:
                                    st.info("dQ/dV curves not available")
                            
                            st.divider()
                        
                        # ============================================================
                        # DEGRADATION TREND ANALYSIS
                        # ============================================================
                        st.markdown("### Degradation Trend Analysis")
                        for di in degradation_interps:
                            st.write("- " + di)
                        
                        st.divider()
                        
                        # ============================================================
                        # EXPORT OPTIONS
                        # ============================================================
                        st.markdown("### Export Options")
                        
                        # Export .mat file
                        if SCIPY_OK:
                            mat_buf = io.BytesIO()
                            mat_data = {
                                'cycle': cycle_data_plot['cycle'].values.astype(float),
                                'discharge_capacity': cycle_data_plot['discharge_capacity'].values.astype(float),
                            }
                            if 'internal_resistance' in cycle_data_plot.columns:
                                mat_data['internal_resistance'] = cycle_data_plot['internal_resistance'].values.astype(float)
                            if 'temperature' in cycle_data_plot.columns:
                                mat_data['temperature'] = cycle_data_plot['temperature'].values.astype(float)
                            
                            # Add time-series data if available
                            if time_series:
                                for cycle_num, ts in time_series.items():
                                    if 'time' in ts:
                                        mat_data[f'time_{cycle_num}'] = ts['time'].astype(float)
                                    if 'current' in ts:
                                        mat_data[f'current_{cycle_num}'] = ts['current'].astype(float)
                                    if 'voltage' in ts:
                                        mat_data[f'voltage_{cycle_num}'] = ts['voltage'].astype(float)
                            
                            savemat(mat_buf, mat_data)
                            mat_buf.seek(0)
                            
                            st.download_button(
                                "Download processed data (.mat)",
                                data=mat_buf.getvalue(),
                                file_name="BatteryLab_cycle_data.mat",
                                mime="application/octet-stream"
                            )
                        
                        # PDF Download
                        pdf_bytes = generate_pdf_report(
                            cycle_data=cycle_data_plot,
                            cycle_features=cycle_features,
                            degradation_interps=degradation_interps,
                            time_series=time_series
                        )
                        st.download_button(
                            "Download Full Report (PDF)",
                            data=pdf_bytes,
                            file_name="BatteryLab_cycle_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        # Original voltage-capacity processing
                        vcol, qcol, cyc = _standardize_columns(df)
                        if vcol is None or qcol is None:
                            st.error("Please include Voltage and Capacity_Ah (or Capacity_mAh).")
                            st.stop()
                        if cyc is None:
                            df["_Cycle"] = "Curve1"
                            cyc = "_Cycle"

                        groups = list(df[cyc].astype(str).unique())
                        features_by_group = {}
                        vc_all_rows, ica_all_rows = [], []

                        for g in groups:
                            sub = df[df[cyc].astype(str) == g]
                            V, Q = _prep_series(sub[vcol], sub[qcol])
                            if len(V) < 5:
                                continue
                            V_robust, dQdV, peak_info, dVdQ_med = _extract_features(V, Q)
                            features_by_group[g] = {
                                "n_samples": int(len(V)),
                                "voltage_range_V": [float(np.nanmin(V)), float(np.nanmax(V))],
                                "cap_range_Ah": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                                "ica_peaks_count": int(peak_info.get("n_peaks", 0)),
                                "ica_peak_voltages_V": peak_info.get("voltages", []),
                                "ica_peak_widths_V": peak_info.get("widths_V", []),
                                "dVdQ_median_abs": dVdQ_med,
                            }
                            vc_all_rows.append(pd.DataFrame({"Voltage": V, "Capacity_Ah": Q, "Cycle": g}))
                            ica_all_rows.append(pd.DataFrame({"Voltage": V_robust, "dQdV": dQdV, "Cycle": g}))

                        if not features_by_group:
                            st.error("Not enough valid data points to analyze.")
                            st.stop()

                        # (1) DATASET QUALITY & RICHNESS
                        st.markdown("### Dataset Quality & Richness")
                        for g, f in features_by_group.items():
                            st.write(
                                f"{g} — {f['n_samples']} points | "
                                f"V range: {f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f} V | "
                                f"Capacity: {f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f} Ah | "
                                f"ICA peaks: {f['ica_peaks_count']}"
                            )

                        richness_notes = []
                        if len(features_by_group) >= 2:
                            richness_notes.append("Multiple curves detected -> enables trend comparisons (fade, peak shifts, impedance).")
                        else:
                            richness_notes.append("Single curve detected -> add an aged or baseline curve for richer insights.")
                        if any(f["n_samples"] < 30 for f in features_by_group.values()):
                            richness_notes.append("Some curves have <30 points -> derivatives may be noisy; consider higher-resolution sampling.")
                        if any(f["ica_peaks_count"] == 0 for f in features_by_group.values()):
                            richness_notes.append("ICA shows few/no peaks -> may indicate smooth kinetics or insufficient resolution.")
                        for rn in richness_notes:
                            st.write("- " + rn)

                        # (2) NEXT-STEP SUGGESTIONS
                        st.markdown("### Next-Step Suggestions")
                        suggestions = []
                        if len(features_by_group) == 1:
                            suggestions = [
                                "Add a comparison curve (e.g., Fresh vs Aged, Cycle 10 vs Cycle 500) to quantify capacity fade and ICA peak shifts.",
                                "Track ICA peak positions/widths across cycles to infer LLI vs LAM vs impedance growth.",
                                "Include IR/temperature columns (if available) to correlate electro-thermal behavior."
                            ]
                        else:
                            suggestions = [
                                "Quantify fade: compute % capacity change between earliest and latest curves.",
                                "Track ICA peak shifts (mV) and broadening (mV) -> LLI and impedance indicators.",
                                "Build a simple regression using extracted features to predict end-of-life or rate performance.",
                                "If you have cycle index/time, add it to the dataset for richer trend modeling."
                            ]
                        for s in suggestions:
                            st.write("- " + s)

                        st.divider()
                        do_plots = st.button("Visualize recommended plots", type="primary")

                        if do_plots:
                            st.markdown("### Visualizations")
                            vc_all = pd.concat(vc_all_rows, ignore_index=True)
                            ica_all = pd.concat(ica_all_rows, ignore_index=True)
                            
                            # Downsample big datasets to prevent browser crash
                            vc_plot = _downsample_series(vc_all, max_points=10000)
                            ica_plot = _downsample_series(ica_all, max_points=10000)

                            vc_chart = (
                                alt.Chart(vc_plot)
                                .mark_line()
                                .encode(
                                    x=alt.X("Voltage:Q", title="Voltage (V)"),
                                    y=alt.Y("Capacity_Ah:Q", title="Capacity (Ah)"),
                                    color=alt.Color("Cycle:N", title="Curve")
                                )
                                .properties(title="Voltage vs Capacity")
                            )

                            ica_chart = (
                                alt.Chart(ica_plot)
                                .mark_line()
                                .encode(
                                    x=alt.X("Voltage:Q", title="Voltage (V)"),
                                    y=alt.Y("dQdV:Q", title="dQ/dV (Ah/V)"),
                                    color=alt.Color("Cycle:N", title="Curve")
                                )
                                .properties(title="ICA: dQ/dV vs Voltage")
                            )

                            c1, c2 = st.columns(2)
                            with c1:
                                st.altair_chart(vc_chart, width='stretch')
                            with c2:
                                st.altair_chart(ica_chart, width='stretch')

                            st.markdown("### Key Features by Curve")
                            st.write(features_by_group)

                            st.markdown("### AI-style Interpretations (dynamic)")
                            interps = []
                            if len(features_by_group) == 1:
                                interps = [
                                    "Distinct ICA peaks often indicate well-defined phase transitions.",
                                    "Broadening of ICA peaks over time is a common sign of rising impedance.",
                                    "Compare this curve to an earlier/later cycle to quantify fade and peak shifts."
                                ]
                            else:
                                gnames = list(features_by_group.keys())[:2]
                                fA, fB = features_by_group[gnames[0]], features_by_group[gnames[1]]
                                featA = {
                                    "cap_range_Ah": fA["cap_range_Ah"],
                                    "ica_peak_voltages_V": fA["ica_peak_voltages_V"],
                                    "ica_peak_widths_V": fA["ica_peak_widths_V"],
                                    "dVdQ_median_abs": fA["dVdQ_median_abs"],
                                }
                                featB = {
                                    "cap_range_Ah": fB["cap_range_Ah"],
                                    "ica_peak_voltages_V": fB["ica_peak_voltages_V"],
                                    "ica_peak_widths_V": fB["ica_peak_widths_V"],
                                    "dVdQ_median_abs": fB["dVdQ_median_abs"],
                                }
                                interps = _compare_two_sets(gnames[0], featA, gnames[1], featB)
                            for b in interps:
                                st.write("- " + b)

                            # ---------------------------
                            # Export: .mat data + .py repro script
                            # ---------------------------
                            def _build_repro_script_py(default_mat_name="BatteryLab_analytics_data.mat"):
                                return f"""# Reproduce BatteryLAB analytics plots
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.io import loadmat
except Exception as e:
    raise SystemExit("This script needs SciPy: pip install scipy") from e

fname = sys.argv[1] if len(sys.argv) > 1 else "{default_mat_name}"
m = loadmat(fname, squeeze_me=True)

# Voltage–Capacity table
V1   = m['vc_voltage']
Q1   = m['vc_capacity_ah']
c1   = m['vc_cycle_idx'].astype(int)
n1   = [str(x) for x in (m['vc_cycle_names'].tolist() if hasattr(m['vc_cycle_names'], 'tolist') else m['vc_cycle_names'])]

# ICA table
V2   = m['ica_voltage']
dqdv = m['ica_dqdv']
c2   = m['ica_cycle_idx'].astype(int)
n2   = [str(x) for x in (m['ica_cycle_names'].tolist() if hasattr(m['ica_cycle_names'], 'tolist') else m['ica_cycle_names'])]

# Plot Voltage–Capacity
plt.figure()
for idx in np.unique(c1):
    mask = (c1 == idx)
    plt.plot(V1[mask], Q1[mask], label=n1[int(idx)])
plt.xlabel('Voltage (V)')
plt.ylabel('Capacity (Ah)')
plt.title('Voltage vs Capacity')
plt.legend()
plt.tight_layout()

# Plot ICA
plt.figure()
for idx in np.unique(c2):
    mask = (c2 == idx)
    plt.plot(V2[mask], dqdv[mask], label=n2[int(idx)])
plt.xlabel('Voltage (V)')
plt.ylabel('dQ/dV (Ah/V)')
plt.title('ICA: dQ/dV vs Voltage')
plt.legend()
plt.tight_layout()
plt.show()
"""

                            # Build .mat payload (handles arbitrary # of curves)
                            vc_cat = vc_all["Cycle"].astype("category")
                            ica_cat = ica_all["Cycle"].astype("category")

                            vc_cycle_idx = vc_cat.cat.codes.to_numpy().astype(np.int32)
                            vc_cycle_names = np.array(vc_cat.cat.categories.tolist(), dtype=object)

                            ica_cycle_idx = ica_cat.cat.codes.to_numpy().astype(np.int32)
                            ica_cycle_names = np.array(ica_cat.cat.categories.tolist(), dtype=object)

                            # Prepare binary buffers for both files
                            py_script_text = _build_repro_script_py()
                            py_bytes = py_script_text.encode("utf-8")

                            if SCIPY_OK:
                                mat_buf = io.BytesIO()
                                savemat(mat_buf, {
                                    # VC table
                                    "vc_voltage":      vc_all["Voltage"].to_numpy().astype(float),
                                    "vc_capacity_ah":  vc_all["Capacity_Ah"].to_numpy().astype(float),
                                    "vc_cycle_idx":    vc_cycle_idx,
                                    "vc_cycle_names":  vc_cycle_names,
                                    # ICA table
                                    "ica_voltage":     ica_all["Voltage"].to_numpy().astype(float),
                                    "ica_dqdv":        ica_all["dQdV"].to_numpy().astype(float),
                                    "ica_cycle_idx":   ica_cycle_idx,
                                    "ica_cycle_names": ica_cycle_names,
                                })
                                mat_buf.seek(0)
                            else:
                                mat_buf = None  # SciPy not available

                            c_dl1, c_dl2 = st.columns(2)
                            with c_dl1:
                                if SCIPY_OK and mat_buf is not None:
                                    st.download_button(
                                        "Download analytics data (.mat)",
                                        data=mat_buf.getvalue(),
                                        file_name="BatteryLab_analytics_data.mat",
                                        mime="application/octet-stream"
                                    )
                                else:
                                    st.info("Install SciPy on the server to enable `.mat` export (pip install scipy).")

                            with c_dl2:
                                st.download_button(
                                    "Download repro plot script (.py)",
                                    data=py_bytes,
                                    file_name="repro_analytics_plots.py",
                                    mime="text/x-python"
                                )

                            # PDF Download
                            pdf_bytes = generate_pdf_report(
                                features_by_group=features_by_group,
                                richness_notes=richness_notes,
                                suggestions=suggestions,
                                interps=interps,
                                vc_all=vc_all,
                                ica_all=ica_all
                            )
                            st.download_button(
                                "Download Full Report (PDF)",
                                data=pdf_bytes,
                                file_name="BatteryLab_analytics_report.pdf",
                                mime="application/pdf"
                            )

                            # Store for Copilot (analytics cache)
                            st.session_state.latest_analytics = {
                                "features_by_group": features_by_group,
                                "vc_all": vc_all,
                                "ica_all": ica_all,
                            }
                        else:
                            st.info("Click **Visualize recommended plots** to render charts, then see Key Features, interpretations, and download the PDF report.")
        else:
            st.info("Upload a CSV with Voltage and Capacity_Ah (or Capacity_mAh). Optional: Cycle column. .mat is supported if SciPy is available.")

    with col_chat:
        render_copilot(context_key="tab2", default_context="Analytics")

# =========================
# TAB 3: Cleaning Module WITH in-tab Copilot
# =========================
with tab3:
    col_main, col_chat = st.columns([0.68, 0.32], gap="large")
    
    with col_main:
        st.subheader("Data Cleaning Module")
        st.write(
            "Upload a CSV or Excel file to clean and download the cleaned version. "
            "Configure cleaning options below to customize the cleaning process."
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload battery data file",
            type=["csv", "xlsx", "xls", "json", "txt", "dat", "h5", "hdf5", "db", "sqlite", "sqlite3", "mat"],
            help="Supports: Excel (.xlsx, .xls), CSV (.csv), JSON (.json), Text (.txt, .dat), HDF5 (.h5, .hdf5), SQLite (.db, .sqlite), MATLAB (.mat)"
        )
        
        if uploaded_file is not None:
            # Read the file
            try:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                is_excel = file_extension in ['xlsx', 'xls']
                is_multi_sheet = False
                all_sheets = []
                channel_sheets = []
                selected_sheets = []
                sheets_data = {}
                
                # Initialize metadata
                metadata = {}
                
                if file_extension == 'csv':
                    df_original = read_file_universal(uploaded_file, file_type='csv')
                    st.session_state.file_type = 'csv'
                elif file_extension in ['json']:
                    df_original = read_file_universal(uploaded_file, file_type='json')
                    st.session_state.file_type = 'json'
                elif file_extension in ['txt', 'dat']:
                    df_original = read_file_universal(uploaded_file, file_type='txt')
                    st.session_state.file_type = 'txt'
                elif file_extension in ['h5', 'hdf5']:
                    # HDF5 requires file path, not buffer - save temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    try:
                        df_original = read_file_universal(tmp_path, file_type='hdf5')
                        st.session_state.file_type = 'hdf5'
                    finally:
                        os.unlink(tmp_path)  # Clean up temp file
                elif file_extension in ['db', 'sqlite', 'sqlite3']:
                    # SQLite requires file path, not buffer - save temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    try:
                        df_original = read_file_universal(tmp_path, file_type='sql')
                        st.session_state.file_type = 'sql'
                    finally:
                        os.unlink(tmp_path)  # Clean up temp file
                elif file_extension == 'mat':
                    # MATLAB files - save temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    try:
                        df_original = read_file_universal(tmp_path, file_type='mat')
                        st.session_state.file_type = 'mat'
                    finally:
                        os.unlink(tmp_path)  # Clean up temp file
                elif is_excel:
                    try:
                        # Try importing openpyxl first
                        import openpyxl
                        from cleaning_module import extract_excel_metadata
                        
                        # Extract metadata from Excel file (Issue 5 Fix)
                        uploaded_file.seek(0)  # Reset file pointer
                        metadata = extract_excel_metadata(uploaded_file)
                        st.session_state.file_metadata = metadata  # Store in session state
                        uploaded_file.seek(0)  # Reset again for reading
                        
                        # Check if Excel file has multiple sheets
                        all_sheets = get_excel_sheets(uploaded_file)
                        channel_sheets = filter_channel_sheets(all_sheets)
                        
                        if len(all_sheets) > 1:
                            is_multi_sheet = True
                            st.session_state.file_type = 'excel_multi'
                            st.session_state.all_sheets = all_sheets
                            st.session_state.channel_sheets = channel_sheets
                            
                            # Show sheet information
                            st.info(f"Found {len(all_sheets)} sheet(s): {', '.join(all_sheets)}")
                            if 'info' in [s.lower() for s in all_sheets]:
                                st.info("'info' sheet detected and will be excluded from processing.")
                            
                            if len(channel_sheets) > 0:
                                st.success(f"Found {len(channel_sheets)} Channel sheet(s): {', '.join(channel_sheets)}")
                                
                                # Sheet selection
                                st.subheader("Sheet Selection")
                                selected_sheets = st.multiselect(
                                    "Select sheets to process",
                                    options=channel_sheets,
                                    default=channel_sheets,  # Select all Channel sheets by default
                                    help="Select which Channel sheets to clean and combine"
                                )
                                
                                if len(selected_sheets) == 0:
                                    st.warning("Please select at least one sheet to process.")
                                    st.stop()
                                
                                # Read selected sheets
                                uploaded_file.seek(0)  # Reset file pointer
                                sheets_data = read_excel_sheets(uploaded_file, selected_sheets)
                                
                                # Combine selected sheets
                                dfs_to_combine = []
                                for sheet_name in selected_sheets:
                                    if sheet_name in sheets_data:
                                        df_sheet = sheets_data[sheet_name].copy()
                                        # Add a column to identify the source sheet
                                        df_sheet.insert(0, 'Sheet_Source', sheet_name)
                                        dfs_to_combine.append(df_sheet)
                                
                                if len(dfs_to_combine) > 0:
                                    df_original = pd.concat(dfs_to_combine, ignore_index=True)
                                    st.success(f"Combined {len(selected_sheets)} sheet(s): {', '.join(selected_sheets)}")
                                else:
                                    st.error("No data found in selected sheets.")
                                    st.stop()
                            else:
                                st.warning("No 'Channel_' sheets found. Processing all sheets except 'info'.")
                                # Process all sheets except 'info'
                                sheets_to_process = [s for s in all_sheets if s.lower() != 'info']
                                if len(sheets_to_process) == 0:
                                    st.error("No sheets available for processing.")
                                    st.stop()
                                
                                selected_sheets = st.multiselect(
                                    "Select sheets to process",
                                    options=sheets_to_process,
                                    default=sheets_to_process,
                                    help="Select which sheets to clean and combine"
                                )
                                
                                if len(selected_sheets) == 0:
                                    st.warning("Please select at least one sheet to process.")
                                    st.stop()
                                
                                uploaded_file.seek(0)
                                sheets_data = read_excel_sheets(uploaded_file, selected_sheets)
                                
                                dfs_to_combine = []
                                for sheet_name in selected_sheets:
                                    if sheet_name in sheets_data:
                                        df_sheet = sheets_data[sheet_name].copy()
                                        df_sheet.insert(0, 'Sheet_Source', sheet_name)
                                        dfs_to_combine.append(df_sheet)
                                
                                if len(dfs_to_combine) > 0:
                                    df_original = pd.concat(dfs_to_combine, ignore_index=True)
                                    st.success(f"Combined {len(selected_sheets)} sheet(s): {', '.join(selected_sheets)}")
                                else:
                                    st.error("No data found in selected sheets.")
                                    st.stop()
                        else:
                            # Single sheet Excel file
                            df_original = pd.read_excel(uploaded_file)
                            st.session_state.file_type = 'excel_single'
                    except ImportError as e:
                        st.error(f"Excel file support requires 'openpyxl' package. Error: {str(e)}")
                        st.info("**To fix this:**\n1. Make sure you're using the virtual environment Python\n2. Install with: `.venv\\Scripts\\python.exe -m pip install openpyxl`\n3. Or activate venv first: `.venv\\Scripts\\Activate.ps1` then `pip install openpyxl`")
                        st.stop()
                    except Exception as e:
                        pass

                        st.error(f"Error reading Excel file: {str(e)}")
                        st.info("**Troubleshooting:**\n- Make sure the file is a valid Excel file (.xlsx or .xls)\n- Check if the file is corrupted\n- Try opening it in Excel first to verify it's valid")
                        st.stop()
                else:
                    st.error(f"Unsupported file format: {file_extension}. Supported formats: CSV, Excel (.xlsx, .xls), JSON, Text (.txt, .dat), HDF5 (.h5, .hdf5), SQLite (.db, .sqlite), MATLAB (.mat)")
                    st.stop()
                
                st.success(f"File loaded successfully! Shape: {df_original.shape[0]} rows × {df_original.shape[1]} columns")
                
                # Store original dataframe and info in session state
                st.session_state.df_original = df_original
                info_original = get_dataframe_info(df_original)
                st.session_state.info_original = info_original
                
                # Display original data info
                with st.expander("Original Data Information", expanded=False):
                    st.write(f"**Shape:** {info_original['shape'][0]} rows × {info_original['shape'][1]} columns")
                    st.write(f"**Memory Usage:** {info_original['memory_usage']:.2f} MB")
                    st.write(f"**Duplicate Rows:** {info_original['duplicate_rows']}")
                    
                    if info_original['null_counts']:
                        st.write("**Missing Values:**")
                        null_df = pd.DataFrame({
                            'Column': list(info_original['null_counts'].keys()),
                            'Null Count': list(info_original['null_counts'].values()),
                            'Null %': [f"{info_original['null_percentages'][k]:.2f}%" 
                                       for k in info_original['null_counts'].keys()]
                        })
                        st.dataframe(null_df, use_container_width=True)
                
                # Preview original data
                with st.expander("Preview Original Data", expanded=False):
                    st.dataframe(df_original.head(10), use_container_width=True)
                
                # Cleaning options
                st.subheader("Cleaning Options")
                
                # Basic options
                col1, col2 = st.columns(2)
                
                with col1:
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
                    remove_empty_rows = st.checkbox("Remove completely empty rows", value=True)
                    remove_empty_cols = st.checkbox("Remove completely empty columns", value=True)
                    standardize_names = st.checkbox("Standardize column names (lowercase, underscores)", value=False)
                    trim_whitespace = st.checkbox("Trim whitespace in text columns", value=True)
                    convert_numeric = st.checkbox("Convert columns to numeric where possible", value=False)
                
                with col2:
                    fill_na_method = st.selectbox(
                        "Handle missing values",
                        options=['drop', 'forward', 'backward', 'interpolate', 'mean', 'median', 'zero', 'custom'],
                        index=0,
                        help="Choose how to handle NaN values. 'interpolate' only fills small gaps (<5 samples)."
                    )
                    
                    fill_na_value = None
                    if fill_na_method == 'custom':
                        fill_na_value = st.number_input("Custom fill value", value=0.0)
                    
                    gap_threshold = st.number_input("Gap threshold (samples)", min_value=1, value=5, 
                                                    help="Gaps larger than this won't be interpolated")
                    
                    remove_outliers = st.checkbox("Remove outliers (statistical)", value=False)
                    outlier_method = None
                    if remove_outliers:
                        outlier_method = st.selectbox(
                            "Outlier detection method",
                            options=['iqr', 'zscore'],
                            index=0,
                            help="IQR: Interquartile Range method, Z-score: Standard deviation method"
                        )
                
                harmonize_schema = st.checkbox("Harmonize to standard schema", value=True)
                detect_cycler = st.checkbox("Auto-detect cycler format", value=True)
                
                col_phase2_1, col_phase2_2 = st.columns(2)
                
                with col_phase2_1:
                    segment_steps = st.checkbox("Segment steps and cycles", value=True)
                    assign_step_types = st.checkbox("Assign step types", value=True)
                    identify_rpt = st.checkbox("Identify RPT cycles", value=True)
                
                with col_phase2_2:
                    detect_anomalies = st.checkbox("Detect statistical anomalies", value=True)
                    anomaly_method = 'zscore'
                    anomaly_window_size = 10
                    if detect_anomalies:
                        anomaly_method = st.selectbox(
                            "Anomaly detection method",
                            options=['zscore', 'lof'],
                            index=0,
                            help="Z-Score: Moving window statistical method. LOF: Local Outlier Factor (requires scikit-learn)"
                        )
                        anomaly_window_size = st.number_input("Anomaly window size", min_value=5, value=10, step=1)
                    
                    correct_anomalies = st.checkbox("Correct detected anomalies", value=True)
                    correction_method = 'savgol'
                    if correct_anomalies:
                        correction_method = st.selectbox(
                            "Correction method",
                            options=['savgol', 'interpolate', 'median_filter'],
                            index=0,
                            help="Savitzky-Golay: Best for preserving curve shape. Interpolate: Linear interpolation. Median: Simple median filter"
                        )
                
                col_missing_1, col_missing_2 = st.columns(2)
                with col_missing_1:
                    detect_missing_data = st.checkbox("Detect missing data gaps", value=True)
                    max_gap_seconds = 60.0
                    if detect_missing_data:
                        max_gap_seconds = st.number_input("Max gap (seconds)", min_value=1.0, value=60.0, step=1.0)
                
                with col_missing_2:
                    impute_missing = st.checkbox("Impute missing data", value=True)
                    imputation_method = 'linear'
                    if impute_missing:
                        imputation_method = st.selectbox(
                            "Imputation method",
                            options=['linear', 'cubic', 'forward_fill'],
                            index=0,
                            help="Linear: Fast interpolation. Cubic: Smoother curves. Forward fill: Simple fill"
                        )
                
                verify_electrochemical = st.checkbox("Verify electrochemical anomalies (CE, Capacity Drift)", value=True)
                
                col_phase3_1, col_phase3_2 = st.columns(2)
                
                with col_phase3_1:
                    resample_capacity_axis = st.checkbox("Resample to uniform capacity axis", value=False)
                    capacity_points = 1000
                    if resample_capacity_axis:
                        capacity_points = st.number_input("Capacity grid points", min_value=100, value=1000, step=100)
                    
                    resample_voltage_axis = st.checkbox("Resample to uniform voltage axis", value=False)
                    voltage_points = 1000
                    if resample_voltage_axis:
                        voltage_points = st.number_input("Voltage grid points", min_value=100, value=1000, step=100)
                
                with col_phase3_2:
                    resample_uniform_frequency = st.checkbox("Resample to uniform frequency (1 Hz)", value=False,
                                                             help="Essential for ICA/dQ/dV analysis. Resamples data to uniform time spacing.")
                    frequency_hz = 1.0
                    if resample_uniform_frequency:
                        frequency_hz = st.number_input("Sampling frequency (Hz)", min_value=0.1, value=1.0, step=0.1,
                                                       help="Target sampling frequency. Default 1 Hz = 1 sample per second.")
                    
                    smooth_for_derivatives = st.checkbox("Smooth for derivative calculation (Savitzky-Golay)", value=True,
                                                        help="Removes spikes and noise. Enabled by default for better data quality.")
                    savgol_window = 51
                    savgol_polyorder = 3
                    if smooth_for_derivatives:
                        savgol_window = st.number_input("Savitzky-Golay window", min_value=5, value=51, step=2, 
                                                       help="Must be odd number. Larger window = more smoothing (default: 51)")
                        savgol_polyorder = st.number_input("Polynomial order", min_value=1, max_value=5, value=3)
                
                # Range filtering options - initialize defaults
                apply_range_filters = False
                range_filters = {}
                
                with st.expander("Value Range Filtering (Set Out-of-Range Values to 0)", expanded=False):
                    st.markdown("""
                    **Battery Data Range Filtering:**
                    - Select columns and set acceptable value ranges
                    - Values outside the range will be replaced with 0 (or a custom value)
                    - Use battery-specific defaults or set custom ranges
                    """)
                    
                    apply_range_filters = st.checkbox("Apply range filters (replace out-of-range with 0)", value=False)
                    
                    if apply_range_filters:
                        # Get numeric columns
                        numeric_cols = [c for c in df_original.columns 
                                       if pd.api.types.is_numeric_dtype(df_original[c])]
                        
                        if numeric_cols:
                            # Import battery defaults
                            from cleaning_module import get_battery_default_ranges
                            battery_defaults = get_battery_default_ranges()
                            
                            # Column selection
                            selected_filter_cols = st.multiselect(
                                "Select columns to apply range filters",
                                options=numeric_cols,
                                default=[],
                                help="Choose which columns to filter. Values outside range will be set to 0."
                            )
                            
                            if selected_filter_cols:
                                st.markdown("---")
                                st.markdown("**Set Range for Each Column**")
                                
                                # Quick apply battery defaults button
                                if st.button("Apply Battery Default Ranges", help="Apply sensible defaults for common battery parameters"):
                                    st.info("Battery defaults applied! Adjust ranges below if needed.")
                                
                                for col in selected_filter_cols:
                                    # Try to match column to battery defaults
                                    col_lower = col.lower()
                                    default_range = None
                                    default_unit = ""
                                    
                                    for pattern, default in battery_defaults.items():
                                        if pattern in col_lower:
                                            default_range = default
                                            default_unit = default.get('unit', '')
                                            break
                                    
                                    # Get current data range
                                    col_min_data = float(df_original[col].min()) if len(df_original) > 0 else 0.0
                                    col_max_data = float(df_original[col].max()) if len(df_original) > 0 else 100.0
                                    
                                    # Use battery default if available, otherwise use data range
                                    if default_range:
                                        default_min = default_range['min']
                                        default_max = default_range['max']
                                        default_replace = default_range.get('replace_with', 0.0)
                                        st.markdown(f"**{col}** {default_unit} (Battery default range)")
                                    else:
                                        default_min = col_min_data
                                        default_max = col_max_data
                                        default_replace = 0.0
                                        st.markdown(f"**{col}**")
                                    
                                    # Create columns for inputs
                                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                                    
                                    with filter_col1:
                                        min_val = st.number_input(
                                            f"Min value",
                                            value=default_min,
                                            key=f"range_min_{col}",
                                            step=0.01 if abs(default_max - default_min) < 100 else 1.0,
                                            help=f"Data range: {col_min_data:.4f} to {col_max_data:.4f}"
                                        )
                                    
                                    with filter_col2:
                                        max_val = st.number_input(
                                            f"Max value",
                                            value=default_max,
                                            key=f"range_max_{col}",
                                            step=0.01 if abs(default_max - default_min) < 100 else 1.0,
                                            help=f"Data range: {col_min_data:.4f} to {col_max_data:.4f}"
                                        )
                                    
                                    with filter_col3:
                                        replace_val = st.number_input(
                                            f"Replace with",
                                            value=default_replace,
                                            key=f"range_replace_{col}",
                                            step=0.01,
                                            help="Value to use for out-of-range data (default: 0)"
                                        )
                                    
                                    # Show data statistics
                                    out_of_range_count = 0
                                    if len(df_original) > 0:
                                        out_of_range_mask = (
                                            (df_original[col] < min_val) | 
                                            (df_original[col] > max_val)
                                        )
                                        out_of_range_count = out_of_range_mask.sum()
                                        out_of_range_pct = (out_of_range_count / len(df_original)) * 100
                                        
                                        if out_of_range_count > 0:
                                            st.warning(f"{out_of_range_count} values ({out_of_range_pct:.1f}%) will be replaced with {replace_val}")
                                        else:
                                            st.success(f"All values are within range")
                                    
                                    range_filters[col] = {
                                        'min': min_val, 
                                        'max': max_val,
                                        'replace_with': replace_val
                                    }
                                    
                                    st.markdown("---")
                        else:
                            st.info("No numeric columns found for range filtering.")
                
                
                # Clean button
                if st.button("Clean Data", type="primary", use_container_width=True):
                    with st.spinner("Cleaning data..."):
                        # Use stored dataframe if available, otherwise use current
                        df_to_clean = st.session_state.get('df_original', df_original)
                        
                        # Get metadata if available (stored in session state or local variable)
                        file_metadata = st.session_state.get('file_metadata', {})
                        if not file_metadata and 'metadata' in locals():
                            file_metadata = metadata
                        
                        cleaning_options = {
                            # Basic options
                            'remove_duplicates': remove_duplicates,
                            'remove_empty_rows': remove_empty_rows,
                            'remove_empty_cols': remove_empty_cols,
                            'fill_na_method': fill_na_method,
                            'fill_na_value': fill_na_value,
                            'gap_threshold': gap_threshold,
                            'remove_outliers': remove_outliers,
                            'outlier_method': outlier_method,
                            'standardize_names': standardize_names,
                            'trim_whitespace': trim_whitespace,
                            'convert_numeric': convert_numeric,
                            # Phase 1: Data Ingestion and Harmonization
                            'harmonize_schema': harmonize_schema,
                            'detect_cycler': detect_cycler,
                            # Phase 2: Anomaly Detection and Correction
                            'segment_steps': segment_steps,
                            'assign_step_types': assign_step_types,
                            'identify_rpt': identify_rpt,
                            'detect_anomalies': detect_anomalies,
                            'anomaly_method': anomaly_method,
                            'anomaly_window_size': anomaly_window_size,
                            'correct_anomalies': correct_anomalies,
                            'correction_method': correction_method,
                            'detect_missing_data': detect_missing_data,
                            'max_gap_seconds': max_gap_seconds,
                            'impute_missing': impute_missing,
                            'imputation_method': imputation_method,
                            'verify_electrochemical': verify_electrochemical,
                            # Phase 3: Preprocessing for Feature Extraction
                            'resample_uniform_frequency': resample_uniform_frequency,
                            'frequency_hz': frequency_hz,
                            'resample_capacity_axis': resample_capacity_axis,
                            'capacity_points': capacity_points,
                            'resample_voltage_axis': resample_voltage_axis,
                            'voltage_points': voltage_points,
                            'smooth_for_derivatives': smooth_for_derivatives,
                            'savgol_window': savgol_window,
                            'savgol_polyorder': savgol_polyorder,
                            'savgol_apply_to_original': True,  # Apply smoothing to original columns
                            # Range filtering
                            'apply_range_filters': apply_range_filters,
                            'range_filters': range_filters
                        }
                        
                        # Get metadata if available
                        file_metadata = metadata if 'metadata' in locals() else {}
                        
                        df_cleaned = clean_dataframe(df_to_clean, cleaning_options, metadata=file_metadata)
                        
                        # Store in session state for download
                        st.session_state.cleaned_df = df_cleaned
                        st.session_state.cleaning_applied = True
                
                # Display cleaned data if available
                if st.session_state.get('cleaning_applied', False) and 'cleaned_df' in st.session_state:
                    df_cleaned = st.session_state.cleaned_df
                    info_original = st.session_state.get('info_original', {})
                    
                    st.subheader("Cleaned Data")
                    
                    # Display cleaned data info
                    with st.expander("Cleaned Data Information", expanded=True):
                        info_cleaned = get_dataframe_info(df_cleaned)
                        st.write(f"**Shape:** {info_cleaned['shape'][0]} rows × {info_cleaned['shape'][1]} columns")
                        st.write(f"**Memory Usage:** {info_cleaned['memory_usage']:.2f} MB")
                        st.write(f"**Duplicate Rows:** {info_cleaned['duplicate_rows']}")
                        
                        # Show changes
                        if info_original and 'shape' in info_original:
                            rows_removed = info_original['shape'][0] - info_cleaned['shape'][0]
                            cols_removed = info_original['shape'][1] - info_cleaned['shape'][1]
                            if rows_removed > 0 or cols_removed > 0:
                                st.info(f"Removed {rows_removed} rows and {cols_removed} columns")
                        
                        if info_cleaned['null_counts']:
                            st.write("**Missing Values (after cleaning):**")
                            null_df = pd.DataFrame({
                                'Column': list(info_cleaned['null_counts'].keys()),
                                'Null Count': list(info_cleaned['null_counts'].values()),
                                'Null %': [f"{info_cleaned['null_percentages'][k]:.2f}%" 
                                           for k in info_cleaned['null_counts'].keys()]
                            })
                            st.dataframe(null_df, use_container_width=True)
                    
                    # Preview cleaned data
                    with st.expander("Preview Cleaned Data", expanded=True):
                        st.dataframe(df_cleaned.head(10), use_container_width=True)
                    
                    # Compatibility validation
                    compatibility = validate_analysis_compatibility(df_cleaned)
                    
                    # Download section
                    st.subheader("Download Cleaned Data")
                    
                    # Show compatibility status
                    if compatibility['is_compatible']:
                        st.success(f"Compatible with Analysis Module (Format: {compatibility['format_type'].replace('_', ' ').title()})")
                    else:
                        st.warning("Data format may not be recognized by Analysis Module. Ensure you have required columns (cycle/discharge_capacity or Voltage/Capacity).")
                    
                    # Show warnings if any
                    if compatibility['warnings']:
                        for warning in compatibility['warnings']:
                            st.info(f"{warning}")
                    
                    # Download options
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        remove_sheet_source = st.checkbox(
                            "Remove 'Sheet_Source' column for analysis compatibility",
                            value=('Sheet_Source' in df_cleaned.columns),
                            help="The Sheet_Source column (added when combining multiple sheets) may interfere with analysis. Remove it for better compatibility."
                        )
                    
                    with col_dl2:
                        download_format = st.radio(
                            "Download format",
                            options=["CSV", "Excel"],
                            index=0,
                            horizontal=True
                        )
                    
                    # Generate download data
                    if download_format == "CSV":
                        csv_data = to_csv_download(df_cleaned, "cleaned_data.csv", remove_sheet_source=remove_sheet_source)
                        st.download_button(
                            label="Download Cleaned CSV",
                            data=csv_data,
                            file_name="cleaned_data.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    elif download_format == "Excel":
                        # Create Excel file in memory
                        excel_buffer = BytesIO()
                        df_export = df_cleaned.copy()
                        if remove_sheet_source and 'Sheet_Source' in df_export.columns:
                            df_export = df_export.drop(columns=['Sheet_Source'])
                        
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df_export.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="Download Cleaned Excel",
                            data=excel_buffer.getvalue(),
                            file_name="cleaned_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True
                        )
                    else:  # MAT format (cell11dataset_for_python.mat)
                        try:
                            mat_filename = f"{cell_id}dataset_for_python.mat"
                            mat_data = to_mat_download(df_cleaned, mat_filename, cell_id=cell_id)
                            st.download_button(
                                label=f"Download {mat_filename}",
                                data=mat_data,
                                file_name=mat_filename,
                                mime="application/octet-stream",
                                type="primary",
                                use_container_width=True
                            )
                            st.info(f"Data will be saved in cell11dataset_for_python.mat format with cell ID: {cell_id}")
                        except ImportError:
                            st.error("SciPy is required for .mat export. Install with: pip install scipy")
                        except Exception as e:
                            st.error(f"Error creating .mat file: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
        else:
            st.info(" Please upload a CSV or Excel file to get started with data cleaning.")
    
    with col_chat:
        render_copilot(context_key="tab3", default_context="Analytics")
