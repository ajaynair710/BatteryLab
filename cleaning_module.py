"""
Advanced Battery Data Cleaning Module for BatteryLab
Implements comprehensive 3-phase cleaning pipeline for cycler data
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Optional imports with fallbacks
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    def fuzz_ratio(s1, s2):
        s1_l, s2_l = str(s1).lower(), str(s2).lower()
        if s1_l == s2_l:
            return 100.0
        if s1_l in s2_l or s2_l in s1_l:
            return 80.0
        return 0.0
    
    def process_extractOne(query, choices, scorer=None):
        best_match = None
        best_score = 0.0
        for choice in choices:
            score = fuzz_ratio(query, choice)
            if score > best_score:
                best_score = score
                best_match = choice
        return (best_match, best_score) if best_match else (None, 0.0)
    
    class MockFuzz:
        ratio = staticmethod(fuzz_ratio)
    fuzz = MockFuzz()
    process = type('MockProcess', (), {'extractOne': staticmethod(process_extractOne)})()

try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# PHASE 1: Data Ingestion and Harmonization
# ============================================================================

STANDARD_SCHEMA = {
    'DateTime': 'datetime',
    'Cycle_Index': 'int',
    'Step_Index': 'int',
    'Current': 'float',
    'Voltage': 'float',
    'Capacity': 'float',
    'Temperature': 'float',
    'Time': 'float'
}

CYCLER_PARSERS = {
    'arbin': {
        'cycle_cols': ['Cycle_Index', 'Cycle', 'Cycle_Number'],
        'step_cols': ['Step_Index', 'Step', 'Step_Number'],
        'current_cols': ['Current', 'I', 'Current(A)', 'Current_A'],
        'voltage_cols': ['Voltage', 'V', 'Voltage(V)', 'Voltage_V'],
        'capacity_cols': ['Capacity', 'Q', 'Capacity(Ah)', 'Capacity_Ah'],
        'temp_cols': ['Temperature', 'Temp', 'Temperature(C)', 'Temperature_C'],
        'time_cols': ['Test_Time', 'Time', 'Test_Time(s)', 'Time(s)']
    },
    'biologic': {
        'cycle_cols': ['cycle number', 'Cycle', 'N'],
        'step_cols': ['step number', 'Step', 'Ns'],
        'current_cols': ['I/mA', 'I', 'Current'],
        'voltage_cols': ['Ewe/V', 'Ewe', 'Voltage'],
        'capacity_cols': ['Q charge/discharge/mAh', 'Q', 'Capacity'],
        'temp_cols': ['T', 'Temperature'],
        'time_cols': ['time/s', 'Time', 't']
    },
    'neware': {
        'cycle_cols': ['Cycle', 'Cycle_Index', 'Cycle Number'],
        'step_cols': ['Step', 'Step_Index', 'Step Number'],
        'current_cols': ['Current', 'I', 'Current(A)'],
        'voltage_cols': ['Voltage', 'V', 'Voltage(V)'],
        'capacity_cols': ['Capacity', 'Q', 'Capacity(Ah)'],
        'temp_cols': ['Temperature', 'Temp'],
        'time_cols': ['Time', 'Test_Time', 'Time(s)']
    },
    'maccor': {
        'cycle_cols': ['Cycle', 'Cyc', 'Cycle Number'],
        'step_cols': ['Step', 'Step Number'],
        'current_cols': ['Amps', 'Current', 'I'],
        'voltage_cols': ['Volts', 'Voltage', 'V'],
        'capacity_cols': ['Amp-hrs', 'Capacity', 'Q'],
        'temp_cols': ['Temp', 'Temperature'],
        'time_cols': ['Time', 'Test Time', 'Time(s)']
    },
    'solartron': {
        'cycle_cols': ['Cycle', 'Cycle Number'],
        'step_cols': ['Step', 'Step Number'],
        'current_cols': ['Current', 'I'],
        'voltage_cols': ['Voltage', 'V'],
        'capacity_cols': ['Capacity', 'Q'],
        'temp_cols': ['Temperature', 'Temp'],
        'time_cols': ['Time', 'Test Time']
    }
}


def detect_cycler_format(df: pd.DataFrame) -> str:
    """
    Detect which cycler format the data is from.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    str
        Detected cycler format ('arbin', 'biologic', 'neware', 'maccor', 'solartron', 'unknown')
    """
    cols_l = [c.lower() for c in df.columns]
    
    for cycler, patterns in CYCLER_PARSERS.items():
        matches = 0
        for key, col_list in patterns.items():
            for col_pattern in col_list:
                if any(col_pattern.lower() in c for c in cols_l):
                    matches += 1
                    break
        if matches >= 4:  # At least 4 out of 7 categories match
            return cycler
    
    return 'unknown'


def harmonize_to_standard_schema(df: pd.DataFrame, cycler_format: str = None) -> pd.DataFrame:
    """
    Map cycler-specific columns to standardized schema.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    cycler_format : str, optional
        Cycler format. If None, auto-detects.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with standardized column names
    """
    if cycler_format is None:
        cycler_format = detect_cycler_format(df)
    
    df_clean = df.copy()
    cols_l = [c.lower() for c in df_clean.columns]
    mapping = {}
    
    if cycler_format != 'unknown' and cycler_format in CYCLER_PARSERS:
        patterns = CYCLER_PARSERS[cycler_format]
        
        # Map cycle
        for pattern in patterns['cycle_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Cycle_Index'
                    break
            if 'Cycle_Index' in mapping.values():
                break
        
        # Map step
        for pattern in patterns['step_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Step_Index'
                    break
            if 'Step_Index' in mapping.values():
                break
        
        # Map current
        for pattern in patterns['current_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Current'
                    break
            if 'Current' in mapping.values():
                break
        
        # Map voltage
        for pattern in patterns['voltage_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Voltage'
                    break
            if 'Voltage' in mapping.values():
                break
        
        # Map capacity
        for pattern in patterns['capacity_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l or ('capacity' in col_l and 'discharge' not in col_l):
                    mapping[df_clean.columns[i]] = 'Capacity'
                    break
            if 'Capacity' in mapping.values():
                break
        
        # Map temperature
        for pattern in patterns['temp_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Temperature'
                    break
            if 'Temperature' in mapping.values():
                break
        
        # Map time
        for pattern in patterns['time_cols']:
            for i, col_l in enumerate(cols_l):
                if pattern.lower() in col_l:
                    mapping[df_clean.columns[i]] = 'Time'
                    break
            if 'Time' in mapping.values():
                break
    
    # Apply mapping
    df_clean = df_clean.rename(columns=mapping)
    
    # Normalize time to seconds
    if 'Time' in df_clean.columns:
        df_clean = normalize_time_to_seconds(df_clean)
    
    # Ensure datetime column exists
    if 'DateTime' not in df_clean.columns:
        if 'Time' in df_clean.columns:
            df_clean['DateTime'] = pd.to_datetime(df_clean.index, errors='coerce')
        else:
            df_clean['DateTime'] = pd.Timestamp.now()
    
    return df_clean


def normalize_time_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize time column to seconds relative to test start.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Time' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized time in seconds
    """
    df_clean = df.copy()
    
    if 'Time' in df_clean.columns:
        time_col = df_clean['Time']
        
        # Detect time unit from column name or values
        time_unit = 'seconds'
        if time_col.max() > 86400:  # Likely in seconds already
            time_unit = 'seconds'
        elif time_col.max() > 1440:  # Likely in minutes
            time_unit = 'minutes'
        elif time_col.max() > 24:  # Likely in hours
            time_unit = 'hours'
        
        # Convert to seconds
        if time_unit == 'minutes':
            df_clean['Time'] = time_col * 60
        elif time_unit == 'hours':
            df_clean['Time'] = time_col * 3600
        
        # Normalize to start at 0
        df_clean['Time'] = df_clean['Time'] - df_clean['Time'].min()
    
    return df_clean


# ============================================================================
# PHASE 2: Anomaly Detection and Correction
# ============================================================================

def segment_steps_and_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and correct step boundaries and cycle isolation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with standardized schema
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected step boundaries and cycle isolation
    """
    df_clean = df.copy()
    
    if 'Step_Index' not in df_clean.columns or 'Current' not in df_clean.columns:
        return df_clean
    
    # Sort by time
    if 'Time' in df_clean.columns:
        df_clean = df_clean.sort_values('Time').reset_index(drop=True)
    
    # Detect step boundaries based on current changes
    if 'Current' in df_clean.columns:
        current_diff = df_clean['Current'].diff().abs()
        step_threshold = current_diff.quantile(0.95)  # 95th percentile as threshold
        
        # Identify step boundaries
        step_boundaries = current_diff > step_threshold
        
        # Correct Step_Index if missing or incorrect
        if 'Step_Index' in df_clean.columns:
            step_idx = 1
            for i in range(1, len(df_clean)):
                if step_boundaries.iloc[i]:
                    step_idx += 1
                df_clean.loc[i, 'Step_Index'] = step_idx
        else:
            step_idx = 1
            step_indices = [1]
            for i in range(1, len(df_clean)):
                if step_boundaries.iloc[i]:
                    step_idx += 1
                step_indices.append(step_idx)
            df_clean['Step_Index'] = step_indices
    
    # Correct Cycle_Index if missing
    if 'Cycle_Index' not in df_clean.columns and 'Step_Index' in df_clean.columns:
        # Assume cycle changes when step resets to 1
        cycle_idx = 1
        cycle_indices = [1]
        prev_step = df_clean['Step_Index'].iloc[0]
        
        for i in range(1, len(df_clean)):
            current_step = df_clean['Step_Index'].iloc[i]
            if current_step < prev_step:  # Step reset indicates new cycle
                cycle_idx += 1
            cycle_indices.append(cycle_idx)
            prev_step = current_step
        
        df_clean['Cycle_Index'] = cycle_indices
    
    return df_clean


def assign_step_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign step types based on current and voltage patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Step_Type' column
    """
    df_clean = df.copy()
    
    if 'Current' not in df_clean.columns or 'Voltage' not in df_clean.columns:
        return df_clean
    
    step_types = []
    
    for idx, row in df_clean.iterrows():
        current = row.get('Current', 0)
        voltage = row.get('Voltage', 0)
        
        if abs(current) < 0.01:  # Near zero current
            step_type = 'Rest'
        elif current > 0.01:  # Positive current
            if voltage > 4.0:  # High voltage
                step_type = 'Constant Voltage Charge'
            else:
                step_type = 'Constant Current Charge'
        elif current < -0.01:  # Negative current
            if voltage < 2.5:  # Low voltage
                step_type = 'Constant Voltage Discharge'
            else:
                step_type = 'Constant Current Discharge'
        else:
            step_type = 'Unknown'
        
        step_types.append(step_type)
    
    df_clean['Step_Type'] = step_types
    return df_clean


def identify_rpt_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Reference Performance Test (RPT) cycles.
    RPTs are typically longer cycles with different protocols.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Is_RPT' boolean column
    """
    df_clean = df.copy()
    
    if 'Cycle_Index' not in df_clean.columns or 'Time' not in df_clean.columns:
        df_clean['Is_RPT'] = False
        return df_clean
    
    # Calculate cycle duration
    cycle_durations = df_clean.groupby('Cycle_Index')['Time'].agg(['min', 'max'])
    cycle_durations['duration'] = cycle_durations['max'] - cycle_durations['min']
    
    # RPT cycles are typically longer (outliers in duration)
    median_duration = cycle_durations['duration'].median()
    std_duration = cycle_durations['duration'].std()
    rpt_threshold = median_duration + 2 * std_duration
    
    # Mark RPT cycles
    is_rpt = df_clean['Cycle_Index'].map(
        lambda x: cycle_durations.loc[x, 'duration'] > rpt_threshold if x in cycle_durations.index else False
    )
    
    df_clean['Is_RPT'] = is_rpt
    return df_clean


def detect_statistical_anomalies(df: pd.DataFrame, window_size: int = 10, method: str = 'zscore') -> pd.DataFrame:
    """
    Detect statistical anomalies using LOF or Z-Score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    window_size : int
        Window size for moving window analysis
    method : str
        'zscore' or 'lof' (Local Outlier Factor)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Anomaly_Flag' column
    """
    df_clean = df.copy()
    df_clean['Anomaly_Flag'] = False
    
    numeric_cols = ['Voltage', 'Current', 'Capacity', 'Temperature']
    available_cols = [c for c in numeric_cols if c in df_clean.columns]
    
    for col in available_cols:
        if method == 'zscore' and len(df_clean) > window_size:
            # Z-Score method on moving window
            rolling_mean = df_clean[col].rolling(window=window_size, center=True).mean()
            rolling_std = df_clean[col].rolling(window=window_size, center=True).std()
            z_scores = (df_clean[col] - rolling_mean) / (rolling_std + 1e-10)
            
            # Flag outliers (|z| > 3)
            df_clean.loc[abs(z_scores) > 3, 'Anomaly_Flag'] = True
        
        elif method == 'lof' and SKLEARN_AVAILABLE and len(df_clean) > 20:
            # Local Outlier Factor method
            try:
                # Use sliding window approach for time series
                for i in range(window_size, len(df_clean) - window_size):
                    window_data = df_clean[col].iloc[i-window_size:i+window_size].values.reshape(-1, 1)
                    if len(window_data) > 5:
                        lof = LocalOutlierFactor(n_neighbors=min(5, len(window_data)-1))
                        pred = lof.fit_predict(window_data)
                        if pred[window_size] == -1:  # Outlier detected
                            df_clean.loc[i, 'Anomaly_Flag'] = True
            except:
                pass
    
    return df_clean


def correct_anomalies(df: pd.DataFrame, method: str = 'savgol') -> pd.DataFrame:
    """
    Correct detected anomalies using smoothing or interpolation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Anomaly_Flag' column
    method : str
        'savgol', 'interpolate', or 'median_filter'
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected values
    """
    df_clean = df.copy()
    
    if 'Anomaly_Flag' not in df_clean.columns:
        return df_clean
    
    numeric_cols = ['Voltage', 'Current', 'Capacity', 'Temperature']
    available_cols = [c for c in numeric_cols if c in df_clean.columns]
    
    for col in available_cols:
        anomaly_mask = df_clean['Anomaly_Flag']
        
        if method == 'savgol' and SCIPY_AVAILABLE:
            # Savitzky-Golay filter for small anomalies
            if anomaly_mask.sum() < len(df_clean) * 0.1:  # Less than 10% anomalies
                try:
                    window_length = min(11, len(df_clean) // 10 * 2 + 1)
                    if window_length >= 5:
                        polyorder = min(3, window_length // 2)
                        smoothed = savgol_filter(df_clean[col].ffill().bfill(), 
                                                 window_length, polyorder)
                        df_clean.loc[anomaly_mask, col] = smoothed[anomaly_mask]
                except:
                    pass
        
        elif method == 'interpolate':
            # Linear interpolation for gaps
            df_clean.loc[anomaly_mask, col] = np.nan
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
        elif method == 'median_filter':
            # Median filter
            window = 5
            for idx in df_clean[anomaly_mask].index:
                start = max(0, idx - window // 2)
                end = min(len(df_clean), idx + window // 2 + 1)
                median_val = df_clean[col].iloc[start:end].median()
                df_clean.loc[idx, col] = median_val
    
    return df_clean


def detect_missing_data_gaps(df: pd.DataFrame, max_gap_seconds: float = 60.0) -> pd.DataFrame:
    """
    Detect missing data gaps in time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Time' column
    max_gap_seconds : float
        Maximum allowed gap in seconds before flagging
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'Missing_Data_Flag' column
    """
    df_clean = df.copy()
    
    if 'Time' not in df_clean.columns:
        df_clean['Missing_Data_Flag'] = False
        return df_clean
    
    df_clean = df_clean.sort_values('Time').reset_index(drop=True)
    time_diff = df_clean['Time'].diff()
    
    # Flag large gaps
    df_clean['Missing_Data_Flag'] = time_diff > max_gap_seconds
    
    return df_clean


def impute_missing_data(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Impute missing data using interpolation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        'linear', 'cubic', or 'forward_fill'
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    df_clean = df.copy()
    
    numeric_cols = ['Voltage', 'Current', 'Capacity', 'Temperature']
    available_cols = [c for c in numeric_cols if c in df_clean.columns]
    
    for col in available_cols:
        if method == 'linear':
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        elif method == 'cubic' and SCIPY_AVAILABLE:
            df_clean[col] = df_clean[col].interpolate(method='cubic', limit_direction='both')
        elif method == 'forward_fill':
            df_clean[col] = df_clean[col].ffill().bfill()
    
    return df_clean


def verify_electrochemical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify electrochemical anomalies (Coulombic Efficiency, Capacity Drift).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with cycle-level or time-series data
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'CE_Anomaly' and 'Capacity_Drift_Anomaly' flags
    """
    df_clean = df.copy()
    
    if 'Cycle_Index' not in df_clean.columns:
        df_clean['CE_Anomaly'] = False
        df_clean['Capacity_Drift_Anomaly'] = False
        return df_clean
    
    # Calculate Coulombic Efficiency per cycle
    cycle_data = df_clean.groupby('Cycle_Index').agg({
        'Current': lambda x: x[x > 0].sum() if (x > 0).any() else 0,  # Charge
        'Capacity': ['max', 'min']
    })
    
    if 'Current' in df_clean.columns and 'Capacity' in df_clean.columns:
        # Calculate CE = Discharge Capacity / Charge Capacity
        charge_capacity = df_clean[df_clean['Current'] > 0].groupby('Cycle_Index')['Capacity'].max()
        discharge_capacity = df_clean[df_clean['Current'] < 0].groupby('Cycle_Index')['Capacity'].max()
        
        ce = discharge_capacity / (charge_capacity + 1e-10)
        
        # Flag cycles with CE > 100.5%
        ce_anomaly_cycles = ce[ce > 1.005].index
        
        df_clean['CE_Anomaly'] = df_clean['Cycle_Index'].isin(ce_anomaly_cycles)
        
        # Detect capacity drift/reset
        if len(discharge_capacity) > 1:
            capacity_diff = discharge_capacity.diff()
            # Flag sudden non-monotonic jumps (>20% increase)
            capacity_jump = capacity_diff > (discharge_capacity * 0.2)
            drift_anomaly_cycles = capacity_jump[capacity_jump].index
            
            df_clean['Capacity_Drift_Anomaly'] = df_clean['Cycle_Index'].isin(drift_anomaly_cycles)
        else:
            df_clean['Capacity_Drift_Anomaly'] = False
    else:
        df_clean['CE_Anomaly'] = False
        df_clean['Capacity_Drift_Anomaly'] = False
    
    return df_clean


# ============================================================================
# PHASE 3: Preprocessing for Feature Extraction
# ============================================================================

def resample_to_uniform_capacity_axis(df: pd.DataFrame, capacity_points: int = 1000) -> pd.DataFrame:
    """
    Resample voltage data to uniform capacity axis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Voltage and Capacity columns
    capacity_points : int
        Number of points in uniform capacity grid
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe with uniform capacity axis
    """
    if 'Voltage' not in df.columns or 'Capacity' not in df.columns:
        return df
    
    df_clean = df.copy()
    
    # Create uniform capacity grid
    if 'Cycle_Index' in df_clean.columns:
        resampled_dfs = []
        
        for cycle in df_clean['Cycle_Index'].unique():
            cycle_data = df_clean[df_clean['Cycle_Index'] == cycle].copy()
            
            if len(cycle_data) < 2:
                continue
            
            # Sort by capacity
            cycle_data = cycle_data.sort_values('Capacity').reset_index(drop=True)
            
            cap_min = cycle_data['Capacity'].min()
            cap_max = cycle_data['Capacity'].max()
            
            if cap_max <= cap_min:
                continue
            
            # Create uniform capacity grid
            uniform_capacity = np.linspace(cap_min, cap_max, capacity_points)
            
            # Interpolate voltage
            if SCIPY_AVAILABLE:
                f_voltage = interp1d(cycle_data['Capacity'].values, 
                                    cycle_data['Voltage'].values,
                                    kind='linear', 
                                    bounds_error=False, 
                                    fill_value='extrapolate')
                uniform_voltage = f_voltage(uniform_capacity)
            else:
                uniform_voltage = np.interp(uniform_capacity, 
                                          cycle_data['Capacity'].values,
                                          cycle_data['Voltage'].values)
            
            # Create resampled dataframe
            resampled = pd.DataFrame({
                'Cycle_Index': cycle,
                'Capacity': uniform_capacity,
                'Voltage': uniform_voltage
            })
            
            resampled_dfs.append(resampled)
        
        if resampled_dfs:
            return pd.concat(resampled_dfs, ignore_index=True)
    
    return df_clean


def resample_to_uniform_voltage_axis(df: pd.DataFrame, voltage_points: int = 1000) -> pd.DataFrame:
    """
    Resample capacity data to uniform voltage axis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Voltage and Capacity columns
    voltage_points : int
        Number of points in uniform voltage grid
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe with uniform voltage axis
    """
    if 'Voltage' not in df.columns or 'Capacity' not in df.columns:
        return df
    
    df_clean = df.copy()
    
    if 'Cycle_Index' in df_clean.columns:
        resampled_dfs = []
        
        for cycle in df_clean['Cycle_Index'].unique():
            cycle_data = df_clean[df_clean['Cycle_Index'] == cycle].copy()
            
            if len(cycle_data) < 2:
                continue
            
            # Sort by voltage
            cycle_data = cycle_data.sort_values('Voltage').reset_index(drop=True)
            
            volt_min = cycle_data['Voltage'].min()
            volt_max = cycle_data['Voltage'].max()
            
            if volt_max <= volt_min:
                continue
            
            # Create uniform voltage grid
            uniform_voltage = np.linspace(volt_min, volt_max, voltage_points)
            
            # Interpolate capacity
            if SCIPY_AVAILABLE:
                f_capacity = interp1d(cycle_data['Voltage'].values,
                                     cycle_data['Capacity'].values,
                                     kind='linear',
                                     bounds_error=False,
                                     fill_value='extrapolate')
                uniform_capacity = f_capacity(uniform_voltage)
            else:
                uniform_capacity = np.interp(uniform_voltage,
                                           cycle_data['Voltage'].values,
                                           cycle_data['Capacity'].values)
            
            # Create resampled dataframe
            resampled = pd.DataFrame({
                'Cycle_Index': cycle,
                'Voltage': uniform_voltage,
                'Capacity': uniform_capacity
            })
            
            resampled_dfs.append(resampled)
        
        if resampled_dfs:
            return pd.concat(resampled_dfs, ignore_index=True)
    
    return df_clean


def smooth_for_derivatives(df: pd.DataFrame, window_length: int = 11, polyorder: int = 3) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter for smoothing before derivative calculation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    window_length : int
        Window length for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with smoothed values
    """
    if not SCIPY_AVAILABLE:
        return df
    
    df_clean = df.copy()
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Smooth voltage and capacity for dQ/dV analysis
    if 'Voltage' in df_clean.columns and len(df_clean) >= window_length:
        try:
            df_clean['Voltage_Smoothed'] = savgol_filter(
                df_clean['Voltage'].ffill().bfill().values,
                window_length,
                polyorder
            )
        except:
            df_clean['Voltage_Smoothed'] = df_clean['Voltage']
    
    if 'Capacity' in df_clean.columns and len(df_clean) >= window_length:
        try:
            df_clean['Capacity_Smoothed'] = savgol_filter(
                df_clean['Capacity'].ffill().bfill().values,
                window_length,
                polyorder
            )
        except:
            df_clean['Capacity_Smoothed'] = df_clean['Capacity']
    
    return df_clean


# ============================================================================
# Main Cleaning Pipeline
# ============================================================================

def clean_dataframe(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    Comprehensive battery data cleaning pipeline implementing 3-phase approach.
    
    Phase 1: Data Ingestion and Harmonization
    Phase 2: Anomaly Detection and Correction
    Phase 3: Preprocessing for Feature Extraction
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    options : dict
        Cleaning options:
        - harmonize_schema: bool (default: True)
        - detect_cycler: bool (default: True)
        - segment_steps: bool (default: True)
        - assign_step_types: bool (default: True)
        - identify_rpt: bool (default: True)
        - detect_anomalies: bool (default: True)
        - anomaly_method: str ('zscore' or 'lof', default: 'zscore')
        - correct_anomalies: bool (default: True)
        - correction_method: str ('savgol', 'interpolate', 'median_filter', default: 'savgol')
        - detect_missing_data: bool (default: True)
        - impute_missing: bool (default: True)
        - imputation_method: str ('linear', 'cubic', 'forward_fill', default: 'linear')
        - verify_electrochemical: bool (default: True)
        - resample_capacity_axis: bool (default: False)
        - capacity_points: int (default: 1000)
        - resample_voltage_axis: bool (default: False)
        - voltage_points: int (default: 1000)
        - smooth_for_derivatives: bool (default: False)
        - savgol_window: int (default: 11)
        - savgol_polyorder: int (default: 3)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Phase 1: Data Ingestion and Harmonization
    if options.get('harmonize_schema', True):
        detect_cycler = options.get('detect_cycler', True)
        cycler_format = detect_cycler_format(df_clean) if detect_cycler else None
        df_clean = harmonize_to_standard_schema(df_clean, cycler_format)
    
    # Phase 2: Anomaly Detection and Correction
    if options.get('segment_steps', True):
        df_clean = segment_steps_and_cycles(df_clean)
    
    if options.get('assign_step_types', True):
        df_clean = assign_step_types(df_clean)
    
    if options.get('identify_rpt', True):
        df_clean = identify_rpt_cycles(df_clean)
    
    if options.get('detect_anomalies', True):
        window_size = options.get('anomaly_window_size', 10)
        method = options.get('anomaly_method', 'zscore')
        df_clean = detect_statistical_anomalies(df_clean, window_size, method)
    
    if options.get('correct_anomalies', True):
        correction_method = options.get('correction_method', 'savgol')
        df_clean = correct_anomalies(df_clean, correction_method)
    
    if options.get('detect_missing_data', True):
        max_gap = options.get('max_gap_seconds', 60.0)
        df_clean = detect_missing_data_gaps(df_clean, max_gap)
    
    if options.get('impute_missing', True):
        imputation_method = options.get('imputation_method', 'linear')
        df_clean = impute_missing_data(df_clean, imputation_method)
    
    if options.get('verify_electrochemical', True):
        df_clean = verify_electrochemical_anomalies(df_clean)
    
    # Phase 3: Preprocessing for Feature Extraction
    if options.get('resample_capacity_axis', False):
        capacity_points = options.get('capacity_points', 1000)
        df_clean = resample_to_uniform_capacity_axis(df_clean, capacity_points)
    
    if options.get('resample_voltage_axis', False):
        voltage_points = options.get('voltage_points', 1000)
        df_clean = resample_to_uniform_voltage_axis(df_clean, voltage_points)
    
    if options.get('smooth_for_derivatives', False):
        window_length = options.get('savgol_window', 11)
        polyorder = options.get('savgol_polyorder', 3)
        df_clean = smooth_for_derivatives(df_clean, window_length, polyorder)
    
    # Apply range filters (replace out-of-range values with 0)
    df_clean = apply_range_filters(df_clean, options)
    
    return df_clean


# ============================================================================
# Utility Functions
# ============================================================================

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """Get summary information about a dataframe."""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    return info


def to_csv_download(df: pd.DataFrame, filename: str = "cleaned_data.csv", remove_sheet_source: bool = False) -> bytes:
    """Convert dataframe to CSV bytes for download."""
    df_export = df.copy()
    
    if remove_sheet_source and 'Sheet_Source' in df_export.columns:
        df_export = df_export.drop(columns=['Sheet_Source'])
    
    output = BytesIO()
    df_export.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    return output.getvalue()


def validate_analysis_compatibility(df: pd.DataFrame) -> dict:
    """Validate if cleaned dataframe is compatible with analysis module."""
    cols_l = [c.lower() for c in df.columns]
    warnings = []
    missing_columns = []
    
    has_cycle = any(c in ["cycle", "cycle_number", "cycle_idx", "cycle_index"] for c in cols_l)
    has_discharge_cap = any(
        "discharge_capacity" in c or "discharge_cap" in c or "qdischarge" in c or
        (c in ["capacity", "cap", "q"] and "discharge" in " ".join(cols_l)) or c == "qdischarge"
        for c in cols_l)
    
    has_voltage = any(c in ["v", "volt", "voltage", "voltage_v"] for c in cols_l)
    has_capacity = any("capacity" in c or c in ["q", "ah", "mah", "capacity_ah", "capacity_mah"] for c in cols_l)
    
    format_type = 'unknown'
    is_compatible = False
    
    if has_cycle and has_discharge_cap:
        format_type = 'cycle_level'
        is_compatible = True
    elif has_voltage and has_capacity:
        format_type = 'voltage_capacity'
        is_compatible = True
    
    if 'Sheet_Source' in df.columns:
        warnings.append("'Sheet_Source' column detected. Consider removing it for analysis compatibility.")
    
    return {
        'is_compatible': is_compatible,
        'format_type': format_type,
        'warnings': warnings,
        'missing_columns': missing_columns
    }


def get_excel_sheets(file_path_or_buffer) -> list:
    """Get list of sheet names from an Excel file."""
    try:
        excel_file = pd.ExcelFile(file_path_or_buffer)
        return excel_file.sheet_names
    except Exception:
        return []


def read_excel_sheets(file_path_or_buffer, sheet_names: list = None) -> dict:
    """Read specific sheets from an Excel file."""
    try:
        excel_file = pd.ExcelFile(file_path_or_buffer)
        if sheet_names is None:
            sheet_names = excel_file.sheet_names
        
        sheets_dict = {}
        for sheet_name in sheet_names:
            if sheet_name in excel_file.sheet_names:
                sheets_dict[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
        return sheets_dict
    except Exception:
        return {}


def filter_channel_sheets(sheet_names: list) -> list:
    """Filter sheet names to get only Channel_ sheets (exclude 'info')."""
    channel_sheets = []
    for sheet in sheet_names:
        if sheet.lower() != 'info' and sheet.lower().startswith('channel_'):
            channel_sheets.append(sheet)
    return sorted(channel_sheets, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 999)


def apply_range_filters(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    """Apply user-specified range filters - replace out-of-range values with 0."""
    if not options.get('apply_range_filters', False):
        return df
    
    df_clean = df.copy()
    range_filters = options.get('range_filters', {})
    
    for col_name, range_config in range_filters.items():
        if col_name in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col_name]):
            min_val = range_config.get('min', None)
            max_val = range_config.get('max', None)
            replace_with = range_config.get('replace_with', 0.0)
            
            mask = pd.Series(True, index=df_clean.index)
            
            if min_val is not None:
                mask = mask & (df_clean[col_name] >= min_val)
            if max_val is not None:
                mask = mask & (df_clean[col_name] <= max_val)
            
            df_clean.loc[~mask, col_name] = replace_with
    
    return df_clean


def get_battery_default_ranges() -> dict:
    """Get default value ranges for common battery parameters."""
    return {
        'voltage': {'min': 2.0, 'max': 4.5, 'replace_with': 0.0, 'unit': 'V'},
        'current': {'min': -50.0, 'max': 50.0, 'replace_with': 0.0, 'unit': 'A'},
        'capacity': {'min': 0.0, 'max': 10.0, 'replace_with': 0.0, 'unit': 'Ah'},
        'discharge_capacity': {'min': 0.0, 'max': 10.0, 'replace_with': 0.0, 'unit': 'Ah'},
        'charge_capacity': {'min': 0.0, 'max': 10.0, 'replace_with': 0.0, 'unit': 'Ah'},
        'temperature': {'min': -40.0, 'max': 90.0, 'replace_with': 25.0, 'unit': '°C'},
        'internal_resistance': {'min': 0.0, 'max': 1.0, 'replace_with': 0.0, 'unit': 'Ohm'},
        'soc': {'min': 0.0, 'max': 100.0, 'replace_with': 0.0, 'unit': '%'},
        'soh': {'min': 0.0, 'max': 100.0, 'replace_with': 100.0, 'unit': '%'},
        'cycle': {'min': 1, 'max': 10000, 'replace_with': 0, 'unit': ''},
        'power': {'min': -500.0, 'max': 500.0, 'replace_with': 0.0, 'unit': 'W'},
        'energy': {'min': 0.0, 'max': 1000.0, 'replace_with': 0.0, 'unit': 'Wh'},
    }


def export_to_cell11_mat_format(df: pd.DataFrame, cell_id: str = "cell11") -> dict:
    """Convert dataframe to cell11dataset_for_python.mat format structure."""
    mat_data = {}
    
    cols_l = [c.lower() for c in df.columns]
    
    cycle_col = None
    for i, c in enumerate(cols_l):
        if c in ["cycle", "cycle_number", "cycle_idx", "cycle_index"]:
            cycle_col = df.columns[i]
            break
    
    cap_col = None
    for i, c in enumerate(cols_l):
        if "discharge_capacity" in c or "discharge_cap" in c or "qdischarge" in c:
            cap_col = df.columns[i]
            break
    if cap_col is None:
        for i, c in enumerate(cols_l):
            if c in ["capacity", "cap", "q"]:
                cap_col = df.columns[i]
                break
    
    voltage_col = None
    current_col = None
    for i, c in enumerate(cols_l):
        if ('voltage' in c or c in ['v', 'volt']) and voltage_col is None:
            voltage_col = df.columns[i]
        if ('current' in c or c in ['a', 'amps', 'i']) and current_col is None:
            current_col = df.columns[i]
    
    if cycle_col and cap_col:
        summary_cols = [cycle_col, cap_col]
        
        for col in df.columns:
            col_l = col.lower()
            if any(x in col_l for x in ['internal_resistance', 'ir', 'resistance', 'temperature', 'temp', 'soc', 'soh']):
                if col not in summary_cols:
                    summary_cols.append(col)
        
        summary = df[summary_cols].copy()
        summary.columns = [c.lower().replace(' ', '_') for c in summary.columns]
        mat_data['summary'] = summary.values if len(summary) > 0 else np.array([])
    
    if cycle_col:
        cycles = {}
        cycles_time_series = {}
        
        unique_cycles = df[cycle_col].dropna().unique()
        
        for cycle_num in unique_cycles:
            cycle_data = df[df[cycle_col] == cycle_num]
            
            cycle_dict = {}
            for col in df.columns:
                if col != cycle_col:
                    values = cycle_data[col].dropna().values
                    if len(values) > 0:
                        cycle_dict[col] = values[0] if len(values) == 1 else values
            
            if cycle_dict:
                cycles[int(cycle_num)] = cycle_dict
            
            if voltage_col and current_col:
                ts_data = {}
                if voltage_col in cycle_data.columns:
                    ts_data['voltage'] = cycle_data[voltage_col].dropna().values
                if current_col in cycle_data.columns:
                    ts_data['current'] = cycle_data[current_col].dropna().values
                
                if ts_data:
                    cycles_time_series[int(cycle_num)] = ts_data
        
        if cycles:
            mat_data['cycles'] = cycles
        if cycles_time_series:
            mat_data['cycles_time_series'] = cycles_time_series
    
    mat_data['policy'] = f"{cell_id}_policy"
    mat_data['policy_readable'] = f"Cleaned data from {cell_id}"
    if cycle_col and len(df) > 0:
        mat_data['cycle_life'] = int(df[cycle_col].max()) if cycle_col in df.columns else 0
    
    return mat_data


def to_mat_download(df: pd.DataFrame, filename: str = "cleaned_data.mat", cell_id: str = "cell11") -> bytes:
    """Convert dataframe to .mat file bytes in cell11dataset_for_python format."""
    try:
        from scipy.io import savemat
        import io
        
        mat_data = export_to_cell11_mat_format(df, cell_id)
        
        output = io.BytesIO()
        savemat(output, mat_data, format='5', oned_as='column')
        output.seek(0)
        return output.getvalue()
    except ImportError:
        raise ImportError("SciPy is required for .mat export. Install with: pip install scipy")
