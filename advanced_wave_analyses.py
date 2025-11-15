#!/usr/bin/env python3
"""
Advanced Wave Analysis Pipeline (Version 3.6 - Robust & Comprehensive)

Analisis data pressure gauge mentah menjadi statistik gelombang ilmiah.

-------------------------------------------------------------------
USAGE EXAMPLE (CONTOH PENGGUNAAN):
-------------------------------------------------------------------

1. Analisis Dasar (dengan path Windows & nama kustom):
   python advanced_wave_analysis_v3.6.py -f "F:\\Kuliah\\CoastalResearch\\Data\\Skripsi\\cores_wg\\data raw\\cores_wg2_coba_fixed.txt" -n "cores_wg2_results"

2. Analisis Lanjutan (Filter Baseline & Waktu Analisis):
   python advanced_wave_analysis_v3.6.py -f "data.txt" -n "hasil_filter" \
     --baseline_start "2025-09-06 13:00:00" \
     --baseline_end "2025-09-06 13:50:00" \
     --analysis_start "2025-09-06 14:00:00"

3. Analisis dengan Validasi ERA5 (Untuk Skripsi Transportasi Sedimen):
   python advanced_wave_analysis_v3.6.py -f "data.txt" -n "hasil_validasi" \
     --analysis_start "2025-09-06 14:00:00" \
     --era5 --era5_file "era5_data.nc" --gauge_lat -8.45 --gauge_lon 112.68

-------------------------------------------------------------------
FITUR UTAMA (v3.6):
-------------------------------------------------------------------
1. [FIX KRITIS v3.6] Mencegah spike Hmax dengan 'cosine taper filter' 
   pada noise frekuensi tinggi di `pressure_to_surface_elevation`.
2. [FIX KRITIS v3.6] Mencegah overestimate Tp dengan mencari 'prominent peak' 
   di `spectral_analysis`, bukan hanya 'max peak'.
3. [NEW v3.6] Menambahkan plot baru: "Komponen Tekanan", "Plot Kedalaman Individu", 
   dan "Plot Elevasi Individu". Plot gabungan tetap ada.
4. [NEW v3.6] Upgrade Validasi ERA5: Menambahkan ekstraksi dan plot untuk
   Arah Gelombang (mwd) dan Periode Puncak (pp1d) untuk skripsi.
5. [EXISTING] Filter waktu `--analysis_start/end` dan `--baseline_start/end`.
6. [EXISTING] Perhitungan Zero-Down-Crossing (v3.4) yang akurat.
7. [EXISTING] Plot Autofit (v3.3).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.integrate import simpson as simps
from scipy.io import savemat
from scipy.optimize import fsolve
import os
import sys
import io
import re
import argparse
from datetime import datetime, timedelta
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [FIX 1 di v3.1] Menambahkan impor dependensi opsional (xarray) ---
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    logger.info("[INFO] Pustaka 'xarray' tidak ditemukan. Pembacaan file .nc ERA5 tidak akan berfungsi.")
# -----------------------------------------------------------

# --- Penanganan Encoding & Plotting ---
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

plt.ion()  # Mode interaktif aktif
warnings.filterwarnings('ignore')

# --- Konfigurasi Parameter ---
class WaveConfig:
    def __init__(self):
        # Parameter fisik
        self.grav = 9.81          # gravitasi [m/s^2]
        self.rho_default = 1025.0 # Densitas default jika suhu tidak ada
        self.fs = None            # Akan dideteksi otomatis
        self.sensor_height = 0.25 # Jarak sensor dari dasar [m]
        self.water_depth = None   # Akan dihitung dari data
        self.avg_atm = None       # Akan dideteksi otomatis
        self.salinity = 35.0      # Salinitas untuk perhitungan densitas
        
        # Parameter filter
        self.waveband_period = 30.0 # Cutoff period highpass [s]
        self.wave_low_freq = 0.05   # Frekuensi rendah [Hz] untuk analisis (Periode 20s)
        self.wave_high_freq = 1.0   # Frekuensi tinggi [Hz] (Periode 1s)
        
        # [NEW v3.6] Taper filter settings to kill high-freq noise
        self.taper_start_freq = 0.8 # Mulai taper pada 0.8 Hz
        
        # Parameter windowing
        self.stat_window_sec = None # Akan dihitung otomatis
        self.window_overlap = 0.5   # 50% overlap
        self.min_window_sec = 600   # Minimum 10 menit
        self.max_window_sec = 1800  # Maksimum 30 menit
        
        # Parameter spektral
        self.nperseg = None         # Akan dihitung otomatis
        self.noverlap = None        # Akan dihitung otomatis
        
        # Parameter baseline
        self.MANUAL_AIR_BASELINE = None # Override (misal 1013.25)
        
        # [NEW] Validation parameters
        self.era5_comparison = False
        self.delft3d_comparison = False
        self.era5_file = None
        self.gauge_lat = None
        self.gauge_lon = None

# --- [IMPROVED] Fungsi Densitas Dinamis ---
def calculate_seawater_density(temp_c, salinity=35.0, pressure=0):
    """
    Menghitung densitas air laut menggunakan TEOS-10 (simplified).
    """
    if not isinstance(temp_c, (pd.Series, np.ndarray, list)):
        return None
    temp_c = np.asarray(temp_c)
    rho_w = 999.842594 + 0.06931752 * temp_c - 0.00137859 * temp_c**2
    a = 0.824493 - 0.0040899 * temp_c + 0.000076438 * temp_c**2 - 0.00000082467 * temp_c**3
    b = -0.00572466 + 0.00010227 * temp_c - 0.0000016546 * temp_c**2
    rho_sw = rho_w + a * salinity + b * salinity**1.5 + 0.00048314 * salinity**2
    if pressure > 0:
        rho_sw = rho_sw * (1 + pressure / (2100 * 1000))
    return rho_sw

# --- Fungsi Helper untuk Timestamp ---
def normalize_timestamp(ts):
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts

# --- Fungsi untuk Generate Custom Output Path ---
def get_output_path(input_file_path, custom_name=None, output_dir=None):
    """
    Generate custom output path untuk hasil analisis.
    """
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.basename(input_file_path)
    input_base_name = os.path.splitext(input_filename)[0]
    
    if custom_name:
        base_name = custom_name
    else:
        base_name = input_base_name
    
    if output_dir:
        output_directory = os.path.normpath(output_dir.strip())
    else:
        output_directory = os.path.join(input_dir, 'results')
    
    try:
        os.makedirs(output_directory, exist_ok=True)
    except Exception as e:
        logger.warning(f"Tidak dapat membuat direktori {output_directory}. Error: {e}")
        output_directory = input_dir
        logger.info(f"Menggunakan fallback direktori: {output_directory}")
        os.makedirs(output_directory, exist_ok=True)
    
    full_path = os.path.join(output_directory, base_name)
    
    return full_path, output_directory

# --- Fungsi Filter Pasut dan Gelombang ---
def separate_tide_waves(pressure, config):
    """
    Pisahkan sinyal pasut dan gelombang menggunakan high-pass filter.
    """
    fc = 1 / config.waveband_period # cutoff frequency
    fn = config.fs / 2              # Nyquist frequency
    order = 4                       # 4th order for steeper roll-off
    
    b, a = butter(order, fc/fn, 'high')
    highpass = filtfilt(b, a, pressure)  # zero-phase filtering
    lowpass = pressure - highpass
    
    return lowpass, highpass

# --- [IMPROVED] Fungsi Dispersion Relation ---
def dispersion_relation(omega, depth, config):
    """
    Selesaikan dispersion relation: ω² = gk·tanh(kh)
    """
    if omega == 0:
        return 0
    kh = omega * np.sqrt(depth / config.grav) # Initial guess
    if kh > 20:
        k = omega**2 / config.grav
    elif kh < 0.5:
        k = omega / np.sqrt(config.grav * depth)
    else:
        k = omega**2 / (config.grav * np.tanh(kh))
    
    for _ in range(20):
        kh = k * depth
        f_k = config.grav * k * np.tanh(kh) - omega**2
        f_k_prime = config.grav * (np.tanh(kh) + kh * (1 - np.tanh(kh)**2))
        if abs(f_k_prime) < 1e-10:
            break
        k_new = k - f_k / f_k_prime
        if abs(k_new - k) < 1e-8:
            break
        k = k_new
    return k

# --- [IMPROVED] Fungsi Transfer Function untuk Koreksi Attenuasi ---
def pressure_transfer_function(freq, depth, config):
    """
    Hitung transfer function untuk koreksi attenuasi tekanan.
    Kp(ω) = cosh(k * z_b) / cosh(k * h)
    """
    omega = 2 * np.pi * freq
    k = np.zeros_like(freq)
    for i, f in enumerate(freq):
        if f != 0:
            k[i] = dispersion_relation(omega[i], depth, config)
    
    height_from_bottom = config.sensor_height
    numerator = np.cosh(k * height_from_bottom)
    denominator = np.cosh(k * depth)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Kp = np.where(denominator > 1e-10, numerator / denominator, 1.0)
    
    Kp[freq == 0] = 1.0
    return Kp

# --- [FIX KRITIS v3.6] Fungsi Konversi Tekanan ke Elevasi Permukaan ---
def pressure_to_surface_elevation(wave_pressure, config, avg_total_depth, rho_series):
    """
    Konversi tekanan gelombang ke elevasi permukaan dengan koreksi attenuasi.
    [FIX v3.6] Menambahkan cosine taper filter untuk membunuh noise frek-tinggi
    yang menyebabkan spike Hmax.
    """
    n = len(wave_pressure)
    rho_avg = np.mean(rho_series)
    freqs = np.fft.fftfreq(n, d=1/config.fs)
    wave_pressure_fft = np.fft.fft(wave_pressure)
    
    # Hitung Kp (Transfer Function)
    Kp = pressure_transfer_function(np.abs(freqs), avg_total_depth, config)
    
    # Hitung eta_fft (Elevasi di domain frekuensi)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_fft = np.where(
            (freqs != 0) & (Kp > 1e-10),
            (wave_pressure_fft * 100) / (rho_avg * config.grav) / Kp,
            0
        )
    
    # --- [FIX v3.6] Taper Filter untuk Mencegah Spike Hmax ---
    # Buat filter taper (transisi mulus ke nol)
    nyquist_freq = config.fs / 2
    taper_filter = np.ones_like(freqs)
    
    # Tentukan frekuensi untuk taper
    f1 = config.taper_start_freq # Mulai taper
    f2 = nyquist_freq # Akhir taper (nol)
    
    if f1 < f2:
        idx_taper = (np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)
        # Cosine taper formula
        taper_filter[idx_taper] = 0.5 * (1 + np.cos(np.pi * (np.abs(freqs[idx_taper]) - f1) / (f2 - f1)))
    
    # Set semua frekuensi di atas f2 (atau di atas high_freq) ke nol
    idx_cut = (np.abs(freqs) > f2)
    taper_filter[idx_cut] = 0.0
    
    # Terapkan filter ke spektrum elevasi
    eta_fft_filtered = eta_fft * taper_filter
    # ----------------------------------------------------
    
    # IFFT untuk kembali ke domain waktu
    eta = np.real(np.fft.ifft(eta_fft_filtered))
    
    return eta

# --- [FIX KRITIS v3.6] Fungsi Analisis Spektral ---
def spectral_analysis(eta, config):
    """
    Analisis spektral menggunakan Welch method.
    [FIX v3.6] Menggunakan find_peaks untuk menemukan 'prominent peak' (Tp)
    dan menghindari 'noise peak' frekuensi rendah.
    """
    n = len(eta)
    if config.nperseg is None:
        config.nperseg = min(2048, n // 4)
        config.nperseg = max(config.nperseg, 256)
    if config.noverlap is None:
        config.noverlap = config.nperseg // 2
    
    freqs, psd = welch(eta, fs=config.fs, nperseg=config.nperseg, 
                       noverlap=config.noverlap, window='hann')
    
    valid_idx = (freqs >= config.wave_low_freq) & (freqs <= config.wave_high_freq)
    
    if not np.any(valid_idx):
        # Kembalikan nol jika tidak ada data
        return {
            'Hm0': 0, 'T01': 0, 'T02': 0, 'Tp': 0, 'epsilon': 0, 'Qp': 0,
            'fp': 0, 'nu': 0, 'freqs': np.array([0]), 'psd': np.array([0]),
            'm0': 0, 'm1': 0, 'm2': 0, 'm4': 0,
            'Tm02': 0, 'Tm10': 0, 'S_max': 0, 'sigma': 0
        }

    freqs_valid = freqs[valid_idx]
    psd_valid = psd[valid_idx]
    
    # --- [FIX v3.6] Logika Baru Pencarian Tp (Robust) ---
    # Cari semua puncak (peaks) di dalam spektrum yang valid
    peaks, properties = find_peaks(psd_valid, prominence=psd_valid.max() * 0.05) # Prominensi min 5% dari max
    
    if len(peaks) > 0:
        # Jika ada puncak yang menonjol, pilih yang paling menonjol
        # (puncak yang paling "jelas" dari dasarnya)
        prominent_peak_idx = peaks[np.argmax(properties['prominences'])]
        fp = freqs_valid[prominent_peak_idx]
        Tp = 1 / fp
    elif len(psd_valid) > 0:
        # Fallback: jika tidak ada puncak menonjol, gunakan max peak (metode lama)
        fp = freqs_valid[np.argmax(psd_valid)]
        Tp = 1 / fp
    else:
        fp = 0
        Tp = 0
    # ----------------------------------------------------

    # Hitung momen spektral
    df = freqs_valid[1] - freqs_valid[0] if len(freqs_valid) > 1 else 1
    m0 = simps(psd_valid, dx=df)
    m1 = simps(psd_valid * freqs_valid, dx=df)
    m2 = simps(psd_valid * freqs_valid**2, dx=df)
    m4 = simps(psd_valid * freqs_valid**4, dx=df)
    m_1 = simps(psd_valid / freqs_valid, dx=df)
    
    Hm0 = 4 * np.sqrt(m0) if m0 > 0 else 0
    T01 = m0 / m1 if m1 > 0 else 0
    T02 = np.sqrt(m0 / m2) if m2 > 0 else 0
    Tm02 = np.sqrt(m_1 / m1) if m_1 > 0 and m1 > 0 else 0
    Tm10 = m_1 / m0 if m_1 > 0 and m0 > 0 else 0
    S_max = np.max(psd_valid) if len(psd_valid) > 0 else 0
    
    epsilon = np.sqrt(1 - (m2**2) / (m0 * m4)) if m0 > 0 and m4 > 0 and (m0*m4) > (m2**2) else 0
    Qp = 2 * simps(psd_valid * (freqs_valid - fp)**2, dx=df) / m0 if fp > 0 and m0 > 0 else 0
    nu = np.sqrt((m2 * m4) / (m1**2) - 1) if m1 > 0 and m4 > 0 else 0
    sigma = np.sqrt(m0 * m2 / m1**2 - 1) if m0 > 0 and m1 > 0 and m2 > 0 else 0
    
    spectral_params = {
        'Hm0': Hm0, 'T01': T01, 'T02': T02, 'Tp': Tp, 'epsilon': epsilon,
        'Qp': Qp, 'fp': fp, 'nu': nu, 'freqs': freqs_valid, 'psd': psd_valid,
        'm0': m0, 'm1': m1, 'm2': m2, 'm4': m4,
        'Tm02': Tm02, 'Tm10': Tm10, 'S_max': S_max, 'sigma': sigma
    }
    
    return spectral_params

# --- [FIX v3.4] Fungsi Statistik Zero-Crossing yang Ditulis Ulang ---
def calculate_wave_statistics(eta, config, min_wave_height=0.01):
    """
    Hitung statistik gelombang menggunakan zero-down-crossing method (Standar).
    """
    zero_crossings = np.where(np.diff(np.sign(eta)))[0]
    wave_heights = []
    wave_periods = []
    
    down_crossings = []
    for i in zero_crossings:
        if eta[i] > 0 and eta[i+1] < 0:
            down_crossings.append(i)
    
    if len(down_crossings) < 2:
        return {'Hs': 0, 'Tm': 0, 'Hmax': 0, 'Havg': 0, 'Nwaves': 0, 'H13': 0, 'H10': 0, 'Hrms': 0}

    for i in range(len(down_crossings) - 1):
        start_idx = down_crossings[i]
        end_idx = down_crossings[i+1]
        segment = eta[start_idx:end_idx+1]
        
        if len(segment) == 0: continue
            
        h_max = np.max(segment)
        h_min = np.min(segment)
        h_wave = h_max - h_min
        
        if h_wave > min_wave_height:
            wave_heights.append(h_wave)
            t_start = (start_idx - eta[start_idx] / (eta[start_idx+1] - eta[start_idx])) / config.fs
            t_end = (end_idx - eta[end_idx] / (eta[end_idx+1] - eta[end_idx])) / config.fs
            wave_periods.append(t_end - t_start)
    
    if not wave_heights:
        return {'Hs': 0, 'Tm': 0, 'Hmax': 0, 'Havg': 0, 'Nwaves': 0, 'H13': 0, 'H10': 0, 'Hrms': 0}
    
    wave_heights = np.array(wave_heights)
    wave_periods = np.array(wave_periods)
    n_waves = len(wave_heights)
    sorted_heights = np.sort(wave_heights)[::-1]
    
    n_top_third = int(np.ceil(n_waves / 3.0)); n_top_third = max(1, n_top_third)
    Hs = np.mean(sorted_heights[:n_top_third])
    n_top_tenth = int(np.ceil(n_waves / 10.0)); n_top_tenth = max(1, n_top_tenth)
    H10 = np.mean(sorted_heights[:n_top_tenth])
    Hrms = np.sqrt(np.mean(wave_heights**2))
    
    stats = {
        'Hs': Hs, 'Tm': np.mean(wave_periods), 'Hmax': np.max(wave_heights),
        'Havg': np.mean(wave_heights), 'Nwaves': n_waves, 'H13': Hs, 'H10': H10, 'Hrms': Hrms
    }
    return stats

# --- [NEW v3.6] Fungsi Pembacaan Data ERA5 (Upgrade Skripsi) ---
def load_era5_data(era5_file, gauge_lat, gauge_lon):
    """
    Memuat dan memproses data ERA5 untuk validasi.
    [NEW v3.6] Ditambahkan 'mwd' (mean wave direction) dan 'pp1d' (peak period).
    """
    try:
        if era5_file.endswith('.nc') and HAS_XARRAY:
            ds = xr.open_dataset(era5_file)
            ds_sel = ds.sel(latitude=gauge_lat, longitude=gauge_lon, method='nearest')
            era5_df = ds_sel.to_dataframe().reset_index()
            era5_df['timestamp'] = pd.to_datetime(era5_df['time'])
        else:
            era5_df = pd.read_csv(era5_file)
            era5_df['timestamp'] = pd.to_datetime(era5_df['timestamp'])
        
        # Standardisasi nama kolom (mencakup lebih banyak variabel)
        column_mapping = {
            'swh': 'Hm0_era5',  # Significant wave height
            'mwp': 'Tm_era5',   # Mean wave period (mwp)
            'pp1d': 'Tp_era5',  # Peak wave period (pp1d)
            'mwd': 'dir_era5',  # Mean wave direction (mwd)
            'si10': 'wind_era5', # 10m wind speed
            'sst': 'temp_era5'   # Sea surface temperature
        }
        
        era5_df.rename(columns=column_mapping, inplace=True)
        
        # Kolom yang diperlukan untuk skripsi transportasi sedimen
        required_cols = ['timestamp', 'Hm0_era5', 'Tp_era5', 'Tm_era5', 'dir_era5']
        available_cols = [col for col in required_cols if col in era5_df.columns]
        missing_cols = [col for col in required_cols if col not in era5_df.columns]
        
        if missing_cols:
            logger.warning(f"[WARN] Kolom ERA5 berikut tidak ditemukan: {missing_cols}")
            
        era5_df = era5_df[available_cols]
        return era5_df
    
    except Exception as e:
        logger.error(f"Error loading ERA5 data: {e}")
        return None

# --- [NEW v3.6] Fungsi Validasi ERA5 (Upgrade Skripsi) ---
def validate_against_era5(df_results, df_spectral, era5_data=None):
    """
    Validasi hasil analisis dengan data ERA5.
    [NEW v3.6] Menambahkan perbandingan Tp vs pp1d (Tp_era5)
    """
    if era5_data is None:
        logger.warning("No ERA5 data provided for validation")
        return None, None
    
    # Gabungkan data spektral (untuk Tp) dan zero-crossing (untuk Tm)
    df_merged_stats = pd.merge(df_spectral, df_results, on='timestamp', suffixes=('_spec', '_zc'))
    
    merged = pd.merge_asof(
        df_merged_stats.sort_values('timestamp'),
        era5_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1 hour')
    )
    
    merged = merged.dropna(subset=['Hs', 'Hm0_era5']) # Perlu Hs (dari ZC) dan Hm0 (dari ERA5)
    
    if len(merged) == 0:
        logger.warning("No overlapping data between wave gauge and ERA5")
        return None, None
    
    # Siapkan DataFrame hasil
    validation_data = {
        'timestamp': merged['timestamp'],
        'Hs_gauge': merged['Hs'],
        'Hm0_gauge': merged['Hm0'],
        'Hm0_era5': merged['Hm0_era5'],
        'Tm_gauge': merged['Tm'],
        'Tm_era5': merged.get('Tm_era5', np.nan),
        'Tp_gauge': merged['Tp'],
        'Tp_era5': merged.get('Tp_era5', np.nan),
        'dir_era5': merged.get('dir_era5', np.nan)
    }
    validation_results = pd.DataFrame(validation_data)
    
    # Hitung statistik RMSE, Bias, Corr
    def get_stats(gauge_col, era5_col):
        if era5_col not in validation_results or validation_results[era5_col].notna().sum() < 2:
            return np.nan, np.nan, np.nan
        
        temp_df = validation_results[[gauge_col, era5_col]].dropna()
        if len(temp_df) < 2:
            return np.nan, np.nan, np.nan
            
        diff = temp_df[gauge_col] - temp_df[era5_col]
        bias = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        corr = np.corrcoef(temp_df[gauge_col], temp_df[era5_col])[0, 1]
        return bias, rmse, corr

    stats = {}
    stats['bias_Hs'], stats['rmse_Hs'], stats['corr_Hs'] = get_stats('Hs', 'Hm0_era5')
    stats['bias_Hm0'], stats['rmse_Hm0'], stats['corr_Hm0'] = get_stats('Hm0', 'Hm0_era5')
    stats['bias_Tm'], stats['rmse_Tm'], stats['corr_Tm'] = get_stats('Tm', 'Tm_era5')
    stats['bias_Tp'], stats['rmse_Tp'], stats['corr_Tp'] = get_stats('Tp', 'Tp_era5')
    
    logger.info(f"Validasi ERA5 - Hs: Bias={stats['bias_Hs']:.3f}m, RMSE={stats['rmse_Hs']:.3f}m, Corr={stats['corr_Hs']:.3f}")
    if not np.isnan(stats['corr_Tm']):
        logger.info(f"Validasi ERA5 - Tm vs mwp: Bias={stats['bias_Tm']:.3f}s, RMSE={stats['rmse_Tm']:.3f}s, Corr={stats['corr_Tm']:.3f}")
    if not np.isnan(stats['corr_Tp']):
        logger.info(f"Validasi ERA5 - Tp vs pp1d: Bias={stats['bias_Tp']:.3f}s, RMSE={stats['rmse_Tp']:.3f}s, Corr={stats['corr_Tp']:.3f}")

    return validation_results, stats

# --- [NEW v3.6] Fungsi Plot Perbandingan ERA5 (Upgrade Skripsi) ---
def plot_era5_comparison(df_results, era5_results, output_dir, base_name):
    """
    Membuat plot perbandingan antara data wave gauge dan ERA5.
    [NEW v3.6] Menambahkan plot untuk Tp vs pp1d (Tp_era5) dan MWD (dir_era5).
    """
    # Cek kolom apa saja yang tersedia
    has_tm = 'Tm_era5' in era5_results.columns and era5_results['Tm_era5'].notna().any()
    has_tp = 'Tp_era5' in era5_results.columns and era5_results['Tp_era5'].notna().any()
    has_dir = 'dir_era5' in era5_results.columns and era5_results['dir_era5'].notna().any()
    
    # Tentukan jumlah baris plot
    num_rows = 1
    if has_tm: num_rows += 1
    if has_tp: num_rows += 1
    if has_dir: num_rows += 1
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), squeeze=False)
    plot_row = 0
    
    # --- Plot 1: Time series Hs ---
    ax = axes[plot_row, 0]
    ax.plot(df_results['timestamp'], df_results['Hs_gauge'], 'b-', label='Gauge Hs (ZC)', alpha=0.7)
    ax.plot(df_results['timestamp'], df_results['Hm0_gauge'], 'c:', label='Gauge Hm0 (Spec)', alpha=0.7)
    ax.plot(era5_results['timestamp'], era5_results['Hm0_era5'], 'r--', label='ERA5 Hm0', alpha=0.7)
    ax.set_title('Significant Wave Height Comparison')
    ax.set_ylabel('Hs (m)')
    ax.legend()
    ax.grid(True)
    
    # --- Plot 2: Scatter plot Hs ---
    ax = axes[plot_row, 1]
    ax.scatter(era5_results['Hm0_era5'], era5_results['Hs_gauge'], alpha=0.6, label=f'Hs vs Hm0_ERA5')
    min_val = min(era5_results['Hm0_era5'].min(), era5_results['Hs_gauge'].min())
    max_val = max(era5_results['Hm0_era5'].max(), era5_results['Hs_gauge'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_title('Hs Scatter Plot')
    ax.set_xlabel('ERA5 Hm0 (m)')
    ax.set_ylabel('Wave Gauge Hs (m)')
    ax.grid(True)
    ax.axis('equal')
    plot_row += 1

    # --- Plot 3 & 4: Periode Puncak (Tp) ---
    if has_tp:
        ax = axes[plot_row, 0]
        ax.plot(df_results['timestamp'], df_results['Tp_gauge'], 'g-', label='Gauge Tp (Spec)', alpha=0.7)
        ax.plot(era5_results['timestamp'], era5_results['Tp_era5'], 'r--', label='ERA5 Tp (pp1d)', alpha=0.7)
        ax.set_title('Peak Period (Tp) Comparison')
        ax.set_ylabel('Tp (s)')
        ax.legend()
        ax.grid(True)
        
        ax = axes[plot_row, 1]
        ax.scatter(era5_results['Tp_era5'], era5_results['Tp_gauge'], alpha=0.6)
        min_val = min(era5_results['Tp_era5'].min(), era5_results['Tp_gauge'].min())
        max_val = max(era5_results['Tp_era5'].max(), era5_results['Tp_gauge'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title('Tp Scatter Plot')
        ax.set_xlabel('ERA5 Tp (pp1d) (s)')
        ax.set_ylabel('Wave Gauge Tp (s)')
        ax.grid(True)
        ax.axis('equal')
        plot_row += 1

    # --- Plot 5 & 6: Periode Rata-rata (Tm) ---
    if has_tm:
        ax = axes[plot_row, 0]
        ax.plot(df_results['timestamp'], df_results['Tm_gauge'], 'b-', label='Gauge Tm (ZC)', alpha=0.7)
        ax.plot(era5_results['timestamp'], era5_results['Tm_era5'], 'r--', label='ERA5 Tm (mwp)', alpha=0.7)
        ax.set_title('Mean Wave Period (Tm) Comparison')
        ax.set_ylabel('Tm (s)')
        ax.legend()
        ax.grid(True)
        
        ax = axes[plot_row, 1]
        ax.scatter(era5_results['Tm_era5'], era5_results['Tm_gauge'], alpha=0.6)
        min_val = min(era5_results['Tm_era5'].min(), era5_results['Tm_gauge'].min())
        max_val = max(era5_results['Tm_era5'].max(), era5_results['Tm_gauge'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title('Tm Scatter Plot')
        ax.set_xlabel('ERA5 Tm (mwp) (s)')
        ax.set_ylabel('Wave Gauge Tm (s)')
        ax.grid(True)
        ax.axis('equal')
        plot_row += 1
        
    # --- Plot 7 & 8: Arah Gelombang (MWD) ---
    if has_dir:
        ax = axes[plot_row, 0]
        ax.plot(era5_results['timestamp'], era5_results['dir_era5'], 'ro', label='ERA5 MWD', markersize=2, alpha=0.7)
        ax.set_title('Mean Wave Direction (ERA5)')
        ax.set_ylabel('Direction (deg)')
        ax.legend()
        ax.grid(True)
        
        # Kosongkan plot scatter karena kita tidak punya data arah dari gauge
        ax = axes[plot_row, 1]
        ax.text(0.5, 0.5, 'Directional data not available\nfrom single-point gauge', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Direction Scatter Plot')
        ax.set_xlabel('ERA5 MWD (deg)')
        ax.set_ylabel('Wave Gauge MWD (deg)')
        plot_row += 1

    # Atur autofit untuk semua plot waktu
    if not era5_results.empty:
        t_min_era5 = era5_results['timestamp'].min(); t_max_era5 = era5_results['timestamp'].max()
        t_min_gauge = df_results['timestamp'].min(); t_max_gauge = df_results['timestamp'].max()
        t_min = min(t_min_era5, t_min_gauge); t_max = max(t_max_era5, t_max_gauge)
        t_margin = (t_max - t_min) * 0.02
        plot_t_min = t_min - t_margin; plot_t_max = t_max + t_margin
        
        for i in range(plot_row):
            axes[i, 0].set_xlim(plot_t_min, plot_t_max)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_era5_comparison.png"), dpi=300)
    plt.close()

# --- [NEW v3.6] Fungsi Validasi untuk Delft3D-FM (Placeholder) ---
def validate_against_delft3d(df_results, df_spectral, delft3d_data=None):
    if delft3d_data is None:
        logger.warning("No Delft3D-FM data provided for validation")
        return None
    logger.warning("Delft3D-FM validation logic is not yet implemented.")
    return None

# --- Fungsi Validasi Teori Gelombang ---
def validate_wave_theory(config, avg_depth):
    logger.info("\n📊 VALIDASI TEORI GELOMBANG:")
    logger.info("="*50)
    T_mean = 1.0 / ((config.wave_low_freq + config.wave_high_freq) / 2)
    L0 = (config.grav / (2 * np.pi)) * T_mean**2
    h = avg_depth
    if h <= 0:
        logger.warning("Peringatan: Kedalaman rata-rata <= 0. Validasi dilewati.")
        return
    kh = 2 * np.pi * h / L0
    if kh > np.pi: logger.info(f"→ Kondisi Perairan Dalam (kh = {kh:.2f} > π)")
    elif kh < np.pi/10: logger.info(f"→ Kondisi Perairan Dangkal (kh = {kh:.2f} < π/10)")
    else: logger.info(f"→ Kondisi Perairan Transisi (π/10 < kh = {kh:.2f} < π)")
    logger.info(f"(Berdasarkan T_mean={T_mean:.1f}s, L0={L0:.1f}m, h={h:.2f}m)")
    if config.sensor_height > h:
        logger.warning(f"⚠️ Sensor height ({config.sensor_height}m) > water depth ({h:.2f}m)")

# --- Fungsi Export Data ---
def export_results(df, df_results, spectral_results, filename_base, output_dir, config):
    """Export hasil ke CSV dan TXT dengan custom output directory"""
    
    csv_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_timeseries.csv")
    df.to_csv(csv_filename, index=False, date_format='%Y-%m-%d %H:%M:%S.%f', float_format='%.4f')
    
    # [NEW v3.6] Gabungkan hasil ZC dan Spektral sebelum menyimpan
    df_stats_combined = pd.merge(df_results, spectral_results, on='timestamp', suffixes=('_zc', '_spec'))
    # Hapus kolom freqs/psd yang tidak ramah CSV
    df_stats_combined = df_stats_combined.drop(columns=['freqs', 'psd'], errors='ignore')

    stats_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_statistics.csv")
    df_stats_combined.to_csv(stats_filename, index=False, date_format='%Y-%m-%d %H:%M:%S.%f', float_format='%.4f')
    
    txt_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_summary.txt")
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("RINGKASAN LAPORAN ANALISIS GELOMBANG\n")
        f.write("="*60 + "\n")
        f.write(f"Tanggal Analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Durasi Data: {df['timestamp'].min()} s/d {df['timestamp'].max()}\n")
        f.write(f"Frekuensi Sampling (fs): {config.fs:.2f} Hz\n")
        f.write(f"Kedalaman Air Rata-rata: {config.water_depth:.3f} m\n")
        f.write(f"Tinggi Sensor dari Dasar: {config.sensor_height:.2f} m\n")
        f.write(f"Baseline Atmosfer yg Digunakan: {config.avg_atm:.2f} mbar\n\n")
        
        f.write("STATISTIK ZERO-CROSSING (Rata-rata Window):\n")
        f.write("-"*40 + "\n")
        f.write(f"Tinggi Gel. Signifikan (Hs): {df_results['Hs'].mean():.3f} ± {df_results['Hs'].std():.3f} m\n")
        f.write(f"Periode Rata-rata (Tm): {df_results['Tm'].mean():.3f} ± {df_results['Tm'].std():.3f} s\n")
        f.write(f"Tinggi Gel. Maksimum (Hmax, absolut): {df_results['Hmax'].max():.3f} m\n\n")
        
        f.write("STATISTIK SPEKTRAL (Rata-rata Window):\n")
        f.write("-"*40 + "\n")
        f.write(f"Tinggi Gel. Spektral (Hm0): {spectral_results['Hm0'].mean():.3f} ± {spectral_results['Hm0'].std():.3f} m\n")
        f.write(f"Periode Puncak (Tp): {spectral_results['Tp'].mean():.3f} ± {spectral_results['Tp'].std():.3f} s\n")
        f.write(f"Periode Zero-crossing (T02): {spectral_results['T02'].mean():.3f} ± {spectral_results['T02'].std():.3f} s\n\n")
        
        f.write("PERBANDINGAN METODE:\n")
        f.write("-"*40 + "\n")
        if spectral_results['Hm0'].mean() > 0:
            f.write(f"Hs/Hm0: {df_results['Hs'].mean()/spectral_results['Hm0'].mean():.3f}\n")
        if spectral_results['T02'].mean() > 0:
            f.write(f"Tm/T02: {df_results['Tm'].mean()/spectral_results['T02'].mean():.3f}\n")
    
    # Export spektrum rata-rata
    spectrum_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_spectrum.csv")
    avg_freqs = spectral_results['freqs'].iloc[0]
    avg_psd = np.mean([s for s in spectral_results['psd']], axis=0)
    spectrum_df = pd.DataFrame({'frequency': avg_freqs, 'psd': avg_psd})
    spectrum_df.to_csv(spectrum_filename, index=False, float_format='%.8f')
    
    return csv_filename, stats_filename, txt_filename, spectrum_filename

# --- Fungsi Export ke MATLAB ---
def export_to_matlab_with_script(df, df_results, df_spectral, filename_base, config, output_dir):
    """
    Export data ke .mat dan generate script MATLAB untuk re-create plot.
    """
    matlab_time = [t.timestamp() for t in df['timestamp']]
    stats_time = [t.timestamp() for t in df_results['timestamp']]
    avg_freqs = df_spectral['freqs'].iloc[0]
    avg_psd = np.mean([s for s in df_spectral['psd']], axis=0)

    mat_data = {
        'time': matlab_time,
        'pressure_mbar': df['pressure_mbar'].values,
        'depth_total': df['depth'].values,
        'h_above_sensor': df['h_above_sensor'].values,
        'surface_elevation': df['surface_elevation'].values,
        'tide_pressure': df['tide_pressure'].values,
        'wave_pressure': df['wave_pressure'].values,
        'stats_time': stats_time,
        'Hs': df_results['Hs'].values, 'Tm': df_results['Tm'].values, 'Hmax': df_results['Hmax'].values,
        'Hm0': df_spectral['Hm0'].values, 'Tp': df_spectral['Tp'].values,
        'T01': df_spectral['T01'].values, 'T02': df_spectral['T02'].values,
        'spectral_freqs': avg_freqs, 'spectral_psd': avg_psd,
        'config': {
            'fs': float(config.fs), 'water_depth_avg': float(config.water_depth),
            'sensor_height': float(config.sensor_height), 'rho_avg': float(df['rho'].mean()),
            'g': float(config.grav), 'atm_baseline': float(config.avg_atm)
        }
    }
    if 'temp_C' in df.columns:
        mat_data['temp_C'] = df['temp_C'].values

    mat_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_data.mat")
    savemat(mat_filename, mat_data)
    logger.info(f"✅ Data diekspor ke {mat_filename}")
    
    base_name = os.path.basename(filename_base)
    # [NEW v3.6] MATLAB script diperbarui untuk menyertakan plot individu
    matlab_script = f"""
% MATLAB script untuk membuat ulang plot analisis gelombang (v3.6)
% Dibuat dari Python pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
load('{os.path.basename(mat_filename)}');
time_dt = datetime(time, 'ConvertFrom', 'posixtime');
stats_dt = datetime(stats_time, 'ConvertFrom', 'posixtime');

%% --- Plot 1: Komponen Tekanan (3-panel) ---
figure('Position', [100, 100, 1200, 800], 'Name', 'Analisis Komponen Tekanan');
ax1 = subplot(3,1,1);
plot(time_dt, pressure_mbar, 'k-', 'LineWidth', 1);
title('Tekanan Total Terukur', 'FontSize', 14); ylabel('Pressure (mbar)'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
ax2 = subplot(3,1,2);
plot(time_dt, tide_pressure, 'b-', 'LineWidth', 1);
title('Komponen Tekanan Pasang Surut (Low-pass)', 'FontSize', 14); ylabel('Pressure (mbar)'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
ax3 = subplot(3,1,3);
plot(time_dt, wave_pressure, 'r-', 'LineWidth', 1);
title('Komponen Tekanan Gelombang (High-pass)', 'FontSize', 14); ylabel('Pressure (mbar)');
xlabel('Waktu'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
linkaxes([ax1, ax2, ax3], 'x');
saveas(gcf, '{base_name}_pressure_components.png');

%% --- Plot 2: Kedalaman dan Elevasi (2-panel, Gabungan) ---
figure('Position', [100, 100, 1200, 800], 'Name', 'Kedalaman dan Elevasi (Gabungan)');
ax1 = subplot(2,1,1);
plot(time_dt, depth_total, 'g-', 'LineWidth', 1.5); hold on;
yline(config.water_depth, 'b--', 'LineWidth', 1.5, 'Label', ['Rata-rata ({config.water_depth:.2f} m)']);
yline(config.sensor_height, 'r--', 'LineWidth', 1.5, 'Label', ['Sensor ({config.sensor_height:.2f} m)']);
title('Kedalaman Air Total (Instan)', 'FontSize', 14); ylabel('Kedalaman (m)');
legend('show', 'Location', 'best'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
ax2 = subplot(2,1,2);
plot(time_dt, surface_elevation, 'r-', 'LineWidth', 1);
title('Elevasi Permukaan (dari Tekanan Gelombang)', 'FontSize', 14); ylabel('Elevasi (m)');
xlabel('Waktu'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
linkaxes([ax1, ax2], 'x');
saveas(gcf, '{base_name}_depth_elevation_combined.png');

%% --- Plot 3: Kedalaman Air Total (Individu) ---
figure('Position', [100, 100, 1200, 600], 'Name', 'Kedalaman Air Total (Individu)');
plot(time_dt, depth_total, 'g-', 'LineWidth', 1.5); hold on;
yline(config.water_depth, 'b--', 'LineWidth', 1.5, 'Label', ['Rata-rata ({config.water_depth:.2f} m)']);
yline(config.sensor_height, 'r--', 'LineWidth', 1.5, 'Label', ['Sensor ({config.sensor_height:.2f} m)']);
title('Kedalaman Air Total (Instan)', 'FontSize', 14); ylabel('Kedalaman (m)');
legend('show', 'Location', 'best'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
saveas(gcf, '{base_name}_depth_individual.png');

%% --- Plot 4: Elevasi Permukaan (Individu) ---
figure('Position', [100, 100, 1200, 600], 'Name', 'Elevasi Permukaan (Individu)');
plot(time_dt, surface_elevation, 'r-', 'LineWidth', 1);
title('Elevasi Permukaan (dari Tekanan Gelombang)', 'FontSize', 14); ylabel('Elevasi (m)');
xlabel('Waktu'); grid on;
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([time_dt(1) time_dt(end)]);
saveas(gcf, '{base_name}_elevation_individual.png');

%% --- Plot 5: Statistik Gelombang (Tinggi) ---
figure('Position', [100, 100, 1200, 600], 'Name', 'Statistik Tinggi Gelombang');
plot(stats_dt, Hs, 'bo-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'Hs (Zero-crossing)'); hold on;
plot(stats_dt, Hm0, 'ro-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'Hm0 (Spectral)');
plot(stats_dt, Hmax, 'g-', 'MarkerSize', 4, 'LineWidth', 1, 'Label', 'Hmax (Zero-crossing)');
title('Statistik Tinggi Gelombang vs Waktu', 'FontSize', 14); ylabel('Tinggi (m)');
xlabel('Waktu'); grid on; legend('show', 'Location', 'best');
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([stats_dt(1) stats_dt(end)]);
saveas(gcf, '{base_name}_wave_height_stats.png');

%% --- Plot 6: Statistik Gelombang (Periode) ---
figure('Position', [100, 100, 1200, 600], 'Name', 'Statistik Periode Gelombang');
plot(stats_dt, Tm, 'bo-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'Tm (Zero-crossing)'); hold on;
plot(stats_dt, Tp, 'go-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'Tp (Spectral Peak)');
plot(stats_dt, T02, 'ro-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'T02 (Spectral Zero-cross)');
plot(stats_dt, T01, 'mo-', 'MarkerSize', 4, 'LineWidth', 1.5, 'Label', 'T01 (Spectral Mean)');
title('Statistik Periode Gelombang vs Waktu', 'FontSize', 14); ylabel('Periode (s)');
xlabel('Waktu'); grid on; legend('show', 'Location', 'best');
datetick('x', 'mm-dd HH:MM', 'keepticks', 'keeplimits'); xlim([stats_dt(1) stats_dt(end)]);
saveas(gcf, '{base_name}_wave_period_stats.png');

%% --- Plot 7: Spektrum Gelombang Rata-rata ---
figure('Position', [100, 100, 1200, 600], 'Name', 'Spektrum Gelombang Rata-rata');
loglog(spectral_freqs, spectral_psd, 'b-', 'LineWidth', 2); hold on;
[peak_psd, peak_idx] = max(spectral_psd);
peak_freq = spectral_freqs(peak_idx);
plot(peak_freq, peak_psd, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'Label', ['Puncak: ' num2str(peak_freq, '%.3f') ' Hz']);
title('Spektrum Kepadatan Daya (PSD) Rata-rata', 'FontSize', 14);
xlabel('Frekuensi (Hz)'); ylabel('PSD (m²/Hz)'); grid on;
xlim([0.04, config.fs/2]); legend('show', 'Location', 'best');
saveas(gcf, '{base_name}_average_spectrum.png');

disp('Semua plot MATLAB telah disimpan!');
"""
    
    script_filename = os.path.join(output_dir, f"{os.path.basename(filename_base)}_recreate_plots.m")
    with open(script_filename, 'w', encoding='utf-8') as f:
        f.write(matlab_script)
    logger.info(f"📝 Skrip MATLAB disimpan: {script_filename}")
    
    return mat_filename, script_filename

# --- Fungsi Smart Windowing ---
def calculate_optimal_window(df, config):
    """Hitung window size optimal berdasarkan data"""
    data_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    logger.info(f"\n📊 DURASI DATA:")
    logger.info(f"   Total Durasi: {data_duration/60:.1f} menit")
    
    if data_duration < 1800:
        optimal_window = min(data_duration * 0.5, config.min_window_sec)
        logger.info(f"   → Data pendek: menggunakan window {optimal_window:.0f}s")
    elif data_duration < 3600:
        optimal_window = 900
        logger.info(f"   → Data sedang: menggunakan window 15 menit")
    else:
        optimal_window = 1800
        logger.info(f"   → Data panjang: menggunakan window 30 menit")
    
    window_points = optimal_window * config.fs
    if window_points < 256:
        logger.warning(f"   ⚠️  Window ({window_points} poin) terlalu kecil. Memperbesar window...")
        optimal_window = 256 / config.fs
        window_points = 256
    
    if config.nperseg is None:
        config.nperseg = min(2048, int(window_points / 2))
        config.nperseg = max(config.nperseg, 256)
    if config.noverlap is None:
        config.noverlap = config.nperseg // 2
    if window_points < config.nperseg:
        logger.warning(f"   ⚠️  Window ({window_points}) < nperseg ({config.nperseg}). Memperkecil nperseg...")
        config.nperseg = int(window_points / 2)
        config.noverlap = int(config.nperseg / 2)

    return optimal_window

# --- Variabel Global untuk Plot Interaktif ---
text_box1, fig1, span1 = None, None, None
text_box2, fig2, span2 = None, None, None
text_box3, fig3, span3 = None, None, None
text_box4, fig4, span4 = None, None, None
text_box5, fig5, span5 = None, None, None
text_box6, fig6, span6 = None, None, None
# [NEW v3.6] Tambah fig7/span7
text_box7, fig7, span7 = None, None, None
df_global, df_results_global, df_spectral_global = None, None, None

# --- Fungsi Callback untuk Plot ---
def onselect_plot(xmin, xmax, text_box, fig, df_select, columns):
    try:
        t_start = normalize_timestamp(mdates.num2date(xmin))
        t_end = normalize_timestamp(mdates.num2date(xmax))
        mask = (df_select['timestamp'] >= t_start) & (df_select['timestamp'] <= t_end)
        df_range = df_select[mask]
        
        if len(df_range) > 0:
            msg = (f"📊 Rentang Dipilih:\n"
                   f"🕐 {t_start.strftime('%Y-%m-%d %H:%M')} → "
                   f"{t_end.strftime('%Y-%m-%d %H:%M')}\n")
            for col, fmt in columns.items():
                if col in df_range:
                    mean_val = df_range[col].mean()
                    max_val = df_range[col].max()
                    msg += f"   {col} (rata-rata): {mean_val:{fmt}}\n"
                    msg += f"   {col} (maks): {max_val:{fmt}}\n"
            text_box.set_text(msg)
            fig.canvas.draw_idle()
        else:
            text_box.set_text("[Tidak ada data dalam rentang]")
            fig.canvas.draw_idle()
    except Exception as e:
        text_box.set_text(f"❌ Error: {str(e)}")
        fig.canvas.draw_idle()

# --- [NEW v3.6] Fungsi Plotting yang Diperbarui ---
def plot_with_spanselector(df, df_results, df_spectral, config):
    """Plot hasil dengan SpanSelector interaktif (v3.6 - Plot Terpisah)"""
    
    global df_global, df_results_global, df_spectral_global
    global text_box1, fig1, span1, text_box2, fig2, span2, text_box3, fig3, span3
    global text_box4, fig4, span4, text_box5, fig5, span5, text_box6, fig6, span6
    global text_box7, fig7, span7
    
    df_global, df_results_global, df_spectral_global = df, df_results, df_spectral
    logger.info("\n📊 Membuat plot interaktif...")
    
    # --- Tentukan rentang waktu untuk autofit ---
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    t_margin = (t_max - t_min) * 0.02 # Buat margin 2%
    plot_t_min, plot_t_max = t_min - t_margin, t_max + t_margin
    
    t_min_stats, t_max_stats = df_results["timestamp"].min(), df_results["timestamp"].max()
    t_margin_stats = (t_max_stats - t_min_stats) * 0.02
    plot_t_min_stats, plot_t_max_stats = t_min_stats - t_margin_stats, t_max_stats + t_margin_stats

    # --- Plot 1: Komponen Tekanan (BARU v3.6) ---
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig1.canvas.manager.set_window_title('Analisis Komponen Tekanan')
    sample_rate = max(1, len(df) // 5000)
    df_sampled = df.iloc[::sample_rate, :]
    
    ax1a.plot(df_sampled["timestamp"], df_sampled["pressure_mbar"], 'k-', label="Tekanan Total")
    ax1a.set_ylabel("Pressure (mbar)"); ax1a.set_title("Komponen Tekanan"); ax1a.grid(True)
    
    ax1b.plot(df_sampled["timestamp"], df_sampled["tide_pressure"], 'b-', label="Tekanan Pasut (Low-pass)")
    ax1b.set_ylabel("Pressure (mbar)"); ax1b.grid(True)
    
    ax1c.plot(df_sampled["timestamp"], df_sampled["wave_pressure"], 'r-', label="Tekanan Gelombang (High-pass)")
    ax1c.set_ylabel("Pressure (mbar)"); ax1c.grid(True)
    ax1c.set_xlim(plot_t_min, plot_t_max)
    
    text_box1 = fig1.text(0.01, 0.98, "Pilih rentang...", transform=fig1.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor='black'))
    span1 = SpanSelector(ax1a, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box1, fig1, df_global, 
                                            {'pressure_mbar': '.2f', 'tide_pressure': '.2f', 'wave_pressure': '.2f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='gray'))

    # --- Plot 2: Kedalaman dan Elevasi (GABUNGAN - Dipertahankan) ---
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig2.canvas.manager.set_window_title('Analisis Kedalaman dan Elevasi (Gabungan)')
    
    ax2a.plot(df_sampled["timestamp"], df_sampled["depth"], 'g-', label="Kedalaman Total Instan", alpha=0.8)
    ax2a.axhline(y=config.water_depth, color='b', linestyle='--', label=f"Kedalaman Rata-rata ({config.water_depth:.2f} m)")
    ax2a.axhline(y=config.sensor_height, color='r', linestyle='--', label=f"Tinggi Sensor ({config.sensor_height:.2f} m)")
    ax2a.set_ylabel("Kedalaman (m)"); ax2a.set_title("Kedalaman Air Total (Pasang Surut + Gelombang)"); ax2a.legend(); ax2a.grid(True)
    
    ax2b.plot(df_sampled["timestamp"], df_sampled["surface_elevation"], 'r-', label="Elevasi Permukaan", alpha=0.7)
    ax2b.set_ylabel("Elevasi (m)"); ax2b.set_title("Elevasi Permukaan (Hanya Gelombang)"); ax2b.grid(True)
    ax2a.set_xlim(plot_t_min, plot_t_max)
    
    text_box2 = fig2.text(0.01, 0.98, "Pilih rentang...", transform=fig2.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor='black'))
    span2 = SpanSelector(ax2a, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box2, fig2, df_global, 
                                            {'depth': '.2f', 'surface_elevation': '.3f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='green'))

    # --- Plot 3: Kedalaman Air Total (INDIVIDU - BARU v3.6) ---
    fig3, ax3 = plt.subplots(figsize=(16, 7))
    fig3.canvas.manager.set_window_title('Kedalaman Air Total (Individu)')
    ax3.plot(df_sampled["timestamp"], df_sampled["depth"], 'g-', label="Kedalaman Total Instan", alpha=0.8)
    ax3.axhline(y=config.water_depth, color='b', linestyle='--', label=f"Kedalaman Rata-rata ({config.water_depth:.2f} m)")
    ax3.axhline(y=config.sensor_height, color='r', linestyle='--', label=f"Tinggi Sensor ({config.sensor_height:.2f} m)")
    ax3.set_ylabel("Kedalaman (m)"); ax3.set_title("Kedalaman Air Total (Pasang Surut + Gelombang)"); ax3.legend(); ax3.grid(True)
    ax3.set_xlim(plot_t_min, plot_t_max)
    
    text_box3 = fig3.text(0.01, 0.98, "Pilih rentang...", transform=fig3.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor='black'))
    span3 = SpanSelector(ax3, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box3, fig3, df_global, {'depth': '.2f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='green'))

    # --- Plot 4: Elevasi Permukaan (INDIVIDU - BARU v3.6) ---
    fig4, ax4 = plt.subplots(figsize=(16, 7))
    fig4.canvas.manager.set_window_title('Elevasi Permukaan (Individu)')
    ax4.plot(df_sampled["timestamp"], df_sampled["surface_elevation"], 'r-', label="Elevasi Permukaan", alpha=0.7)
    ax4.set_ylabel("Elevasi (m)"); ax4.set_title("Elevasi Permukaan (Hanya Gelombang)"); ax4.grid(True)
    ax4.set_xlim(plot_t_min, plot_t_max)
    
    text_box4 = fig4.text(0.01, 0.98, "Pilih rentang...", transform=fig4.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor='black'))
    span4 = SpanSelector(ax4, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box4, fig4, df_global, {'surface_elevation': '.3f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='red'))

    # --- Plot 5: Statistik Tinggi Gelombang ---
    fig5, ax5 = plt.subplots(figsize=(16, 7))
    fig5.canvas.manager.set_window_title('Statistik Tinggi Gelombang')
    ax5.plot(df_results['timestamp'], df_results['Hs'], 'bo-', label='Hs (Zero-crossing)', markersize=4, alpha=0.7)
    ax5.plot(df_spectral['timestamp'], df_spectral['Hm0'], 'ro:', label='Hm0 (Spectral)', markersize=4, alpha=0.7)
    ax5.plot(df_results['timestamp'], df_results['Hmax'], 'g-', label='Hmax (Zero-crossing)', markersize=2, alpha=0.5)
    ax5.set_ylabel('Tinggi Gelombang (m)'); ax5.set_title('Perbandingan Statistik Tinggi Gelombang'); ax5.grid(True); ax5.legend()
    ax5.set_xlim(plot_t_min_stats, plot_t_max_stats)
    
    text_box5 = fig5.text(0.01, 0.98, "Pilih rentang...", transform=fig5.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8))
    span5 = SpanSelector(ax5, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box5, fig5, 
                                            pd.merge(df_results, df_spectral, on='timestamp'), 
                                            {'Hs': '.3f', 'Hm0': '.3f', 'Hmax': '.3f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='cyan'))

    # --- Plot 6: Statistik Periode Gelombang ---
    fig6, ax6 = plt.subplots(figsize=(16, 7))
    fig6.canvas.manager.set_window_title('Statistik Periode Gelombang')
    ax6.plot(df_results['timestamp'], df_results['Tm'], 'bo-', label='Tm (Zero-crossing)', markersize=4, alpha=0.7)
    ax6.plot(df_spectral['timestamp'], df_spectral['Tp'], 'go:', label='Tp (Spectral Peak)', markersize=4, alpha=0.7)
    ax6.plot(df_spectral['timestamp'], df_spectral['T02'], 'ro:', label='T02 (Spectral Zero-cross)', markersize=4, alpha=0.7)
    ax6.set_ylabel('Periode (s)'); ax6.set_title('Perbandingan Statistik Periode Gelombang'); ax6.grid(True); ax6.legend()
    ax6.set_xlim(plot_t_min_stats, plot_t_max_stats)
    
    text_box6 = fig6.text(0.01, 0.98, "Pilih rentang...", transform=fig6.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8))
    span6 = SpanSelector(ax6, lambda xmin, xmax: onselect_plot(xmin, xmax, text_box6, fig6, 
                                            pd.merge(df_results, df_spectral, on='timestamp'), 
                                            {'Tm': '.2f', 'Tp': '.2f', 'T02': '.2f'}),
        'horizontal', useblit=True, interactive=True, props=dict(alpha=0.3, facecolor='magenta'))
        
    # --- Plot 7: Spektrum Rata-rata ---
    fig7, ax7 = plt.subplots(figsize=(16, 7))
    fig7.canvas.manager.set_window_title('Spektrum Gelombang Rata-rata')
    
    avg_freqs = df_spectral['freqs'].iloc[0]
    avg_psd = np.mean([s for s in df_spectral['psd']], axis=0)
    
    ax7.loglog(avg_freqs, avg_psd, 'b-', linewidth=2)
    peak_idx = np.argmax(avg_psd)
    peak_freq = avg_freqs[peak_idx]
    peak_psd_val = avg_psd[peak_idx]
    ax7.plot(peak_freq, peak_psd_val, 'ro', markersize=8, label=f'Puncak (Max): {peak_freq:.3f} Hz')
    
    # [NEW v3.6] Tampilkan Puncak Prominen (Tp)
    tp_freq_avg = 1.0 / df_spectral['Tp'].mean()
    if 0 < tp_freq_avg < config.fs / 2:
        tp_psd_val = np.interp(tp_freq_avg, avg_freqs, avg_psd)
        ax7.plot(tp_freq_avg, tp_psd_val, 'gx', markersize=10, markeredgewidth=3, label=f'Puncak (Prominen Tp): {tp_freq_avg:.3f} Hz')

    ax7.set_xlabel('Frekuensi (Hz)'); ax7.set_ylabel('PSD (m²/Hz)')
    ax7.set_title('Spektrum Kepadatan Daya (PSD) Rata-rata')
    ax7.grid(True, which="both"); ax7.legend()
    ax7.set_xlim(config.wave_low_freq * 0.8, config.wave_high_freq * 1.2)
    
    text_box7 = fig7.text(0.01, 0.98, "", transform=fig7.transFigure,
                          fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8))
    
    def onselect_spectrum(xmin, xmax):
        try:
            mask = (avg_freqs >= xmin) & (avg_freqs <= xmax)
            if np.any(mask):
                freqs_range = avg_freqs[mask]
                psd_range = avg_psd[mask]
                m0_range = simps(psd_range, dx=(freqs_range[1]-freqs_range[0]))
                Hm0_range = 4 * np.sqrt(m0_range)
                msg = (f"📊 Rentang Frekuensi Dipilih:\n"
                       f"   {xmin:.3f} Hz → {xmax:.3f} Hz\n"
                       f"   m0 (area): {m0_range:.4f} m²\n"
                       f"   Hm0 (area): {Hm0_range:.3f} m")
                text_box7.set_text(msg)
            else:
                text_box7.set_text("[Tidak ada data spektral di rentang]")
            fig7.canvas.draw_idle()
        except Exception as e:
            text_box7.set_text(f"❌ Error: {str(e)}")
            fig7.canvas.draw_idle()

    span7 = SpanSelector(ax7, onselect_spectrum, 'horizontal', useblit=True, interactive=True,
        props=dict(alpha=0.3, facecolor='grey'))

    logger.info("\n" + "="*60)
    logger.info("📊 Plot interaktif siap!")
    logger.info("💡 Klik dan seret (drag) pada plot untuk memilih rentang data.")
    logger.info("="*60)
    
    plt.show(block=True)
    
    # Return list of spans agar tetap interaktif
    return [span1, span2, span3, span4, span5, span6, span7]

# --- Fungsi Analisis Utama ---
def analyze_wave_data(file_path, config, analysis_start=None, analysis_end=None, baseline_start=None, baseline_end=None):
    """
    Fungsi utama untuk analisis data gelombang.
    [IMPROVED v3.5] Menambahkan filter waktu untuk analisis dan baseline.
    """
    logger.info(f"📂 Membaca data dari: {file_path}")
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
    except FileNotFoundError:
        logger.error(f"KESALAHAN: File tidak ditemukan di {file_path}")
        return None, None, None, config
    except Exception as e:
        logger.error(f"KESALAHAN: Gagal membaca file. Error: {e}")
        return None, None, None, config
        
    required_columns = ['timestamp', 'pressure_mbar']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"KESALAHAN: Kolom yang diperlukan tidak ditemukan: {missing_columns}")
        return None, None, None, config
    
    # --- [FIX v3.2] Parsing Timestamp dengan Format Eksplisit ---
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        logger.info("   [INFO] Timestamp berhasil dibaca dengan format milidetik.")
    except ValueError:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            logger.warning("   [WARN] Timestamp dibaca tanpa milidetik.")
        except Exception as e:
            logger.error(f"KESALAHAN: Gagal mem-parsing timestamp. Cek format data.")
            logger.error(f"   Error: {e}")
            return None, None, None, config
    
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # --- [NEW v3.5] Logika Baseline Atmosfer yang Diperbarui ---
    if config.MANUAL_AIR_BASELINE is not None:
        air_baseline = config.MANUAL_AIR_BASELINE
        baseline_source = "MANUAL"
    elif baseline_start and baseline_end:
        try:
            baseline_start_dt = pd.to_datetime(baseline_start)
            baseline_end_dt = pd.to_datetime(baseline_end)
            baseline_df = df[(df['timestamp'] >= baseline_start_dt) & (df['timestamp'] <= baseline_end_dt)]
            
            if baseline_df.empty:
                logger.error(f"KESALAHAN: Tidak ada data baseline ditemukan antara {baseline_start} and {baseline_end}.")
                return None, None, None, config
                
            air_baseline = baseline_df['pressure_mbar'].mean()
            baseline_source = f"MANUAL RANGE ({baseline_start} to {baseline_end})"
        except Exception as e:
            logger.error(f"KESALAHAN: Gagal mem-parsing waktu baseline. Gunakan format 'YYYY-MM-DD HH:MM:SS'. Error: {e}")
            return None, None, None, config
    else:
        min_pressure = df["pressure_mbar"].min()
        p1_pressure = df["pressure_mbar"].quantile(0.01)
        if p1_pressure - min_pressure > 5:
            air_baseline = p1_pressure
            baseline_source = "OTOMATIS (Persentil 1)"
            logger.warning(f"   [WARN] Minimum pressure ({min_pressure:.2f} mbar) terlihat outlier. Menggunakan 1st percentile.")
        else:
            air_baseline = min_pressure
            baseline_source = "OTOMATIS (Minimum)"
            
        if air_baseline < 950 or air_baseline > 1050: 
            logger.warning(f"   [WARN] Baseline otomatis {air_baseline:.2f} mbar terlihat tidak wajar (di luar 950-1050).")
            logger.warning("          Jika ini salah (misal, sensor sudah di air), gunakan '--baseline_start' dan '--baseline_end' untuk menentukan periode 'in-air'.")
        
    config.avg_atm = air_baseline
    logger.info(f"   [INFO] Baseline atmosfer: {config.avg_atm:.2f} mbar ({baseline_source})")
    # -------------------------------------------------------------

    # --- [NEW v3.5] Filter Data untuk Analisis ---
    if analysis_start:
        try:
            analysis_start_dt = pd.to_datetime(analysis_start)
            df = df[df['timestamp'] >= analysis_start_dt].copy()
            logger.info(f"   [INFO] Memfilter data. Analisis dimulai dari: {analysis_start}")
        except Exception as e:
            logger.error(f"KESALAHAN: Gagal mem-parsing --analysis_start: {e}")
            return None, None, None, config
            
    if analysis_end:
        try:
            analysis_end_dt = pd.to_datetime(analysis_end)
            df = df[df['timestamp'] <= analysis_end_dt].copy()
            logger.info(f"   [INFO] Memfilter data. Analisis berakhir pada: {analysis_end}")
        except Exception as e:
            logger.error(f"KESALAHAN: Gagal mem-parsing --analysis_end: {e}")
            return None, None, None, config

    df = df.reset_index(drop=True)
    if df.empty:
        logger.error("KESALAHAN: Tidak ada data tersisa setelah filtering waktu. Periksa rentang --analysis_start/end.")
        return None, None, None, config
    # -------------------------------------------------------------
    
    time_diff = df['timestamp'].diff().median().total_seconds()
    if time_diff > 0:
        config.fs = 1.0 / time_diff
        logger.info(f"   [INFO] Frekuensi sampling (fs) terdeteksi: {config.fs:.2f} Hz")
    else:
        logger.error(f"KESALAHAN: Gagal mendeteksi fs. Perbedaan waktu median adalah {time_diff}.")
        return None, None, None, config

    if 'temp_C' in df.columns:
        df['rho'] = calculate_seawater_density(df['temp_C'], config.salinity)
        df['rho'] = df['rho'].fillna(config.rho_default)
        logger.info(f"   [INFO] Densitas dinamis dihitung. Rata-rata: {df['rho'].mean():.2f} kg/m³")
    else:
        df['rho'] = config.rho_default
        logger.info(f"   [WARN] Kolom 'temp_C' tidak ditemukan. Menggunakan rho default: {config.rho_default} kg/m³")

    logger.info("🔄 Menghitung kedalaman air...")
    df['h_above_sensor'] = ((df['pressure_mbar'] - config.avg_atm) * 100) / (df['rho'] * config.grav)
    
    negative_count = (df['h_above_sensor'] < 0).sum()
    if negative_count > 0:
        logger.warning(f"   [WARN] {negative_count} nilai negatif di h_above_sensor (mungkin karena gelombang). Mengubah ke 0.")
        df.loc[df['h_above_sensor'] < 0, 'h_above_sensor'] = 0
    
    df['depth'] = df['h_above_sensor'] + config.sensor_height
    avg_total_depth = df['depth'].mean()
    config.water_depth = avg_total_depth
    
    logger.info(f"\n📏 INFORMASI KEDALAMAN (setelah filter):")
    logger.info(f"   Kedalaman Air Rata-rata (Total): {avg_total_depth:.3f} m")
    logger.info(f"   Tinggi Sensor dari Dasar: {config.sensor_height:.2f} m")
    
    validate_wave_theory(config, avg_total_depth)
    
    logger.info("\n🌊 Memisahkan komponen pasut dan gelombang...")
    tide_pressure, wave_pressure = separate_tide_waves(df['pressure_mbar'], config)
    df['tide_pressure'] = tide_pressure
    df['wave_pressure'] = wave_pressure
    
    logger.info("📈 Konversi tekanan gelombang ke elevasi permukaan...")
    df['surface_elevation'] = pressure_to_surface_elevation(wave_pressure, config, avg_total_depth, df['rho'])
    
    config.stat_window_sec = calculate_optimal_window(df, config)
    
    logger.info(f"\n📊 Menghitung statistik gelombang (window: {config.stat_window_sec}s)...")
    window_size = int(config.stat_window_sec * config.fs)
    step_size = int(window_size * (1 - config.window_overlap))
    results = []
    spectral_results = []
    
    if window_size > len(df):
        logger.error(f"KESALAHAN: Data (panjang {len(df)}) lebih pendek dari window ({window_size}).")
        return None, None, None, config

    window_count = 0
    for i in range(0, len(df) - window_size + 1, step_size):
        start = i
        end = i + window_size
        eta_window = df['surface_elevation'].iloc[start:end].values
        if len(eta_window) == 0:
            continue
            
        stats = calculate_wave_statistics(eta_window, config)
        spectral = spectral_analysis(eta_window, config)
        mid_idx = start + window_size // 2
        
        results.append({
            'timestamp': df['timestamp'].iloc[mid_idx],
            'Hs': stats['Hs'], 'Tm': stats['Tm'], 'Hmax': stats['Hmax'],
            'Havg': stats['Havg'], 'Nwaves': stats['Nwaves']
        })
        
        spectral_results.append({
            'timestamp': df['timestamp'].iloc[mid_idx],
            'Hm0': spectral['Hm0'], 'Tp': spectral['Tp'], 'T01': spectral['T01'],
            'T02': spectral['T02'], 'epsilon': spectral['epsilon'], 'Qp': spectral['Qp'],
            'freqs': spectral['freqs'], 'psd': spectral['psd'],
            'Tm02': spectral['Tm02'], 'Tm10': spectral['Tm10'], 'S_max': spectral['S_max'], 'sigma': spectral['sigma']
        })
        window_count += 1
    
    if window_count == 0:
        logger.error("KESALAHAN: Tidak ada window yang diproses. Cek panjang data dan ukuran window.")
        return None, None, None, config

    logger.info(f"   ✅ Memproses {window_count} window")
    
    df_results = pd.DataFrame(results)
    df_spectral = pd.DataFrame(spectral_results)
    
    return df, df_results, df_spectral, config

# --- Fungsi Main dengan Argparse ---
def main():
    parser = argparse.ArgumentParser(
        description="Advanced Wave Analysis Pipeline (v3.6 - Robust & Comprehensive)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Contoh Penggunaan:

1. Analisis Dasar (dengan path Windows & nama kustom):
   python %(prog)s -f "F:\\Kuliah\\CoastalResearch\\Data\\Skripsi\\cores_wg\\data raw\\cores_wg2_coba_fixed.txt" -n "cores_wg2_results"

2. Analisis Lanjutan (Filter Baseline & Waktu Analisis):
   python %(prog)s -f "data.txt" -n "hasil_filter" \
     --baseline_start "2025-09-06 13:00:00" \
     --baseline_end "2025-09-06 13:50:00" \
     --analysis_start "2025-09-06 14:00:00"

3. Analisis dengan Validasi ERA5 (Untuk Skripsi Transportasi Sedimen):
   python %(prog)s -f "data.txt" -n "hasil_validasi" \
     --analysis_start "2025-09-06 14:00:00" \
     --era5 --era5_file "era5_data.nc" --gauge_lat -8.45 --gauge_lon 112.68
"""
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path ke file input CSV (misal: 'data_input.txt')"
    )
    parser.add_argument(
        "-o", "--outdir",
        default=None,
        help="Direktori output kustom. Default: [direktori_input]/results"
    )
    parser.add_argument(
        "-n", "--name",
        default=None,
        help="Nama dasar kustom untuk file output. Default: [nama_file_input]"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Manual override untuk baseline tekanan atmosfer (mbar). Default: Otomatis"
    )
    # [NEW v3.5] Argumen untuk filter waktu
    parser.add_argument(
        "--analysis_start",
        default=None,
        help="Waktu mulai analisis (Format: 'YYYY-MM-DD HH:MM:SS'). Memotong data sebelum waktu ini."
    )
    parser.add_argument(
        "--analysis_end",
        default=None,
        help="Waktu selesai analisis (Format: 'YYYY-MM-DD HH:MM:SS'). Memotong data setelah waktu ini."
    )
    parser.add_argument(
        "--baseline_start",
        default=None,
        help="Waktu mulai untuk kalkulasi baseline 'in-air' (Format: 'YYYY-MM-DD HH:MM:SS')."
    )
    parser.add_argument(
        "--baseline_end",
        default=None,
        help="Waktu selesai untuk kalkulasi baseline 'in-air' (Format: 'YYYY-MM-DD HH:MM:SS')."
    )
    # Argumen standar
    parser.add_argument(
        "--offset",
        type=float,
        default=0.25,
        help="Tinggi sensor dari dasar laut (m). Default: 0.25"
    )
    parser.add_argument(
        "--salinity",
        type=float,
        default=35.0,
        help="Asumsi salinitas (PSU). Default: 35.0"
    )
    # Argumen ERA5
    parser.add_argument(
        "--era5",
        action="store_true",
        help="Enable ERA5 comparison"
    )
    parser.add_argument(
        "--era5_file",
        default=None,
        help="Path ke file data ERA5 (NetCDF atau CSV)"
    )
    parser.add_argument(
        "--gauge_lat",
        type=float,
        default=None,
        help="Latitude lokasi wave gauge"
    )
    parser.add_argument(
        "--gauge_lon",
        type=float,
        default=None,
        help="Longitude lokasi wave gauge"
    )
    parser.add_argument(
        "--delft3d",
        action="store_true",
        help="Enable Delft3D-FM comparison"
    )
    
    args = parser.parse_args()
    
    # Inisialisasi konfigurasi
    config = WaveConfig()
    config.MANUAL_AIR_BASELINE = args.baseline
    config.sensor_height = args.offset
    config.salinity = args.salinity
    config.era5_comparison = args.era5
    config.delft3d_comparison = args.delft3d
    config.era5_file = args.era5_file
    config.gauge_lat = args.gauge_lat
    config.gauge_lon = args.gauge_lon
    
    # [FIX v3.4] Pindahkan definisi output_path ke sini
    output_path, output_directory = get_output_path(args.file, args.name, args.outdir)
    
    logger.info(f"🚀 Memulai Analisis Gelombang (v3.6)...")
    logger.info(f"   [CONFIG] File Input: {args.file}")
    logger.info(f"   [CONFIG] Direktori Output: {output_directory}")
    logger.info(f"   [CONFIG] Nama Output: {os.path.basename(output_path)}")
    logger.info(f"   [CONFIG] Tinggi Sensor: {config.sensor_height} m")
    
    # Load data ERA5 jika diperlukan
    era5_data = None
    if config.era5_comparison and config.era5_file and config.gauge_lat is not None and config.gauge_lon is not None:
        logger.info(f"Loading ERA5 data from {config.era5_file}")
        era5_data = load_era5_data(config.era5_file, config.gauge_lat, config.gauge_lon)
        if era5_data is not None:
            logger.info(f"Loaded {len(era5_data)} ERA5 records")
        else:
            logger.error("Failed to load ERA5 data")
    
    # Analisis data
    # [NEW v3.5] Teruskan argumen filter waktu
    df, df_results, df_spectral, config = analyze_wave_data(
        args.file, config,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end
    )
    
    if df is None:
        logger.error("Analisis gagal. Keluar.")
        sys.exit(1)
    
    # Export hasil
    logger.info("\n💾 Mengekspor hasil...")
    export_results(df, df_results, df_spectral, output_path, output_directory, config)
    
    # Export ke MATLAB
    export_to_matlab_with_script(df, df_results, df_spectral, output_path, config, output_directory)
    
    # [NEW] Validation against ERA5 if requested
    if config.era5_comparison and era5_data is not None:
        logger.info("\n🔍 Memvalidasi dengan data ERA5...")
        era5_results, era5_stats = validate_against_era5(df_results, df_spectral, era5_data)
        if era5_results is not None:
            era5_filename = os.path.join(output_directory, f"{os.path.basename(output_path)}_era5_validation.csv")
            era5_results.to_csv(era5_filename, index=False)
            logger.info(f"   ✅ Hasil validasi ERA5 disimpan ke {era5_filename}")
            
            # [NEW v3.6] Gabungkan data untuk plot ERA5
            df_plot_era5 = pd.merge(df_results, df_spectral, on='timestamp', suffixes=('_zc', '_spec'))
            plot_era5_comparison(df_plot_era5, era5_results, output_directory, os.path.basename(output_path))
    
    # [NEW] Validation against Delft3D-FM if requested
    if config.delft3d_comparison:
        logger.info("\n🔍 Memvalidasi dengan data Delft3D-FM...")
        delft3d_results = validate_against_delft3d(df_results, df_spectral) # delft3d_data=... perlu ditambahkan
        if delft3d_results:
            delft3d_filename = os.path.join(output_directory, f"{os.path.basename(output_path)}_delft3d_validation.csv")
            delft3d_results.to_csv(delft3d_filename, index=False)
            logger.info(f"   ✅ Hasil validasi Delft3D-FM disimpan ke {delft3d_filename}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ANALISIS SELESAI")
    logger.info("="*60)
    
    # Tampilkan ringkasan statistik
    logger.info("\n📊 RINGKASAN STATISTIK GELOMBANG (Rata-rata dari semua window):")
    logger.info("="*50)
    logger.info("METODE ZERO-CROSSING (v3.4 - Akurat):")
    logger.info(f"   Tinggi Gel. Signifikan (Hs): {df_results['Hs'].mean():.3f} m")
    logger.info(f"   Periode Rata-rata (Tm): {df_results['Tm'].mean():.3f} s")
    logger.info(f"   Tinggi Gel. Maksimum (Hmax, absolut): {df_results['Hmax'].max():.3f} m")
    logger.info("\nMETODE SPEKTRAL (v3.6 - Robust Tp):")
    logger.info(f"   Tinggi Gel. Spektral (Hm0): {df_spectral['Hm0'].mean():.3f} m")
    logger.info(f"   Periode Puncak (Tp): {df_spectral['Tp'].mean():.3f} s")
    logger.info(f"   Periode Zero-crossing (T02): {df_spectral['T02'].mean():.3f} s")
    
    # [NEW] Comparison between methods
    logger.info("\nPERBANDINGAN METODE:")
    if df_spectral['Hm0'].mean() > 0: # Hindari Sembagi nol
        logger.info(f"   Hs/Hm0: {df_results['Hs'].mean()/df_spectral['Hm0'].mean():.3f}")
    if df_spectral['T02'].mean() > 0:
        logger.info(f"   Tm/T02: {df_results['Tm'].mean()/df_spectral['T02'].mean():.3f}")
    
    # Tampilkan plot interaktif
    plot_with_spanselector(df, df_results, df_spectral, config)

if __name__ == "__main__":
    main()