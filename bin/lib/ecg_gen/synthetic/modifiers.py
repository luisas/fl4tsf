"""
ECG signal modifiers.
Only signal modification functions belong here: artifacts, events, physiological changes.
All medication/comorbidity logic and main processing pipeline is in core.py.

Registry layout:
    MODIFIERS["physiological"][name]
    MODIFIERS["artifact"][name]  
    MODIFIERS["event"][name]
"""

from __future__ import annotations
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import neurokit3 as nk
from scipy.signal import resample_poly
# =============================================================================
# Constants
# =============================================================================

BASE_QT_MS = 400  # Baseline QT interval in milliseconds
NOMINAL_ECG_MV = 1.0  # Nominal ECG amplitude for noise scaling

try:
    profile
except NameError:
    def profile(f): return f

# =============================================================================
# Registry System (for signal modifications only)
# =============================================================================
MODIFIERS: Dict[str, Dict[str, Any]] = {
    "physiological": {},
    "artifact": {},
    "event": {},
    "morphology": {},  
}

def register_modifier(category: str, name: str):
    """Decorator: @register_modifier('artifact', 'baseline_wander')"""
    def _inner(fn):
        MODIFIERS.setdefault(category, {})[name] = fn
        return fn
    return _inner

# =============================================================================
# Fast Noise Helper Functions
# =============================================================================

def _add_sinusoidal_noise(ecg_signal: np.ndarray, sampling_rate: int, 
                          severity: float, base_frequency: float,
                          num_components: int = 3) -> np.ndarray:
    """
    Fast helper to add noise composed of several sinusoids for natural feel.
    Replaces slow nk.signal_distort for this purpose.
    """
    duration = len(ecg_signal) / sampling_rate
    time_axis = np.linspace(0, duration, len(ecg_signal), endpoint=False)

    noise = np.zeros_like(ecg_signal, dtype=np.float32)
    for _ in range(num_components):
        # Randomize frequency, amplitude, and phase for each component
        rand_freq = base_frequency * (1 + np.random.uniform(-0.4, 0.4))
        rand_amp = severity * np.random.uniform(0.5, 1.5) / num_components
        phase = np.random.uniform(0, 2 * np.pi)
        noise += rand_amp * np.sin(2 * np.pi * rand_freq * time_axis + phase)
    return ecg_signal + noise

# =============================================================================
# PHYSIOLOGICAL MODIFIERS
# =============================================================================

@register_modifier("physiological", "circadian_modulation")
def circadian_modulation(
    time_axis: np.ndarray,
    ecg_signal: np.ndarray,
    *,
    hour_of_day: float,
    archetype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the entire ECG trace by a smooth circadian envelope."""
    phase = 2 * np.pi * hour_of_day / 24
    amp = 0.15 if archetype != "athlete" else 0.25
    scale = 1.0 + amp * np.sin(phase - np.pi / 3)  # peak ≈ 16:00
    return time_axis, ecg_signal * scale

# =============================================================================
# ARTIFACT MODIFIERS
# =============================================================================

@register_modifier("artifact", "baseline_wander")
def add_baseline_wander(ecg_signal: np.ndarray, *, sampling_rate: int, severity: float = 0.05) -> np.ndarray:
    """Add low-frequency baseline wander using fast sinusoidal method."""
    return _add_sinusoidal_noise(ecg_signal, sampling_rate, severity, base_frequency=0.1)

@register_modifier("artifact", "powerline_noise")
def add_powerline_noise(ecg_signal: np.ndarray, *, sampling_rate: int, 
                       severity: float = 0.01, frequency_hz: int = 60) -> np.ndarray:
    """Add powerline interference at specific frequency."""
    duration = len(ecg_signal) / sampling_rate
    time_axis = np.linspace(0, duration, len(ecg_signal), endpoint=False)
    noise = severity * np.sin(2 * np.pi * frequency_hz * time_axis)
    return ecg_signal + noise.astype(np.float32)

@register_modifier("artifact", "motion_artifact")
def add_motion_artifact(ecg_signal: np.ndarray, *, sampling_rate: int, severity: float = 0.02) -> np.ndarray:
    """Add motion-related artifacts using fast sinusoidal method."""
    return _add_sinusoidal_noise(ecg_signal, sampling_rate, severity, base_frequency=1.5)

# =============================================================================
# EVENT MODIFIERS
# =============================================================================

@register_modifier("event", "add_pvc")
def add_pvc_events(
    ecg_signal: np.ndarray,
    *,
    sampling_rate: int,
    event_times: Sequence[float],
) -> np.ndarray:
    """Insert PVCs at given second timestamps."""
    if not event_times:
        return ecg_signal
    
    idx = (np.asarray(event_times) * sampling_rate).astype(int)
    idx = idx[(idx >= 0) & (idx < len(ecg_signal))]
    if len(idx) == 0:
        return ecg_signal
    
    try:
        return nk.ecg_add_ectopics(ecg_signal, sampling_rate=sampling_rate, ectopic_index=idx)
    except Exception:
        # Simple inversion fallback
        pvc_len = sampling_rate // 8
        out = ecg_signal.copy()
        for i in idx:
            end_idx = min(i + pvc_len, len(out))
            out[i:end_idx] *= -1.5
        return out

@register_modifier("event", "add_atrial_fib")
def add_atrial_fibrillation(
    ecg_signal: np.ndarray,
    *,
    sampling_rate: int,
    start_s: float,
    duration_s: float,
) -> np.ndarray:
    """Blend an irregular high-HR segment to mimic AF."""
    start_idx = int(start_s * sampling_rate)
    end_idx = int((start_s + duration_s) * sampling_rate)
    
    if end_idx - start_idx < sampling_rate or end_idx > len(ecg_signal):
        return ecg_signal

    # Estimate baseline HR
    segment = ecg_signal[max(0, start_idx - sampling_rate):start_idx]
    try:
        _, info = nk.ecg_peaks(segment, sampling_rate=sampling_rate)
        r_peaks = info.get("ECG_R_Peaks", [])
        if len(r_peaks) > 1:
            rr = np.diff(r_peaks) / sampling_rate
            mean_hr = 60 / rr.mean()
        else:
            mean_hr = 80.0
    except Exception:
        mean_hr = 80.0

    try:
        af_sig = nk.ecg_simulate(
            duration=duration_s,
            sampling_rate=sampling_rate,
            heart_rate=mean_hr * 1.2,
            heart_rate_std=mean_hr * 0.3,
        )
    except Exception:
        # Fallback: simple noise
        af_sig = ecg_signal[start_idx:end_idx] * 1.3 + np.random.normal(0, 0.1, end_idx - start_idx)

    fade_len = min(sampling_rate // 4, (end_idx - start_idx) // 4)
    out = ecg_signal.copy()
    
    if fade_len > 0 and len(af_sig) > 2 * fade_len:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = fade_in[::-1]
        
        out[start_idx:start_idx + fade_len] = (
            (1 - fade_in) * out[start_idx:start_idx + fade_len]
            + fade_in * af_sig[:fade_len]
        )
        out[start_idx + fade_len:end_idx - fade_len] = af_sig[fade_len:-fade_len]
        out[end_idx - fade_len:end_idx] = (
            fade_out * af_sig[-fade_len:] + (1 - fade_out) * out[end_idx - fade_len:end_idx]
        )
    else:
        out[start_idx:end_idx] = af_sig[:end_idx - start_idx]
    
    return out

# =============================================================================
# MORPHOLOGY MODIFIERS (Add to existing modifiers.py)
# =============================================================================

@register_modifier("morphology", "old_mi")
def add_old_mi_morphology(ecg_signal: np.ndarray, *, sampling_rate: int) -> np.ndarray:
    """
    Apply old MI morphology using in-place beat template modification.
    Creates pathological Q-waves and poor R-wave progression.
    """
    try:
        # Find R-peaks to segment beats
        _, rpeaks_info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate, correct_artifacts=False)
        rpeaks = rpeaks_info['ECG_R_Peaks']
        
        if len(rpeaks) < 3:
            return ecg_signal
        
        # Work directly on copy to avoid O(N²) memory allocation
        modified_signal = ecg_signal.copy()
        
        # Segment beats around R-peaks for template creation
        beat_window = int(0.6 * sampling_rate)  # 600ms window
        half_window = beat_window // 2
        beats = []
        
        for rpeak in rpeaks:
            start = rpeak - half_window
            end = rpeak + half_window
            if start >= 0 and end < len(ecg_signal):  # Only complete beats
                beats.append(ecg_signal[start:end])
        
        if len(beats) < 2:
            return ecg_signal
            
        # Create average template
        template = np.mean(beats, axis=0)
        
        # Pre-calculate modification parameters
        r_peak_idx = half_window
        r_amplitude = np.max(np.abs(template))
        
        # 1. Q-wave modification parameters
        q_start = max(0, r_peak_idx - int(0.08 * sampling_rate))
        q_end = min(beat_window, r_peak_idx - int(0.02 * sampling_rate))
        q_depth = 0.3 * r_amplitude
        
        # 2. R-wave reduction parameters  
        r_start = max(0, r_peak_idx - int(0.02 * sampling_rate))
        r_end = min(beat_window, r_peak_idx + int(0.04 * sampling_rate))
        
        # 3. T-wave modification parameters
        t_start = max(0, r_peak_idx + int(0.15 * sampling_rate))
        t_end = min(beat_window, r_peak_idx + int(0.35 * sampling_rate))
        
        # Apply modifications in-place to each beat
        for rpeak in rpeaks:
            beat_start = rpeak - half_window
            beat_end = rpeak + half_window
            
            if beat_start >= 0 and beat_end < len(modified_signal):
                # Apply Q-wave carving
                if q_end > q_start:
                    q_center = (q_start + q_end) // 2
                    q_width = q_end - q_start
                    for i in range(q_start, q_end):
                        gaussian = np.exp(-0.5 * ((i - q_center) / (q_width/4)) ** 2)
                        modified_signal[beat_start + i] -= q_depth * gaussian
                
                # Apply R-wave reduction
                modified_signal[beat_start + r_start:beat_start + r_end] *= 0.6
                
                # Apply T-wave flattening
                if t_end > t_start:
                    modified_signal[beat_start + t_start:beat_start + t_end] *= 0.2
        
        # Clip to reasonable ECG amplitudes to avoid validation failures
        return np.clip(modified_signal, -5.0, 5.0).astype(np.float32)
        
    except Exception:
        return ecg_signal


@register_modifier("morphology", "chronic_af")  
def add_chronic_atrial_fibrillation(ecg_signal: np.ndarray, *, sampling_rate: int) -> np.ndarray:
    """
    Replace the entire signal with chronic atrial fibrillation rhythm.
    Creates irregularly irregular rhythm with no discernible P-waves.
    """
    try:
        # Estimate baseline HR from original signal
        _, rpeaks_info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        rpeaks = rpeaks_info['ECG_R_Peaks']
        
        if len(rpeaks) > 1:
            rr_intervals = np.diff(rpeaks) / sampling_rate
            mean_hr = 60 / np.mean(rr_intervals)
        else:
            mean_hr = 80.0
        
        # AF characteristics: faster rate, high variability
        af_hr = np.clip(mean_hr * 1.2, 90, 150)  # AF typically 90-150 bpm
        af_hrv = af_hr * 0.3  # High variability (30% of mean HR)
        
        # Generate AF signal with high irregularity
        duration = len(ecg_signal) / sampling_rate
        af_signal = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=af_hr,
            heart_rate_std=af_hrv,
            random_state=42
        )
        
        # Resample to exact target length if needed (preserves HRV characteristics)
        if len(af_signal) != len(ecg_signal):
            # Use scipy's resample_poly for high-quality resampling
            target_len = len(ecg_signal)
            af_signal = resample_poly(af_signal, target_len, len(af_signal))
            
            # Ensure exact length match
            if len(af_signal) > target_len:
                af_signal = af_signal[:target_len]
            elif len(af_signal) < target_len:
                # If still short, repeat the last few samples
                padding_needed = target_len - len(af_signal)
                if padding_needed > 0:
                    padding = np.tile(af_signal[-min(100, len(af_signal)):], 
                                    (padding_needed // min(100, len(af_signal)) + 1))[:padding_needed]
                    af_signal = np.concatenate([af_signal, padding])
        
        # Clip to reasonable ECG amplitudes to avoid validation failures
        return np.clip(af_signal, -3.0, 3.0).astype(np.float32)
        
    except Exception:
        return ecg_signal

# =============================================================================
# Public Interface
# =============================================================================

def apply_modifier(category: str, name: str, *args, **kwargs):
    """Apply a registered modifier function."""
    if category not in MODIFIERS or name not in MODIFIERS[category]:
        raise ValueError(f"Modifier '{category}/{name}' not registered")
    return MODIFIERS[category][name](*args, **kwargs)

def list_modifiers() -> Dict[str, list]:
    """List all available modifiers by category."""
    return {cat: list(d.keys()) for cat, d in MODIFIERS.items()}