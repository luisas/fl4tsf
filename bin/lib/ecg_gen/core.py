"""
Core ECG generation logic.
Orchestrates the entire pipeline: config loading, sampling, generation, modification, validation.
Contains medication/comorbidity effects and main signal processing functions.
"""

import yaml
import numpy as np
import torch
from scipy.stats import dirichlet
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple, List
import neurokit3 as nk
from . import modifiers
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
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
# Configuration Loading
# =============================================================================

def load_configs() -> Dict[str, Any]:
    """Load all YAML configuration files."""
    config_path = Path(__file__).parent / "configs"
    configs = {}
    
    for config_file in config_path.glob("*.yaml"):
        with open(config_file, 'r') as stream:
            configs[config_file.stem] = yaml.safe_load(stream)
    
    return configs

# =============================================================================
# Medication & Comorbidity Registry (separate from signal modifiers)
# =============================================================================

CONDITION_REGISTRY: Dict[str, Dict[str, Any]] = {"medication": {}, "comorbidity": {}}

def register_condition(kind: str, name: str):
    """Decorator to register medication/comorbidity effects."""
    def _inner(fn):
        CONDITION_REGISTRY.setdefault(kind, {})[name] = fn
        return fn
    return _inner

# Medication Effects
@register_condition("medication", "beta_blocker")
def apply_beta_blocker(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] *= 0.75
    params["heart_rate_std"] *= 1.25
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.02
    return params

@register_condition("medication", "ace_inhibitor")
def apply_ace_inhibitor(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] *= 0.95
    params["heart_rate_std"] *= 1.15
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.01
    return params

@register_condition("medication", "antiarrhythmic")
def apply_antiarrhythmic(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] *= 0.90
    params["heart_rate_std"] *= 0.85
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.08
    return params

@register_condition("medication", "diuretic")
def apply_diuretic(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] *= 1.05
    params["heart_rate_std"] *= 0.90
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.03
    return params

@register_condition("medication", "digoxin")
def apply_digoxin(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] *= 0.80
    params["heart_rate_std"] *= 0.75
    params["qt_offset"] = params.get("qt_offset", 0.0) - 0.02
    return params

# Comorbidity Effects
@register_condition("comorbidity", "diabetes")
def apply_diabetes(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] += 5  # Use offset instead of multiplier for more realistic effect
    params["heart_rate_std"] *= 0.80
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.01
    return params

@register_condition("comorbidity", "copd")
def apply_copd(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] += 7
    params["heart_rate_std"] *= 0.85
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.0
    return params

@register_condition("comorbidity", "renal_disease")
def apply_renal_disease(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] += 4
    params["heart_rate_std"] *= 0.90
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.02
    return params

@register_condition("comorbidity", "sleep_apnea")
def apply_sleep_apnea(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] += 3
    params["heart_rate_std"] *= 1.15  # Cyclical brady-tachy increases variability
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.0
    return params

@register_condition("comorbidity", "heart_failure")
def apply_heart_failure(params: Dict[str, Any]) -> Dict[str, Any]:
    params = params.copy()
    params["heart_rate"] += 12
    params["heart_rate_std"] *= 0.60
    params["qt_offset"] = params.get("qt_offset", 0.0) + 0.025
    return params

def apply_condition_effects(params: Dict[str, float], effect_list: List[str], condition_type: str) -> Dict[str, float]:
    """Apply medication or comorbidity effects in sequence."""
    modified_params = params.copy()
    
    for effect_name in effect_list:
        if effect_name in CONDITION_REGISTRY[condition_type]:
            modified_params = CONDITION_REGISTRY[condition_type][effect_name](modified_params)
        else:
            print(f"Warning: {condition_type.title()} '{effect_name}' not registered, skipping")
    
    return modified_params

# =============================================================================
# Sampling Functions
# =============================================================================

def sample_client_distributions(n_clients: int, archetypes_config: Dict[str, Any], alpha: float) -> np.ndarray:
    """Sample Dirichlet distributions for archetype mix per client with normalization."""
    archetype_names = list(archetypes_config.keys())
    matrix = dirichlet.rvs([alpha] * len(archetype_names), size=n_clients)
    # Normalize to prevent zero probabilities
    return (matrix + 1e-12) / (matrix + 1e-12).sum(axis=1, keepdims=True)

def sample_patient_archetypes(client_dist: np.ndarray, n_patients: int, archetype_names: List[str]) -> List[str]:
    """Sample patient archetypes for a client based on its distribution."""
    client_dist = client_dist / client_dist.sum()  # Normalize
    return np.random.choice(archetype_names, size=n_patients, p=client_dist).tolist()

def sample_physiological_params(archetype: str, archetypes_config: Dict[str, Any]) -> Dict[str, float]:
    """Sample physiological parameters for a patient of given archetype."""
    config = archetypes_config[archetype]
    
    # Build covariance matrix and sample
    std_devs = np.array(config['sigma_std_devs'])
    corr_matrix = np.array(config['sigma_corr_matrix'])
    cov_matrix = np.diag(std_devs) @ corr_matrix @ np.diag(std_devs)
    
    hr, hrv_ms, qt_target_ms = np.random.multivariate_normal(config['mu'], cov_matrix)
    
    # Apply physiological constraints
    hr = np.clip(hr, 40, 200)
    hrv_ms = np.clip(hrv_ms, 0.5, 150)
    qt_target_ms = np.clip(qt_target_ms, 350, 500)
    
    # Convert HRV from ms to bpm for NeuroKit
    hrv_bpm = np.clip((hrv_ms / 1000) * (hr / 60) ** 2, 0.1, 10.0)
    
    return {
        'heart_rate': hr,
        'heart_rate_std': hrv_bpm,
        'qt_target_ms': qt_target_ms,
        'qt_stretch': qt_target_ms / BASE_QT_MS,
        'qt_offset': 0.0
    }

def sample_hardware_params(archetype: str, hardware_config: Dict[str, Any], geo_config: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hardware/recording parameters for given archetype."""
    config = hardware_config[archetype]
    
    hw_params = {
        'sampling_rate': np.random.randint(*config['sampling_rate_hz']),
        'noise_amplitude_mv': np.random.uniform(*config['noise_amplitude_mv']),
        'powerline_noise_prob': config['powerline_noise_prob'],
        'motion_artifact_prob': config['motion_artifact_prob'],
        'powerline_frequency_hz': geo_config['powerline_frequency_hz']
    }
    
    # ADC resolution is optional
    if 'adc_resolution_bits' in config:
        hw_params['adc_resolution_bits'] = np.random.randint(config['adc_resolution_bits'][0], 
                                                            config['adc_resolution_bits'][1] + 1)
    
    return hw_params

def sample_conditions(archetype: str, conditions_config: Dict[str, Any], condition_type: str) -> List[str]:
    """Sample medication or comorbidities based on archetype-specific prevalence."""
    active_conditions = []
    
    prevalence_key = 'archetype_prevalence' if condition_type == 'medication' else 'prevalence_by_archetype'
    
    for condition_name, condition_config in conditions_config.items():
        prevalence = condition_config[prevalence_key].get(archetype, 0.0)
        if np.random.rand() < prevalence:
            active_conditions.append(condition_name)
    
    return active_conditions

# =============================================================================
# Variable Duration Functions
# =============================================================================

def sample_recording_duration(archetype: str, duration_config: Dict[str, Any], events_config: Dict[str, Any], 
                             patient_seed: int) -> Tuple[float, Dict[str, Any]]:
    """Sample recording duration based on archetype and potential cardiac events."""
    np.random.seed(patient_seed)
    
    if archetype not in duration_config:
        base_duration = np.random.uniform(60, 300)
        return base_duration, {}
    
    config = duration_config[archetype]
    
    # Sample base duration
    base_duration = np.random.uniform(*config['base_duration_range'])
    
    # Check for continuous monitoring (for very sick patients)
    if np.random.rand() < config['continuous_monitoring_prob']:
        continuous_duration = np.random.uniform(base_duration * 2, base_duration * 5)
        max_limit = duration_config.get('max_duration_limits', {}).get(archetype, 3600)
        base_duration = min(continuous_duration, max_limit)
    
    # Sample potential events during base recording period
    event_schedule = sample_events_in_timeframe(
        archetype, base_duration, events_config, patient_seed
    )
    
    # Extend duration if significant events detected
    final_duration = base_duration
    if event_schedule:
        extension_needed = calculate_event_extension(
            event_schedule, duration_config, archetype
        )
        final_duration = min(
            base_duration + extension_needed,
            duration_config.get('max_duration_limits', {}).get(archetype, 7200)
        )
    
    return final_duration, event_schedule


def sample_events_in_timeframe(archetype: str, duration_sec: float, 
                             events_config: Dict[str, Any], patient_seed: int) -> Dict[str, List[float]]:
    """Sample cardiac events that occur during the recording timeframe."""
    np.random.seed(patient_seed + 1000)  # Different seed for events
    
    event_schedule = {}
    duration_minutes = duration_sec / 60.0
    
    for event_name, event_config in events_config.items():
        if archetype not in event_config.get('archetype_prevalence_multiplier', {}):
            continue
            
        # Calculate event probability for this duration and archetype
        base_prob_per_min = event_config['base_prob_per_minute']
        archetype_multiplier = event_config['archetype_prevalence_multiplier'][archetype]
        effective_prob = base_prob_per_min * archetype_multiplier * duration_minutes
        
        # Sample if this event type occurs
        if np.random.rand() < min(effective_prob, 0.8):  # Cap at 80% to avoid unrealistic rates
            event_times = []
            
            # For clustered events (like PVC storms, AF episodes)
            if 'cluster_prob_if_one_occurs' in event_config:
                cluster_prob = event_config['cluster_prob_if_one_occurs']
                
                # First event occurs randomly in first 80% of recording
                first_event_time = np.random.uniform(0, duration_sec * 0.8)
                event_times.append(first_event_time)
                
                # Additional clustered events
                current_time = first_event_time
                while (np.random.rand() < cluster_prob and 
                       current_time < duration_sec * 0.9):
                    cluster_interval = np.random.uniform(60, 600)
                    current_time += cluster_interval
                    if current_time < duration_sec:
                        event_times.append(current_time)
                    cluster_prob *= 0.7  # Decreasing probability for more events
            else:
                # Single isolated event
                event_time = np.random.uniform(0, duration_sec * 0.8)
                event_times.append(event_time)
            
            if event_times:
                event_schedule[event_name] = event_times
    
    return event_schedule

@profile
def calculate_event_extension(event_schedule: Dict[str, List[float]], 
                            duration_config: Dict[str, Any], archetype: str) -> float:
    """Calculate how much to extend recording based on detected events."""
    if not event_schedule:
        return 0.0
    
    extension_config = duration_config.get('event_extensions', {})
    archetype_config = duration_config.get(archetype, {})
    
    total_extension = 0.0
    
    for event_type, event_times in event_schedule.items():
        if event_type in extension_config:
            event_ext_config = extension_config[event_type]
            min_ext = event_ext_config['min_extension_sec']
            max_ext = event_ext_config['max_extension_sec']
            extension = np.random.uniform(min_ext, max_ext)
        else:
            if 'extension_on_event' in archetype_config:
                extension = np.random.uniform(*archetype_config['extension_on_event'])
            else:
                extension = 300  # Default 5 min extension
        
        # Multiple events of same type don't stack linearly
        total_extension = max(total_extension, extension)
    
    # Multiple different event types can add some cumulative extension
    if len(event_schedule) > 1:
        total_extension *= 1.2  # 20% bonus for multiple event types
    
    return total_extension

@profile
def apply_events_to_signal(ecg_signal: np.ndarray, sampling_rate: int, 
                         event_schedule: Dict[str, List[float]]) -> np.ndarray:
    """Apply cardiac events to the ECG signal at scheduled times."""
    if not event_schedule:
        return ecg_signal
    
    modified_signal = ecg_signal.copy()
    
    for event_type, event_times in event_schedule.items():
        try:
            if event_type == 'pvc_isolated' or event_type == 'pvc_storm':
                modified_signal = modifiers.apply_modifier(
                    'event', 'add_pvc',
                    modified_signal,
                    sampling_rate=sampling_rate,
                    event_times=event_times
                )
            elif event_type == 'atrial_fibrillation':
                for event_time in event_times:
                    duration = np.random.uniform(30, 300)  # 30 sec to 5 min AF episodes
                    modified_signal = modifiers.apply_modifier(
                        'event', 'add_atrial_fib',
                        modified_signal,
                        sampling_rate=sampling_rate,
                        start_s=event_time,
                        duration_s=duration
                    )
            elif event_type == 'baseline_wander':
                # This is already applied in hardware artifacts, skip to avoid double application
                continue
            else:
                print(f"Event type '{event_type}' not yet implemented for signal modification")
                
        except Exception as e:
            print(f"Warning: Failed to apply {event_type} event: {e}")
            continue
    
    return modified_signal

# =============================================================================
# Signal Processing Functions
# =============================================================================
@profile
def apply_qt_stretch(ecg: np.ndarray, base_sr: int, qt_scale: float) -> Tuple[np.ndarray, int]:
    """Apply QT interval stretching by resampling signal."""
    if np.isclose(qt_scale, 1.0):
        return ecg.astype(np.float32), base_sr

    try:
        new_length = int(round(len(ecg) * qt_scale))
        if new_length < 20:  # Prevent resampling to tiny length
            return ecg.astype(np.float32), base_sr
        ecg_resampled = resample_poly(ecg, 
                                      new_length, 
                                      len(ecg)).astype(np.float32)
        # ecg_resampled = nk.signal_resample(
        #     ecg, 
        #     desired_length=new_length,
        #     method="poly"  # Fast resampling method
        # ).astype(np.float32)
        
        # Compute new sampling rate to maintain original duration
        original_duration = len(ecg) / base_sr
        new_sr = int(round(new_length / original_duration))
        return ecg_resampled, new_sr
    except Exception:
        return ecg.astype(np.float32), base_sr

@profile
def apply_signal_artifacts(ecg: np.ndarray, hw_params: Dict[str, Any], sampling_rate: int) -> np.ndarray:
    """Apply hardware-related artifacts to ECG signal using fast modifier functions."""
    modified_ecg = ecg.copy()
    noise_fraction = hw_params['noise_amplitude_mv'] / NOMINAL_ECG_MV
    
    # Apply artifacts using the modifier system for consistency
    # Powerline interference
    if np.random.rand() < hw_params['powerline_noise_prob']:
        powerline_severity = min(0.01, 0.2 * noise_fraction)
        modified_ecg = modifiers.apply_modifier(
            "artifact", "powerline_noise", 
            modified_ecg,
            sampling_rate=sampling_rate,
            severity=powerline_severity,
            frequency_hz=hw_params['powerline_frequency_hz']
        )
    
    # Motion artifacts
    if np.random.rand() < hw_params['motion_artifact_prob']:
        modified_ecg = modifiers.apply_modifier(
            "artifact", "motion_artifact",
            modified_ecg,
            sampling_rate=sampling_rate,
            severity=noise_fraction
        )
    
    # Baseline wander (always present)
    modified_ecg = modifiers.apply_modifier(
        "artifact", "baseline_wander",
        modified_ecg,
        sampling_rate=sampling_rate,
        severity=noise_fraction * 0.5
    )
    
    return modified_ecg

@profile
def apply_adc_quantization(ecg: np.ndarray, adc_bits: int) -> np.ndarray:
    """Apply ADC quantization effects if specified."""
    if adc_bits <= 0:
        return ecg
    
    adc_levels = 2 ** adc_bits
    signal_range = np.ptp(ecg)
    
    if signal_range > 0:
        resolution = signal_range / adc_levels
        return np.round(ecg / resolution) * resolution
    
    return ecg


def apply_parameter_effects(phys_params: Dict[str, float], 
                          parameter_conditions: List[str], 
                          timing_conditions: List[str],
                          condition_type: str,
                          configs: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply parameter effects and prepare morphology_params for timing-based conditions.
    
    Args:
        phys_params: Physiological parameters dictionary
        parameter_conditions: List of parameter-type conditions  
        timing_conditions: List of timing-type conditions
        condition_type: "comorbidity" or "medication"
        configs: Full configuration dictionary
        
    Returns:
        Tuple of (modified_phys_params, morphology_params_dict)
    """
    modified_params = phys_params.copy()
    morphology_params = None
    
    # Apply parameter effects using existing system
    if parameter_conditions:
        modified_params = apply_condition_effects(modified_params, parameter_conditions, condition_type)
    
    # Prepare morphology_params for timing-based conditions
    if timing_conditions:
        # For now, handle one timing condition (can be extended)
        # In practice, timing conditions should rarely co-occur
        timing_condition = timing_conditions[0]  # Use first one
        if len(timing_conditions) > 1:
            # Silently use first condition in multiprocessing context
            pass
            
        condition_config = configs[f"{condition_type}"][timing_condition]
        if "morphology_params" in condition_config:
            morphology_params = condition_config["morphology_params"].copy()
            
            # Convert any factors to actual parameters
            if "qt_prolongation_factor" in morphology_params:
                factor = morphology_params.pop("qt_prolongation_factor")
                morphology_params.update({
                    'ti': np.array([-70, -15, 0, 15, 100 * factor]),
                    'ai': np.array([1.2, -5, 30, -7.5, 0.75]),
                    'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.4 * factor])
                })
                
            if "qt_shortening_factor" in morphology_params:
                factor = morphology_params.pop("qt_shortening_factor") 
                morphology_params.update({
                    'ti': np.array([-70, -15, 0, 15, 100 * factor]),
                    'ai': np.array([1.2, -5, 30, -7.5, 0.75]),
                    'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.4 * factor])
                })
    
    return modified_params, morphology_params


def apply_chronic_morphologies(ecg_signal: np.ndarray, 
                             sampling_rate: int, 
                             morphology_conditions: List[str],
                             configs: Dict[str, Any]) -> np.ndarray:
    """
    Apply morphology-based chronic conditions via post-processing.
    
    Args:
        ecg_signal: Base ECG signal
        sampling_rate: Sampling rate in Hz
        morphology_conditions: List of morphology-type conditions
        configs: Configuration dictionary
        
    Returns:
        Modified ECG signal with morphological changes applied
    """
    if not morphology_conditions:
        return ecg_signal
        
    modified_signal = ecg_signal.copy()
    
    for condition in morphology_conditions:
        try:
            if condition == 'old_myocardial_infarction':
                modified_signal = modifiers.apply_modifier(
                    'morphology', 'old_mi',
                    modified_signal,
                    sampling_rate=sampling_rate
                )
            elif condition == 'chronic_atrial_fibrillation':
                modified_signal = modifiers.apply_modifier(
                    'morphology', 'chronic_af',
                    modified_signal,
                    sampling_rate=sampling_rate
                )
            else:
                # Silently skip unknown conditions in multiprocessing context
                continue
                
        except Exception:
            # Silently continue on failure to avoid print spam in workers
            continue
    
    # Validate and clip amplitudes after morphology operations
    # Apply same checks as validate_ecg_signal to prevent validation failures
    rms = np.sqrt(np.mean(modified_signal**2))
    if rms < 0.02 or rms > 10.0:
        # If RMS is out of range, scale signal to reasonable range
        if rms > 0:
            scale_factor = 1.0 / rms if rms > 2.0 else 2.0 / rms if rms < 0.1 else 1.0
            modified_signal *= scale_factor
    
    # Final amplitude clipping to ensure signal stays within realistic ECG range
    return np.clip(modified_signal, -8.0, 8.0).astype(np.float32)
# =============================================================================
# Validation
# =============================================================================
@profile
def validate_ecg_signal(ecg: np.ndarray, sampling_rate: int, phys_params: Dict[str, float]) -> bool:
    """Validate ECG signal for clinical realism."""
    try:
        # Basic signal checks
        if len(ecg) == 0 or np.any(np.isnan(ecg)):
            return False
        
        # Amplitude validation
        rms = np.sqrt(np.mean(ecg**2))
        if rms < 0.02 or rms > 10.0:  # Reasonable RMS range in mV
            return False
        
        # Heart rate validation via R-peak detection (using fast method)
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method='emrich2023', correct_artifacts = False)
        if len(rpeaks['ECG_R_Peaks']) < 2:
            return False
        
        rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / sampling_rate
        avg_hr = 60 / np.mean(rr_intervals)
        if avg_hr < 25 or avg_hr > 300:
            return False
        
        # QTc validation
        target_qt_ms = phys_params.get('qt_target_ms', BASE_QT_MS)
        qtc_ms = target_qt_ms / np.sqrt(60 / avg_hr)  # Bazett's formula
        if qtc_ms < 250 or qtc_ms > 700:
            return False
        
        return True
        
    except Exception:
        return False
