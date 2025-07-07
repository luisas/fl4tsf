
import json
import numpy as np
import neurokit3 as nk
from pathlib import Path

# --- Constants ---
TARGET_FS = 1000
DURATION_SEC = 1  # Each template will be 1 second long
R_PEAK_INDEX = 300 
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def save_template(
    waveform: np.ndarray,
    metadata: dict,
    output_dir: Path
):
    """Saves a template waveform (.npy) and its metadata (.json) sidecar."""
    label = metadata["label"]
    variant = metadata["variant"]
    seed = metadata.get("seed", "na")
    
    # Create subdirectory if it doesn't exist
    template_dir = output_dir / label
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    base_filename = f"{label}_{variant}_fs{TARGET_FS}_seed{seed}"
    npy_path = template_dir / f"{base_filename}.npy"
    json_path = template_dir / f"{base_filename}.json"
    
    # Save files
    np.save(npy_path, waveform)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ Saved template: {npy_path}")



def _approx_qrs_width_ms(lead: np.ndarray, r_idx: int, fs: int) -> float:
    """Rough width: where abs(derivative) drops <20% of max inside ±150 ms."""
    window = int(0.15 * fs)
    deriv = np.abs(np.diff(lead))
    max_d = deriv[r_idx - 5 : r_idx + 5].max()
    thr = 0.2 * max_d
    # search left
    left = r_idx
    while left > 0 and deriv[left] > thr:
        left -= 1
    # search right
    right = r_idx
    while right < len(deriv) - 1 and deriv[right] > thr:
        right += 1
    return (right - left) / (fs / 1000.0)
def validate_and_get_delineation(waveform: np.ndarray, label: str) -> dict:
    """Validate 1‑s template; fallback to heuristic width if delineator fails."""

    lead_ii = waveform[1, :]

    # 1) Try fast peak‑based delineation
    try:
        _, waves = nk.ecg_delineate(
            lead_ii,
            rpeaks=[R_PEAK_INDEX],
            sampling_rate=TARGET_FS,
            method="peak",
        )
        qrs_on = waves["ECG_R_Onsets"][0]
        qrs_off = waves["ECG_R_Offsets"][0]
        if np.isnan(qrs_on) or np.isnan(qrs_off):
            raise ValueError("NaN landmarks")
        qrs_width_ms = (qrs_off - qrs_on) / (TARGET_FS / 1000.0)
    except Exception:
        # Fallback heuristic measurement
        qrs_width_ms = _approx_qrs_width_ms(lead_ii, R_PEAK_INDEX, TARGET_FS)

    # 2) Sanity checks
    patho = label.lower()
    if patho in {"pvc", "rbbb", "lbbb"} and qrs_width_ms < 120:
        raise ValueError(f"{label}: QRS {qrs_width_ms:.1f} ms < 120 ms")
    if patho == "sinus" and qrs_width_ms > 110:
        raise ValueError(f"Sinus: QRS {qrs_width_ms:.1f} ms > 110 ms")

    return {"qrs_width_ms": round(qrs_width_ms)}


def create_base_metadata(label: str, variant: str, seed: int, description: str, validation_data: dict) -> dict:
    """Creates the common metadata structure."""
    return {
        "label": label,
        "variant": variant,
        "fs": TARGET_FS,
        "n_leads": len(LEAD_ORDER),
        "lead_order": LEAD_ORDER,
        "duration_sec": DURATION_SEC,
        "r_peak_index": R_PEAK_INDEX,
        "seed": seed,
        "source": "neurokit3_multilead_simulation",
        "description": description,
        "validation": validation_data
    }