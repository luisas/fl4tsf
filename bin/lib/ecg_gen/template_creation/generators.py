# template_creation/generators.py
import numpy as np
import neurokit3 as nk

from pathlib import Path
from .utils import (
    TARGET_FS,
    R_PEAK_INDEX,
    DURATION_SEC,
)

# ---------------------------------------------------------------------------#
# Helper utilities (private; used by several generators)                     #
# ---------------------------------------------------------------------------#

def _stretch_segment(seg: np.ndarray, factor: float) -> np.ndarray:
    """
    Linearly stretches a 2-D segment (leads × samples) by `factor`.
    Keeps the first and last samples identical; intermediate points are
    resampled with np.interp. Only used for widening QRS complexes.
    """
    n_old = seg.shape[1]
    n_new = int(round(n_old * factor))
    x_old = np.linspace(0, 1, n_old)
    x_new = np.linspace(0, 1, n_new)
    return np.stack([np.interp(x_new, x_old, lead) for lead in seg], axis=0)

def _widen_qrs(wave: np.ndarray, waves_dict: dict, factor: float = 1.6) -> np.ndarray:
    """
    Returns a copy of `wave` with its QRS complex linearly stretched.
    `waves_dict` must come from nk.ecg_delineate on Lead II.
    """
    qrs_on = waves_dict["ECG_R_Onsets"][0]
    qrs_off = waves_dict["ECG_R_Offsets"][0]
    pre  = wave[:, :qrs_on]
    qrs  = wave[:, qrs_on:qrs_off]
    post = wave[:, qrs_off:]

    qrs_wide = _stretch_segment(qrs, factor)
    # Re-assemble and trim/pad to exact 1-second length
    new_wave = np.concatenate([pre, qrs_wide, post], axis=1)
    if new_wave.shape[1] > wave.shape[1]:
        new_wave = new_wave[:, :wave.shape[1]]
    elif new_wave.shape[1] < wave.shape[1]:
        pad = wave.shape[1] - new_wave.shape[1]
        new_wave = np.pad(new_wave, ((0, 0), (0, pad)))
    return new_wave

def _suppress_p_wave(wave: np.ndarray, waves_dict: dict) -> np.ndarray:
    """
    Flattens the P-wave segment (sets it to baseline of Lead II).
    """
    p_on  = waves_dict["ECG_P_Onsets"][0]
    p_off = waves_dict["ECG_P_Offsets"][0]
    baseline = wave[:, p_on - 5 : p_on].mean(axis=1, keepdims=True)
    wave[:, p_on:p_off] = baseline
    return wave

def _carve_pathological_q_wave(
    wave: np.ndarray,
    waves_dict: dict,
    *,
    leads: list[int],
    depth_frac: float = 0.3,
    width_ms: int = 40,
) -> np.ndarray:
    """
    Creates a negative Q-wave by subtracting a Gaussian bump on selected leads.
    depth_frac  – fraction of that lead's R-peak amplitude to carve out
    width_ms    – width of the carved region in milliseconds
    """
    qrs_on = waves_dict["ECG_R_Onsets"][0]
    width_samples = int(round(width_ms * TARGET_FS / 1000))
    x = np.linspace(-1, 1, width_samples)
    gaussian = np.exp(-4 * x**2)           # centred, width≈0.5

    carved = wave.copy()
    for ld in leads:
        r_amp = wave[ld, R_PEAK_INDEX]     # assume R at centre
        depth = depth_frac * abs(r_amp)
        idx = slice(max(qrs_on - width_samples // 2, 0),
                     max(qrs_on - width_samples // 2, 0) + width_samples)
        carved[ld, idx] -= depth * gaussian
    return carved


def _reduce_r_wave_amplitude(
    wave: np.ndarray,
    waves_dict: dict,
    *,
    leads: list[int],
    factor: float = 0.6,
) -> np.ndarray:
    """Scales the R-wave segment inside QRS window on selected leads."""
    qrs_on, qrs_off = waves_dict["ECG_R_Onsets"][0], waves_dict["ECG_R_Offsets"][0]
    attenuated = wave.copy()
    attenuated[leads, qrs_on:qrs_off] *= factor
    return attenuated


def _flatten_or_invert_t_wave(
    wave: np.ndarray,
    waves_dict: dict,
    *,
    leads: list[int],
    factor: float = -0.3,
) -> np.ndarray:
    """
    Multiplies the T-wave segment by `factor`.
    factor = 0   → flatten
    factor < 0   → invert and scale
    """
    t_on = waves_dict["ECG_T_Onsets"][0]
    modified = wave.copy()
    modified[leads, t_on:] *= factor
    return modified


def _prolong_pr_interval(
    wave: np.ndarray,
    waves_dict: dict,
    *,
    new_pr_ms: int = 260,
) -> np.ndarray:
    """
    Inserts additional isoelectric baseline between P-offset and QRS-onset to
    achieve a target PR interval (ms). Keeps overall beat length constant by
    trimming the tail if needed.
    """
    p_off = waves_dict["ECG_P_Offsets"][0]
    qrs_on = waves_dict["ECG_R_Onsets"][0]
    current_pr = qrs_on - p_off
    target_pr = int(round(new_pr_ms * TARGET_FS / 1000))

    if target_pr <= current_pr:
        return wave  # already long enough

    delta = target_pr - current_pr
    prolonged = np.pad(wave, ((0, 0), (0, delta)))[:, : wave.shape[1]]  # pad then trim

    # fill the inserted gap with baseline
    baseline = wave[:, p_off - 5 : p_off].mean(axis=1, keepdims=True)
    prolonged[:, p_off : p_off + delta] = baseline
    return prolonged


# ---------------------------------------------------------------------------#
# Public generator functions                                                 #
# ---------------------------------------------------------------------------#
def get_clean_sinus_beat(
    *,
    seed: int,
    fs: int = TARGET_FS,
    duration_sec: int = 20,
    pre_ms: int = 300,
    post_ms: int = 700,
) -> tuple[np.ndarray, dict]:
    """
    Simulate a multilead ECG, delineate once, and return (beat, local_waves)
    with the R-peak centred at R_PEAK_INDEX. Guarantees no NaNs in landmarks.
    """
    # 1) Simulate long trace
    sig_df = nk.ecg_simulate(
        duration=duration_sec,
        sampling_rate=fs,
        method="multilead",
        heart_rate=70,
        random_state=seed,
    )
    sig = sig_df.T.values                                     # shape (12, N)

    # 2) Detect R-peaks on Lead II (index 1 ⇒ conventional)
    _, rpeak_info = nk.ecg_peaks(sig[2, :], sampling_rate=fs)
    rpeaks = rpeak_info["ECG_R_Peaks"]
    # 3) One-shot delineation (wavelet) WITH supplied R-peaks
    _, waves = nk.ecg_delineate(
        sig[1, :],
        rpeaks=rpeaks,
        sampling_rate=fs,
        method="dwt",
    )

    # 4) Find first beat whose key landmarks are non-NaN
    def _is_complete(idx: int) -> bool:
        needed_keys = [
            "ECG_P_Onsets",
            "ECG_P_Offsets",
            "ECG_R_Onsets",
            "ECG_R_Offsets",
            "ECG_T_Peaks",
            "ECG_T_Offsets",
        ]
        return all(
            len(waves[k]) > idx and not np.isnan(waves[k][idx])
            for k in needed_keys
        )

    for i in range(len(rpeaks) - 1):          # ensure there *is* a next P-onset
        if _is_complete(i) and _is_complete(i + 1):
            break
    else:
        raise RuntimeError("No fully-delineated beat found; try another seed.")

    # 5) Define crop window around the chosen R-peak
    r_peak = rpeaks[i]
    win_start = max(0, r_peak - int(pre_ms * fs / 1000))
    win_end = win_start + int((pre_ms + post_ms) * fs / 1000)

    beat = sig[:, win_start:win_end].copy()

    # 6) Localise landmarks to the cropped beat
    local_waves: dict[str, list[int]] = {}
    local_waves["ECG_R_Peaks"] = [int(r_peak - win_start)]  # R-peak is always present
    for key, arr in waves.items():
        if len(arr) > i and not np.isnan(arr[i]):
            local_waves[key] = [int(arr[i] - win_start)]

    # 7) Centre R-peak at R_PEAK_INDEX
    shift = R_PEAK_INDEX - local_waves["ECG_R_Peaks"][0]
    if shift > 0:
        beat = np.pad(beat, ((0, 0), (shift, 0)))[:, : beat.shape[1]]
    elif shift < 0:
        beat = beat[:, -shift :]
    for v in local_waves.values():
        v[0] += shift

    return beat, local_waves


def generate_sinus_template(seed: int) -> np.ndarray:
    base, _ = get_clean_sinus_beat(seed=seed)
    return base


def generate_pvc_template(origin: str, seed: int) -> np.ndarray:
    """
    Generates a premature ventricular complex (PVC) beat.
    `origin` in {"lv_origin", "rv_origin"} sets V1 polarity.
    """
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)

    # 2) Widen QRS and suppress P-wave
    pvc = base.copy()
    pvc = _suppress_p_wave(pvc, w)          # No preceding P
    pvc = _widen_qrs(pvc, w, factor=1.8)    # Broad complex

    # 3) Polarity manipulation in V1 (lead index 6)
    v1 = pvc[6, :]
    if origin == "lv_origin":
        pvc[6, :] = np.abs(v1)              # Positive in V1
    elif origin == "rv_origin":
        pvc[6, :] = -np.abs(v1)             # Negative in V1

    # 4) Discordant T-wave: invert T across all leads
    t_on = w["ECG_T_Onsets"][0]
    pvc[:, t_on:] *= -1

    return pvc


def generate_rbbb_template(seed: int) -> np.ndarray:
    """
    Generates a right-bundle branch block (RBBB) beat: wide QRS with rSR'
    pattern in V1 and broad S-wave in I, V6.
    """
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)
    rbbb = _widen_qrs(base.copy(), w, factor=1.4)

    # rSR′ in V1 (lead 6): insert small notch after mid-QRS
    v1 = rbbb[6, :]
    qrs_on, qrs_off = w["ECG_R_Onsets"][0], w["ECG_R_Offsets"][0]
    mid = (qrs_on + qrs_off) // 2
    rbbb[6, mid:mid + 15] += 0.3 * np.max(v1)  # create R′

    # Broad S in I (lead 0) and V6 (lead 11): deepen negative tail
    for lead_idx in (0, 11):
        s_tail = rbbb[lead_idx, mid + 10 : qrs_off + 25]
        rbbb[lead_idx, mid + 10 : qrs_off + 25] = s_tail - 0.2 * np.abs(s_tail)

    return rbbb


def generate_lbbb_template(seed: int) -> np.ndarray:
    """
    Generates a left-bundle branch block (LBBB) beat: QS or rS in V1,
    broad monophasic R in I, V5-V6, plus discordant ST/T.
    """
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)
    lbbb = _widen_qrs(base.copy(), w, factor=1.4)

    # V1 (lead 6): invert early part to create QS pattern
    qrs_on = w["ECG_R_Onsets"][0]
    lbbb[6, qrs_on : qrs_on + 70] *= -1

    # Leads I (0) and V5-V6 (10, 11): amplify R and suppress S
    for lead_idx in (0, 10, 11):
        qrs_seg = lbbb[lead_idx, qrs_on : qrs_on + 100]
        lbbb[lead_idx, qrs_on : qrs_on + 100] = np.abs(qrs_seg)  # monophasic R

    # Discordant T: invert T in V1-V3 only
    t_on = w["ECG_T_Onsets"][0]
    for lead_idx in (6, 7, 8):
        lbbb[lead_idx, t_on:] *= -1

    return lbbb

def generate_old_mi_template(location: str, seed: int) -> np.ndarray:
    """
    Creates a beat with old-MI morphology.
    location ∈ {'anterior', 'inferior', 'lateral'}
    """
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)
    if location == "anterior":
        q_leads = t_leads = [6, 7, 8, 9]          # V1-V4
    elif location == "inferior":
        q_leads = t_leads = [1, 2, 5]             # II, III, aVF
    elif location == "lateral":
        q_leads = t_leads = [0, 4, 10, 11]        # I, aVL, V5-V6
    else:
        raise ValueError("location must be anterior|inferior|lateral")

    mi = _carve_pathological_q_wave(base, w, leads=q_leads)
    mi = _reduce_r_wave_amplitude(mi, w, leads=q_leads, factor=0.5)
    mi = _flatten_or_invert_t_wave(mi, w, leads=t_leads, factor=-0.4)
    return mi


def generate_first_degree_av_block_template(seed: int, pr_ms: int = 260) -> np.ndarray:
    """Returns a sinus beat with PR prolonged to `pr_ms`."""
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)
    return _prolong_pr_interval(base, w, new_pr_ms=pr_ms)


def generate_af_qrs_template(seed: int) -> np.ndarray:
    """
    Generates a single beat suitable for AF datasets: QRS-T with no P-wave.
    """
    # 1) Get clean sinus beat
    base, w = get_clean_sinus_beat(seed=seed)
    return _suppress_p_wave(base.copy(), w)
