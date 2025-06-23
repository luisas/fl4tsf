"""generator.py
===================
Unified, parallel‑only ECG data generator.

This file supersedes the old *generate_clients_sequential* / *generate_clients_parallel*
setup.  All public consumers should now just call
``generate_clients(...)`` – the implementation always runs with a
``ProcessPoolExecutor`` (multi‑RPC).  For very tiny workloads we simply
fallback to a **single worker**; this keeps the code path identical and
removes the need for two different functions.

External helper utilities (e.g. ``load_configs``, ``sample_client_distributions``)
are assumed to live in sibling modules – exactly as in the original codebase.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import yaml
import neurokit3 as nk

try:
    profile
except NameError:
    def profile(f): return f
# -----------------------------------------------------------------------------
# Constants & small utilities
# -----------------------------------------------------------------------------
BASE_QT_MS = 400  # used for stretch factor calculations
from .core import (
    load_configs,
    sample_client_distributions,
    sample_patient_archetypes,
    sample_physiological_params,
    sample_hardware_params,
    sample_conditions,
    apply_condition_effects,
    apply_parameter_effects,        
    apply_chronic_morphologies,     
    sample_recording_duration,
    apply_qt_stretch,
    apply_events_to_signal,
    apply_signal_artifacts,
    apply_adc_quantization,
    validate_ecg_signal,
)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
_Assignment = Tuple[int, int, str]  # (client_id, patient_id, archetype)


def _create_assignments(
    num_clients: int,
    num_patients_per_client: int,
    archetype_dists: np.ndarray,
    archetype_names: Sequence[str],
) -> List[_Assignment]:
    """Return a flat list of (client, patient, archetype) tuples."""
    assignments: List[_Assignment] = []
    for c in range(num_clients):
        chosen = sample_patient_archetypes(
            archetype_dists[c], num_patients_per_client, archetype_names
        )
        assignments.extend((c, p, a) for p, a in enumerate(chosen))
    return assignments


def _chunk(seq: Sequence[_Assignment], n_chunks: int) -> List[List[_Assignment]]:
    """Split *seq* into *n_chunks* nearly‑equal parts."""
    if n_chunks <= 1:
        return [list(seq)]
    k, m = divmod(len(seq), n_chunks)
    return [list(seq[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n_chunks)]


# -----------------------------------------------------------------------------
# Worker function
# -----------------------------------------------------------------------------

@profile
def _worker(
    batch: List[_Assignment],
    *,
    cfg: Dict[str, Any],
    duration_cfg: Dict[str, Any],
    base_duration: float,
    variable_duration: bool,
    base_seed: int,
    client_geos: np.ndarray,
    client_arch_dists: np.ndarray,
) -> List[Dict[str, Any]]:
    np.random.seed(base_seed)
    rng = np.random.default_rng(base_seed)

    results: List[Dict[str, Any]] = []

    for client_id, patient_id, archetype in batch:
        patient_seed = base_seed + client_id * 10_000 + patient_id
        geo_cfg = cfg["geography"][client_geos[client_id]]

        # 1. Duration & event schedule
        if variable_duration:
            rec_dur, evt_sched = sample_recording_duration(
                archetype, duration_cfg, cfg.get("event", {}), patient_seed
            )
            rec_dur = max(rec_dur, base_duration)
        else:
            rec_dur = base_duration
            evt_sched = {}

        # 2. Physiological + hardware parameters
        phys = sample_physiological_params(archetype, cfg["archetype"])
        hw = sample_hardware_params(archetype, cfg["hardware"], geo_cfg)

        # 3. Sample and partition conditions by effect_type
        comorbs = sample_conditions(archetype, cfg.get("comorbidity", {}), "comorbidity")
        
        # Partition comorbidity by effect_type
        parameter_comorbs = []
        timing_comorbs = []
        morphology_comorbs = []
        
        for condition in comorbs:
            effect_type = cfg["comorbidity"][condition].get("effect_type", "parameter")
            if effect_type == "parameter":
                parameter_comorbs.append(condition)
            elif effect_type == "timing":
                timing_comorbs.append(condition)
            elif effect_type == "morphology":
                morphology_comorbs.append(condition)
        
        # Medication (typically all parameter-based)
        meds = sample_conditions(archetype, cfg["medication"], "medication")

        # 4. Apply parameter effects and prepare timing-based morphology_params
        phys, morphology_params = apply_parameter_effects(
            phys, parameter_comorbs, timing_comorbs, "comorbidity", cfg
        )
        
        # Apply medication effects (always parameter-based)
        if meds:
            phys = apply_condition_effects(phys, meds, "medication")

        # 5. Apply QT offset → stretch (from parameter effects)
        if (off := phys.get("qt_offset", 0.0)) != 0.0:
            tgt = phys.get("qt_target_ms", BASE_QT_MS) + off * 1000
            tgt = np.clip(tgt, 300, 600)
            phys["qt_target_ms"] = tgt
            phys["qt_stretch"] = tgt / BASE_QT_MS

        phys["heart_rate"] = np.clip(phys["heart_rate"], 30, 220)
        phys["heart_rate_std"] = np.clip(phys["heart_rate_std"], 0.05, 15.0)

        # 6. Simulate BASELINE ECG with timing modifications (if any)
        try:
            ecg = nk.ecg_simulate(
                duration=float(rec_dur),
                sampling_rate=int(hw["sampling_rate"]),
                heart_rate=float(phys["heart_rate"]),
                heart_rate_std=float(phys["heart_rate_std"]),
                random_state=int(patient_seed),
                morphology_params=morphology_params  # Pass timing modifications to solver
            )
        except Exception:
            continue  # Skip failed ECG generation silently in multiprocessing context

        # 7. Apply QT stretch (for any remaining adjustments)
        ecg, final_sr = apply_qt_stretch(ecg, hw["sampling_rate"], phys["qt_stretch"])
        # 8. Apply CHRONIC MORPHOLOGY changes (post-simulation)
        if morphology_comorbs:
            ecg = apply_chronic_morphologies(ecg, final_sr, morphology_comorbs, cfg)

        # 9. Apply transient events and artifacts (existing pipeline)
        if evt_sched:
            ecg = apply_events_to_signal(ecg, final_sr, evt_sched)
        ecg = apply_signal_artifacts(ecg, hw, final_sr)
        if hw.get("adc_resolution_bits"):
            ecg = apply_adc_quantization(ecg, hw["adc_resolution_bits"])
        
        # 10. Validation
        # if not validate_ecg_signal(ecg, final_sr, phys):
        #     continue  # Skip invalid signals silently in multiprocessing context
        # 11. Create output
        t_axis = np.linspace(0, len(ecg) / final_sr, len(ecg))
        meta = {
            "client_id": client_id,
            "patient_id": patient_id,
            "archetype": archetype,
            "geography": client_geos[client_id],
            "medication": meds,
            "comorbidity": comorbs,
            "parameter_comorbidity": parameter_comorbs,
            "timing_comorbidity": timing_comorbs,
            "morphology_comorbidity": morphology_comorbs,
            "sampling_rate": final_sr,
            "actual_duration_sec": len(ecg) / final_sr,
            "planned_duration_sec": rec_dur,
            "base_duration_sec": base_duration,
            "variable_duration_used": variable_duration,
            **{k: v for k, v in phys.items() if k != "qt_offset"},
            **{k: v for k, v in hw.items() if k != "sampling_rate"},
        }
        
        if evt_sched:
            meta.update(
                events_detected=evt_sched,
                total_events=sum(len(v) for v in evt_sched.values()),
                event_types=list(evt_sched.keys()),
            )
        else:
            meta.update(events_detected={}, total_events=0, event_types=[])

        results.append(
            {
                "metadata": meta,
                "signal": torch.tensor(ecg, dtype=torch.float32).unsqueeze(-1),
                "timestamps": torch.tensor(t_axis, dtype=torch.float32),
            }
        )
    return results

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@profile
def generate_clients(
    *,
    num_clients: int,
    num_patients_per_client: int,
    duration_sec: float,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    variable_duration: bool = True,
    max_workers: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield dictionaries containing simulated ECG signals + rich metadata."""

    cfg = load_configs()

    duration_cfg: Dict[str, Any] = {}
    if variable_duration:
        dur_file = Path(__file__).with_suffix("").parent / "configs/recording_duration.yaml"
        if dur_file.exists():
            duration_cfg = yaml.safe_load(dur_file.read_text())
        else:
            print("[ecg_generator] recording_duration.yaml missing – using fixed durations")
            variable_duration = False

    # Per‑client settings
    archetype_names = list(cfg["archetype"].keys())
    geo_names = np.array(list(cfg["geography"].keys()))
    geo_probs = np.array([cfg["geography"][g]["weight"] for g in geo_names])

    rng = np.random.default_rng(seed)
    client_geos = rng.choice(geo_names, size=num_clients, p=geo_probs, replace=True)
    client_arch_dists = sample_client_distributions(num_clients, cfg["archetype"], dirichlet_alpha)

    assignments = _create_assignments(num_clients, num_patients_per_client, client_arch_dists, archetype_names)

    # Decide on worker count
    total = len(assignments)
    cpu_cnt = 6
    max_workers = cpu_cnt if max_workers is None else min(max_workers, cpu_cnt)
    n_workers = max(1, min(max_workers, total))
    batches = _chunk(assignments, n_workers)
    
    # TEMPORARY: Run a single batch synchronously to profile _worker
    # for idx, batch in enumerate(batches[:1]):  # Just one batch
    #     result = _worker(
    #         batch,
    #         cfg=cfg,
    #         duration_cfg=duration_cfg,
    #         base_duration=duration_sec,
    #         variable_duration=variable_duration,
    #         base_seed=seed + idx * 1_000_000,
    #         client_geos=client_geos,
    #         client_arch_dists=client_arch_dists,
    #     )
    #     for patient in result:
    #         yield patient


    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _worker,
                batch,
                cfg=cfg,
                duration_cfg=duration_cfg,
                base_duration=duration_sec,
                variable_duration=variable_duration,
                base_seed=seed + idx * 1_000_000,
                client_geos=client_geos,
                client_arch_dists=client_arch_dists,
            )
            for idx, batch in enumerate(batches)
        ]

        for fut in as_completed(futures):
            for patient in fut.result():
                yield patient
