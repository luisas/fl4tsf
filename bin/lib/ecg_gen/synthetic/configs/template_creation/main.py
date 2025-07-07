from pathlib import Path
from generators import (
    generate_sinus_template,
    generate_pvc_template,
    generate_rbbb_template,
    generate_old_mi_template,
    generate_first_degree_av_block_template,
    generate_af_qrs_template
)
from utils import save_template, validate_and_get_delineation, create_base_metadata

def create_all_templates():
    """Main function to generate and save all defined templates."""
    output_dir = Path(__file__).parent / "templates"
    print(f"Templates will be saved to: {output_dir}\n")

    # --- Sinus Templates ---
    seed = 0
    label = "sinus"
    variant = "default"
    waveform = generate_sinus_template(seed=seed)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "Standard 12-lead sinus rhythm beat.", validation_data
    )
    save_template(waveform, meta, output_dir)
    
    # --- PVC Templates ---
    # LV Origin
    seed = 17
    label = "pvc"
    variant = "lv_origin"
    waveform = generate_pvc_template(origin="lv", seed=seed)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "PVC with Left Ventricular origin (positive in V1).", validation_data
    )
    save_template(waveform, meta, output_dir)
    
    # RV Origin
    seed = 8
    label = "pvc"
    variant = "rv_origin"
    waveform = generate_pvc_template(origin="rv", seed=seed)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "PVC with Right Ventricular origin (negative in V1).", validation_data
    )
    save_template(waveform, meta, output_dir)
    
    # --- RBBB Template ---
    seed = 42
    label = "rbbb"
    variant = "typical"
    waveform = generate_rbbb_template(seed=seed)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "Typical Right Bundle Branch Block morphology.", validation_data
    )
    save_template(waveform, meta, output_dir)

    # --- Old MI Templates ---
    for i, location in enumerate(['anterior', 'inferior', 'lateral']):
        seed = 100 + i # e.g., a hash or predefined dict
        label = "old_mi"
        variant = location
        waveform = generate_old_mi_template(location=location, seed=seed)
        validation_data = validate_and_get_delineation(waveform, label)
        meta = create_base_metadata(
            label, variant, seed, f"Typical {location} old MI morphology.", validation_data
        )
        save_template(waveform, meta, output_dir)

    # --- First-Degree AV Block Template ---
    seed = 301
    label = "first_degree_av_block"
    variant = "pr260ms"
    waveform = generate_first_degree_av_block_template(seed=seed, pr_ms=260)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "First degree AV block morphology.", validation_data
    )
    save_template(waveform, meta, output_dir)

    # --- AF QRS Template ---
    seed = 202
    label = "af_qrs"
    variant = "default"
    waveform = generate_af_qrs_template(seed=seed)
    validation_data = validate_and_get_delineation(waveform, label)
    meta = create_base_metadata(
        label, variant, seed, "Typical AF QRS morphology.", validation_data
    )
    save_template(waveform, meta, output_dir)

if __name__ == "__main__":
    create_all_templates()