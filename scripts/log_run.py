#!/usr/bin/env python3
import os, csv, datetime, subprocess

HEADER = [
    # identity & context
    "utc_time", "image", "git_commit",
    # args used (from run_pipeline kwargs / CLI)
    "um_per_px", "nucleus_diam_um",
    "tissue_method", "min_tissue_obj_um2",
    "pclip_low", "pclip_high",
    "target_area_lo", "target_area_hi",
    "min_area_um2", "min_area_factor", "fill_holes",
    "q_width", "k_s", "k_v",
    "nucleus_dilate_px", "smooth_replaced", "k_px_for_smooth",
    # derived during run: kernel in px
    "k_px",
    # artifact info (from info['artifact'])
    "artifact_method", "artifact_tail", "artifact_area_frac", "artifact_n_components",
    # filter info (from info['filter'])
    "post_min_area_um2", "post_min_area_px", "n_before", "n_after", "n_removed",
    # pink/background replacement (from info['pink'])
    "pink_mu_h_deg", "pink_replaced_frac", "pink_core_frac",
    "pink_S_med", "pink_V_med", "pink_S_mad", "pink_V_mad",
    "pink_s_lo", "pink_s_hi", "pink_v_lo", "pink_v_hi",
]

def _git_commit_or_empty():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return ""

def append_run_row(csv_path, image_path, args_dict, info_dict):
    """
    csv_path: e.g., 'data/tables/run_metadata.csv'
    image_path: the processed image path (string)
    args_dict: exactly what you passed to run_pipeline (dict)
    info_dict: out['info'] from run_pipeline (dict with 'artifact', 'filter', 'pink', 'k_px')
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    art = info_dict.get("artifact", {}) or {}
    flt = info_dict.get("filter", {}) or {}
    pnk = info_dict.get("pink", {}) or {}

    # unpack tuples from args
    pclip = args_dict.get("pclip_low_high", (5, 95))
    targ  = args_dict.get("target_area_range", (0.005, 0.30))

    row = {
        "utc_time": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "image": os.path.basename(image_path),
        "git_commit": _git_commit_or_empty(),

        "um_per_px": args_dict.get("um_per_px"),
        "nucleus_diam_um": args_dict.get("nucleus_diam_um"),
        "tissue_method": str(args_dict.get("tissue_method")),
        "min_tissue_obj_um2": args_dict.get("min_tissue_obj_um2"),

        "pclip_low": pclip[0],
        "pclip_high": pclip[1],
        "target_area_lo": targ[0],
        "target_area_hi": targ[1],

        "min_area_um2": args_dict.get("min_area_um2"),
        "min_area_factor": args_dict.get("min_area_factor"),
        "fill_holes": bool(args_dict.get("fill_holes")),

        "q_width": args_dict.get("q_width"),
        "k_s": args_dict.get("k_s"),
        "k_v": args_dict.get("k_v"),
        "nucleus_dilate_px": args_dict.get("nucleus_dilate_px"),
        "smooth_replaced": bool(args_dict.get("smooth_replaced")),
        "k_px_for_smooth": args_dict.get("k_px_for_smooth"),

        "k_px": info_dict.get("k_px"),

        "artifact_method": art.get("method"),
        "artifact_tail": art.get("tail"),
        "artifact_area_frac": art.get("area_frac"),
        "artifact_n_components": art.get("n_components"),

        "post_min_area_um2": flt.get("min_area_um2"),
        "post_min_area_px": flt.get("min_area_px"),
        "n_before": flt.get("n_before"),
        "n_after": flt.get("n_after"),
        "n_removed": flt.get("removed"),

        "pink_mu_h_deg": pnk.get("mu_h"),
        "pink_replaced_frac": pnk.get("replaced_frac"),
        "pink_core_frac": pnk.get("pink_frac_core"),
        "pink_S_med": pnk.get("S_med"),
        "pink_V_med": pnk.get("V_med"),
        "pink_S_mad": pnk.get("S_mad"),
        "pink_V_mad": pnk.get("V_mad"),
        "pink_s_lo": pnk.get("s_lo"),
        "pink_s_hi": pnk.get("s_hi"),
        "pink_v_lo": pnk.get("v_lo"),
        "pink_v_hi": pnk.get("v_hi"),
    }

    # write header if file not exists / empty
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
