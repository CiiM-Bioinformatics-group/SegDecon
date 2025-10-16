#!/usr/bin/env python3
"""
SegDecon — Histology Image Processing & Nuclei Segmentation (Refactor)
---------------------------------------------------------------------
This is a **pure refactor** of the related notebook cells into a clean, importable pipeline.
Logic and default parameter values are preserved. The file exposes:

- Utility functions: ensure_uint8, as_rgb, show, adaptive_kernel_px
- Core steps: get_tissue_mask, robust_hue_artifact_mask, filter_small_nuclei,
              keep_nuclei_and_pink_no_bh
- A single orchestration entry point: run_pipeline(...)
- A CLI (`python segdecon_histology_pipeline.py --help`) for reproducible runs

Notes:
- No algorithmic changes; only structure and clarity improvements.
- All defaults match the original cell values.
"""
from __future__ import annotations
import os, json, math, gc
import numpy as np
import cv2
from typing import Dict, Tuple, Any
from skimage import color, morphology, measure, exposure, util
from skimage.filters import threshold_otsu, threshold_yen, threshold_triangle
from scipy import ndimage as ndi
from scipy.stats import median_abs_deviation as mad
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    return util.img_as_ubyte(img)


def as_rgb(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return ensure_uint8(img)


def show(img, title: str = "", cmap: str | None = None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def adaptive_kernel_px(um_per_px: float, nucleus_diam_um: float = 10.0) -> int:
    d_px = max(3, int(round(nucleus_diam_um / max(um_per_px, 1e-6))))
    if d_px % 2 == 0:
        d_px += 1
    return d_px

# -------------------------
# Core steps
# -------------------------

def get_tissue_mask(rgb, method: str = "S", um_per_px: float = 0.5, min_obj_um2: float = 400.0):
    """Automatic tissue mask.
    method: "S" (HSV saturation) or starting with "h" (hematoxylin channel).
    """
    rgb = as_rgb(rgb)
    if method.lower().startswith("h"):  # hematoxylin
        hed = color.rgb2hed(rgb)
        chan = hed[..., 0]
        # invert to make tissue brighter for thresholding
        chan = exposure.rescale_intensity(-chan,
                                          in_range=(np.percentile(-chan, 2), np.percentile(-chan, 98)))
    else:  # "S"
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        chan = hsv[..., 1] / 255.0

    thr = threshold_otsu(chan)
    mask = chan > thr

    # resolution-aware area threshold
    area_px_min = int(round((min_obj_um2 / (um_per_px ** 2))))
    mask = morphology.remove_small_objects(mask, area_px_min)
    mask = morphology.remove_small_holes(mask, area_px_min)

    k = adaptive_kernel_px(um_per_px, nucleus_diam_um=8.0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, se) > 0
    return mask


def robust_hue_artifact_mask(rgb, tissue_mask, pclip=(5, 95), target_area_range=(0.005, 0.30)):
    """Hue-based artifact mask with percentile clipping, multi-threshold candidates, and
    a target area range selector. Falls back to quantiles if needed.
    Returns (mask: bool array, info: dict).
    """
    rgb = as_rgb(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32)  # [0,179] OpenCV

    tissue_idx = tissue_mask > 0
    h_vals = H[tissue_idx]
    if h_vals.size == 0:
        return np.zeros(H.shape, bool), {"method": None, "area_frac": 0.0}

    lo, hi = np.percentile(h_vals, pclip)
    h_clip = np.clip(H, lo, hi)
    h_norm = (h_clip - lo) / max(hi - lo, 1e-6)  # [0,1]

    thr_candidates = []
    for mname, mfun in (("yen", threshold_yen), ("triangle", threshold_triangle), ("otsu", threshold_otsu)):
        try:
            thr = float(mfun(h_norm[tissue_idx]))
            thr_candidates.append((mname, thr))
        except Exception:
            pass

    def eval_mask(thr, pick_low):
        if pick_low:
            m = (h_norm <= thr) & tissue_idx
        else:
            m = (h_norm >= thr) & tissue_idx
        af = float(m.sum()) / float(tissue_idx.sum())
        return m, af

    best = None
    for mname, thr in thr_candidates:
        for tail in ("high", "low"):
            m, af = eval_mask(thr, pick_low=(tail == "low"))
            if target_area_range[0] <= af <= target_area_range[1]:
                labeled = measure.label(m)
                ncomp = labeled.max()
                score = af + 0.001 * ncomp  # area first, lightly penalize fragmentation
                if (best is None) or (score < best[0]):
                    best = (score, mname, thr, tail, af, ncomp, m)

    if best is None:
        # fallback: choose side with smaller area at extreme quantiles
        q = 0.95
        thr_high = np.quantile(h_norm[tissue_idx], q)
        thr_low = np.quantile(h_norm[tissue_idx], 1 - q)
        m_high, af_high = eval_mask(thr_high, pick_low=False)
        m_low, af_low = eval_mask(thr_low, pick_low=True)
        if af_high <= af_low:
            return m_high, {"method": "quantile", "thr": thr_high, "tail": "high", "area_frac": af_high, "pclip": pclip}
        else:
            return m_low, {"method": "quantile", "thr": thr_low, "tail": "low", "area_frac": af_low, "pclip": pclip}

    _, mname, thr, tail, af, ncomp, mask = best
    info = {
        "method": mname,
        "thr": thr,
        "tail": tail,
        "area_frac": af,
        "n_components": int(ncomp),
        "pclip": pclip,
        "target_area_range": target_area_range,
    }
    return mask, info


def filter_small_nuclei(
    nuc_mask_bool,
    um_per_px: float,
    nucleus_diam_um: float = 10.0,
    min_area_um2: float | None = None,
    min_area_factor: float = 0.35,
    fill_holes: bool = True,
):
    """Remove tiny nuclei/fragments using a resolution-aware area threshold.
    The threshold is min_area_um2 if provided, else (pi*(d/2)^2)*min_area_factor.
    Returns (mask: bool, info: dict).
    """
    nuc_mask_bool = nuc_mask_bool.astype(bool)
    expected_um2 = np.pi * (nucleus_diam_um / 2.0) ** 2
    thr_um2 = float(min_area_um2) if (min_area_um2 is not None) else expected_um2 * float(min_area_factor)
    thr_px = int(round(thr_um2 / (um_per_px ** 2)))

    n_before = measure.label(nuc_mask_bool).max()
    out = morphology.remove_small_objects(nuc_mask_bool, min_size=max(1, thr_px))
    if fill_holes:
        out = morphology.remove_small_holes(out, area_threshold=max(1, thr_px))
    n_after = measure.label(out).max()

    info = {
        "n_before": int(n_before),
        "n_after": int(n_after),
        "removed": int(n_before - n_after),
        "min_area_um2": round(thr_um2, 2),
        "min_area_px": int(thr_px),
        "nucleus_diam_um": float(nucleus_diam_um),
        "um_per_px": float(um_per_px),
    }
    return out, info


def keep_nuclei_and_pink_no_bh(
    rgb,
    tissue_mask,
    nuclei_mask,  # here pass artifact_mask_filt
    q_width: float = 0.30,
    k_s: float = 2.0,
    k_v: float = 2.0,
    nucleus_dilate_px: int = 0,
    smooth_replaced: bool = True,
    k_px_for_smooth: int = 9,
):
    """Keep nuclei and replace non-nuclear tissue pixels with a median pink reference.
    Hue statistics computed with circular mean; S/V constrained by MAD.
    Returns (rgb_out, info: dict, masks: dict)
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32)  # 0–179
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0

    # optional dilation safety margin around nuclei
    if nucleus_dilate_px > 0:
        k = nucleus_dilate_px if nucleus_dilate_px % 2 == 1 else nucleus_dilate_px + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        nuclei_safe = cv2.dilate(nuclei_mask.astype(np.uint8), se).astype(bool)
    else:
        nuclei_safe = nuclei_mask.astype(bool)

    domain = (tissue_mask.astype(bool) & (~nuclei_safe))
    if domain.sum() == 0:
        return rgb.copy(), {"note": "empty domain"}, {"replace_mask": np.zeros(H.shape, bool)}

    # 1) circular (S-weighted) hue mean
    theta = H[domain] / 180.0 * 2 * np.pi
    w = np.clip(S[domain], 0.05, None)
    mu_x = np.sum(w * np.cos(theta)) / (w.sum() + 1e-9)
    mu_y = np.sum(w * np.sin(theta)) / (w.sum() + 1e-9)
    mu = np.arctan2(mu_y, mu_x)

    # 2) pink band: circular hue distance quantile + MAD constraints in S/V + broad quantile bounds
    theta_all = H / 180.0 * 2 * np.pi
    d = np.abs(np.angle(np.exp(1j * (theta_all - mu))))
    d_thr = np.quantile(d[domain], q_width)

    S_dom, V_dom = S[domain], V[domain]
    S_med, V_med = np.median(S_dom), np.median(V_dom)
    S_mad = mad(S_dom, scale="normal") + 1e-6
    V_mad = mad(V_dom, scale="normal") + 1e-6
    s_lo, s_hi = np.percentile(S_dom, (5, 99.5))
    v_lo, v_hi = np.percentile(V_dom, (2, 99.8))

    pink_mask_core = (
        (d <= d_thr)
        & (np.abs(S - S_med) <= k_s * S_mad)
        & (np.abs(V - V_med) <= k_v * V_mad)
        & (S >= s_lo)
        & (S <= s_hi)
        & (V >= v_lo)
        & (V <= v_hi)
        & tissue_mask
        & (~nuclei_safe)
    )

    # 3) replacement: non-nuclear & non-pink tissue pixels
    replace_mask = domain & (~pink_mask_core)

    # 4) reference pink from median HSV of core region (fallback to domain if too small)
    sel = pink_mask_core if pink_mask_core.sum() >= 500 else domain
    med_h = np.median(H[sel]).astype(np.uint8)
    med_s = (np.median(S[sel] * 255)).astype(np.uint8)
    med_v = (np.median(V[sel] * 255)).astype(np.uint8)
    pink_ref = np.array([med_h, med_s, med_v], dtype=np.uint8)

    out_hsv = hsv.copy()
    out_hsv[replace_mask] = pink_ref
    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2RGB)

    # 5) light bilateral smoothing only on replaced regions
    if smooth_replaced and replace_mask.any():
        sm = cv2.bilateralFilter(out, d=max(5, k_px_for_smooth), sigmaColor=40, sigmaSpace=max(5, k_px_for_smooth))
        out = np.where(replace_mask[..., None], sm, out)

    info = {
        "mu_h": float(((mu % (2 * np.pi)) / (2 * np.pi) * 180.0)),
        "q_width": float(q_width),
        "d_thr": float(d_thr),
        "S_med": float(S_med),
        "V_med": float(V_med),
        "S_mad": float(S_mad),
        "V_mad": float(V_mad),
        "s_lo": float(s_lo),
        "s_hi": float(s_hi),
        "v_lo": float(v_lo),
        "v_hi": float(v_hi),
        "pink_frac_core": float(pink_mask_core.sum()) / float(tissue_mask.sum() + 1e-9),
        "replaced_frac": float(replace_mask.sum()) / float(tissue_mask.sum() + 1e-9),
        "nucleus_dilate_px": int(nucleus_dilate_px),
    }
    return out, info, {"replace_mask": replace_mask, "nonpink_mask": (domain & (~pink_mask_core)), "nuclei_mask_used": nuclei_safe}

# -------------------------
# Orchestration
# -------------------------

def read_rgb(image_path: str):
    """Read image from path, prefer tifffile when available, fallback to cv2.
    Returns uint8 RGB.
    """
    try:
        from tifffile import imread as tiff_imread
        rgb = tiff_imread(image_path)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
    except Exception:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return as_rgb(rgb)


def run_pipeline(
    image_path: str,
    # acquisition & geometry
    um_per_px: float = 0.5,
    nucleus_diam_um: float = 10.0,
    # tissue masking
    tissue_method: str = "S",  # or "hematoxylin"
    min_tissue_obj_um2: float = 400.0,
    # hue artifact mask
    pclip_low_high: Tuple[float, float] = (5, 95),
    target_area_range: Tuple[float, float] = (0.005, 0.30),
    # small nuclei filter
    min_area_um2: float | None = None,
    min_area_factor: float = 0.35,
    fill_holes: bool = True,
    # pink replacement params
    q_width: float = 0.30,
    k_s: float = 2.0,
    k_v: float = 2.0,
    nucleus_dilate_px: int = 0,
    smooth_replaced: bool = True,
    k_px_for_smooth: int | None = None,
    # outputs
    show_preview: bool = False,
    save_json: str | None = None,
):
    """Run the full preprocessing pipeline and return outputs ready for StarDist.
    Returns dict with keys: rgb, tissue_mask, artifact_mask, artifact_mask_filt,
    img_keep_pink, info dicts, masks dicts, k_px.
    """
    rgb = read_rgb(image_path)

    tissue_mask = get_tissue_mask(rgb, method=tissue_method, um_per_px=um_per_px, min_obj_um2=min_tissue_obj_um2)

    artifact_mask, art_info = robust_hue_artifact_mask(
        rgb, tissue_mask, pclip=pclip_low_high, target_area_range=target_area_range
    )

    artifact_mask_filt, filt_info = filter_small_nuclei(
        artifact_mask, um_per_px, nucleus_diam_um=nucleus_diam_um,
        min_area_um2=min_area_um2, min_area_factor=min_area_factor, fill_holes=fill_holes
    )

    k_px = adaptive_kernel_px(um_per_px, nucleus_diam_um=nucleus_diam_um)
    k_px_for_smooth = k_px if (k_px_for_smooth is None) else int(k_px_for_smooth)

    img_keep_pink, info_pink, masks_pink = keep_nuclei_and_pink_no_bh(
        rgb, tissue_mask, artifact_mask_filt,
        q_width=q_width, k_s=k_s, k_v=k_v,
        nucleus_dilate_px=nucleus_dilate_px,
        smooth_replaced=smooth_replaced, k_px_for_smooth=k_px_for_smooth
    )

    if show_preview:
        try:
            show(img_keep_pink, f"Scheme 1 (no black-hat) | {json.dumps(info_pink, indent=0)[:120]}…")
        except Exception:
            pass

    out = {
        "rgb": rgb,
        "tissue_mask": tissue_mask,
        "artifact_mask": artifact_mask,
        "artifact_mask_filt": artifact_mask_filt,
        "img_keep_pink": img_keep_pink,
        "info": {
            "artifact": art_info,
            "filter": filt_info,
            "pink": info_pink,
            "k_px": int(k_px),
        },
        "masks": masks_pink,
    }

    if save_json:
        serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in out["info"].items()
        }
        with open(save_json, "w") as f:
            json.dump(serializable, f, indent=2)

    return out


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SegDecon histology preprocessing pipeline (refactor)")
    p.add_argument("image_path", type=str, help="Path to input RGB/H&E image")
    p.add_argument("--um_per_px", type=float, default=0.5)
    p.add_argument("--nucleus_diam_um", type=float, default=10.0)

    p.add_argument("--tissue_method", type=str, default="S", choices=["S", "hematoxylin", "H", "h"])
    p.add_argument("--min_tissue_obj_um2", type=float, default=400.0)

    p.add_argument("--pclip_low_high", type=float, nargs=2, default=(5, 95))
    p.add_argument("--target_area_range", type=float, nargs=2, default=(0.005, 0.30))

    p.add_argument("--min_area_um2", type=float, default=None)
    p.add_argument("--min_area_factor", type=float, default=0.35)
    p.add_argument("--fill_holes", action="store_true", default=True)
    p.add_argument("--no-fill_holes", dest="fill_holes", action="store_false")

    p.add_argument("--q_width", type=float, default=0.30)
    p.add_argument("--k_s", type=float, default=2.0)
    p.add_argument("--k_v", type=float, default=2.0)
    p.add_argument("--nucleus_dilate_px", type=int, default=0)
    p.add_argument("--smooth_replaced", action="store_true", default=True)
    p.add_argument("--no-smooth_replaced", dest="smooth_replaced", action="store_false")
    p.add_argument("--k_px_for_smooth", type=int, default=None)

    p.add_argument("--show_preview", action="store_true", default=False)
    p.add_argument("--save_json", type=str, default=None)

    args = p.parse_args()

    out = run_pipeline(
        image_path=args.image_path,
        um_per_px=args.um_per_px,
        nucleus_diam_um=args.nucleus_diam_um,
        tissue_method=args.tissue_method,
        min_tissue_obj_um2=args.min_tissue_obj_um2,
        pclip_low_high=tuple(args.pclip_low_high),
        target_area_range=tuple(args.target_area_range),
        min_area_um2=args.min_area_um2,
        min_area_factor=args.min_area_factor,
        fill_holes=args.fill_holes,
        q_width=args.q_width,
        k_s=args.k_s,
        k_v=args.k_v,
        nucleus_dilate_px=args.nucleus_dilate_px,
        smooth_replaced=args.smooth_replaced,
        k_px_for_smooth=args.k_px_for_smooth,
        show_preview=args.show_preview,
        save_json=args.save_json,
    )

    # Ready to feed `out["img_keep_pink"]` to StarDist in your environment.
    # Example: save intermediate visualization
    try:
        import imageio.v2 as iio
        preview_path = os.path.splitext(os.path.basename(args.image_path))[0] + "_preproc_preview.png"
        iio.imwrite(preview_path, out["img_keep_pink"])
        print(f"Saved preview to: {preview_path}")
    except Exception:
        pass

    print(json.dumps(out["info"], indent=2))


# --- optional: per-run logging to CSV (recommended) ---
try:
    from scripts.eval.log_run import append_run_row
    run_csv = "data/tables/run_metadata.csv"
    args_dict = {
        "um_per_px": args.um_per_px,
        "nucleus_diam_um": args.nucleus_diam_um,
        "tissue_method": args.tissue_method,
        "min_tissue_obj_um2": args.min_tissue_obj_um2,
        "pclip_low_high": tuple(args.pclip_low_high),
        "target_area_range": tuple(args.target_area_range),
        "min_area_um2": args.min_area_um2,
        "min_area_factor": args.min_area_factor,
        "fill_holes": args.fill_holes,
        "q_width": args.q_width,
        "k_s": args.k_s,
        "k_v": args.k_v,
        "nucleus_dilate_px": args.nucleus_dilate_px,
        "smooth_replaced": args.smooth_replaced,
        "k_px_for_smooth": args.k_px_for_smooth,
    }
    append_run_row(run_csv, args.image_path, args_dict, out["info"])
    print(f"Appended run metadata → {run_csv}")
except Exception as e:
    print(f"[warn] could not append run metadata: {e}")


"""
how to use it
---------------------------------------------------------------------
#1.terminal
python segdecon_histology_pipeline.py /data/HE_slide.tif \
  --um_per_px 0.5 \
  --nucleus_diam_um 10 \
  --tissue_method S \
  --pclip_low_high 5 95 \
  --target_area_range 0.005 0.30 \
  --min_area_factor 0.35 \
  --q_width 0.30 --k_s 2.0 --k_v 2.0 \
  --nucleus_dilate_px 0 \
  --smooth_replaced \
  --show_preview \
  --save_json run_info.json

#2. python
from segdecon_histology_stack import run_pipeline

out = run_pipeline(
    image_path="/path/to/your_image.tif",   # 
    um_per_px=0.5,
    nucleus_diam_um=10.0,
    tissue_method="S",                      # or "hematoxylin"
    pclip_low_high=(5, 95),
    target_area_range=(0.005, 0.30),
    min_area_factor=0.35,                   # or min_area_um2=...
    q_width=0.30, k_s=2.0, k_v=2.0,
    nucleus_dilate_px=0,
    smooth_replaced=True,
    show_preview=False                     
)

# input for StarDist ：
img_for_stardist = out["img_keep_pink"]
"""
