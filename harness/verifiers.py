"""Deterministic Physical-Engineering Code-Gate Verifiers & Revision Authority Parser."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


AUTHORITY_RANK: Dict[str, int] = {
    "preliminary": 10,
    "permit": 20,
    "bid": 30,
    "ifc": 40,
    "field_directive": 50,
    "rfi_response": 60,
    "rfi": 60,
    "addendum": 70,
    "addenda": 70,
    "ahj_letter": 80,
    "ahj": 80,
}


def classify_document_authority(filename: str) -> Tuple[str, int]:
    """Return (authority_label, rank) for a project document filename."""
    lower = filename.lower()
    best_label = "base"
    best_rank = 0
    for key, rank in AUTHORITY_RANK.items():
        if key in lower and rank > best_rank:
            best_label = key
            best_rank = rank
    return best_label, best_rank


def rank_project_files(filenames: Iterable[str]) -> List[Dict[str, Any]]:
    """Sort project filenames from highest revision authority to lowest."""
    ranked = []
    for name in filenames:
        label, rank = classify_document_authority(name)
        ranked.append({"file": name, "authority": label, "rank": rank})
    ranked.sort(key=lambda x: (-x["rank"], x["file"]))
    return ranked


def verify_aisc_w_beam(
    span_ft: float,
    w_dead_klf: float,
    w_live_klf: float,
    zx_in3: float,
    ix_in4: float,
    d_in: float,
    tw_in: float,
    fy_ksi: float = 50.0,
    e_ksi: float = 29000.0,
    deflection_limit_ratio: float = 360.0,
) -> Dict[str, Any]:
    """AISC 360-16 LRFD check for a simply supported uniformly loaded W-beam."""
    if min(span_ft, zx_in3, ix_in4, d_in, tw_in, fy_ksi, e_ksi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    w_u_klf = 1.2 * w_dead_klf + 1.6 * w_live_klf
    m_u_kip_in = (w_u_klf * (span_ft**2) / 8.0) * 12.0
    phi_mn_kip_in = 0.90 * fy_ksi * zx_in3
    flex_dcr = m_u_kip_in / phi_mn_kip_in

    v_u_kip = w_u_klf * span_ft / 2.0
    phi_vn_kip = 0.90 * 0.6 * fy_ksi * d_in * tw_in
    shear_dcr = v_u_kip / phi_vn_kip

    w_live_kip_per_in = w_live_klf / 12.0
    l_in = span_ft * 12.0
    delta_live_in = (5.0 * w_live_kip_per_in * (l_in**4)) / (384.0 * e_ksi * ix_in4)
    delta_allow_in = l_in / deflection_limit_ratio
    defl_dcr = delta_live_in / delta_allow_in

    passed = (flex_dcr <= 1.0) and (shear_dcr <= 1.0) and (defl_dcr <= 1.0)
    return {
        "pass": passed,
        "flex_dcr": round(flex_dcr, 4),
        "shear_dcr": round(shear_dcr, 4),
        "defl_dcr": round(defl_dcr, 4),
    }


def verify_footing_bearing_and_settlement(
    applied_pressure_ksf: float,
    ultimate_bearing_ksf: float,
    predicted_settlement_in: float,
    allowable_settlement_in: float = 1.0,
    min_fs: float = 3.0,
) -> Dict[str, Any]:
    """Civil/Geotech spread-footing bearing safety factor and settlement gate."""
    if applied_pressure_ksf <= 0 or ultimate_bearing_ksf <= 0:
        return {"pass": False, "reason": "non_positive_bearing_input"}
    fs = ultimate_bearing_ksf / applied_pressure_ksf
    passed = (fs >= min_fs) and (predicted_settlement_in <= allowable_settlement_in)
    return {
        "pass": passed,
        "factor_of_safety": round(fs, 4),
        "settlement_ratio": round(predicted_settlement_in / allowable_settlement_in, 4),
    }


def verify_nec_voltage_drop(
    current_a: float,
    length_ft: float,
    cmil: float,
    voltage_v: float,
    three_phase: bool = True,
    resistivity_ohm_cmil_ft: float = 12.9,
    max_drop_pct: float = 3.0,
) -> Dict[str, Any]:
    """NEC branch/feeder voltage-drop check (copper default K=12.9 ohm-cmil/ft)."""
    if min(cmil, voltage_v) <= 0 or current_a < 0 or length_ft < 0:
        return {"pass": False, "reason": "invalid_electrical_input"}
    factor = math.sqrt(3.0) if three_phase else 2.0
    v_drop = (factor * resistivity_ohm_cmil_ft * current_a * length_ft) / cmil
    pct_drop = (v_drop / voltage_v) * 100.0
    return {
        "pass": pct_drop <= max_drop_pct,
        "voltage_drop_v": round(v_drop, 4),
        "voltage_drop_pct": round(pct_drop, 4),
    }


def check_aabb_clashes(
    boxes: Sequence[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]],
    min_clearance: float = 0.0,
) -> List[Tuple[str, str]]:
    """Detect 3D Axis-Aligned Bounding Box (AABB) MEP clashes with optional clearance."""
    clashes: List[Tuple[str, str]] = []
    n = len(boxes)
    for i in range(n):
        id_a, min_a, max_a = boxes[i]
        for j in range(i + 1, n):
            id_b, min_b, max_b = boxes[j]
            overlap = all(
                (min_a[k] - min_clearance < max_b[k]) and (min_b[k] - min_clearance < max_a[k])
                for k in range(3)
            )
            if overlap:
                clashes.append((id_a, id_b))
    return clashes


def verify_aisc_column_compression(
    p_dead_kips: float,
    p_live_kips: float,
    kl_x_ft: float,
    kl_y_ft: float,
    rx_in: float,
    ry_in: float,
    ag_in2: float,
    fy_ksi: float = 50.0,
    e_ksi: float = 29000.0,
    max_kl_r: float = 200.0,
) -> Dict[str, Any]:
    """AISC 360-16 Chapter E LRFD check for column flexural buckling."""
    inputs = [p_dead_kips, p_live_kips, kl_x_ft, kl_y_ft, rx_in, ry_in, ag_in2, fy_ksi, e_ksi, max_kl_r]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(p_dead_kips, p_live_kips) < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(kl_x_ft, kl_y_ft, rx_in, ry_in, ag_in2, fy_ksi, e_ksi, max_kl_r) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    kl_r_x = (kl_x_ft * 12.0) / rx_in
    kl_r_y = (kl_y_ft * 12.0) / ry_in
    kl_r = max(kl_r_x, kl_r_y)

    p_u = 1.2 * p_dead_kips + 1.6 * p_live_kips
    fe = (math.pi ** 2 * e_ksi) / (kl_r ** 2)
    threshold = 4.71 * math.sqrt(e_ksi / fy_ksi)

    if kl_r <= threshold:
        f_cr = (0.658 ** (fy_ksi / fe)) * fy_ksi
    else:
        f_cr = 0.877 * fe

    phi_pn = 0.90 * f_cr * ag_in2
    dcr = p_u / phi_pn if phi_pn > 0 else float("inf")
    passed = (dcr <= 1.0) and (kl_r <= max_kl_r)

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_pn_kips": round(phi_pn, 4),
        "kl_r": round(kl_r, 4),
        "p_u_kips": round(p_u, 4),
    }


def verify_ashrae_duct_sizing(
    airflow_cfm: float,
    width_in: float,
    height_in: float,
    max_velocity_fpm: float = 1200.0,
    max_pressure_drop_per_100ft: float = 0.10,
    max_aspect_ratio: float = 4.0,
) -> Dict[str, Any]:
    """ASHRAE Fundamentals rectangular duct airflow velocity, aspect ratio, and friction rate check."""
    inputs = [airflow_cfm, width_in, height_in, max_velocity_fpm, max_pressure_drop_per_100ft, max_aspect_ratio]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(inputs) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_airflow"}

    area_ft2 = (width_in * height_in) / 144.0
    velocity_fpm = airflow_cfm / area_ft2
    velocity_dcr = velocity_fpm / max_velocity_fpm

    # ASHRAE Huebscher circular equivalent diameter for rectangular ducts
    d_e_in = 1.30 * ((width_in * height_in) ** 0.625) / ((width_in + height_in) ** 0.250)

    # ASHRAE standard friction head loss equation (in. wg per 100 ft of duct)
    friction_loss_per_100ft = 0.109136 * (airflow_cfm ** 1.90) / (d_e_in ** 5.02)
    friction_dcr = friction_loss_per_100ft / max_pressure_drop_per_100ft

    aspect_ratio = max(width_in, height_in) / min(width_in, height_in)
    passed = (velocity_dcr <= 1.0) and (friction_dcr <= 1.0) and (aspect_ratio <= max_aspect_ratio)

    return {
        "pass": passed,
        "velocity_fpm": round(velocity_fpm, 2),
        "equivalent_diameter_in": round(d_e_in, 2),
        "pressure_drop_per_100ft": round(friction_loss_per_100ft, 4),
        "aspect_ratio": round(aspect_ratio, 2),
        "velocity_dcr": round(velocity_dcr, 4),
        "friction_dcr": round(friction_dcr, 4),
    }


def verify_aci_beam_flexure(
    mu_kip_ft: float,
    b_in: float,
    d_in: float,
    as_in2: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    es_psi: float = 29000000.0,
) -> Dict[str, Any]:
    """ACI 318-19 LRFD flexural capacity check for rectangular reinforced concrete beams."""
    inputs = [mu_kip_ft, b_in, d_in, as_in2, fc_psi, fy_psi, es_psi]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if mu_kip_ft < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(b_in, d_in, as_in2, fc_psi, fy_psi, es_psi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    # ACI 318-19 Section 9.6.1.2 Minimum flexural reinforcement
    as_min_in2 = max(
        (3.0 * math.sqrt(fc_psi) / fy_psi) * b_in * d_in,
        (200.0 / fy_psi) * b_in * d_in,
    )

    # Equivalent rectangular compressive stress block depth (a) and neutral axis (c)
    a_in = (as_in2 * fy_psi) / (0.85 * fc_psi * b_in)
    if fc_psi <= 4000.0:
        beta_1 = 0.85
    elif fc_psi >= 8000.0:
        beta_1 = 0.65
    else:
        beta_1 = 0.85 - 0.05 * ((fc_psi - 4000.0) / 1000.0)

    c_in = a_in / beta_1
    if c_in <= 0:
        return {"pass": False, "reason": "invalid_neutral_axis_depth"}

    # Net tensile strain in extreme tension steel
    eps_ty = fy_psi / es_psi
    eps_t = 0.003 * (d_in - c_in) / c_in

    # ACI 318-19 Table 21.2.2 Strength reduction factor (phi) for moment
    if eps_t >= (eps_ty + 0.003):
        phi = 0.90
    elif eps_t <= eps_ty:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * ((eps_t - eps_ty) / 0.003)

    # Nominal and design moment strength
    mn_kip_in = as_in2 * fy_psi * (d_in - a_in / 2.0) / 1000.0
    phi_mn_kip_ft = (phi * mn_kip_in) / 12.0
    flex_dcr = mu_kip_ft / phi_mn_kip_ft if phi_mn_kip_ft > 0 else float("inf")

    # ACI 318-19 Section 9.3.3.1 requires tension-controlled ductility eps_t >= 0.004
    passed = (flex_dcr <= 1.0) and (as_in2 >= as_min_in2) and (eps_t >= 0.004)

    return {
        "pass": passed,
        "dcr": round(flex_dcr, 4),
        "phi_mn_kip_ft": round(phi_mn_kip_ft, 2),
        "as_min_in2": round(as_min_in2, 3),
        "net_tensile_strain": round(eps_t, 5),
        "phi": round(phi, 3),
    }


def verify_retaining_wall_stability(
    stem_height_ft: float,
    footing_thickness_ft: float,
    toe_width_ft: float,
    stem_thickness_ft: float,
    heel_width_ft: float,
    soil_gamma_pcf: float = 120.0,
    concrete_gamma_pcf: float = 150.0,
    soil_friction_angle_deg: float = 30.0,
    base_friction_coeff: float = 0.45,
    surcharge_psf: float = 0.0,
    min_fs_overturning: float = 2.0,
    min_fs_sliding: float = 1.5,
) -> Dict[str, Any]:
    """Rankine cantilever retaining wall stability check for overturning, sliding, and eccentricity."""
    inputs = [
        stem_height_ft,
        footing_thickness_ft,
        toe_width_ft,
        stem_thickness_ft,
        heel_width_ft,
        soil_gamma_pcf,
        concrete_gamma_pcf,
        soil_friction_angle_deg,
        base_friction_coeff,
        surcharge_psf,
        min_fs_overturning,
        min_fs_sliding,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            stem_height_ft,
            footing_thickness_ft,
            stem_thickness_ft,
            heel_width_ft,
            soil_gamma_pcf,
            concrete_gamma_pcf,
            base_friction_coeff,
            min_fs_overturning,
            min_fs_sliding,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if toe_width_ft < 0 or surcharge_psf < 0:
        return {"pass": False, "reason": "negative_dimension_or_surcharge"}
    if not (0.0 < soil_friction_angle_deg < 90.0):
        return {"pass": False, "reason": "invalid_friction_angle"}

    phi_rad = math.radians(soil_friction_angle_deg)
    ka = (1.0 - math.sin(phi_rad)) / (1.0 + math.sin(phi_rad))

    h_total = stem_height_ft + footing_thickness_ft
    b_total = toe_width_ft + stem_thickness_ft + heel_width_ft

    # Lateral earth pressure & surcharge driving forces
    p_soil = 0.5 * ka * soil_gamma_pcf * (h_total**2)
    p_sur = ka * surcharge_psf * h_total
    p_ah = p_soil + p_sur
    m_ot = p_soil * (h_total / 3.0) + p_sur * (h_total / 2.0)

    # Resisting vertical forces & moments about toe edge
    w_footing = b_total * footing_thickness_ft * concrete_gamma_pcf
    m_footing = w_footing * (b_total / 2.0)

    w_stem = stem_thickness_ft * stem_height_ft * concrete_gamma_pcf
    m_stem = w_stem * (toe_width_ft + stem_thickness_ft / 2.0)

    w_soil = heel_width_ft * stem_height_ft * soil_gamma_pcf
    m_soil = w_soil * (toe_width_ft + stem_thickness_ft + heel_width_ft / 2.0)

    w_sur = surcharge_psf * heel_width_ft
    m_sur = w_sur * (toe_width_ft + stem_thickness_ft + heel_width_ft / 2.0)

    v_total = w_footing + w_stem + w_soil + w_sur
    m_resisting = m_footing + m_stem + m_soil + m_sur

    fs_ot = m_resisting / m_ot if m_ot > 0 else float("inf")
    f_resisting_sliding = v_total * base_friction_coeff
    fs_sl = f_resisting_sliding / p_ah if p_ah > 0 else float("inf")

    # Location of resultant from toe and eccentricity relative to base centerline
    x_res = (m_resisting - m_ot) / v_total
    e = abs((b_total / 2.0) - x_res)
    kern_limit = b_total / 6.0

    passed = (fs_ot >= min_fs_overturning) and (fs_sl >= min_fs_sliding) and (e <= kern_limit)
    return {
        "pass": passed,
        "fs_overturning": round(fs_ot, 4),
        "fs_sliding": round(fs_sl, 4),
        "eccentricity_ft": round(e, 4),
        "kern_limit_ft": round(kern_limit, 4),
        "ka": round(ka, 4),
        "lateral_force_lb_per_ft": round(p_ah, 2),
    }


def verify_aci_beam_shear(
    vu_kips: float,
    bw_in: float,
    d_in: float,
    av_in2: float,
    spacing_in: float,
    fc_psi: float,
    fyt_psi: float = 60000.0,
    lambda_factor: float = 1.0,
) -> Dict[str, Any]:
    """ACI 318-19 Chapter 22 LRFD one-way shear capacity check for reinforced concrete beams."""
    inputs = [vu_kips, bw_in, d_in, av_in2, spacing_in, fc_psi, fyt_psi, lambda_factor]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if av_in2 < 0:
        return {"pass": False, "reason": "negative_reinforcement_area"}
    if min(bw_in, d_in, spacing_in, fc_psi, fyt_psi, lambda_factor) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    # ACI 318-19 Table 20.2.2.4a yield strength upper limit for shear design
    fyt_effective = min(fyt_psi, 60000.0)
    sqrt_fc = math.sqrt(fc_psi)

    # Concrete shear strength Vc (ACI 318-19 Table 22.5.5.1)
    vc_kips = (2.0 * lambda_factor * sqrt_fc * bw_in * d_in) / 1000.0

    # Steel stirrup shear strength Vs (ACI 318-19 Eq. 22.5.10.5.3)
    vs_kips = (av_in2 * fyt_effective * d_in / spacing_in) / 1000.0 if av_in2 > 0 else 0.0

    # Section shear capacity upper limit (ACI 318-19 Section 22.5.1.2: Vs <= 8*sqrt(f'c)*bw*d)
    vs_max_kips = (8.0 * sqrt_fc * bw_in * d_in) / 1000.0
    if vs_kips > vs_max_kips:
        return {
            "pass": False,
            "reason": "section_shear_exceeded",
            "vs_kips": round(vs_kips, 2),
            "vs_max_kips": round(vs_max_kips, 2),
        }

    # Strength reduction factor phi for shear
    phi = 0.75
    phi_vn_kips = phi * (vc_kips + vs_kips)
    shear_dcr = vu_kips / phi_vn_kips if phi_vn_kips > 0 else float("inf")

    # Maximum stirrup spacing (ACI 318-19 Table 9.7.6.2.2)
    vs_thresh_kips = (4.0 * sqrt_fc * bw_in * d_in) / 1000.0
    if vs_kips <= vs_thresh_kips:
        s_max_in = min(d_in / 2.0, 24.0)
    else:
        s_max_in = min(d_in / 4.0, 12.0)

    # Minimum shear reinforcement (ACI 318-19 Table 9.6.3.4)
    av_s_min = max(
        (0.75 * sqrt_fc * bw_in) / fyt_effective,
        (50.0 * bw_in) / fyt_effective,
    )
    av_s_provided = av_in2 / spacing_in

    # If Vu > 0.5 * phi * Vc, transverse reinforcement is required by code
    transverse_required = vu_kips > (0.5 * phi * vc_kips)
    reinf_adequate = True
    if transverse_required:
        reinf_adequate = (av_s_provided >= av_s_min) and (spacing_in <= s_max_in)

    passed = (shear_dcr <= 1.0) and reinf_adequate
    return {
        "pass": passed,
        "dcr": round(shear_dcr, 4),
        "phi_vn_kips": round(phi_vn_kips, 2),
        "vc_kips": round(vc_kips, 2),
        "vs_kips": round(vs_kips, 2),
        "s_max_in": round(s_max_in, 2),
        "av_s_min": round(av_s_min, 4),
        "av_s_provided": round(av_s_provided, 4),
    }


def verify_manning_pipe_flow(
    q_design_cfs: float,
    diameter_in: float,
    slope_ft_per_ft: float,
    manning_n: float = 0.013,
    min_velocity_fps: float = 2.0,
    max_velocity_fps: float = 10.0,
    max_flow_ratio: float = 0.80,
) -> Dict[str, Any]:
    """Civil/Water Resources gravity sewer pipe sizing and self-cleansing velocity check via Manning's formula."""
    inputs = [
        q_design_cfs,
        diameter_in,
        slope_ft_per_ft,
        manning_n,
        min_velocity_fps,
        max_velocity_fps,
        max_flow_ratio,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if q_design_cfs < 0:
        return {"pass": False, "reason": "negative_flow_input"}
    if (
        min(
            diameter_in,
            slope_ft_per_ft,
            manning_n,
            min_velocity_fps,
            max_velocity_fps,
            max_flow_ratio,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_flow"}
    if min_velocity_fps >= max_velocity_fps:
        return {"pass": False, "reason": "invalid_velocity_limits"}

    d_ft = diameter_in / 12.0
    area_ft2 = (math.pi * (d_ft**2)) / 4.0
    hydraulic_radius_ft = d_ft / 4.0

    # Manning's equation for circular pipe flowing full (US customary units)
    v_full_fps = (1.486 / manning_n) * (hydraulic_radius_ft ** (2.0 / 3.0)) * math.sqrt(slope_ft_per_ft)
    q_full_cfs = v_full_fps * area_ft2

    flow_ratio = q_design_cfs / q_full_cfs if q_full_cfs > 0 else float("inf")
    flow_dcr = flow_ratio / max_flow_ratio

    velocity_ok = (min_velocity_fps <= v_full_fps <= max_velocity_fps)
    capacity_ok = flow_ratio <= max_flow_ratio
    passed = velocity_ok and capacity_ok

    return {
        "pass": passed,
        "q_full_cfs": round(q_full_cfs, 3),
        "v_full_fps": round(v_full_fps, 3),
        "flow_ratio": round(flow_ratio, 4),
        "flow_dcr": round(flow_dcr, 4),
        "velocity_ok": velocity_ok,
    }


def verify_nec_conductor_ampacity(
    base_ampacity_a: float,
    continuous_load_a: float,
    non_continuous_load_a: float = 0.0,
    ambient_temp_c: float = 30.0,
    num_conductors: int = 3,
    temp_rating_c: float = 75.0,
) -> Dict[str, Any]:
    """NEC 310.15/310.16 conductor ampacity adjustment, correction, and continuous load check."""
    inputs = [
        base_ampacity_a,
        continuous_load_a,
        non_continuous_load_a,
        ambient_temp_c,
        float(num_conductors),
        temp_rating_c,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if base_ampacity_a <= 0:
        return {"pass": False, "reason": "non_positive_ampacity"}
    if continuous_load_a < 0 or non_continuous_load_a < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (continuous_load_a + non_continuous_load_a) <= 0:
        return {"pass": False, "reason": "zero_load"}
    if num_conductors < 1:
        return {"pass": False, "reason": "invalid_conductor_count"}
    if temp_rating_c not in (60.0, 75.0, 90.0):
        return {"pass": False, "reason": "invalid_temperature_rating"}
    if ambient_temp_c >= temp_rating_c:
        return {"pass": False, "reason": "ambient_exceeds_rating"}

    # NEC 310.15(B) ambient temperature correction factor formula
    if ambient_temp_c == 30.0:
        temp_factor = 1.0
    else:
        temp_factor = math.sqrt((temp_rating_c - ambient_temp_c) / (temp_rating_c - 30.0))

    # NEC Table 310.15(C)(1) adjustment factors for more than three current-carrying conductors
    if num_conductors <= 3:
        bundle_factor = 1.0
    elif num_conductors <= 6:
        bundle_factor = 0.80
    elif num_conductors <= 9:
        bundle_factor = 0.70
    elif num_conductors <= 20:
        bundle_factor = 0.50
    elif num_conductors <= 30:
        bundle_factor = 0.45
    elif num_conductors <= 40:
        bundle_factor = 0.40
    else:
        bundle_factor = 0.35

    derated_ampacity_a = base_ampacity_a * temp_factor * bundle_factor

    # NEC 210.19(A)(1) & 215.2(A)(1) 125% continuous + 100% non-continuous load sizing
    required_ampacity_a = 1.25 * continuous_load_a + non_continuous_load_a
    ampacity_dcr = required_ampacity_a / derated_ampacity_a if derated_ampacity_a > 0 else float("inf")

    passed = ampacity_dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(ampacity_dcr, 4),
        "derated_ampacity_a": round(derated_ampacity_a, 2),
        "required_ampacity_a": round(required_ampacity_a, 2),
        "temp_factor": round(temp_factor, 4),
        "bundle_factor": round(bundle_factor, 2),
    }


def verify_transformer_fault_current(
    transformer_kva: float,
    secondary_voltage_v: float,
    impedance_pct: float,
    aic_rating_a: float,
    three_phase: bool = True,
    motor_contribution_multiplier: float = 0.0,
    utility_sc_mva: float = float("inf"),
) -> Dict[str, Any]:
    """NEC 110.9/110.10 & IEEE 141 transformer short-circuit current and equipment AIC gate."""
    inputs = [
        transformer_kva,
        secondary_voltage_v,
        impedance_pct,
        aic_rating_a,
        motor_contribution_multiplier,
        utility_sc_mva,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(transformer_kva, secondary_voltage_v, impedance_pct, aic_rating_a) <= 0:
        return {"pass": False, "reason": "non_positive_rating_or_voltage"}
    if motor_contribution_multiplier < 0:
        return {"pass": False, "reason": "negative_motor_contribution"}
    if utility_sc_mva <= 0:
        return {"pass": False, "reason": "non_positive_utility_mva"}

    # Full load amperes (FLA)
    if three_phase:
        fla_a = (transformer_kva * 1000.0) / (math.sqrt(3.0) * secondary_voltage_v)
    else:
        fla_a = (transformer_kva * 1000.0) / secondary_voltage_v

    z_xfmr_pu = impedance_pct / 100.0
    z_utility_pu = (
        (transformer_kva / (utility_sc_mva * 1000.0))
        if not math.isinf(utility_sc_mva)
        else 0.0
    )
    z_total_pu = z_xfmr_pu + z_utility_pu

    isc_transformer_a = fla_a / z_total_pu
    isc_motor_a = motor_contribution_multiplier * fla_a
    total_fault_current_a = isc_transformer_a + isc_motor_a

    aic_dcr = total_fault_current_a / aic_rating_a
    passed = aic_dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(aic_dcr, 4),
        "total_fault_current_a": round(total_fault_current_a, 2),
        "transformer_fault_a": round(isc_transformer_a, 2),
        "motor_contribution_a": round(isc_motor_a, 2),
        "fla_a": round(fla_a, 2),
    }


def verify_hydronic_pipe_sizing(
    flow_gpm: float,
    inner_diameter_in: float,
    pipe_length_ft: float = 100.0,
    c_factor: float = 130.0,
    max_velocity_fps: float = 8.0,
    min_velocity_fps: float = 2.0,
    max_head_loss_per_100ft: float = 4.0,
) -> Dict[str, Any]:
    """ASHRAE Fundamentals closed-loop hydronic pipe velocity, friction head loss, and pressure drop check via Hazen-Williams."""
    inputs = [
        flow_gpm,
        inner_diameter_in,
        pipe_length_ft,
        c_factor,
        max_velocity_fps,
        min_velocity_fps,
        max_head_loss_per_100ft,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(inputs) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_flow"}
    if min_velocity_fps >= max_velocity_fps:
        return {"pass": False, "reason": "invalid_velocity_limits"}

    area_ft2 = (math.pi * ((inner_diameter_in / 12.0) ** 2)) / 4.0
    q_cfs = flow_gpm / 448.831169
    velocity_fps = q_cfs / area_ft2
    vel_dcr = velocity_fps / max_velocity_fps

    # Hazen-Williams friction head loss in ft w.g. per 100 ft of pipe (US customary)
    h_f_100 = 1044.0 * (flow_gpm**1.852) / ((c_factor**1.852) * (inner_diameter_in**4.87))
    friction_dcr = h_f_100 / max_head_loss_per_100ft

    total_head_loss_ft = h_f_100 * (pipe_length_ft / 100.0)
    # 1 ft w.g. head = 0.4335 psi for standard density water
    pressure_drop_psi = total_head_loss_ft * 0.4335

    velocity_ok = min_velocity_fps <= velocity_fps <= max_velocity_fps
    head_loss_ok = friction_dcr <= 1.0
    passed = velocity_ok and head_loss_ok

    return {
        "pass": passed,
        "velocity_fps": round(velocity_fps, 3),
        "head_loss_per_100ft": round(h_f_100, 4),
        "total_head_loss_ft": round(total_head_loss_ft, 4),
        "pressure_drop_psi": round(pressure_drop_psi, 4),
        "velocity_dcr": round(vel_dcr, 4),
        "friction_dcr": round(friction_dcr, 4),
        "velocity_ok": velocity_ok,
    }


def verify_asce7_seismic_base_shear(
    seismic_weight_kips: float,
    s_ds: float,
    s_d1: float,
    fundamental_period_s: float,
    r_factor: float,
    v_capacity_kips: float,
    importance_factor: float = 1.0,
    tl_period_s: float = 8.0,
    s1: float = 0.0,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 12.8 Equivalent Lateral Force (ELF) seismic base shear gate."""
    inputs = [
        seismic_weight_kips,
        s_ds,
        s_d1,
        fundamental_period_s,
        r_factor,
        v_capacity_kips,
        importance_factor,
        tl_period_s,
        s1,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            seismic_weight_kips,
            s_ds,
            s_d1,
            fundamental_period_s,
            r_factor,
            v_capacity_kips,
            importance_factor,
            tl_period_s,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_hazard"}
    if s1 < 0:
        return {"pass": False, "reason": "negative_spectral_acceleration"}

    r_over_ie = r_factor / importance_factor

    # ASCE 7-16 Eq. 12.8-2: unconstrained seismic response coefficient
    cs_unclamped = s_ds / r_over_ie

    # ASCE 7-16 Eq. 12.8-3 and Eq. 12.8-4: upper bound limits on Cs
    if fundamental_period_s <= tl_period_s:
        cs_max = s_d1 / (fundamental_period_s * r_over_ie)
    else:
        cs_max = (s_d1 * tl_period_s) / ((fundamental_period_s**2) * r_over_ie)

    # ASCE 7-16 Eq. 12.8-5 and Eq. 12.8-6: lower bound limits on Cs
    cs_min = max(0.044 * s_ds * importance_factor, 0.01)
    if s1 >= 0.6:
        cs_min = max(cs_min, (0.5 * s1) / r_over_ie)

    cs_governing = min(cs_unclamped, cs_max)
    cs = max(cs_governing, cs_min)

    # ASCE 7-16 Eq. 12.8-1: seismic base shear V = Cs * W
    base_shear_kips = cs * seismic_weight_kips
    dcr = base_shear_kips / v_capacity_kips
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "base_shear_kips": round(base_shear_kips, 2),
        "cs": round(cs, 5),
        "cs_unclamped": round(cs_unclamped, 5),
        "cs_max": round(cs_max, 5),
        "cs_min": round(cs_min, 5),
    }


def verify_ashrae_ventilation_rate(
    floor_area_sqft: float,
    occupant_count: float,
    rp_cfm_per_person: float,
    ra_cfm_per_sqft: float,
    supplied_oa_cfm: float,
    ez_effectiveness: float = 1.0,
    supply_air_cfm: float = 0.0,
) -> Dict[str, Any]:
    """ASHRAE 62.1-2019/2022 Ventilation Rate Procedure (VRP) outdoor airflow and distribution check."""
    inputs = [
        floor_area_sqft,
        occupant_count,
        rp_cfm_per_person,
        ra_cfm_per_sqft,
        supplied_oa_cfm,
        ez_effectiveness,
        supply_air_cfm,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if floor_area_sqft <= 0 or supplied_oa_cfm <= 0 or ez_effectiveness <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_flow"}
    if occupant_count < 0:
        return {"pass": False, "reason": "negative_occupancy"}
    if rp_cfm_per_person < 0 or ra_cfm_per_sqft < 0:
        return {"pass": False, "reason": "negative_rate_coefficient"}
    if supply_air_cfm < 0:
        return {"pass": False, "reason": "negative_supply_air"}

    # ASHRAE 62.1 Eq. 6-1: Breathing zone outdoor airflow
    v_bz = (occupant_count * rp_cfm_per_person) + (floor_area_sqft * ra_cfm_per_sqft)
    if v_bz <= 0:
        return {"pass": False, "reason": "zero_required_ventilation"}

    # ASHRAE 62.1 Eq. 6-2: Zone outdoor airflow
    v_oz = v_bz / ez_effectiveness

    # Sizing demand-to-capacity ratio for outdoor air supply
    dcr = v_oz / supplied_oa_cfm

    # Primary outdoor air fraction check (ASHRAE 62.1 Section 6.2.5.1)
    z_p = None
    fraction_ok = True
    if supply_air_cfm > 0:
        if supplied_oa_cfm > supply_air_cfm:
            return {
                "pass": False,
                "reason": "supplied_oa_exceeds_total_supply_air",
                "dcr": round(dcr, 4),
                "v_oz_cfm": round(v_oz, 2),
            }
        z_p = v_oz / supply_air_cfm
        if z_p > 1.0:
            fraction_ok = False

    passed = (dcr <= 1.0) and fraction_ok

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "v_bz_cfm": round(v_bz, 2),
        "v_oz_cfm": round(v_oz, 2),
        "supplied_oa_cfm": round(supplied_oa_cfm, 2),
        "z_p": round(z_p, 4) if z_p is not None else None,
    }


def verify_aisc_bolted_connection(
    vu_kips: float,
    num_bolts: int,
    bolt_diameter_in: float,
    f_nv_ksi: float,
    plate_thickness_in: float,
    fu_ksi: float,
    clear_distance_lc_in: float,
    num_shear_planes: int = 1,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """AISC 360-16 Chapter J3 LRFD check for bolted shear connections (shear rupture & bearing/tearout)."""
    inputs = [
        vu_kips,
        float(num_bolts),
        bolt_diameter_in,
        f_nv_ksi,
        plate_thickness_in,
        fu_ksi,
        clear_distance_lc_in,
        float(num_shear_planes),
        phi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if num_bolts < 1 or num_shear_planes not in (1, 2):
        return {"pass": False, "reason": "invalid_bolt_count_or_shear_planes"}
    if (
        min(bolt_diameter_in, f_nv_ksi, plate_thickness_in, fu_ksi, clear_distance_lc_in, phi)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    # AISC 360-16 Section J3.6 Nominal bolt shear strength
    ab_in2 = (math.pi * (bolt_diameter_in**2)) / 4.0
    rn_shear_single = f_nv_ksi * ab_in2 * num_shear_planes
    phi_rnv_kips = phi * num_bolts * rn_shear_single

    # AISC 360-16 Section J3.10 Bearing and tearout strength on connected material
    rn_tearout = 1.2 * clear_distance_lc_in * plate_thickness_in * fu_ksi
    rn_bearing_max = 2.4 * bolt_diameter_in * plate_thickness_in * fu_ksi
    rn_bearing_single = min(rn_tearout, rn_bearing_max)
    phi_rnb_kips = phi * num_bolts * rn_bearing_single

    phi_rn_governing_kips = min(phi_rnv_kips, phi_rnb_kips)
    dcr = vu_kips / phi_rn_governing_kips if phi_rn_governing_kips > 0 else float("inf")
    limit_state = "bolt_shear" if phi_rnv_kips <= phi_rnb_kips else "plate_bearing"

    passed = dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_rn_kips": round(phi_rn_governing_kips, 2),
        "phi_rnv_kips": round(phi_rnv_kips, 2),
        "phi_rnb_kips": round(phi_rnb_kips, 2),
        "governing_limit_state": limit_state,
    }


def verify_pump_npsh(
    npsh_required_ft: float,
    static_suction_head_ft: float,
    friction_head_loss_ft: float,
    atmospheric_pressure_psia: float = 14.696,
    vapor_pressure_psia: float = 0.363,
    specific_gravity: float = 1.0,
    min_margin_ratio: float = 1.20,
    min_margin_ft: float = 2.0,
) -> Dict[str, Any]:
    """Hydraulic Institute (ANSI/HI 9.6.1) pump Net Positive Suction Head Available (NPSHa) and cavitation gate."""
    inputs = [
        npsh_required_ft,
        static_suction_head_ft,
        friction_head_loss_ft,
        atmospheric_pressure_psia,
        vapor_pressure_psia,
        specific_gravity,
        min_margin_ratio,
        min_margin_ft,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if npsh_required_ft <= 0:
        return {"pass": False, "reason": "non_positive_npsh_required"}
    if friction_head_loss_ft < 0:
        return {"pass": False, "reason": "negative_friction_loss"}
    if min(atmospheric_pressure_psia, specific_gravity, min_margin_ratio) <= 0:
        return {"pass": False, "reason": "non_positive_pressure_or_ratio"}
    if vapor_pressure_psia < 0:
        return {"pass": False, "reason": "negative_vapor_pressure"}
    if min_margin_ft < 0:
        return {"pass": False, "reason": "negative_margin_distance"}
    if vapor_pressure_psia >= atmospheric_pressure_psia:
        return {"pass": False, "reason": "vapor_pressure_exceeds_atmospheric"}

    # Pressure head conversion (ft of liquid) = (p * 144) / (rho_water * SG)
    density_pcf = 62.4 * specific_gravity
    h_atm = (atmospheric_pressure_psia * 144.0) / density_pcf
    h_vp = (vapor_pressure_psia * 144.0) / density_pcf

    npsh_available_ft = h_atm + static_suction_head_ft - friction_head_loss_ft - h_vp
    if npsh_available_ft <= 0:
        return {
            "pass": False,
            "reason": "zero_or_negative_npsh_available",
            "npsh_available_ft": round(npsh_available_ft, 4),
            "npsh_required_ft": round(npsh_required_ft, 4),
        }

    margin_ratio = npsh_available_ft / npsh_required_ft
    margin_ft = npsh_available_ft - npsh_required_ft
    req_npsh_with_margin = max(npsh_required_ft * min_margin_ratio, npsh_required_ft + min_margin_ft)
    dcr = req_npsh_with_margin / npsh_available_ft

    passed = (margin_ratio >= min_margin_ratio) and (margin_ft >= min_margin_ft)
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "npsh_available_ft": round(npsh_available_ft, 3),
        "npsh_required_ft": round(npsh_required_ft, 3),
        "margin_ft": round(margin_ft, 3),
        "margin_ratio": round(margin_ratio, 4),
        "h_atm_ft": round(h_atm, 3),
        "h_vp_ft": round(h_vp, 3),
    }


def verify_asce7_wind_pressure(
    basic_wind_speed_mph: float,
    height_z_ft: float,
    mean_roof_height_ft: float,
    exposure_category: str = "C",
    p_capacity_psf: float = 30.0,
    kd_factor: float = 0.85,
    kzt_factor: float = 1.0,
    ke_factor: float = 1.0,
    gust_factor_g: float = 0.85,
    cp_windward: float = 0.80,
    gc_pi: float = 0.18,
    min_pressure_psf: float = 16.0,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 26.10 & 27.3.1 MWFRS design wind pressure gate."""
    inputs = [
        basic_wind_speed_mph,
        height_z_ft,
        mean_roof_height_ft,
        p_capacity_psf,
        kd_factor,
        kzt_factor,
        ke_factor,
        gust_factor_g,
        cp_windward,
        gc_pi,
        min_pressure_psf,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            basic_wind_speed_mph,
            height_z_ft,
            mean_roof_height_ft,
            p_capacity_psf,
            kd_factor,
            kzt_factor,
            ke_factor,
            gust_factor_g,
            min_pressure_psf,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_velocity"}
    if height_z_ft > mean_roof_height_ft:
        return {"pass": False, "reason": "height_exceeds_roof_height"}

    exposure = exposure_category.strip().upper()
    # ASCE 7 Table 26.11-1 terrain exposure constants (alpha, zg)
    exposure_constants: Dict[str, Tuple[float, float]] = {
        "B": (7.0, 1200.0),
        "C": (9.5, 900.0),
        "D": (11.5, 700.0),
    }
    if exposure not in exposure_constants:
        return {"pass": False, "reason": "invalid_exposure_category"}

    alpha, zg = exposure_constants[exposure]

    def calc_kz(z: float) -> float:
        z_eff = max(15.0, min(z, zg))
        return 2.01 * ((z_eff / zg) ** (2.0 / alpha))

    kz = calc_kz(height_z_ft)
    kh = calc_kz(mean_roof_height_ft)

    # ASCE 7 Eq. 26.10-1 velocity pressure qz (psf)
    coeff = 0.00256 * kzt_factor * kd_factor * ke_factor * (basic_wind_speed_mph**2)
    qz = coeff * kz
    qh = coeff * kh

    # ASCE 7 Eq. 27.3-1 MWFRS net design wind pressure
    p_external = qz * gust_factor_g * cp_windward
    p_comb1 = p_external + (qh * abs(gc_pi))
    p_comb2 = p_external - (qh * abs(gc_pi))
    p_net_calc = max(abs(p_comb1), abs(p_comb2))

    # ASCE 7 Section 27.1.5 minimum design wind pressure threshold
    p_governing = max(p_net_calc, min_pressure_psf)
    dcr = p_governing / p_capacity_psf
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "p_design_psf": round(p_governing, 2),
        "p_net_psf": round(p_net_calc, 2),
        "qz_psf": round(qz, 2),
        "qh_psf": round(qh, 2),
        "kz": round(kz, 4),
        "kh": round(kh, 4),
    }


def verify_nec_solar_busbar_interconnection(
    busbar_rating_a: float,
    main_breaker_a: float,
    inverter_rated_ac_current_a: float,
    pv_breaker_a: float = 0.0,
    opposite_end: bool = True,
    allowance_pct: float = 120.0,
) -> Dict[str, Any]:
    """NEC 705.12(B) & NEC 690.8 solar PV backfed breaker and busbar interconnection gate."""
    inputs = [
        busbar_rating_a,
        main_breaker_a,
        inverter_rated_ac_current_a,
        pv_breaker_a,
        allowance_pct,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(busbar_rating_a, main_breaker_a, inverter_rated_ac_current_a, allowance_pct) <= 0:
        return {"pass": False, "reason": "non_positive_rating_or_current"}
    if pv_breaker_a < 0:
        return {"pass": False, "reason": "negative_breaker_input"}

    min_required_pv_ocpd = 1.25 * inverter_rated_ac_current_a
    if pv_breaker_a > 0:
        if pv_breaker_a < min_required_pv_ocpd:
            return {
                "pass": False,
                "reason": "inverter_ocpd_undersized",
                "pv_breaker_a": round(pv_breaker_a, 2),
                "min_required_pv_breaker_a": round(min_required_pv_ocpd, 2),
            }
        pv_breaker_effective = pv_breaker_a
    else:
        pv_breaker_effective = min_required_pv_ocpd

    factor = (allowance_pct / 100.0) if opposite_end else 1.0
    max_allowed_supply_a = busbar_rating_a * factor
    total_supply_a = main_breaker_a + pv_breaker_effective
    max_pv_breaker_a = max_allowed_supply_a - main_breaker_a

    dcr = total_supply_a / max_allowed_supply_a if max_allowed_supply_a > 0 else float("inf")
    passed = (dcr <= 1.0) and (max_pv_breaker_a >= pv_breaker_effective)

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "total_supply_a": round(total_supply_a, 2),
        "max_allowed_supply_a": round(max_allowed_supply_a, 2),
        "pv_breaker_a": round(pv_breaker_effective, 2),
        "max_pv_breaker_a": round(max_pv_breaker_a, 2),
        "min_required_pv_breaker_a": round(min_required_pv_ocpd, 2),
        "opposite_end": opposite_end,
    }


def verify_aisc_fillet_weld(
    pu_kips: float,
    leg_size_in: float,
    length_in: float,
    f_exx_ksi: float = 70.0,
    base_metal_fu_ksi: float = 65.0,
    base_metal_t_in: float = 0.50,
    load_angle_deg: float = 0.0,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """AISC 360-16 Section J2.4 & J4 LRFD fillet weld shear capacity and base metal rupture check."""
    inputs = [
        pu_kips,
        leg_size_in,
        length_in,
        f_exx_ksi,
        base_metal_fu_ksi,
        base_metal_t_in,
        load_angle_deg,
        phi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(leg_size_in, length_in, f_exx_ksi, base_metal_fu_ksi, base_metal_t_in, phi)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if not (0.0 <= load_angle_deg <= 90.0):
        return {"pass": False, "reason": "invalid_load_angle"}

    # AISC 360-16 Section J2.2b end-loaded long fillet weld reduction factor beta
    length_to_leg = length_in / leg_size_in
    if length_to_leg <= 100.0:
        beta = 1.0
    elif length_to_leg <= 300.0:
        beta = 1.2 - 0.002 * length_to_leg
    else:
        beta = 0.60

    # AISC 360-16 Section J2.4 Eq. J2-5 directional strength enhancement factor
    theta_rad = math.radians(load_angle_deg)
    k_ds = 1.0 + 0.50 * (math.sin(theta_rad) ** 1.5)

    # Effective throat thickness for standard 90-degree fillet weld
    throat_in = 0.707107 * leg_size_in
    a_we_in2 = throat_in * length_in

    # Nominal and design weld shear strength
    f_nw_ksi = 0.60 * f_exx_ksi * k_ds
    rn_weld_kips = beta * f_nw_ksi * a_we_in2
    phi_rn_weld_kips = phi * rn_weld_kips

    # AISC 360-16 Section J4.2 base metal shear rupture strength
    rn_base_kips = 0.60 * base_metal_fu_ksi * base_metal_t_in * length_in
    phi_rn_base_kips = phi * rn_base_kips

    phi_rn_gov_kips = min(phi_rn_weld_kips, phi_rn_base_kips)
    limit_state = "weld_shear" if phi_rn_weld_kips <= phi_rn_base_kips else "base_metal_shear"
    dcr = pu_kips / phi_rn_gov_kips if phi_rn_gov_kips > 0 else float("inf")

    passed = dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_rn_kips": round(phi_rn_gov_kips, 2),
        "phi_rn_weld_kips": round(phi_rn_weld_kips, 2),
        "phi_rn_base_kips": round(phi_rn_base_kips, 2),
        "beta": round(beta, 4),
        "k_ds": round(k_ds, 4),
        "governing_limit_state": limit_state,
    }


def verify_nec_conduit_fill(
    conduit_inner_diameter_in: float,
    conductor_diameters_in: Sequence[float],
    conduit_length_in: float = 120.0,
    max_fill_override_pct: float = 0.0,
) -> Dict[str, Any]:
    """NEC Chapter 9 Table 1 and Note 4 raceway/conduit conductor fill capacity check."""
    scalars = [conduit_inner_diameter_in, conduit_length_in, max_fill_override_pct]
    if any(math.isnan(val) for val in scalars) or any(math.isnan(d) for d in conductor_diameters_in):
        return {"pass": False, "reason": "nan_input"}
    if conduit_inner_diameter_in <= 0:
        return {"pass": False, "reason": "non_positive_conduit_diameter"}
    if conduit_length_in <= 0:
        return {"pass": False, "reason": "non_positive_conduit_length"}
    if not conductor_diameters_in:
        return {"pass": False, "reason": "empty_conductor_list"}
    if any(d <= 0 for d in conductor_diameters_in):
        return {"pass": False, "reason": "non_positive_conductor_diameter"}
    if any(d > conduit_inner_diameter_in for d in conductor_diameters_in):
        return {"pass": False, "reason": "conductor_exceeds_conduit_diameter"}
    if not (0.0 <= max_fill_override_pct <= 100.0):
        return {"pass": False, "reason": "invalid_fill_override"}

    conduit_area_sqin = (math.pi * (conduit_inner_diameter_in ** 2)) / 4.0
    conductor_areas = [(math.pi * (d ** 2)) / 4.0 for d in conductor_diameters_in]
    total_conductor_area_sqin = sum(conductor_areas)
    actual_fill_pct = (total_conductor_area_sqin / conduit_area_sqin) * 100.0

    num_conductors = len(conductor_diameters_in)
    if max_fill_override_pct > 0.0:
        allowable_fill_pct = max_fill_override_pct
    elif conduit_length_in <= 24.0:
        # NEC Chapter 9 Note 4 nipple exception (length <= 24 in / 600 mm)
        allowable_fill_pct = 60.0
    elif num_conductors == 1:
        allowable_fill_pct = 53.0
    elif num_conductors == 2:
        allowable_fill_pct = 31.0
    else:
        allowable_fill_pct = 40.0

    dcr = actual_fill_pct / allowable_fill_pct if allowable_fill_pct > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "actual_fill_pct": round(actual_fill_pct, 2),
        "allowable_fill_pct": round(allowable_fill_pct, 2),
        "total_conductor_area_sqin": round(total_conductor_area_sqin, 4),
        "conduit_area_sqin": round(conduit_area_sqin, 4),
        "conductor_count": num_conductors,
    }


def verify_aisc_tension_member(
    pu_kips: float,
    ag_in2: float,
    an_in2: float,
    u_shear_lag: float,
    length_in: float,
    rx_in: float,
    ry_in: float,
    fy_ksi: float = 50.0,
    fu_ksi: float = 65.0,
    max_slenderness: float = 300.0,
) -> Dict[str, Any]:
    """AISC 360-16 Chapter D LRFD check for tension members (gross yielding, net rupture, and slenderness)."""
    inputs = [
        pu_kips,
        ag_in2,
        an_in2,
        u_shear_lag,
        length_in,
        rx_in,
        ry_in,
        fy_ksi,
        fu_ksi,
        max_slenderness,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(ag_in2, an_in2, length_in, rx_in, ry_in, fy_ksi, fu_ksi, max_slenderness) <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if an_in2 > ag_in2:
        return {"pass": False, "reason": "net_area_exceeds_gross_area"}
    if not (0.0 < u_shear_lag <= 1.0):
        return {"pass": False, "reason": "invalid_shear_lag_factor"}

    # AISC 360-16 Section D1 slenderness limit recommendation (L/r <= 300)
    r_min = min(rx_in, ry_in)
    slenderness = length_in / r_min

    # AISC 360-16 Section D2(a) Tensile yielding in the gross section (phi = 0.90)
    phi_t_yield = 0.90
    phi_pny_kips = phi_t_yield * fy_ksi * ag_in2

    # AISC 360-16 Section D2(b) Tensile rupture in the net section (phi = 0.75, Ae = U * An)
    ae_in2 = u_shear_lag * an_in2
    phi_t_rupture = 0.75
    phi_pnr_kips = phi_t_rupture * fu_ksi * ae_in2

    phi_pn_kips = min(phi_pny_kips, phi_pnr_kips)
    limit_state = "gross_yielding" if phi_pny_kips <= phi_pnr_kips else "net_rupture"
    dcr = pu_kips / phi_pn_kips if phi_pn_kips > 0 else float("inf")

    passed = (dcr <= 1.0) and (slenderness <= max_slenderness)
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_pn_kips": round(phi_pn_kips, 2),
        "phi_pny_kips": round(phi_pny_kips, 2),
        "phi_pnr_kips": round(phi_pnr_kips, 2),
        "governing_limit_state": limit_state,
        "effective_net_area_in2": round(ae_in2, 3),
        "slenderness_ratio": round(slenderness, 2),
    }


def verify_clay_consolidation_settlement(
    layer_thickness_ft: float,
    initial_void_ratio: float,
    compression_index_cc: float,
    recompression_index_cr: float,
    initial_effective_stress_psf: float,
    preconsolidation_stress_psf: float,
    stress_increase_psf: float,
    allowable_settlement_in: float = 1.0,
) -> Dict[str, Any]:
    """Terzaghi 1D clay consolidation settlement and allowable deformation gate."""
    inputs = [
        layer_thickness_ft,
        initial_void_ratio,
        compression_index_cc,
        recompression_index_cr,
        initial_effective_stress_psf,
        preconsolidation_stress_psf,
        stress_increase_psf,
        allowable_settlement_in,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            layer_thickness_ft,
            initial_void_ratio,
            compression_index_cc,
            recompression_index_cr,
            initial_effective_stress_psf,
            preconsolidation_stress_psf,
            allowable_settlement_in,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_soil_parameter"}
    if stress_increase_psf < 0:
        return {"pass": False, "reason": "negative_stress_increase"}
    if recompression_index_cr > compression_index_cc:
        return {"pass": False, "reason": "recompression_index_exceeds_compression_index"}

    ocr = preconsolidation_stress_psf / initial_effective_stress_psf
    sigma_f = initial_effective_stress_psf + stress_increase_psf

    if stress_increase_psf == 0.0:
        s_c_ft = 0.0
        state = "normally_consolidated" if ocr <= 1.0 else "overconsolidated_elastic"
    elif ocr <= 1.0:
        state = "normally_consolidated"
        s_c_ft = (
            (compression_index_cc * layer_thickness_ft / (1.0 + initial_void_ratio))
            * math.log10(sigma_f / initial_effective_stress_psf)
        )
    elif sigma_f <= preconsolidation_stress_psf:
        state = "overconsolidated_elastic"
        s_c_ft = (
            (recompression_index_cr * layer_thickness_ft / (1.0 + initial_void_ratio))
            * math.log10(sigma_f / initial_effective_stress_psf)
        )
    else:
        state = "overconsolidated_virgin"
        s_cr = (
            (recompression_index_cr * layer_thickness_ft / (1.0 + initial_void_ratio))
            * math.log10(preconsolidation_stress_psf / initial_effective_stress_psf)
        )
        s_cc = (
            (compression_index_cc * layer_thickness_ft / (1.0 + initial_void_ratio))
            * math.log10(sigma_f / preconsolidation_stress_psf)
        )
        s_c_ft = s_cr + s_cc

    settlement_in = s_c_ft * 12.0
    dcr = settlement_in / allowable_settlement_in if allowable_settlement_in > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "settlement_in": round(settlement_in, 4),
        "ocr": round(ocr, 3),
        "consolidation_state": state,
        "sigma_final_psf": round(sigma_f, 2),
    }
