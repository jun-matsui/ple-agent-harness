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


def verify_nec_motor_branch_circuit(
    motor_fla_a: float,
    conductor_ampacity_a: float,
    ocpd_rating_a: float,
    ocpd_type: str = "inverse_time_breaker",
    overload_rating_a: float = 0.0,
    service_factor: float = 1.15,
    allow_next_standard_ocpd: bool = True,
) -> Dict[str, Any]:
    """NEC Article 430 motor branch circuit sizing check (conductor, OCPD, and overload protection)."""
    scalars = [motor_fla_a, conductor_ampacity_a, ocpd_rating_a, service_factor, overload_rating_a]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if min(motor_fla_a, conductor_ampacity_a, ocpd_rating_a) <= 0:
        return {"pass": False, "reason": "non_positive_current_or_ampacity"}
    if overload_rating_a < 0:
        return {"pass": False, "reason": "negative_overload_rating"}
    if service_factor < 1.0:
        return {"pass": False, "reason": "invalid_service_factor"}

    # NEC Table 430.52 maximum rating or setting multipliers
    ocpd_multipliers: Dict[str, float] = {
        "inverse_time_breaker": 2.50,
        "dual_element_fuse": 1.75,
        "non_time_delay_fuse": 3.00,
        "instantaneous_trip": 8.00,
    }
    type_key = ocpd_type.strip().lower()
    if type_key not in ocpd_multipliers:
        return {"pass": False, "reason": "invalid_ocpd_type"}

    # NEC 430.22 branch circuit conductor sizing (125% of FLA for continuous duty)
    min_conductor_ampacity_a = 1.25 * motor_fla_a
    conductor_dcr = min_conductor_ampacity_a / conductor_ampacity_a
    conductor_ok = conductor_ampacity_a >= min_conductor_ampacity_a

    # NEC 430.52 branch circuit short-circuit and ground-fault protection
    raw_max_ocpd_a = ocpd_multipliers[type_key] * motor_fla_a
    standard_ratings = [
        15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
        110, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
        500, 600, 700, 800, 1000, 1200,
    ]

    if allow_next_standard_ocpd:
        # NEC 430.52(C)(1) Exception 1 next higher standard rating (NEC 240.6(A))
        if raw_max_ocpd_a <= standard_ratings[0]:
            max_allowed_ocpd_a = float(standard_ratings[0])
        elif raw_max_ocpd_a >= standard_ratings[-1]:
            max_allowed_ocpd_a = raw_max_ocpd_a
        else:
            max_allowed_ocpd_a = float(
                next(r for r in standard_ratings if r >= (raw_max_ocpd_a - 1e-6))
            )
    else:
        max_allowed_ocpd_a = raw_max_ocpd_a

    ocpd_dcr = ocpd_rating_a / max_allowed_ocpd_a if max_allowed_ocpd_a > 0 else float("inf")
    ocpd_ok = ocpd_rating_a <= max_allowed_ocpd_a

    # NEC 430.32 separate motor overload protection sizing
    if overload_rating_a > 0.0:
        max_overload_factor = 1.25 if service_factor >= 1.15 else 1.15
        max_overload_a = max_overload_factor * motor_fla_a
        overload_dcr = overload_rating_a / max_overload_a if max_overload_a > 0 else float("inf")
        overload_ok = overload_rating_a <= max_overload_a
    else:
        max_overload_a = None
        overload_dcr = None
        overload_ok = True

    passed = conductor_ok and ocpd_ok and overload_ok
    return {
        "pass": passed,
        "conductor_dcr": round(conductor_dcr, 4),
        "ocpd_dcr": round(ocpd_dcr, 4),
        "min_conductor_ampacity_a": round(min_conductor_ampacity_a, 2),
        "max_allowed_ocpd_a": round(max_allowed_ocpd_a, 2),
        "max_overload_a": round(max_overload_a, 2) if max_overload_a is not None else None,
        "overload_dcr": round(overload_dcr, 4) if overload_dcr is not None else None,
        "conductor_ok": conductor_ok,
        "ocpd_ok": ocpd_ok,
        "overload_ok": overload_ok,
    }


def verify_stormwater_rational_runoff(
    drainage_areas_acres: Sequence[float],
    runoff_coefficients: Sequence[float],
    rainfall_intensity_in_hr: float,
    capacity_cfs: float,
    frequency_factor_cf: float = 1.0,
    time_of_concentration_min: float = 5.0,
    min_time_of_concentration_min: float = 5.0,
) -> Dict[str, Any]:
    """ASCE/WEF & DOT Rational Method peak stormwater runoff and drainage conveyance capacity check."""
    scalars = [
        rainfall_intensity_in_hr,
        capacity_cfs,
        frequency_factor_cf,
        time_of_concentration_min,
        min_time_of_concentration_min,
    ]
    if any(math.isnan(val) for val in scalars) or any(
        math.isnan(a) for a in drainage_areas_acres
    ) or any(math.isnan(c) for c in runoff_coefficients):
        return {"pass": False, "reason": "nan_input"}
    if not drainage_areas_acres or not runoff_coefficients:
        return {"pass": False, "reason": "empty_subcatchment_list"}
    if len(drainage_areas_acres) != len(runoff_coefficients):
        return {"pass": False, "reason": "length_mismatch"}
    if min(drainage_areas_acres) <= 0:
        return {"pass": False, "reason": "non_positive_area"}
    if rainfall_intensity_in_hr <= 0 or capacity_cfs <= 0:
        return {"pass": False, "reason": "non_positive_intensity_or_capacity"}
    if any(not (0.0 < c <= 1.0) for c in runoff_coefficients):
        return {"pass": False, "reason": "invalid_runoff_coefficient"}
    if frequency_factor_cf < 1.0:
        return {"pass": False, "reason": "invalid_frequency_factor"}
    if min_time_of_concentration_min <= 0 or time_of_concentration_min <= 0:
        return {"pass": False, "reason": "non_positive_tc"}
    if time_of_concentration_min < min_time_of_concentration_min:
        return {"pass": False, "reason": "tc_below_minimum"}

    total_area_acres = sum(drainage_areas_acres)
    composite_c = (
        sum(a * c for a, c in zip(drainage_areas_acres, runoff_coefficients))
        / total_area_acres
    )
    # Design runoff coefficient adjusted by storm frequency factor Cf (capped at 1.0)
    design_c = min(composite_c * frequency_factor_cf, 1.0)

    # Rational formula: Q = C * I * A (cfs = unitless * in/hr * acres)
    q_peak_cfs = design_c * rainfall_intensity_in_hr * total_area_acres
    dcr = q_peak_cfs / capacity_cfs
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "q_peak_cfs": round(q_peak_cfs, 3),
        "composite_c": round(composite_c, 4),
        "design_c": round(design_c, 4),
        "total_area_acres": round(total_area_acres, 3),
    }


def verify_aci_punching_shear(
    vu_kips: float,
    c1_in: float,
    c2_in: float,
    d_in: float,
    fc_psi: float,
    column_location: str = "interior",
    bo_in: float = 0.0,
    lambda_factor: float = 1.0,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """ACI 318-19 Section 22.6 two-way slab punching shear capacity and size effect check."""
    scalars = [vu_kips, c1_in, c2_in, d_in, fc_psi, bo_in, lambda_factor, phi]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if bo_in < 0:
        return {"pass": False, "reason": "negative_perimeter_input"}
    if min(c1_in, c2_in, d_in, fc_psi, lambda_factor, phi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    loc = column_location.strip().lower()
    alpha_map: Dict[str, float] = {"interior": 40.0, "edge": 30.0, "corner": 20.0}
    if loc not in alpha_map:
        return {"pass": False, "reason": "invalid_column_location"}
    alpha_s = alpha_map[loc]

    # Calculate critical perimeter b0 at d/2 if not explicitly overridden
    if bo_in > 0.0:
        b0 = bo_in
    else:
        if loc == "interior":
            b0 = 2.0 * (c1_in + d_in) + 2.0 * (c2_in + d_in)
        elif loc == "edge":
            b0 = 2.0 * (c1_in + d_in / 2.0) + (c2_in + d_in)
        else:  # corner
            b0 = (c1_in + d_in / 2.0) + (c2_in + d_in / 2.0)

    beta = max(c1_in, c2_in) / min(c1_in, c2_in)
    sqrt_fc = math.sqrt(fc_psi)

    # ACI 318-19 Section 22.5.5.1.3 size effect factor lambda_s
    lambda_s = min(1.0, math.sqrt(2.0 / (1.0 + 0.10 * d_in)))

    # ACI 318-19 Table 22.6.5.2 two-way shear stress equations
    vc1 = 4.0 * lambda_s * lambda_factor * sqrt_fc
    vc2 = (2.0 + 4.0 / beta) * lambda_s * lambda_factor * sqrt_fc
    vc3 = ((alpha_s * d_in / b0) + 2.0) * lambda_s * lambda_factor * sqrt_fc

    vc = min(vc1, vc2, vc3)
    if vc == vc1:
        governing_eq = "4_sqrt_fc"
    elif vc == vc2:
        governing_eq = "aspect_ratio_beta"
    else:
        governing_eq = "perimeter_alpha_s"

    vn_kips = (vc * b0 * d_in) / 1000.0
    phi_vn_kips = phi * vn_kips
    dcr = vu_kips / phi_vn_kips if phi_vn_kips > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_vn_kips": round(phi_vn_kips, 2),
        "vn_kips": round(vn_kips, 2),
        "vc_psi": round(vc, 2),
        "b0_in": round(b0, 2),
        "beta": round(beta, 3),
        "lambda_s": round(lambda_s, 4),
        "governing_equation": governing_eq,
    }


def verify_hvac_cooling_load_capacity(
    sensible_load_btuh: float,
    latent_load_btuh: float,
    room_temp_db_f: float,
    supply_temp_db_f: float,
    design_airflow_cfm: float,
    coil_capacity_tons: float,
    elevation_ft: float = 0.0,
    room_volume_cuft: float = 0.0,
    min_ach: float = 0.0,
) -> Dict[str, Any]:
    """ASHRAE Fundamentals & Standard 183 HVAC cooling load, supply airflow, and coil capacity gate."""
    inputs = [
        sensible_load_btuh,
        latent_load_btuh,
        room_temp_db_f,
        supply_temp_db_f,
        design_airflow_cfm,
        coil_capacity_tons,
        elevation_ft,
        room_volume_cuft,
        min_ach,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(sensible_load_btuh, design_airflow_cfm, coil_capacity_tons) <= 0:
        return {"pass": False, "reason": "non_positive_load_or_capacity"}
    if latent_load_btuh < 0:
        return {"pass": False, "reason": "negative_latent_load"}
    if room_temp_db_f <= supply_temp_db_f:
        return {"pass": False, "reason": "supply_temp_not_below_room_temp"}
    if elevation_ft < -1000.0 or elevation_ft > 30000.0:
        return {"pass": False, "reason": "invalid_elevation"}
    if room_volume_cuft < 0 or min_ach < 0:
        return {"pass": False, "reason": "negative_volume_or_ach"}

    # US Standard Atmosphere barometric pressure and air density correction ratio
    p_atm_psia = 14.696 * ((1.0 - 0.0000068756 * elevation_ft) ** 5.2559)
    density_ratio = p_atm_psia / 14.696
    c_s = 1.08 * density_ratio

    delta_t = room_temp_db_f - supply_temp_db_f
    required_cfm = sensible_load_btuh / (c_s * delta_t)
    airflow_dcr = required_cfm / design_airflow_cfm

    q_total_btuh = sensible_load_btuh + latent_load_btuh
    tons_required = q_total_btuh / 12000.0
    capacity_dcr = tons_required / coil_capacity_tons
    shr = sensible_load_btuh / q_total_btuh

    ach = None
    ach_ok = True
    if room_volume_cuft > 0.0:
        ach = (design_airflow_cfm * 60.0) / room_volume_cuft
        if min_ach > 0.0:
            ach_ok = ach >= min_ach

    passed = (airflow_dcr <= 1.0) and (capacity_dcr <= 1.0) and ach_ok
    return {
        "pass": passed,
        "airflow_dcr": round(airflow_dcr, 4),
        "capacity_dcr": round(capacity_dcr, 4),
        "q_total_btuh": round(q_total_btuh, 2),
        "tons_required": round(tons_required, 2),
        "required_cfm": round(required_cfm, 2),
        "shr": round(shr, 4),
        "density_ratio": round(density_ratio, 4),
        "ach": round(ach, 2) if ach is not None else None,
    }


def verify_aisc_column_base_plate(
    pu_kips: float,
    b_in: float,
    n_in: float,
    tp_in: float,
    col_d_in: float,
    col_bf_in: float,
    fc_psi: float,
    fy_ksi: float = 36.0,
    pedestal_area_sqin: float = 0.0,
    phi_bearing: float = 0.65,
    phi_flexure: float = 0.90,
) -> Dict[str, Any]:
    """AISC 360-16 Section J8 & AISC Design Guide 1 column base plate bearing and thickness check."""
    inputs = [
        pu_kips,
        b_in,
        n_in,
        tp_in,
        col_d_in,
        col_bf_in,
        fc_psi,
        fy_ksi,
        pedestal_area_sqin,
        phi_bearing,
        phi_flexure,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(b_in, n_in, tp_in, col_d_in, col_bf_in, fc_psi, fy_ksi, phi_bearing, phi_flexure)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if col_d_in > n_in or col_bf_in > b_in:
        return {"pass": False, "reason": "column_exceeds_base_plate"}

    a1_sqin = b_in * n_in
    if pedestal_area_sqin < 0:
        return {"pass": False, "reason": "negative_pedestal_area"}
    if pedestal_area_sqin > 0 and pedestal_area_sqin < a1_sqin:
        return {"pass": False, "reason": "pedestal_area_less_than_plate_area"}

    # AISC 360-16 Section J8 concrete bearing capacity
    a2_sqin = pedestal_area_sqin if pedestal_area_sqin > 0 else a1_sqin
    area_confinement_factor = min(math.sqrt(a2_sqin / a1_sqin), 2.0)
    fc_ksi = fc_psi / 1000.0
    pp_kips = 0.85 * fc_ksi * a1_sqin * area_confinement_factor
    phi_pp_kips = phi_bearing * pp_kips
    bearing_dcr = pu_kips / phi_pp_kips if phi_pp_kips > 0 else float("inf")

    # AISC Design Guide 1 Thornton cantilever dimensions m, n, and n'
    m_in = (n_in - 0.95 * col_d_in) / 2.0
    n_cant_in = (b_in - 0.80 * col_bf_in) / 2.0
    n_prime_in = math.sqrt(col_d_in * col_bf_in) / 4.0
    l_crit_in = max(m_in, n_cant_in, n_prime_in)

    # Required base plate thickness (AISC DG 1 Eq. 3.3.13)
    fp_ksi = pu_kips / a1_sqin
    t_min_in = l_crit_in * math.sqrt((2.0 * fp_ksi) / (phi_flexure * fy_ksi))
    thickness_dcr = t_min_in / tp_in
    gov_dcr = max(bearing_dcr, thickness_dcr)

    passed = (bearing_dcr <= 1.0) and (thickness_dcr <= 1.0)
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "bearing_dcr": round(bearing_dcr, 4),
        "thickness_dcr": round(thickness_dcr, 4),
        "t_min_in": round(t_min_in, 3),
        "phi_pp_kips": round(phi_pp_kips, 2),
        "critical_cantilever_in": round(l_crit_in, 3),
        "bearing_pressure_ksi": round(fp_ksi, 4),
    }


def verify_aci_rebar_development_length(
    provided_length_in: float,
    bar_size_num: int = 8,
    bar_diameter_in: float = 0.0,
    fy_psi: float = 60000.0,
    fc_psi: float = 4000.0,
    cb_in: float = 0.0,
    ktr_in: float = 0.0,
    confinement_override: float = 0.0,
    casting_position: str = "other",
    coating_type: str = "uncoated",
    is_lightweight: bool = False,
    lambda_factor: float = 1.0,
) -> Dict[str, Any]:
    """ACI 318-19 Section 25.4.2 tension development length (ld) check for deformed rebar."""
    scalars = [
        provided_length_in,
        bar_diameter_in,
        fy_psi,
        fc_psi,
        cb_in,
        ktr_in,
        confinement_override,
        lambda_factor,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if provided_length_in <= 0:
        return {"pass": False, "reason": "non_positive_provided_length"}
    if fy_psi <= 0 or fc_psi <= 0 or lambda_factor <= 0:
        return {"pass": False, "reason": "non_positive_material_properties"}
    if cb_in < 0 or ktr_in < 0 or confinement_override < 0:
        return {"pass": False, "reason": "negative_confinement_input"}

    bar_diameters: Dict[int, float] = {
        3: 0.375,
        4: 0.500,
        5: 0.625,
        6: 0.750,
        7: 0.875,
        8: 1.000,
        9: 1.128,
        10: 1.270,
        11: 1.410,
        14: 1.693,
        18: 2.257,
    }
    if bar_diameter_in > 0.0:
        db = bar_diameter_in
    elif bar_size_num in bar_diameters:
        db = bar_diameters[bar_size_num]
    else:
        return {"pass": False, "reason": "invalid_bar_size"}

    # Casting position factor psi_t (ACI 318-19 Table 25.4.2.4)
    pos = casting_position.strip().lower()
    if pos in ("top",):
        psi_t = 1.3
    elif pos in ("other", "bottom"):
        psi_t = 1.0
    else:
        return {"pass": False, "reason": "invalid_casting_position"}

    # Coating factor psi_e (ACI 318-19 Table 25.4.2.4)
    coat = coating_type.strip().lower()
    if coat in ("uncoated", "galvanized", "black"):
        psi_e = 1.0
    elif coat in ("epoxy_low_cover", "epoxy_low"):
        psi_e = 1.5
    elif coat in ("epoxy_high_cover", "epoxy_high", "epoxy"):
        psi_e = 1.2
    else:
        return {"pass": False, "reason": "invalid_coating_type"}

    # ACI 318-19 Section 25.4.2.4: product of psi_t and psi_e need not exceed 1.7
    psi_t_e = min(psi_t * psi_e, 1.7)

    # Reinforcement size factor psi_s (ACI 318-19 Table 25.4.2.4)
    psi_s = 0.8 if db <= 0.75 else 1.0

    # Reinforcement grade factor psi_g (ACI 318-19 Table 25.4.2.4)
    if fy_psi <= 60000.0:
        psi_g = 1.0
    elif fy_psi <= 80000.0:
        psi_g = 1.15
    else:
        psi_g = 1.30

    # Lightweight concrete modification factor lambda
    lambda_val = (lambda_factor if lambda_factor < 1.0 else 0.75) if is_lightweight else lambda_factor

    # Concrete compressive strength square-root limit (ACI 318-19 Section 25.4.1.4: sqrt(f'c) <= 100 psi)
    sqrt_fc = min(math.sqrt(fc_psi), 100.0)

    # Confinement term (cb + Ktr) / db capped at 2.5 (ACI 318-19 Section 25.4.2.3)
    if confinement_override > 0.0:
        confinement_term = min(confinement_override, 2.5)
    elif cb_in > 0.0:
        confinement_term = min((cb_in + ktr_in) / db, 2.5)
    else:
        confinement_term = 2.5

    # ACI 318-19 Eq. 25.4.2.3a tension development length
    ld_calc = (
        (3.0 / 40.0)
        * (fy_psi / (lambda_val * sqrt_fc))
        * (psi_t_e * psi_s * psi_g / confinement_term)
    ) * db

    # ACI 318-19 Section 25.4.2.1 minimum development length of 12.0 inches
    ld_required = max(ld_calc, 12.0)
    dcr = ld_required / provided_length_in
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "ld_required_in": round(ld_required, 2),
        "ld_calc_in": round(ld_calc, 2),
        "bar_diameter_in": round(db, 3),
        "confinement_term": round(confinement_term, 3),
        "psi_t_e": round(psi_t_e, 3),
        "psi_s": round(psi_s, 2),
        "psi_g": round(psi_g, 2),
    }


def verify_asce7_roof_snow_load(
    pg_psf: float,
    roof_slope_deg: float = 0.0,
    p_capacity_psf: float = 30.0,
    ce_exposure_factor: float = 1.0,
    ct_thermal_factor: float = 1.0,
    importance_factor: float = 1.0,
    is_slippery_surface: bool = False,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Chapter 7 design roof snow load, slope factor Cs, and minimum load gate."""
    scalars = [
        pg_psf,
        roof_slope_deg,
        p_capacity_psf,
        ce_exposure_factor,
        ct_thermal_factor,
        importance_factor,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if min(pg_psf, p_capacity_psf, ce_exposure_factor, ct_thermal_factor, importance_factor) <= 0:
        return {"pass": False, "reason": "non_positive_loading_or_capacity"}
    if not (0.0 <= roof_slope_deg <= 90.0):
        return {"pass": False, "reason": "invalid_roof_slope"}

    # ASCE 7-16 Eq. 7.3-1 Flat roof snow load pf
    pf_psf = 0.70 * ce_exposure_factor * ct_thermal_factor * importance_factor * pg_psf

    # ASCE 7-16 Section 7.4 Roof slope factor Cs
    # Determine threshold angle theta_1 based on thermal factor Ct and surface lubricity
    if ct_thermal_factor <= 1.0:
        theta_1 = 5.0 if is_slippery_surface else 30.0
    elif ct_thermal_factor >= 1.2:
        theta_1 = 15.0 if is_slippery_surface else 45.0
    else:
        # Linear interpolation for 1.0 < Ct < 1.2
        frac = (ct_thermal_factor - 1.0) / 0.2
        theta_1_slip = 5.0 + 10.0 * frac
        theta_1_other = 30.0 + 15.0 * frac
        theta_1 = theta_1_slip if is_slippery_surface else theta_1_other

    theta_2 = 70.0
    if roof_slope_deg <= theta_1:
        cs = 1.0
    elif roof_slope_deg >= theta_2:
        cs = 0.0
    else:
        cs = 1.0 - ((roof_slope_deg - theta_1) / (theta_2 - theta_1))

    # Sloped roof snow load ps
    ps_psf = cs * pf_psf

    # ASCE 7-16 Section 7.3.4 Minimum roof snow load pm for low-slope roofs (<= 15 deg)
    is_low_slope = roof_slope_deg <= 15.0
    if is_low_slope:
        pm_psf = (importance_factor * pg_psf) if pg_psf <= 20.0 else (20.0 * importance_factor)
        p_design_psf = max(ps_psf, pm_psf)
    else:
        pm_psf = 0.0
        p_design_psf = ps_psf

    dcr = p_design_psf / p_capacity_psf if p_capacity_psf > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "p_design_psf": round(p_design_psf, 2),
        "pf_psf": round(pf_psf, 2),
        "ps_psf": round(ps_psf, 2),
        "pm_psf": round(pm_psf, 2),
        "cs": round(cs, 4),
        "is_low_slope": is_low_slope,
    }


def verify_aisc_beam_column_interaction(
    pu_kips: float,
    mux_kip_ft: float,
    muy_kip_ft: float,
    phi_pn_kips: float,
    phi_mnx_kip_ft: float,
    phi_mny_kip_ft: float,
    cm_x: float = 1.0,
    cm_y: float = 1.0,
    pe1_x_kips: float = 0.0,
    pe1_y_kips: float = 0.0,
    b1_x_override: float = 0.0,
    b1_y_override: float = 0.0,
) -> Dict[str, Any]:
    """AISC 360-16 Chapter H & Appendix 8 combined axial compression and biaxial flexure interaction check."""
    inputs = [
        pu_kips,
        mux_kip_ft,
        muy_kip_ft,
        phi_pn_kips,
        phi_mnx_kip_ft,
        phi_mny_kip_ft,
        cm_x,
        cm_y,
        pe1_x_kips,
        pe1_y_kips,
        b1_x_override,
        b1_y_override,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(pu_kips, mux_kip_ft, muy_kip_ft) < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(phi_pn_kips, phi_mnx_kip_ft, phi_mny_kip_ft) <= 0:
        return {"pass": False, "reason": "non_positive_capacity"}
    if not (0.0 < cm_x <= 1.0) or not (0.0 < cm_y <= 1.0):
        return {"pass": False, "reason": "invalid_cm_factor"}
    if pe1_x_kips < 0 or pe1_y_kips < 0:
        return {"pass": False, "reason": "negative_euler_buckling"}
    if (b1_x_override > 0 and b1_x_override < 1.0) or (b1_y_override > 0 and b1_y_override < 1.0):
        return {"pass": False, "reason": "invalid_b1_override"}

    # AISC 360-16 Appendix 8 Section 8.2.1 B1 moment magnification for x-axis
    if b1_x_override > 0.0:
        b1_x = b1_x_override
    elif pe1_x_kips > 0.0:
        if pu_kips >= pe1_x_kips:
            return {"pass": False, "reason": "axial_load_exceeds_euler_buckling_x"}
        b1_x = max(1.0, cm_x / (1.0 - (pu_kips / pe1_x_kips)))
    else:
        b1_x = 1.0

    # AISC 360-16 Appendix 8 Section 8.2.1 B1 moment magnification for y-axis
    if b1_y_override > 0.0:
        b1_y = b1_y_override
    elif pe1_y_kips > 0.0:
        if pu_kips >= pe1_y_kips:
            return {"pass": False, "reason": "axial_load_exceeds_euler_buckling_y"}
        b1_y = max(1.0, cm_y / (1.0 - (pu_kips / pe1_y_kips)))
    else:
        b1_y = 1.0

    m_rx = b1_x * mux_kip_ft
    m_ry = b1_y * muy_kip_ft

    p_ratio = pu_kips / phi_pn_kips
    flex_ratio = (m_rx / phi_mnx_kip_ft) + (m_ry / phi_mny_kip_ft)

    # AISC 360-16 Equations H1-1a and H1-1b
    if p_ratio >= 0.20:
        governing_eq = "H1-1a"
        dcr = p_ratio + (8.0 / 9.0) * flex_ratio
    else:
        governing_eq = "H1-1b"
        dcr = (p_ratio / 2.0) + flex_ratio

    passed = dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "governing_equation": governing_eq,
        "axial_ratio": round(p_ratio, 4),
        "flexure_ratio": round(flex_ratio, 4),
        "b1_x": round(b1_x, 4),
        "b1_y": round(b1_y, 4),
        "m_rx_kip_ft": round(m_rx, 2),
        "m_ry_kip_ft": round(m_ry, 2),
    }


def verify_aci_column_axial_load(
    pu_kips: float,
    ag_in2: float,
    ast_in2: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    transverse_type: str = "tied",
    core_diameter_in: float = 0.0,
    spiral_pitch_in: float = 0.0,
    spiral_bar_dia_in: float = 0.0,
    fyt_psi: float = 60000.0,
) -> Dict[str, Any]:
    """ACI 318-19 Section 10.6, 22.4.2, & Table 21.2.2 short column axial compressive strength check."""
    scalars = [
        pu_kips,
        ag_in2,
        ast_in2,
        fc_psi,
        fy_psi,
        core_diameter_in,
        spiral_pitch_in,
        spiral_bar_dia_in,
        fyt_psi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(ag_in2, ast_in2, fc_psi, fy_psi, fyt_psi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if ast_in2 >= ag_in2:
        return {"pass": False, "reason": "steel_area_exceeds_gross_area"}

    t_type = transverse_type.strip().lower()
    if t_type not in ("tied", "spiral"):
        return {"pass": False, "reason": "invalid_transverse_type"}

    # ACI 318-19 Section 10.6.1.1 longitudinal steel ratio limits (0.01 <= rho_g <= 0.08)
    rho_g = ast_in2 / ag_in2
    if not (0.01 <= rho_g <= 0.08):
        return {
            "pass": False,
            "reason": "reinforcement_ratio_out_of_bounds",
            "rho_g": round(rho_g, 4),
        }

    # ACI 318-19 Eq. 22.4.2.2 nominal axial strength at zero eccentricity
    fc_ksi = fc_psi / 1000.0
    fy_ksi = fy_psi / 1000.0
    p0_kips = 0.85 * fc_ksi * (ag_in2 - ast_in2) + fy_ksi * ast_in2

    # ACI 318-19 Eq. 22.4.2.1 and Table 21.2.2 reduction factors
    if t_type == "tied":
        alpha = 0.80
        phi = 0.65
    else:
        alpha = 0.85
        phi = 0.75

    pn_max_kips = alpha * p0_kips
    phi_pn_max_kips = phi * pn_max_kips
    dcr = pu_kips / phi_pn_max_kips if phi_pn_max_kips > 0 else float("inf")

    # Spiral confinement check per ACI 318-19 Table 10.7.6.5.2 & Section 25.7.3.1
    spiral_ok = True
    rho_s_provided = None
    rho_s_min = None
    if t_type == "spiral" and core_diameter_in > 0.0:
        ach_in2 = (math.pi * (core_diameter_in**2)) / 4.0
        if ach_in2 >= ag_in2:
            return {"pass": False, "reason": "core_area_exceeds_gross_area"}
        rho_s_min = 0.45 * ((ag_in2 / ach_in2) - 1.0) * (fc_psi / fyt_psi)

        if spiral_pitch_in > 0.0 and spiral_bar_dia_in > 0.0:
            a_sp_in2 = (math.pi * (spiral_bar_dia_in**2)) / 4.0
            rho_s_provided = (4.0 * a_sp_in2) / (spiral_pitch_in * core_diameter_in)
            clear_spacing_in = spiral_pitch_in - spiral_bar_dia_in
            spacing_ok = 1.0 <= clear_spacing_in <= 3.0
            spiral_ok = (rho_s_provided >= rho_s_min) and spacing_ok
        else:
            spiral_ok = False

    passed = (dcr <= 1.0) and spiral_ok
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_pn_max_kips": round(phi_pn_max_kips, 2),
        "p0_kips": round(p0_kips, 2),
        "rho_g": round(rho_g, 4),
        "transverse_type": t_type,
        "phi": phi,
        "rho_s_provided": round(rho_s_provided, 5) if rho_s_provided is not None else None,
        "rho_s_min": round(rho_s_min, 5) if rho_s_min is not None else None,
    }


def verify_darcy_weisbach_pipe_flow(
    flow_rate_gpm: float,
    diameter_in: float,
    pipe_length_ft: float,
    roughness_in: float = 0.0018,
    fluid_density_pcf: float = 62.4,
    kinematic_viscosity_ft2_s: float = 1.08e-5,
    minor_loss_k: float = 0.0,
    max_velocity_fps: float = 10.0,
    max_pressure_drop_psi: float = 15.0,
) -> Dict[str, Any]:
    """Darcy-Weisbach fluid pipe friction head loss, velocity, and pressure drop check via Swamee-Jain."""
    inputs = [
        flow_rate_gpm,
        diameter_in,
        pipe_length_ft,
        roughness_in,
        fluid_density_pcf,
        kinematic_viscosity_ft2_s,
        minor_loss_k,
        max_velocity_fps,
        max_pressure_drop_psi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if flow_rate_gpm < 0:
        return {"pass": False, "reason": "negative_flow_rate"}
    if (
        min(
            diameter_in,
            pipe_length_ft,
            roughness_in,
            fluid_density_pcf,
            kinematic_viscosity_ft2_s,
            max_velocity_fps,
            max_pressure_drop_psi,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_fluid_property"}
    if minor_loss_k < 0:
        return {"pass": False, "reason": "negative_minor_loss"}

    d_ft = diameter_in / 12.0
    area_ft2 = (math.pi * (d_ft**2)) / 4.0
    q_cfs = flow_rate_gpm / 448.831169
    velocity_fps = q_cfs / area_ft2 if area_ft2 > 0 else 0.0
    g = 32.174

    if velocity_fps == 0.0:
        reynolds = 0.0
        f = 0.0
        regime = "quiescent"
    else:
        reynolds = (velocity_fps * d_ft) / kinematic_viscosity_ft2_s
        if reynolds <= 2000.0:
            regime = "laminar"
            f = 64.0 / reynolds
        else:
            regime = "transitional" if reynolds < 4000.0 else "turbulent"
            rel_roughness = roughness_in / diameter_in
            term = (rel_roughness / 3.7) + (5.74 / (reynolds**0.9))
            f = 0.25 / (math.log10(term) ** 2)

    # Darcy-Weisbach major friction head loss and minor loss
    v_head_ft = (velocity_fps**2) / (2.0 * g)
    h_f_ft = (f * (pipe_length_ft / d_ft) * v_head_ft) if d_ft > 0 else 0.0
    h_m_ft = minor_loss_k * v_head_ft
    total_head_loss_ft = h_f_ft + h_m_ft

    # Pressure drop delta P = (rho * h_L) / 144 psi
    pressure_drop_psi = (fluid_density_pcf * total_head_loss_ft) / 144.0

    vel_dcr = velocity_fps / max_velocity_fps
    press_dcr = pressure_drop_psi / max_pressure_drop_psi
    gov_dcr = max(vel_dcr, press_dcr)

    passed = (vel_dcr <= 1.0) and (press_dcr <= 1.0)
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "velocity_fps": round(velocity_fps, 3),
        "reynolds_number": round(reynolds, 1),
        "friction_factor": round(f, 5),
        "flow_regime": regime,
        "head_loss_friction_ft": round(h_f_ft, 3),
        "head_loss_minor_ft": round(h_m_ft, 3),
        "total_head_loss_ft": round(total_head_loss_ft, 3),
        "pressure_drop_psi": round(pressure_drop_psi, 4),
    }


def verify_asce7_story_drift_and_pdelta(
    story_height_in: float,
    elastic_drift_in: float,
    cd_factor: float,
    importance_factor_ie: float = 1.0,
    risk_category: str = "II",
    allowable_drift_ratio_override: float = 0.0,
    total_vertical_load_px_kips: float = 0.0,
    story_shear_vx_kips: float = 0.0,
    shear_demand_capacity_ratio_beta: float = 1.0,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 12.8.6, 12.8.7, & 12.12.1 seismic story drift and P-delta stability check."""
    inputs = [
        story_height_in,
        elastic_drift_in,
        cd_factor,
        importance_factor_ie,
        allowable_drift_ratio_override,
        total_vertical_load_px_kips,
        story_shear_vx_kips,
        shear_demand_capacity_ratio_beta,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if story_height_in <= 0 or cd_factor <= 0 or importance_factor_ie <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_factor"}
    if elastic_drift_in < 0:
        return {"pass": False, "reason": "negative_drift_input"}
    if total_vertical_load_px_kips < 0 or story_shear_vx_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if shear_demand_capacity_ratio_beta <= 0:
        return {"pass": False, "reason": "non_positive_beta_ratio"}

    # ASCE 7 Table 12.12-1 allowable story drift ratios
    cat = risk_category.strip().upper()
    category_limits: Dict[str, float] = {"I": 0.020, "II": 0.020, "III": 0.015, "IV": 0.010}
    if allowable_drift_ratio_override > 0.0:
        allowable_ratio = allowable_drift_ratio_override
    elif cat in category_limits:
        allowable_ratio = category_limits[cat]
    else:
        return {"pass": False, "reason": "invalid_risk_category"}

    delta_allowable_in = allowable_ratio * story_height_in

    # ASCE 7 Eq. 12.8-15 design story drift Delta = Cd * delta_xe / Ie
    delta_unmagnified_in = (cd_factor * elastic_drift_in) / importance_factor_ie

    # ASCE 7 Section 12.8.7 P-Delta stability coefficient theta = (Px * delta_xe) / (Vx * hsx)
    if total_vertical_load_px_kips > 0.0 and story_shear_vx_kips > 0.0:
        theta = (total_vertical_load_px_kips * elastic_drift_in) / (
            story_shear_vx_kips * story_height_in
        )
        theta_max = min(0.50 / (shear_demand_capacity_ratio_beta * cd_factor), 0.25)
        stability_ok = theta <= theta_max
        pdelta_required = theta > 0.10
        if theta >= 1.0:
            a_d = float("inf")
        elif pdelta_required:
            a_d = 1.0 / (1.0 - theta)
        else:
            a_d = 1.0
    else:
        theta = None
        theta_max = None
        stability_ok = True
        pdelta_required = False
        a_d = 1.0

    delta_design_in = a_d * delta_unmagnified_in if not math.isinf(a_d) else float("inf")
    drift_dcr = delta_design_in / delta_allowable_in if delta_allowable_in > 0 else float("inf")

    passed = (drift_dcr <= 1.0) and stability_ok
    return {
        "pass": passed,
        "dcr": round(drift_dcr, 4),
        "design_drift_in": round(delta_design_in, 4),
        "unmagnified_drift_in": round(delta_unmagnified_in, 4),
        "allowable_drift_in": round(delta_allowable_in, 4),
        "stability_coeff_theta": round(theta, 4) if theta is not None else None,
        "theta_max": round(theta_max, 4) if theta_max is not None else None,
        "pdelta_amplifier": round(a_d, 4),
        "pdelta_required": pdelta_required,
        "stability_ok": stability_ok,
    }


def verify_nec_grounding_electrode_conductor(
    service_conductor_kcmil: float,
    gec_provided_kcmil: float,
    service_material: str = "copper",
    gec_material: str = "copper",
) -> Dict[str, Any]:
    """NEC 250.66 Grounding Electrode Conductor (GEC) minimum sizing check for AC services."""
    scalars = [service_conductor_kcmil, gec_provided_kcmil]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if service_conductor_kcmil <= 0 or gec_provided_kcmil <= 0:
        return {"pass": False, "reason": "non_positive_conductor_area"}

    s_mat = service_material.strip().lower()
    g_mat = gec_material.strip().lower()
    valid_materials = {"copper", "cu", "aluminum", "al"}
    if s_mat not in valid_materials or g_mat not in valid_materials:
        return {"pass": False, "reason": "invalid_conductor_material"}

    is_s_cu = s_mat in {"copper", "cu"}
    is_g_cu = g_mat in {"copper", "cu"}

    # NEC Table 250.66 standard wire areas in kcmil
    # 8 AWG = 16.51 kcmil, 6 AWG = 26.24 kcmil, 4 AWG = 41.74 kcmil,
    # 2 AWG = 66.36 kcmil, 1/0 AWG = 105.6 kcmil, 2/0 AWG = 133.1 kcmil,
    # 3/0 AWG = 167.8 kcmil, 250 kcmil = 250.0 kcmil
    if is_s_cu:
        if service_conductor_kcmil <= 41.74:  # <= 2 AWG Cu
            req_kcmil = 16.51 if is_g_cu else 26.24  # 8 AWG Cu or 6 AWG Al
        elif service_conductor_kcmil <= 105.6:  # 1 AWG or 1/0 AWG Cu
            req_kcmil = 26.24 if is_g_cu else 41.74  # 6 AWG Cu or 4 AWG Al
        elif service_conductor_kcmil <= 167.8:  # 2/0 or 3/0 AWG Cu
            req_kcmil = 41.74 if is_g_cu else 66.36  # 4 AWG Cu or 2 AWG Al
        elif service_conductor_kcmil <= 600.0:  # Over 3/0 thru 350 kcmil, or over 350 thru 600 kcmil Cu
            if service_conductor_kcmil <= 350.0:
                req_kcmil = 66.36 if is_g_cu else 105.6  # 2 AWG Cu or 1/0 Al
            else:
                req_kcmil = 105.6 if is_g_cu else 167.8  # 1/0 Cu or 3/0 Al
        elif service_conductor_kcmil <= 1100.0:
            req_kcmil = 133.1 if is_g_cu else 250.0  # 2/0 Cu or 250 kcmil Al
        else:
            # Over 1100 kcmil: 12.5% of total area per NEC 250.66 Note 1
            multiplier = 0.125 if is_g_cu else (0.125 * (66.36 / 41.74))
            req_kcmil = service_conductor_kcmil * multiplier
    else:  # Service is Aluminum
        if service_conductor_kcmil <= 66.36:  # <= 1/0 AWG Al
            req_kcmil = 16.51 if is_g_cu else 26.24
        elif service_conductor_kcmil <= 167.8:  # 2/0 or 3/0 AWG Al
            req_kcmil = 26.24 if is_g_cu else 41.74
        elif service_conductor_kcmil <= 250.0:  # 4/0 AWG or 250 kcmil Al
            req_kcmil = 41.74 if is_g_cu else 66.36
        elif service_conductor_kcmil <= 500.0:  # Over 250 thru 500 kcmil Al
            req_kcmil = 66.36 if is_g_cu else 105.6
        elif service_conductor_kcmil <= 900.0:  # Over 500 thru 900 kcmil Al
            req_kcmil = 105.6 if is_g_cu else 167.8
        elif service_conductor_kcmil <= 1750.0:  # Over 900 thru 1750 kcmil Al
            req_kcmil = 133.1 if is_g_cu else 250.0
        else:
            multiplier = (0.125 * (41.74 / 66.36)) if is_g_cu else 0.125
            req_kcmil = service_conductor_kcmil * multiplier

    dcr = req_kcmil / gec_provided_kcmil
    passed = dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "min_gec_kcmil": round(req_kcmil, 2),
        "provided_gec_kcmil": round(gec_provided_kcmil, 2),
        "service_material": s_mat,
        "gec_material": g_mat,
    }


def verify_aashto_stopping_sight_distance(
    design_speed_mph: float,
    provided_sight_distance_ft: float,
    grade_pct: float = 0.0,
    reaction_time_s: float = 2.5,
    deceleration_rate_fps2: float = 11.2,
    g1_pct: float = 0.0,
    g2_pct: float = 0.0,
    curve_length_ft: float = 0.0,
    curve_type: str = "crest",
) -> Dict[str, Any]:
    """AASHTO Green Book stopping sight distance (SSD) and crest/sag vertical curve length check."""
    scalars = [
        design_speed_mph,
        provided_sight_distance_ft,
        grade_pct,
        reaction_time_s,
        deceleration_rate_fps2,
        g1_pct,
        g2_pct,
        curve_length_ft,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if min(design_speed_mph, provided_sight_distance_ft, reaction_time_s, deceleration_rate_fps2) <= 0:
        return {"pass": False, "reason": "non_positive_speed_or_geometry"}
    if curve_length_ft < 0:
        return {"pass": False, "reason": "negative_curve_length"}

    c_type = curve_type.strip().lower()
    if c_type not in ("crest", "sag"):
        return {"pass": False, "reason": "invalid_curve_type"}

    g_standard = 32.174
    f_ratio = deceleration_rate_fps2 / g_standard
    denom = f_ratio + (grade_pct / 100.0)
    if denom <= 0.0:
        return {"pass": False, "reason": "excessive_downgrade_cannot_stop"}

    # AASHTO Eq. 3-1: Brake reaction distance (1.467 * V * t)
    d_reaction_ft = 1.467 * design_speed_mph * reaction_time_s
    # AASHTO Eq. 3-2: Braking distance on grade (V^2 / (30 * (a/g +/- G)))
    d_braking_ft = (design_speed_mph**2) / (30.0 * denom)
    ssd_required_ft = d_reaction_ft + d_braking_ft

    ssd_dcr = ssd_required_ft / provided_sight_distance_ft
    ssd_ok = ssd_dcr <= 1.0

    # Vertical curve check if grades and curve length are supplied
    curve_ok = True
    curve_dcr = None
    l_min_ft = None
    k_value = None
    a_diff = abs(g1_pct - g2_pct)

    if a_diff > 0.0 and curve_length_ft > 0.0:
        s = ssd_required_ft
        if c_type == "crest":
            # AASHTO standard driver eye height h1=3.5 ft, object height h2=2.0 ft -> C = 2158
            c_coeff = 2158.0
            l_calc_s_le_l = (a_diff * (s**2)) / c_coeff
            if s <= l_calc_s_le_l:
                l_min_ft = l_calc_s_le_l
            else:
                l_min_ft = 2.0 * s - (c_coeff / a_diff)
        else:  # sag curve
            # Headlight beam criteria: h=2.0 ft, 1-deg upward divergence -> C = 400 + 3.5*S
            c_sag = 400.0 + 3.5 * s
            l_calc_sag = (a_diff * (s**2)) / c_sag
            if s <= l_calc_sag:
                l_min_ft = l_calc_sag
            else:
                l_min_ft = 2.0 * s - (c_sag / a_diff)

        l_min_ft = max(l_min_ft, 0.0)
        k_value = curve_length_ft / a_diff if a_diff > 0 else float("inf")
        curve_dcr = l_min_ft / curve_length_ft if curve_length_ft > 0 else float("inf")
        curve_ok = curve_dcr <= 1.0

    gov_dcr = max(ssd_dcr, curve_dcr) if curve_dcr is not None else ssd_dcr
    passed = ssd_ok and curve_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "ssd_required_ft": round(ssd_required_ft, 2),
        "d_reaction_ft": round(d_reaction_ft, 2),
        "d_braking_ft": round(d_braking_ft, 2),
        "ssd_dcr": round(ssd_dcr, 4),
        "l_min_ft": round(l_min_ft, 2) if l_min_ft is not None else None,
        "curve_dcr": round(curve_dcr, 4) if curve_dcr is not None else None,
        "k_value": round(k_value, 2) if k_value is not None else None,
    }


def verify_aisc_unbraced_beam_flexure(
    mu_kip_ft: float,
    unbraced_length_ft: float,
    zx_in3: float,
    sx_in3: float,
    ry_in: float,
    rts_in: float,
    ho_in: float,
    j_in4: float,
    cb: float = 1.0,
    fy_ksi: float = 50.0,
    e_ksi: float = 29000.0,
    phi: float = 0.90,
) -> Dict[str, Any]:
    """AISC 360-16 Section F2 LRFD flexural capacity and lateral-torsional buckling (LTB) check."""
    inputs = [
        mu_kip_ft,
        unbraced_length_ft,
        zx_in3,
        sx_in3,
        ry_in,
        rts_in,
        ho_in,
        j_in4,
        cb,
        fy_ksi,
        e_ksi,
        phi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if mu_kip_ft < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(unbraced_length_ft, zx_in3, sx_in3, ry_in, rts_in, ho_in, j_in4, cb, fy_ksi, e_ksi, phi)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if sx_in3 > zx_in3:
        return {"pass": False, "reason": "sx_exceeds_zx"}

    # AISC 360-16 Eq. F2-1 plastic moment capacity
    mp_kip_in = fy_ksi * zx_in3
    lb_in = unbraced_length_ft * 12.0

    # AISC 360-16 Eq. F2-5 limiting unbraced length for plastic yielding Lp
    lp_in = 1.76 * ry_in * math.sqrt(e_ksi / fy_ksi)

    # AISC 360-16 Eq. F2-6 limiting unbraced length for inelastic LTB Lr
    ratio_j = j_in4 / (sx_in3 * ho_in)
    term_fltb = (0.70 * fy_ksi) / e_ksi
    inside_sqrt = ratio_j + math.sqrt((ratio_j**2) + 6.76 * (term_fltb**2))
    lr_in = 1.95 * rts_in * (e_ksi / (0.70 * fy_ksi)) * math.sqrt(inside_sqrt)

    # Determine nominal flexural strength Mn per AISC Section F2
    if lb_in <= lp_in:
        mn_kip_in = mp_kip_in
        limit_state = "plastic_yielding"
    elif lb_in <= lr_in:
        # AISC Eq. F2-2 inelastic LTB
        mn_calc = cb * (mp_kip_in - (mp_kip_in - 0.70 * fy_ksi * sx_in3) * ((lb_in - lp_in) / (lr_in - lp_in)))
        mn_kip_in = min(mn_calc, mp_kip_in)
        limit_state = "inelastic_ltb"
    else:
        # AISC Eq. F2-3 & F2-4 elastic LTB
        lb_rts = lb_in / rts_in
        fcr_ksi = (cb * (math.pi**2) * e_ksi / (lb_rts**2)) * math.sqrt(1.0 + 0.078 * ratio_j * (lb_rts**2))
        mn_calc = fcr_ksi * sx_in3
        mn_kip_in = min(mn_calc, mp_kip_in)
        limit_state = "elastic_ltb"

    phi_mn_kip_ft = (phi * mn_kip_in) / 12.0
    dcr = mu_kip_ft / phi_mn_kip_ft if phi_mn_kip_ft > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_mn_kip_ft": round(phi_mn_kip_ft, 2),
        "mp_kip_ft": round((phi * mp_kip_in) / 12.0, 2),
        "lp_ft": round(lp_in / 12.0, 2),
        "lr_ft": round(lr_in / 12.0, 2),
        "governing_limit_state": limit_state,
    }


def verify_aisc_web_local_yielding_and_crippling(
    pu_kips: float,
    d_in: float,
    tw_in: float,
    tf_in: float,
    k_des_in: float,
    lb_bearing_in: float,
    distance_to_end_in: float,
    fy_ksi: float = 50.0,
    e_ksi: float = 29000.0,
) -> Dict[str, Any]:
    """AISC 360-16 Section J10.2 & J10.3 LRFD beam web local yielding and crippling check."""
    inputs = [
        pu_kips,
        d_in,
        tw_in,
        tf_in,
        k_des_in,
        lb_bearing_in,
        distance_to_end_in,
        fy_ksi,
        e_ksi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(d_in, tw_in, tf_in, k_des_in, lb_bearing_in, fy_ksi, e_ksi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if distance_to_end_in < 0:
        return {"pass": False, "reason": "negative_distance_to_end"}
    if k_des_in > d_in or tf_in > d_in:
        return {"pass": False, "reason": "flange_or_k_exceeds_depth"}

    # AISC 360-16 Section J10.2 Web Local Yielding (phi = 1.00)
    phi_yield = 1.00
    if distance_to_end_in > d_in:
        rn_yield = fy_ksi * tw_in * (5.0 * k_des_in + lb_bearing_in)
    else:
        rn_yield = fy_ksi * tw_in * (2.5 * k_des_in + lb_bearing_in)
    phi_rn_yield = phi_yield * rn_yield

    # AISC 360-16 Section J10.3 Web Local Crippling (phi = 0.75)
    phi_crip = 0.75
    crippling_const = math.sqrt((e_ksi * fy_ksi * tf_in) / tw_in)
    lb_over_d = lb_bearing_in / d_in
    tw_over_tf_term = (tw_in / tf_in) ** 1.5

    if distance_to_end_in >= (d_in / 2.0):
        # Concentrated load applied at distance >= d / 2 from member end (Eq. J10-4)
        rn_crip = 0.80 * (tw_in**2) * (1.0 + 3.0 * lb_over_d * tw_over_tf_term) * crippling_const
    else:
        # Concentrated load applied at distance < d / 2 from member end (Eq. J10-5a & J10-5b)
        if lb_over_d <= 0.20:
            rn_crip = 0.40 * (tw_in**2) * (1.0 + 3.0 * lb_over_d * tw_over_tf_term) * crippling_const
        else:
            rn_crip = 0.40 * (tw_in**2) * (1.0 + (4.0 * lb_over_d - 0.20) * tw_over_tf_term) * crippling_const

    phi_rn_crip = phi_crip * rn_crip

    phi_rn_gov = min(phi_rn_yield, phi_rn_crip)
    limit_state = "web_local_yielding" if phi_rn_yield <= phi_rn_crip else "web_local_crippling"
    dcr = pu_kips / phi_rn_gov if phi_rn_gov > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_rn_kips": round(phi_rn_gov, 2),
        "phi_rn_yielding_kips": round(phi_rn_yield, 2),
        "phi_rn_crippling_kips": round(phi_rn_crip, 2),
        "governing_limit_state": limit_state,
        "stiffeners_required": not passed,
    }


def verify_asce7_seismic_force_distribution(
    story_weights_kips: Sequence[float],
    story_elevations_ft: Sequence[float],
    story_shear_capacities_kips: Sequence[float],
    base_shear_kips: float,
    fundamental_period_s: float,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 12.8.3 vertical distribution of seismic forces and story shear capacity check."""
    scalars = [base_shear_kips, fundamental_period_s]
    if any(math.isnan(val) for val in scalars) or any(
        math.isnan(x) for seq in (story_weights_kips, story_elevations_ft, story_shear_capacities_kips) for x in seq
    ):
        return {"pass": False, "reason": "nan_input"}

    if not story_weights_kips or not story_elevations_ft or not story_shear_capacities_kips:
        return {"pass": False, "reason": "empty_story_list"}

    n = len(story_weights_kips)
    if len(story_elevations_ft) != n or len(story_shear_capacities_kips) != n:
        return {"pass": False, "reason": "mismatched_story_data"}

    if base_shear_kips <= 0 or fundamental_period_s <= 0:
        return {"pass": False, "reason": "non_positive_loading_or_capacity"}

    if any(w <= 0 for w in story_weights_kips) or any(c <= 0 for c in story_shear_capacities_kips):
        return {"pass": False, "reason": "non_positive_loading_or_capacity"}

    for i, h in enumerate(story_elevations_ft):
        prev = 0.0 if i == 0 else story_elevations_ft[i - 1]
        if h <= prev:
            return {"pass": False, "reason": "non_increasing_elevations"}

    # ASCE 7-16 Section 12.8.3 structural period exponent k
    if fundamental_period_s <= 0.5:
        k = 1.0
    elif fundamental_period_s >= 2.5:
        k = 2.0
    else:
        k = 1.0 + (fundamental_period_s - 0.5) / 2.0

    # ASCE 7-16 Eq. 12.8-12 vertical distribution factor Cvx
    w_h_k = [w * (h ** k) for w, h in zip(story_weights_kips, story_elevations_ft)]
    total_w_h_k = sum(w_h_k)
    if total_w_h_k <= 0:
        return {"pass": False, "reason": "zero_weight_elevation_product"}

    c_vx = [val / total_w_h_k for val in w_h_k]
    story_forces = [c * base_shear_kips for c in c_vx]

    # ASCE 7-16 Section 12.8.4 story shear demand Vx summation from top down
    story_shears = [0.0] * n
    running_shear = 0.0
    for i in range(n - 1, -1, -1):
        running_shear += story_forces[i]
        story_shears[i] = running_shear

    story_dcrs = [v / cap for v, cap in zip(story_shears, story_shear_capacities_kips)]
    governing_dcr = max(story_dcrs)
    gov_story_idx = story_dcrs.index(governing_dcr) + 1

    passed = governing_dcr <= 1.0
    return {
        "pass": passed,
        "dcr": round(governing_dcr, 4),
        "governing_story": gov_story_idx,
        "k_exponent": round(k, 4),
        "story_forces_kips": [round(f, 2) for f in story_forces],
        "story_shears_kips": [round(v, 2) for v in story_shears],
        "story_dcrs": [round(d, 4) for d in story_dcrs],
    }


def verify_asme_b313_pipe_thickness(
    design_pressure_psi: float,
    outer_diameter_in: float,
    nominal_wall_thickness_in: float,
    allowable_stress_psi: float = 20000.0,
    weld_efficiency_e: float = 1.0,
    corrosion_allowance_in: float = 0.0625,
    mechanical_allowance_in: float = 0.0,
    mill_tolerance_pct: float = 12.5,
    y_coefficient: float = 0.40,
    w_factor: float = 1.0,
) -> Dict[str, Any]:
    """ASME B31.3 Section 304.1.2 straight metallic pipe wall thickness and MAWP check."""
    inputs = [
        design_pressure_psi,
        outer_diameter_in,
        nominal_wall_thickness_in,
        allowable_stress_psi,
        weld_efficiency_e,
        corrosion_allowance_in,
        mechanical_allowance_in,
        mill_tolerance_pct,
        y_coefficient,
        w_factor,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            design_pressure_psi,
            outer_diameter_in,
            nominal_wall_thickness_in,
            allowable_stress_psi,
            w_factor,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_pressure_or_dimensions"}
    if corrosion_allowance_in < 0 or mechanical_allowance_in < 0:
        return {"pass": False, "reason": "negative_allowance"}
    if not (0.0 < weld_efficiency_e <= 1.0):
        return {"pass": False, "reason": "invalid_weld_efficiency"}
    if not (0.0 <= mill_tolerance_pct < 100.0):
        return {"pass": False, "reason": "invalid_mill_tolerance"}
    if not (0.0 <= y_coefficient <= 1.0):
        return {"pass": False, "reason": "invalid_y_coefficient"}
    if nominal_wall_thickness_in >= (outer_diameter_in / 2.0):
        return {"pass": False, "reason": "wall_thickness_exceeds_radius"}

    # ASME B31.3 Eq. 3a pressure design thickness t
    denom = 2.0 * (allowable_stress_psi * weld_efficiency_e * w_factor + design_pressure_psi * y_coefficient)
    if denom <= 0:
        return {"pass": False, "reason": "invalid_stress_pressure_combination"}
    t_press_in = (design_pressure_psi * outer_diameter_in) / denom

    total_allowance_in = corrosion_allowance_in + mechanical_allowance_in
    t_min_req_in = t_press_in + total_allowance_in

    mill_factor = 1.0 - (mill_tolerance_pct / 100.0)
    t_nom_req_in = t_min_req_in / mill_factor

    # Effective corroded available wall thickness
    t_avail_min_in = nominal_wall_thickness_in * mill_factor
    t_eff_in = t_avail_min_in - total_allowance_in
    if t_eff_in <= 0:
        return {
            "pass": False,
            "reason": "allowances_exceed_available_thickness",
            "t_avail_min_in": round(t_avail_min_in, 4),
            "total_allowance_in": round(total_allowance_in, 4),
        }

    # Maximum Allowable Working Pressure (MAWP) for pipe
    mawp_denom = outer_diameter_in - 2.0 * y_coefficient * t_eff_in
    if mawp_denom <= 0:
        return {"pass": False, "reason": "invalid_geometry_for_mawp"}
    mawp_psi = (2.0 * allowable_stress_psi * weld_efficiency_e * w_factor * t_eff_in) / mawp_denom

    dcr = t_nom_req_in / nominal_wall_thickness_in
    passed = (dcr <= 1.0) and (design_pressure_psi <= mawp_psi)

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "t_nom_required_in": round(t_nom_req_in, 4),
        "t_min_required_in": round(t_min_req_in, 4),
        "t_pressure_in": round(t_press_in, 4),
        "mawp_psi": round(mawp_psi, 2),
        "effective_thickness_in": round(t_eff_in, 4),
    }


def verify_meyerhof_bearing_capacity(
    applied_load_kips: float,
    footing_width_ft: float,
    footing_length_ft: float,
    embedment_depth_ft: float,
    soil_cohesion_psf: float,
    soil_friction_angle_deg: float,
    soil_unit_weight_pcf: float,
    water_table_depth_ft: float = 999.0,
    load_eccentricity_b_ft: float = 0.0,
    load_eccentricity_l_ft: float = 0.0,
    min_fs: float = 3.0,
) -> Dict[str, Any]:
    """Meyerhof general bearing capacity check for shallow footings with shape, depth, and water table effects."""
    inputs = [
        applied_load_kips,
        footing_width_ft,
        footing_length_ft,
        embedment_depth_ft,
        soil_cohesion_psf,
        soil_friction_angle_deg,
        soil_unit_weight_pcf,
        water_table_depth_ft,
        load_eccentricity_b_ft,
        load_eccentricity_l_ft,
        min_fs,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if applied_load_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(footing_width_ft, footing_length_ft, soil_unit_weight_pcf, min_fs) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_soil_property"}
    if embedment_depth_ft < 0 or soil_cohesion_psf < 0 or water_table_depth_ft < 0:
        return {"pass": False, "reason": "negative_depth_or_cohesion"}
    if not (0.0 <= soil_friction_angle_deg < 50.0):
        return {"pass": False, "reason": "invalid_friction_angle"}
    if load_eccentricity_b_ft < 0 or load_eccentricity_l_ft < 0:
        return {"pass": False, "reason": "negative_eccentricity"}

    # Effective dimensions accounting for Meyerhof eccentricity reduction
    b_prime = footing_width_ft - 2.0 * load_eccentricity_b_ft
    l_prime = footing_length_ft - 2.0 * load_eccentricity_l_ft
    if b_prime <= 0 or l_prime <= 0:
        return {"pass": False, "reason": "excessive_eccentricity"}

    b_eff = min(b_prime, l_prime)
    l_eff = max(b_prime, l_prime)
    effective_area_sqft = b_eff * l_eff

    phi_rad = math.radians(soil_friction_angle_deg)
    # Bearing capacity factors
    if soil_friction_angle_deg == 0.0:
        n_q = 1.0
        n_c = 5.14
        n_gamma = 0.0
    else:
        n_q = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45.0 + soil_friction_angle_deg / 2.0)) ** 2)
        n_c = (n_q - 1.0) / math.tan(phi_rad)
        n_gamma = 2.0 * (n_q + 1.0) * math.tan(phi_rad)

    # Meyerhof shape factors
    s_c = 1.0 + (b_eff / l_eff) * (n_q / n_c)
    s_q = 1.0 + (b_eff / l_eff) * math.tan(phi_rad)
    s_gamma = 1.0 - 0.4 * (b_eff / l_eff)

    # Meyerhof depth factors
    k_ratio = (embedment_depth_ft / b_eff) if (embedment_depth_ft / b_eff) <= 1.0 else math.atan(embedment_depth_ft / b_eff)
    d_c = 1.0 + 0.4 * k_ratio
    d_q = 1.0 + 2.0 * math.tan(phi_rad) * ((1.0 - math.sin(phi_rad)) ** 2) * k_ratio
    d_gamma = 1.0

    # Groundwater table correction on surcharge q and soil unit weight gamma
    gamma_w = 62.4
    if water_table_depth_ft >= (embedment_depth_ft + b_eff):
        q_surcharge = soil_unit_weight_pcf * embedment_depth_ft
        gamma_eff = soil_unit_weight_pcf
    elif water_table_depth_ft <= embedment_depth_ft:
        gamma_sub = max(soil_unit_weight_pcf - gamma_w, 0.0)
        q_surcharge = soil_unit_weight_pcf * water_table_depth_ft + gamma_sub * (embedment_depth_ft - water_table_depth_ft)
        gamma_eff = gamma_sub
    else:
        gamma_sub = max(soil_unit_weight_pcf - gamma_w, 0.0)
        d_w_below = water_table_depth_ft - embedment_depth_ft
        q_surcharge = soil_unit_weight_pcf * embedment_depth_ft
        gamma_eff = gamma_sub + (d_w_below / b_eff) * (soil_unit_weight_pcf - gamma_sub)

    # Meyerhof ultimate bearing capacity (psf)
    q_ult_psf = (
        soil_cohesion_psf * n_c * s_c * d_c
        + q_surcharge * n_q * s_q * d_q
        + 0.5 * gamma_eff * b_eff * n_gamma * s_gamma * d_gamma
    )
    q_allow_psf = q_ult_psf / min_fs

    q_applied_psf = (applied_load_kips * 1000.0) / effective_area_sqft if effective_area_sqft > 0 else float("inf")
    fs_actual = q_ult_psf / q_applied_psf if q_applied_psf > 0 else float("inf")
    dcr = q_applied_psf / q_allow_psf if q_allow_psf > 0 else float("inf")

    passed = (dcr <= 1.0) and (fs_actual >= min_fs)
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "actual_fs": round(fs_actual, 3),
        "q_applied_psf": round(q_applied_psf, 2),
        "q_allowable_psf": round(q_allow_psf, 2),
        "q_ultimate_psf": round(q_ult_psf, 2),
        "effective_width_ft": round(b_eff, 2),
        "effective_length_ft": round(l_eff, 2),
    }


def verify_aashto_horizontal_curve(
    design_speed_mph: float,
    radius_ft: float,
    superelevation_pct: float,
    max_superelevation_pct: float = 8.0,
    f_max_override: float = 0.0,
    ssd_ft: float = 0.0,
    hso_provided_ft: float = 0.0,
) -> Dict[str, Any]:
    """AASHTO Green Book horizontal curve radius, side friction, superelevation, and sightline offset (HSO) check."""
    scalars = [
        design_speed_mph,
        radius_ft,
        superelevation_pct,
        max_superelevation_pct,
        f_max_override,
        ssd_ft,
        hso_provided_ft,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if min(design_speed_mph, radius_ft, max_superelevation_pct) <= 0:
        return {"pass": False, "reason": "non_positive_speed_or_geometry"}
    if superelevation_pct < 0:
        return {"pass": False, "reason": "negative_superelevation"}
    if f_max_override < 0:
        return {"pass": False, "reason": "negative_friction_factor"}
    if ssd_ft < 0 or hso_provided_ft < 0:
        return {"pass": False, "reason": "negative_sight_distance_or_offset"}

    # AASHTO Table 3-7 maximum side friction factor f_max as a function of design speed
    if f_max_override > 0.0:
        f_max = f_max_override
    elif design_speed_mph <= 50.0:
        f_max = 0.19 - 0.001 * design_speed_mph
    else:
        f_max = max(0.14 - 0.002 * (design_speed_mph - 50.0), 0.05)

    e_max = max_superelevation_pct / 100.0
    e_actual = superelevation_pct / 100.0

    # AASHTO Eq. 3-6: Minimum curve radius R_min = V^2 / (15 * (e_max + f_max))
    r_min_ft = (design_speed_mph**2) / (15.0 * (e_max + f_max))
    radius_dcr = r_min_ft / radius_ft

    # Side friction demand f_demand = V^2 / (15 * R) - e
    f_demand = (design_speed_mph**2) / (15.0 * radius_ft) - e_actual
    friction_dcr = (f_demand / f_max) if f_max > 0 else float("inf")
    superelev_dcr = superelevation_pct / max_superelevation_pct

    # AASHTO Eq. 3-36: Horizontal Sightline Offset (HSO / Middle Ordinate M)
    hso_required_ft = None
    hso_dcr = None
    hso_ok = True
    if ssd_ft > 0.0:
        half_angle_deg = (28.64788975654116 * ssd_ft) / radius_ft
        if half_angle_deg >= 90.0:
            return {"pass": False, "reason": "sight_distance_subtends_excessive_angle"}
        hso_required_ft = radius_ft * (1.0 - math.cos(math.radians(half_angle_deg)))
        if hso_provided_ft > 0.0:
            hso_dcr = hso_required_ft / hso_provided_ft
            hso_ok = hso_dcr <= 1.0

    dcr_components = [radius_dcr, friction_dcr, superelev_dcr]
    if hso_dcr is not None:
        dcr_components.append(hso_dcr)
    gov_dcr = max(dcr_components)

    passed = (
        (radius_ft >= r_min_ft)
        and (f_demand <= f_max)
        and (superelevation_pct <= max_superelevation_pct)
        and hso_ok
    )

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "r_min_ft": round(r_min_ft, 2),
        "f_max": round(f_max, 4),
        "f_demand": round(f_demand, 4),
        "hso_required_ft": round(hso_required_ft, 2) if hso_required_ft is not None else None,
        "hso_dcr": round(hso_dcr, 4) if hso_dcr is not None else None,
    }


def verify_nec_equipment_grounding_conductor(
    ocpd_rating_a: float,
    egc_provided_kcmil: float,
    egc_material: str = "copper",
    actual_ungrounded_kcmil: float = 0.0,
    min_ungrounded_kcmil: float = 0.0,
) -> Dict[str, Any]:
    """NEC 250.122 Equipment Grounding Conductor (EGC) sizing and 250.122(B) proportional upsize gate."""
    scalars = [ocpd_rating_a, egc_provided_kcmil, actual_ungrounded_kcmil, min_ungrounded_kcmil]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if ocpd_rating_a <= 0 or egc_provided_kcmil <= 0:
        return {"pass": False, "reason": "non_positive_rating_or_area"}
    if actual_ungrounded_kcmil < 0 or min_ungrounded_kcmil < 0:
        return {"pass": False, "reason": "negative_conductor_area"}

    g_mat = egc_material.strip().lower()
    valid_materials = {"copper", "cu", "aluminum", "al"}
    if g_mat not in valid_materials:
        return {"pass": False, "reason": "invalid_conductor_material"}
    is_cu = g_mat in {"copper", "cu"}

    if (actual_ungrounded_kcmil > 0.0) ^ (min_ungrounded_kcmil > 0.0):
        return {"pass": False, "reason": "incomplete_ungrounded_conductor_data"}
    if actual_ungrounded_kcmil > 0.0 and actual_ungrounded_kcmil < min_ungrounded_kcmil:
        return {"pass": False, "reason": "actual_area_less_than_minimum"}

    # NEC Table 250.122 standard thresholds: (max_ocpd_a, cu_kcmil, al_kcmil)
    table_250_122 = [
        (15.0, 4.11, 6.53),
        (20.0, 6.53, 10.38),
        (60.0, 10.38, 16.51),
        (100.0, 16.51, 26.24),
        (200.0, 26.24, 41.74),
        (300.0, 41.74, 66.36),
        (400.0, 52.62, 83.69),
        (500.0, 66.36, 105.6),
        (600.0, 83.69, 133.1),
        (800.0, 105.6, 167.8),
        (1000.0, 133.1, 211.6),
        (1200.0, 167.8, 250.0),
        (1600.0, 211.6, 350.0),
        (2000.0, 250.0, 400.0),
        (2500.0, 350.0, 600.0),
        (3000.0, 400.0, 600.0),
        (4000.0, 500.0, 750.0),
        (5000.0, 700.0, 1200.0),
        (6000.0, 800.0, 1200.0),
    ]

    if ocpd_rating_a > 6000.0:
        return {"pass": False, "reason": "ocpd_exceeds_code_table"}

    base_egc_kcmil = 0.0
    for limit_a, cu_area, al_area in table_250_122:
        if ocpd_rating_a <= limit_a:
            base_egc_kcmil = cu_area if is_cu else al_area
            break

    # NEC 250.122(B) proportional adjustment for upsized phase conductors
    if actual_ungrounded_kcmil > 0.0 and min_ungrounded_kcmil > 0.0:
        growth_ratio = actual_ungrounded_kcmil / min_ungrounded_kcmil
        required_egc_kcmil = base_egc_kcmil * max(1.0, growth_ratio)
    else:
        growth_ratio = 1.0
        required_egc_kcmil = base_egc_kcmil

    dcr = required_egc_kcmil / egc_provided_kcmil
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "required_egc_kcmil": round(required_egc_kcmil, 2),
        "base_egc_kcmil": round(base_egc_kcmil, 2),
        "provided_egc_kcmil": round(egc_provided_kcmil, 2),
        "upsize_ratio": round(growth_ratio, 4),
        "egc_material": g_mat,
    }


def verify_asce7_nonstructural_component_seismic(
    w_p_kips: float,
    s_ds: float,
    ap_factor: float,
    rp_factor: float,
    z_height_ft: float,
    h_roof_ft: float,
    f_p_capacity_kips: float,
    importance_factor_ip: float = 1.0,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 13.3.1 seismic design force (Fp) and anchorage capacity gate for nonstructural components."""
    inputs = [
        w_p_kips,
        s_ds,
        ap_factor,
        rp_factor,
        z_height_ft,
        h_roof_ft,
        f_p_capacity_kips,
        importance_factor_ip,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(w_p_kips, s_ds, ap_factor, rp_factor, h_roof_ft, f_p_capacity_kips) <= 0
    ):
        return {"pass": False, "reason": "non_positive_loading_or_capacity"}
    if z_height_ft < 0:
        return {"pass": False, "reason": "negative_attachment_height"}
    if z_height_ft > h_roof_ft:
        return {"pass": False, "reason": "attachment_height_exceeds_roof_height"}
    if importance_factor_ip not in (1.0, 1.5):
        return {"pass": False, "reason": "invalid_importance_factor"}
    if ap_factor < 1.0 or rp_factor < 1.0:
        return {"pass": False, "reason": "invalid_component_factors"}

    # Elevation ratio z / h (ASCE 7 Section 13.3.1)
    height_ratio = min(max(z_height_ft / h_roof_ft, 0.0), 1.0)

    # ASCE 7 Eq. 13.3-1 unconstrained horizontal seismic force
    rp_over_ip = rp_factor / importance_factor_ip
    f_p_unclamped = (
        (0.4 * ap_factor * s_ds * w_p_kips / rp_over_ip)
        * (1.0 + 2.0 * height_ratio)
    )

    # ASCE 7 Eq. 13.3-2 upper bound limit and Eq. 13.3-3 lower bound limit
    f_p_max = 1.6 * s_ds * importance_factor_ip * w_p_kips
    f_p_min = 0.3 * s_ds * importance_factor_ip * w_p_kips

    if f_p_unclamped > f_p_max:
        f_p_gov = f_p_max
        gov_limit = "max_limit"
    elif f_p_unclamped < f_p_min:
        f_p_gov = f_p_min
        gov_limit = "min_limit"
    else:
        f_p_gov = f_p_unclamped
        gov_limit = "unclamped"

    # Concurrent vertical seismic force Fpv (ASCE 7 Section 13.3.1 & 12.4.2.2)
    f_pv = 0.20 * s_ds * w_p_kips

    dcr = f_p_gov / f_p_capacity_kips
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "f_p_kips": round(f_p_gov, 3),
        "f_p_unclamped_kips": round(f_p_unclamped, 3),
        "f_p_max_kips": round(f_p_max, 3),
        "f_p_min_kips": round(f_p_min, 3),
        "f_pv_kips": round(f_pv, 3),
        "governing_limit": gov_limit,
    }


def verify_nfpa13_fire_sprinkler_demand(
    hazard_classification: str,
    coverage_area_per_sprinkler_sqft: float,
    available_flow_gpm: float,
    available_pressure_psi: float,
    design_area_sqft: float = 0.0,
    density_gpm_per_sqft: float = 0.0,
    k_factor: float = 5.6,
    system_piping_loss_psi: float = 25.0,
    elevation_loss_psi: float = 0.0,
    hose_allowance_gpm: float = 0.0,
    is_dry_system: bool = False,
    min_head_pressure_psi: float = 7.0,
) -> Dict[str, Any]:
    """NFPA 13 (2019/2022) Chapter 19 & 27 hydraulic demand and water supply adequacy gate."""
    scalars = [
        coverage_area_per_sprinkler_sqft,
        available_flow_gpm,
        available_pressure_psi,
        design_area_sqft,
        density_gpm_per_sqft,
        k_factor,
        system_piping_loss_psi,
        elevation_loss_psi,
        hose_allowance_gpm,
        min_head_pressure_psi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            coverage_area_per_sprinkler_sqft,
            available_flow_gpm,
            available_pressure_psi,
            k_factor,
            min_head_pressure_psi,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_capacity"}
    if (
        design_area_sqft < 0
        or density_gpm_per_sqft < 0
        or system_piping_loss_psi < 0
        or elevation_loss_psi < 0
        or hose_allowance_gpm < 0
    ):
        return {"pass": False, "reason": "negative_hydraulic_parameter"}

    hazard_key = hazard_classification.strip().lower().replace(" ", "_")
    # NFPA 13 Table 19.3.3.1.1 & Table 10.2.4.2.1: (density, default_area, default_hose, max_coverage_sqft)
    hazard_data: Dict[str, Tuple[float, float, float, float]] = {
        "light": (0.10, 1500.0, 100.0, 225.0),
        "light_hazard": (0.10, 1500.0, 100.0, 225.0),
        "ordinary_1": (0.15, 1500.0, 250.0, 130.0),
        "ordinary_hazard_group_1": (0.15, 1500.0, 250.0, 130.0),
        "oh1": (0.15, 1500.0, 250.0, 130.0),
        "ordinary_2": (0.20, 1500.0, 250.0, 130.0),
        "ordinary_hazard_group_2": (0.20, 1500.0, 250.0, 130.0),
        "oh2": (0.20, 1500.0, 250.0, 130.0),
        "extra_1": (0.30, 2500.0, 500.0, 100.0),
        "extra_hazard_group_1": (0.30, 2500.0, 500.0, 100.0),
        "eh1": (0.30, 2500.0, 500.0, 100.0),
        "extra_2": (0.40, 2500.0, 500.0, 100.0),
        "extra_hazard_group_2": (0.40, 2500.0, 500.0, 100.0),
        "eh2": (0.40, 2500.0, 500.0, 100.0),
    }
    if hazard_key not in hazard_data:
        return {"pass": False, "reason": "invalid_hazard_classification"}

    code_density, code_area, code_hose, max_coverage_sqft = hazard_data[hazard_key]
    if coverage_area_per_sprinkler_sqft > max_coverage_sqft:
        return {"pass": False, "reason": "sprinkler_coverage_exceeds_maximum"}

    eff_density = density_gpm_per_sqft if density_gpm_per_sqft > 0.0 else code_density
    base_area = design_area_sqft if design_area_sqft > 0.0 else code_area
    # NFPA 13 Section 19.3.3.2.3: 30% area penalty for dry-pipe systems
    eff_area = base_area * 1.30 if is_dry_system else base_area
    eff_hose = hose_allowance_gpm if hose_allowance_gpm > 0.0 else code_hose

    num_sprinklers = int(math.ceil(eff_area / coverage_area_per_sprinkler_sqft))

    # Minimum discharge flow per sprinkler
    q_density = eff_density * coverage_area_per_sprinkler_sqft
    q_min_press = k_factor * math.sqrt(min_head_pressure_psi)
    q_head = max(q_density, q_min_press)

    # Hydraulic pressure demand at most remote sprinkler head (P = (Q / K)^2)
    p_head = (q_head / k_factor) ** 2

    q_sprinklers = num_sprinklers * q_head
    q_total_demand = q_sprinklers + eff_hose
    p_total_demand = p_head + system_piping_loss_psi + elevation_loss_psi

    flow_dcr = q_total_demand / available_flow_gpm
    press_dcr = p_total_demand / available_pressure_psi
    gov_dcr = max(flow_dcr, press_dcr)

    passed = (gov_dcr <= 1.0) and (p_head >= min_head_pressure_psi)
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "flow_dcr": round(flow_dcr, 4),
        "pressure_dcr": round(press_dcr, 4),
        "q_total_demand_gpm": round(q_total_demand, 2),
        "q_sprinklers_gpm": round(q_sprinklers, 2),
        "p_total_demand_psi": round(p_total_demand, 2),
        "p_head_psi": round(p_head, 2),
        "num_sprinklers": num_sprinklers,
        "design_area_sqft": round(eff_area, 1),
        "design_density_gpm_sqft": round(eff_density, 3),
        "hose_allowance_gpm": round(eff_hose, 1),
    }


def verify_aisc_block_shear(
    pu_kips: float,
    agv_in2: float,
    anv_in2: float,
    ant_in2: float,
    agt_in2: float = 0.0,
    fy_ksi: float = 50.0,
    fu_ksi: float = 65.0,
    u_bs: float = 1.0,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """AISC 360-16 Section J4.3 LRFD check for block shear rupture on connection failure paths."""
    scalars = [pu_kips, agv_in2, anv_in2, ant_in2, agt_in2, fy_ksi, fu_ksi, u_bs, phi]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(agv_in2, anv_in2, ant_in2, fy_ksi, fu_ksi, phi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if agt_in2 < 0:
        return {"pass": False, "reason": "negative_gross_tension_area"}
    if anv_in2 > agv_in2:
        return {"pass": False, "reason": "net_shear_area_exceeds_gross"}
    if agt_in2 > 0.0 and ant_in2 > agt_in2:
        return {"pass": False, "reason": "net_tension_area_exceeds_gross"}
    if not (0.0 < u_bs <= 1.0):
        return {"pass": False, "reason": "invalid_shear_lag_factor"}

    # AISC 360-16 Eq. J4-5 block shear nominal strength
    tension_rupture_kips = u_bs * fu_ksi * ant_in2
    rn_rupture_kips = 0.60 * fu_ksi * anv_in2 + tension_rupture_kips
    rn_yielding_kips = 0.60 * fy_ksi * agv_in2 + tension_rupture_kips

    rn_gov_kips = min(rn_rupture_kips, rn_yielding_kips)
    limit_state = (
        "shear_rupture"
        if rn_rupture_kips <= rn_yielding_kips
        else "shear_yielding"
    )

    phi_rn_kips = phi * rn_gov_kips
    dcr = pu_kips / phi_rn_kips if phi_rn_kips > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_rn_kips": round(phi_rn_kips, 2),
        "rn_kips": round(rn_gov_kips, 2),
        "rn_rupture_kips": round(rn_rupture_kips, 2),
        "rn_yielding_kips": round(rn_yielding_kips, 2),
        "governing_limit_state": limit_state,
        "tension_term_kips": round(tension_rupture_kips, 2),
    }


def verify_asce7_snow_drift(
    pg_psf: float,
    lu_fetch_ft: float,
    step_height_ft: float,
    p_capacity_psf: float,
    pf_lower_psf: float = 0.0,
    ce_exposure_factor: float = 1.0,
    ct_thermal_factor: float = 1.0,
    importance_factor: float = 1.0,
    is_windward: bool = False,
) -> Dict[str, Any]:
    """ASCE 7-16/7-22 Section 7.7 & 7.8 leeward and windward snow drift surcharge gate for roof elevation steps."""
    scalars = [
        pg_psf,
        lu_fetch_ft,
        step_height_ft,
        p_capacity_psf,
        pf_lower_psf,
        ce_exposure_factor,
        ct_thermal_factor,
        importance_factor,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            pg_psf,
            lu_fetch_ft,
            step_height_ft,
            p_capacity_psf,
            ce_exposure_factor,
            ct_thermal_factor,
            importance_factor,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_loading_or_geometry"}
    if pf_lower_psf < 0:
        return {"pass": False, "reason": "negative_balanced_snow_load"}

    # ASCE 7-16 Eq. 7.7-1 snow density gamma (pcf), capped at 30 pcf
    gamma_pcf = min(0.13 * pg_psf + 14.0, 30.0)

    # Balanced uniform snow load on lower roof (Eq. 7.3-1) if not explicitly provided
    if pf_lower_psf > 0.0:
        pf = pf_lower_psf
    else:
        pf = 0.70 * ce_exposure_factor * ct_thermal_factor * importance_factor * pg_psf

    # Balanced snow depth on lower roof (ft)
    hb_ft = pf / gamma_pcf
    # Clear height difference between upper roof and lower roof balanced snow surface
    hc_ft = step_height_ft - hb_ft

    # ASCE 7-16 Section 7.7.1: if hc / hb < 0.2 or hc <= 0, drift need not be evaluated
    if hc_ft <= 0.0 or (hc_ft / hb_ft) < 0.20:
        p_total = pf
        dcr = p_total / p_capacity_psf
        return {
            "pass": dcr <= 1.0,
            "dcr": round(dcr, 4),
            "p_total_psf": round(p_total, 2),
            "p_drift_psf": 0.0,
            "pf_lower_psf": round(pf, 2),
            "drift_height_ft": 0.0,
            "drift_width_ft": 0.0,
            "snow_density_pcf": round(gamma_pcf, 2),
            "drift_required": False,
        }

    # ASCE 7-16 Figure 7-9 & Eq. 7.7-2 / 7.7-3 unconstrained drift height hd
    hd_calc = 0.43 * (lu_fetch_ft ** (1.0 / 3.0)) * ((pg_psf + 10.0) ** 0.25) - 1.50
    if is_windward:
        # Windward drift height is 3/4 of leeward drift
        hd_unclamped = 0.75 * max(hd_calc, 0.0)
    else:
        hd_unclamped = max(hd_calc, 0.0)

    if hd_unclamped <= 0.0:
        p_total = pf
        dcr = p_total / p_capacity_psf
        return {
            "pass": dcr <= 1.0,
            "dcr": round(dcr, 4),
            "p_total_psf": round(p_total, 2),
            "p_drift_psf": 0.0,
            "pf_lower_psf": round(pf, 2),
            "drift_height_ft": 0.0,
            "drift_width_ft": 0.0,
            "snow_density_pcf": round(gamma_pcf, 2),
            "drift_required": False,
        }

    # Check truncation by clear height hc (ASCE 7-16 Section 7.7.1)
    if hd_unclamped > hc_ft:
        hd_gov = hc_ft
        # When hd > hc, drift width widens: w = 4 * hd^2 / hc, capped at 8 * hc
        w_drift_ft = min(4.0 * (hd_unclamped ** 2) / hc_ft, 8.0 * hc_ft)
    else:
        hd_gov = hd_unclamped
        w_drift_ft = 4.0 * hd_gov

    # Peak drift surcharge pressure pd (psf)
    pd_psf = hd_gov * gamma_pcf
    p_total_psf = pf + pd_psf
    dcr = p_total_psf / p_capacity_psf
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "p_total_psf": round(p_total_psf, 2),
        "p_drift_psf": round(pd_psf, 2),
        "pf_lower_psf": round(pf, 2),
        "drift_height_ft": round(hd_gov, 3),
        "drift_width_ft": round(w_drift_ft, 2),
        "clear_height_ft": round(hc_ft, 3),
        "snow_density_pcf": round(gamma_pcf, 2),
        "drift_required": True,
    }


def verify_nrcs_tr55_runoff(
    rainfall_depth_in: float,
    curve_number_cn: float,
    drainage_area_acres: float,
    storage_capacity_acre_ft: float,
    initial_abstraction_ratio: float = 0.20,
    allowable_runoff_depth_in: float = 0.0,
) -> Dict[str, Any]:
    """USDA NRCS (SCS) TR-55 National Engineering Handbook Part 630 runoff depth, volume, and storage gate."""
    scalars = [
        rainfall_depth_in,
        curve_number_cn,
        drainage_area_acres,
        storage_capacity_acre_ft,
        initial_abstraction_ratio,
        allowable_runoff_depth_in,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if rainfall_depth_in < 0:
        return {"pass": False, "reason": "negative_rainfall_depth"}
    if drainage_area_acres <= 0 or storage_capacity_acre_ft <= 0:
        return {"pass": False, "reason": "non_positive_area_or_capacity"}
    if not (30.0 <= curve_number_cn <= 100.0):
        return {"pass": False, "reason": "invalid_curve_number"}
    if not (0.01 <= initial_abstraction_ratio <= 0.50):
        return {"pass": False, "reason": "invalid_abstraction_ratio"}
    if allowable_runoff_depth_in < 0:
        return {"pass": False, "reason": "negative_allowable_depth"}

    # Potential maximum soil retention after runoff begins (S, in inches)
    s_in = (1000.0 / curve_number_cn) - 10.0
    ia_in = initial_abstraction_ratio * s_in

    # NRCS TR-55 Eq. 2-1 direct runoff depth (Q, in inches)
    if rainfall_depth_in <= ia_in:
        q_in = 0.0
    else:
        denom = rainfall_depth_in + (1.0 - initial_abstraction_ratio) * s_in
        q_in = ((rainfall_depth_in - ia_in) ** 2) / denom if denom > 0 else 0.0

    v_runoff_acft = (q_in / 12.0) * drainage_area_acres
    v_runoff_cuft = v_runoff_acft * 43560.0

    volume_dcr = v_runoff_acft / storage_capacity_acre_ft
    if allowable_runoff_depth_in > 0.0:
        depth_dcr = q_in / allowable_runoff_depth_in
        gov_dcr = max(volume_dcr, depth_dcr)
        depth_ok = depth_dcr <= 1.0
    else:
        gov_dcr = volume_dcr
        depth_ok = True

    runoff_fraction = (q_in / rainfall_depth_in) if rainfall_depth_in > 0 else 0.0
    passed = (volume_dcr <= 1.0) and depth_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "runoff_depth_in": round(q_in, 4),
        "runoff_volume_acre_ft": round(v_runoff_acft, 4),
        "runoff_volume_cuft": round(v_runoff_cuft, 2),
        "potential_retention_s_in": round(s_in, 4),
        "initial_abstraction_ia_in": round(ia_in, 4),
        "runoff_fraction": round(runoff_fraction, 4),
        "runoff_generated": q_in > 0.0,
    }


def verify_open_channel_hydraulic_jump(
    discharge_cfs: float,
    channel_width_ft: float,
    upstream_depth_ft: float,
    provided_basin_length_ft: float,
    tailwater_depth_ft: float = 0.0,
    min_froude_number: float = 1.7,
    basin_length_factor: float = 6.0,
    g_fps2: float = 32.174,
) -> Dict[str, Any]:
    """Civil/Water Resources hydraulic jump sequent depth, energy dissipation, and stilling basin length check."""
    scalars = [
        discharge_cfs,
        channel_width_ft,
        upstream_depth_ft,
        provided_basin_length_ft,
        tailwater_depth_ft,
        min_froude_number,
        basin_length_factor,
        g_fps2,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            discharge_cfs,
            channel_width_ft,
            upstream_depth_ft,
            provided_basin_length_ft,
            basin_length_factor,
            g_fps2,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_flow"}
    if tailwater_depth_ft < 0:
        return {"pass": False, "reason": "negative_tailwater_depth"}
    if min_froude_number < 1.0:
        return {"pass": False, "reason": "invalid_min_froude_number"}

    # Upstream supercritical flow characteristics
    area_1 = channel_width_ft * upstream_depth_ft
    v1 = discharge_cfs / area_1
    fr1 = v1 / math.sqrt(g_fps2 * upstream_depth_ft)
    if fr1 <= 1.0:
        return {
            "pass": False,
            "reason": "upstream_flow_subcritical",
            "froude_upstream": round(fr1, 4),
        }

    # Belanger equation for hydraulic jump sequent depth (y2)
    y2 = 0.5 * upstream_depth_ft * (math.sqrt(1.0 + 8.0 * (fr1**2)) - 1.0)

    # Downstream subcritical flow characteristics
    area_2 = channel_width_ft * y2
    v2 = discharge_cfs / area_2
    fr2 = v2 / math.sqrt(g_fps2 * y2)

    # Upstream and downstream specific energy & energy dissipation head loss
    e1 = upstream_depth_ft + (v1**2) / (2.0 * g_fps2)
    e2 = y2 + (v2**2) / (2.0 * g_fps2)
    delta_e = ((y2 - upstream_depth_ft) ** 3) / (4.0 * upstream_depth_ft * y2)
    efficiency = delta_e / e1 if e1 > 0 else 0.0

    # USBR stilling basin required length
    req_basin_length_ft = basin_length_factor * y2
    length_dcr = req_basin_length_ft / provided_basin_length_ft
    length_ok = length_dcr <= 1.0

    # USBR jump classification based on upstream Froude number
    if fr1 < 1.7:
        jump_class = "undular"
    elif fr1 < 2.5:
        jump_class = "weak"
    elif fr1 < 4.5:
        jump_class = "oscillating"
    elif fr1 < 9.0:
        jump_class = "steady"
    else:
        jump_class = "strong"

    froude_ok = fr1 >= min_froude_number

    # Tailwater adequacy check (preventing sweepout or drowning)
    tw_ratio = None
    tw_ok = True
    if tailwater_depth_ft > 0.0:
        tw_ratio = tailwater_depth_ft / y2
        tw_ok = 0.95 <= tw_ratio <= 1.15

    passed = length_ok and froude_ok and tw_ok
    return {
        "pass": passed,
        "dcr": round(length_dcr, 4),
        "froude_upstream": round(fr1, 3),
        "froude_downstream": round(fr2, 3),
        "sequent_depth_ft": round(y2, 3),
        "head_loss_ft": round(delta_e, 3),
        "dissipation_efficiency": round(efficiency, 4),
        "required_basin_length_ft": round(req_basin_length_ft, 2),
        "jump_classification": jump_class,
        "tailwater_ratio": round(tw_ratio, 3) if tw_ratio is not None else None,
        "length_ok": length_ok,
        "froude_ok": froude_ok,
        "tailwater_ok": tw_ok,
    }


def verify_heat_exchanger_lmtd(
    q_design_btuh: float,
    th_in_f: float,
    th_out_f: float,
    tc_in_f: float,
    tc_out_f: float,
    u_overall_btu_hr_sqft_f: float,
    area_sqft: float,
    flow_type: str = "counterflow",
    min_f_factor: float = 0.75,
) -> Dict[str, Any]:
    """ASHRAE/AHRI heat exchanger thermal rating, LMTD, and F-correction factor check."""
    scalars = [
        q_design_btuh,
        th_in_f,
        th_out_f,
        tc_in_f,
        tc_out_f,
        u_overall_btu_hr_sqft_f,
        area_sqft,
        min_f_factor,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if min(q_design_btuh, u_overall_btu_hr_sqft_f, area_sqft, min_f_factor) <= 0:
        return {"pass": False, "reason": "non_positive_duty_or_geometry"}
    if th_in_f <= th_out_f:
        return {"pass": False, "reason": "hot_stream_does_not_cool"}
    if tc_out_f <= tc_in_f:
        return {"pass": False, "reason": "cold_stream_does_not_heat"}
    if th_in_f <= tc_in_f:
        return {"pass": False, "reason": "hot_inlet_not_above_cold_inlet"}

    ft = flow_type.strip().lower()
    if ft not in ("counterflow", "parallel", "shell_tube_1_2"):
        return {"pass": False, "reason": "invalid_flow_type"}

    if ft == "parallel":
        dt1 = th_in_f - tc_in_f
        dt2 = th_out_f - tc_out_f
        if dt2 <= 0:
            return {"pass": False, "reason": "parallel_flow_temperature_cross"}
    else:
        dt1 = th_in_f - tc_out_f
        dt2 = th_out_f - tc_in_f
        if dt1 <= 0 or dt2 <= 0:
            return {"pass": False, "reason": "temperature_cross_violation"}

    if abs(dt1 - dt2) < 1e-6:
        lmtd = dt1
    else:
        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

    if ft in ("counterflow", "parallel"):
        f_factor = 1.0
    else:
        denom_p = th_in_f - tc_in_f
        p = (tc_out_f - tc_in_f) / denom_p
        denom_r = tc_out_f - tc_in_f
        r = (th_in_f - th_out_f) / denom_r

        if p >= 1.0 or (p * r) >= 1.0:
            return {"pass": False, "reason": "thermal_effectiveness_out_of_range"}

        term_sqrt = math.sqrt(r**2 + 1.0)
        num_arg = (1.0 - p) / (1.0 - p * r)
        if num_arg <= 0:
            return {"pass": False, "reason": "invalid_log_argument_in_f_factor"}

        if abs(r - 1.0) < 1e-6:
            den_arg = (2.0 - p * (2.0 - math.sqrt(2.0))) / (2.0 - p * (2.0 + math.sqrt(2.0)))
            if den_arg <= 0:
                return {"pass": False, "reason": "temperature_cross_in_shell_pass"}
            f_factor = (math.sqrt(2.0) * p / (1.0 - p)) / math.log(den_arg)
        else:
            den_arg = (2.0 - p * (r + 1.0 - term_sqrt)) / (2.0 - p * (r + 1.0 + term_sqrt))
            if den_arg <= 0:
                return {"pass": False, "reason": "temperature_cross_in_shell_pass"}
            f_factor = (term_sqrt * math.log(num_arg) / (r - 1.0)) / math.log(den_arg)

    f_factor = max(0.0, min(f_factor, 1.0))
    q_available_btuh = u_overall_btu_hr_sqft_f * area_sqft * f_factor * lmtd
    area_required_sqft = q_design_btuh / (u_overall_btu_hr_sqft_f * f_factor * lmtd)
    dcr = q_design_btuh / q_available_btuh if q_available_btuh > 0 else float("inf")

    passed = (dcr <= 1.0) and (f_factor >= min_f_factor)
    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "q_available_btuh": round(q_available_btuh, 2),
        "area_required_sqft": round(area_required_sqft, 2),
        "lmtd_f": round(lmtd, 2),
        "f_factor": round(f_factor, 4),
        "flow_type": ft,
    }


def verify_asme_bpvc_vessel_thickness(
    design_pressure_psi: float,
    inside_diameter_in: float,
    nominal_thickness_in: float,
    allowable_stress_psi: float = 20000.0,
    joint_efficiency_e: float = 1.0,
    corrosion_allowance_in: float = 0.0625,
    component_type: str = "cylindrical_shell",
    crown_radius_in: float = 0.0,
    knuckle_radius_in: float = 0.0,
) -> Dict[str, Any]:
    """ASME BPVC Section VIII Division 1 UG-27 & UG-32 shell and head thickness & MAWP gate."""
    scalars = [
        design_pressure_psi,
        inside_diameter_in,
        nominal_thickness_in,
        allowable_stress_psi,
        joint_efficiency_e,
        corrosion_allowance_in,
        crown_radius_in,
        knuckle_radius_in,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            design_pressure_psi,
            inside_diameter_in,
            nominal_thickness_in,
            allowable_stress_psi,
            joint_efficiency_e,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_pressure_or_geometry"}
    if corrosion_allowance_in < 0 or crown_radius_in < 0 or knuckle_radius_in < 0:
        return {"pass": False, "reason": "negative_allowance_or_radius"}
    if not (0.0 < joint_efficiency_e <= 1.0):
        return {"pass": False, "reason": "invalid_joint_efficiency"}

    c_type = component_type.strip().lower().replace("-", "_")
    valid_types = {"cylindrical_shell", "2_to_1_ellipsoidal", "ellipsoidal", "hemispherical", "torispherical"}
    if c_type not in valid_types:
        return {"pass": False, "reason": "invalid_component_type"}

    # ASME Section VIII-1 thin-shell limit (t <= 0.5 * R or P <= 0.385 * S * E)
    radius_in = inside_diameter_in / 2.0
    if nominal_thickness_in >= radius_in:
        return {"pass": False, "reason": "thickness_exceeds_inside_radius"}
    if design_pressure_psi >= (allowable_stress_psi * joint_efficiency_e):
        return {"pass": False, "reason": "pressure_exceeds_allowable_stress"}

    t_eff_in = nominal_thickness_in - corrosion_allowance_in
    if t_eff_in <= 0:
        return {
            "pass": False,
            "reason": "corrosion_allowance_exceeds_nominal_thickness",
            "nominal_thickness_in": round(nominal_thickness_in, 4),
            "corrosion_allowance_in": round(corrosion_allowance_in, 4),
        }

    se = allowable_stress_psi * joint_efficiency_e
    p = design_pressure_psi
    r = radius_in
    d = inside_diameter_in

    if c_type == "cylindrical_shell":
        # UG-27(c)(1) circumferential stress (longitudinal joint)
        denom = se - 0.60 * p
        if denom <= 0:
            return {"pass": False, "reason": "invalid_stress_pressure_state"}
        t_press_in = (p * r) / denom
        mawp_psi = (se * t_eff_in) / (r + 0.60 * t_eff_in)
    elif c_type in ("2_to_1_ellipsoidal", "ellipsoidal"):
        # UG-32(d) 2:1 ellipsoidal head
        denom = 2.0 * se - 0.20 * p
        if denom <= 0:
            return {"pass": False, "reason": "invalid_stress_pressure_state"}
        t_press_in = (p * d) / denom
        mawp_psi = (2.0 * se * t_eff_in) / (d + 0.20 * t_eff_in)
    elif c_type == "hemispherical":
        # UG-32(f) hemispherical head
        denom = 2.0 * se - 0.20 * p
        if denom <= 0:
            return {"pass": False, "reason": "invalid_stress_pressure_state"}
        t_press_in = (p * r) / denom
        mawp_psi = (2.0 * se * t_eff_in) / (r + 0.20 * t_eff_in)
    else:  # torispherical
        # UG-32(e) ASME flanged and dished head
        l_crown = crown_radius_in if crown_radius_in > 0.0 else inside_diameter_in
        r_knuckle = knuckle_radius_in if knuckle_radius_in > 0.0 else (0.06 * inside_diameter_in)
        if r_knuckle <= 0:
            return {"pass": False, "reason": "invalid_knuckle_radius"}
        m_factor = 0.25 * (3.0 + math.sqrt(l_crown / r_knuckle))
        denom = 2.0 * se - 0.20 * p
        if denom <= 0:
            return {"pass": False, "reason": "invalid_stress_pressure_state"}
        t_press_in = (p * l_crown * m_factor) / denom
        mawp_psi = (2.0 * se * t_eff_in) / (m_factor * l_crown + 0.20 * t_eff_in)

    t_req_in = t_press_in + corrosion_allowance_in
    dcr = t_req_in / nominal_thickness_in
    passed = (dcr <= 1.0) and (design_pressure_psi <= mawp_psi)

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "t_required_in": round(t_req_in, 4),
        "t_pressure_in": round(t_press_in, 4),
        "t_effective_in": round(t_eff_in, 4),
        "mawp_psi": round(mawp_psi, 2),
        "component_type": c_type,
    }


def verify_infinite_slope_stability(
    slope_angle_deg: float,
    depth_z_ft: float,
    soil_cohesion_psf: float,
    soil_friction_angle_deg: float,
    soil_unit_weight_pcf: float,
    water_table_height_ft: float = 0.0,
    sat_unit_weight_pcf: float = 0.0,
    seismic_coeff_kh: float = 0.0,
    min_fs: float = 1.5,
) -> Dict[str, Any]:
    """Geotechnical infinite slope stability and steady seepage factor of safety check."""
    scalars = [
        slope_angle_deg,
        depth_z_ft,
        soil_cohesion_psf,
        soil_friction_angle_deg,
        soil_unit_weight_pcf,
        water_table_height_ft,
        sat_unit_weight_pcf,
        seismic_coeff_kh,
        min_fs,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if depth_z_ft <= 0 or soil_unit_weight_pcf <= 0 or min_fs <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_soil_property"}
    if not (0.0 < slope_angle_deg < 90.0):
        return {"pass": False, "reason": "invalid_slope_angle"}
    if not (0.0 <= soil_friction_angle_deg < 90.0):
        return {"pass": False, "reason": "invalid_friction_angle"}
    if soil_cohesion_psf < 0.0:
        return {"pass": False, "reason": "negative_cohesion"}
    if water_table_height_ft < 0.0:
        return {"pass": False, "reason": "negative_water_table_height"}
    if water_table_height_ft > depth_z_ft:
        return {"pass": False, "reason": "water_table_exceeds_depth"}
    if seismic_coeff_kh < 0.0:
        return {"pass": False, "reason": "negative_seismic_coefficient"}
    if sat_unit_weight_pcf < 0.0:
        return {"pass": False, "reason": "negative_sat_unit_weight"}

    gamma_moist = soil_unit_weight_pcf
    gamma_sat = sat_unit_weight_pcf if sat_unit_weight_pcf > 0.0 else gamma_moist
    gamma_w = 62.4

    hw = water_table_height_ft
    z = depth_z_ft
    beta_rad = math.radians(slope_angle_deg)
    phi_rad = math.radians(soil_friction_angle_deg)

    cos_b = math.cos(beta_rad)
    sin_b = math.sin(beta_rad)

    # Total weight per unit base area on slope failure plane
    total_soil_wt = gamma_moist * (z - hw) + gamma_sat * hw

    # Driving shear stress along the failure plane
    tau_driving = total_soil_wt * (sin_b * cos_b + seismic_coeff_kh * (cos_b ** 2))

    # Normal total stress on the failure plane
    sigma_total = total_soil_wt * (cos_b ** 2 - seismic_coeff_kh * sin_b * cos_b)

    # Pore water pressure for steady seepage parallel to the slope
    u_pore = gamma_w * hw * (cos_b ** 2)
    sigma_effective = max(0.0, sigma_total - u_pore)

    # Mohr-Coulomb available shear strength
    tau_resisting = soil_cohesion_psf + sigma_effective * math.tan(phi_rad)

    if tau_driving <= 0.0:
        fs = float("inf")
        dcr = 0.0
    else:
        fs = tau_resisting / tau_driving
        dcr = (min_fs / fs) if fs > 0.0 else float("inf")

    passed = (fs >= min_fs) and (dcr <= 1.0)
    return {
        "pass": passed,
        "factor_of_safety": round(fs, 4),
        "dcr": round(dcr, 4),
        "tau_resisting_psf": round(tau_resisting, 2),
        "tau_driving_psf": round(tau_driving, 2),
        "sigma_effective_psf": round(sigma_effective, 2),
        "pore_pressure_psf": round(u_pore, 2),
    }


def verify_water_hammer_surge(
    pipe_inner_diameter_in: float,
    wall_thickness_in: float,
    pipe_length_ft: float,
    valve_closure_time_s: float,
    static_pressure_psi: float,
    pipe_pressure_rating_psi: float,
    flow_velocity_fps: float = 0.0,
    flow_rate_gpm: float = 0.0,
    pipe_modulus_psi: float = 400000.0,
    fluid_bulk_modulus_psi: float = 300000.0,
    fluid_density_pcf: float = 62.4,
    surge_allowance_pct: float = 50.0,
    g_fps2: float = 32.174,
) -> Dict[str, Any]:
    """Joukowsky transient fluid water hammer surge pressure, acoustic wave speed, and pipe pressure rating check."""
    scalars = [
        pipe_inner_diameter_in,
        wall_thickness_in,
        pipe_length_ft,
        valve_closure_time_s,
        static_pressure_psi,
        pipe_pressure_rating_psi,
        flow_velocity_fps,
        flow_rate_gpm,
        pipe_modulus_psi,
        fluid_bulk_modulus_psi,
        fluid_density_pcf,
        surge_allowance_pct,
        g_fps2,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            pipe_inner_diameter_in,
            wall_thickness_in,
            pipe_length_ft,
            pipe_pressure_rating_psi,
            pipe_modulus_psi,
            fluid_bulk_modulus_psi,
            fluid_density_pcf,
            g_fps2,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if valve_closure_time_s <= 0:
        return {"pass": False, "reason": "non_positive_closure_time"}
    if static_pressure_psi < 0:
        return {"pass": False, "reason": "negative_static_pressure"}
    if surge_allowance_pct < 0:
        return {"pass": False, "reason": "negative_surge_allowance"}
    if flow_velocity_fps < 0 or flow_rate_gpm < 0:
        return {"pass": False, "reason": "negative_flow_input"}

    # Resolve flow velocity
    if flow_velocity_fps > 0.0:
        v = flow_velocity_fps
    elif flow_rate_gpm > 0.0:
        area_ft2 = (math.pi * ((pipe_inner_diameter_in / 12.0) ** 2)) / 4.0
        v = (flow_rate_gpm / 448.831169) / area_ft2
    else:
        return {"pass": False, "reason": "zero_flow_input"}

    # Unconfined acoustic wave speed in fluid a0 = sqrt(Kf / rho)
    rho = fluid_density_pcf / g_fps2
    k_psf = fluid_bulk_modulus_psi * 144.0
    a0 = math.sqrt(k_psf / rho)

    # Conduit elastic constraint effect: a = a0 / sqrt(1 + (Kf / E) * (D / t))
    elastic_term = 1.0 + (fluid_bulk_modulus_psi / pipe_modulus_psi) * (pipe_inner_diameter_in / wall_thickness_in)
    a = a0 / math.sqrt(elastic_term)

    # Critical closure time Tc = 2 * L / a
    t_c = (2.0 * pipe_length_ft) / a
    is_rapid = valve_closure_time_s <= t_c

    # Joukowsky maximum potential surge delta_P = rho * a * delta_V
    delta_p_joukowsky_psi = (fluid_density_pcf * a * v) / (144.0 * g_fps2)

    if is_rapid:
        closure_regime = "rapid"
        delta_p_surge_psi = delta_p_joukowsky_psi
    else:
        closure_regime = "slow"
        delta_p_surge_psi = delta_p_joukowsky_psi * (t_c / valve_closure_time_s)

    total_max_pressure_psi = static_pressure_psi + delta_p_surge_psi
    p_allowable_psi = pipe_pressure_rating_psi * (1.0 + surge_allowance_pct / 100.0)

    dcr = total_max_pressure_psi / p_allowable_psi if p_allowable_psi > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "total_pressure_psi": round(total_max_pressure_psi, 2),
        "surge_pressure_psi": round(delta_p_surge_psi, 2),
        "joukowsky_surge_psi": round(delta_p_joukowsky_psi, 2),
        "allowable_pressure_psi": round(p_allowable_psi, 2),
        "wave_speed_fps": round(a, 2),
        "critical_time_s": round(t_c, 4),
        "closure_regime": closure_regime,
        "flow_velocity_fps": round(v, 3),
    }


def verify_weir_discharge_capacity(
    head_ft: float,
    crest_length_ft: float = 0.0,
    weir_type: str = "suppressed_rectangular",
    required_discharge_cfs: float = 0.0,
    notch_angle_deg: float = 90.0,
    crest_height_ft: float = 0.0,
    tailwater_head_ft: float = 0.0,
    broad_crest_breadth_ft: float = 0.0,
    freeboard_ft: float = 0.0,
    min_freeboard_ft: float = 0.0,
) -> Dict[str, Any]:
    """Civil/Water Resources sharp- and broad-crested weir discharge capacity and Villemonte submergence gate."""
    scalars = [
        head_ft,
        crest_length_ft,
        required_discharge_cfs,
        notch_angle_deg,
        crest_height_ft,
        tailwater_head_ft,
        broad_crest_breadth_ft,
        freeboard_ft,
        min_freeboard_ft,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if head_ft <= 0:
        return {"pass": False, "reason": "non_positive_head"}
    if required_discharge_cfs < 0:
        return {"pass": False, "reason": "negative_required_flow"}
    if crest_height_ft < 0:
        return {"pass": False, "reason": "negative_crest_height"}
    if tailwater_head_ft < 0:
        return {"pass": False, "reason": "negative_tailwater_head"}
    if tailwater_head_ft >= head_ft:
        return {"pass": False, "reason": "tailwater_drowns_weir"}
    if freeboard_ft < 0 or min_freeboard_ft < 0:
        return {"pass": False, "reason": "negative_freeboard"}
    if broad_crest_breadth_ft < 0:
        return {"pass": False, "reason": "negative_breadth"}

    w_type = weir_type.strip().lower().replace("-", "_")
    valid_types = {
        "suppressed_rectangular",
        "contracted_rectangular",
        "v_notch",
        "cipolletti",
        "broad_crested",
    }
    if w_type not in valid_types:
        return {"pass": False, "reason": "invalid_weir_type"}

    if w_type in ("suppressed_rectangular", "contracted_rectangular", "cipolletti", "broad_crested"):
        if crest_length_ft <= 0:
            return {"pass": False, "reason": "non_positive_crest_length"}

    if w_type == "v_notch":
        if not (10.0 <= notch_angle_deg <= 150.0):
            return {"pass": False, "reason": "invalid_notch_angle"}

    g = 32.174
    # Unsubmerged ideal discharge calculation
    if w_type == "suppressed_rectangular":
        q_free = 3.33 * crest_length_ft * (head_ft ** 1.5)
        n_sub_exp = 1.5
        l_effective = crest_length_ft
    elif w_type == "contracted_rectangular":
        # Francis formula accounting for two end contractions
        l_effective = crest_length_ft - 0.20 * head_ft
        if l_effective <= 0:
            return {"pass": False, "reason": "head_exceeds_allowable_contraction"}
        q_free = 3.33 * l_effective * (head_ft ** 1.5)
        n_sub_exp = 1.5
    elif w_type == "v_notch":
        theta_rad = math.radians(notch_angle_deg)
        cd_v = 0.585
        q_free = (8.0 / 15.0) * cd_v * math.sqrt(2.0 * g) * math.tan(theta_rad / 2.0) * (head_ft ** 2.5)
        n_sub_exp = 2.5
        l_effective = 2.0 * head_ft * math.tan(theta_rad / 2.0)
    elif w_type == "cipolletti":
        # 4:1 side slope trapezoidal weir compensating for end contractions
        q_free = 3.367 * crest_length_ft * (head_ft ** 1.5)
        n_sub_exp = 1.5
        l_effective = crest_length_ft
    else:  # broad_crested
        c_bc = 0.385 * math.sqrt(2.0 * g)
        q_free = c_bc * crest_length_ft * (head_ft ** 1.5)
        n_sub_exp = 1.5
        l_effective = crest_length_ft

    # Villemonte submergence reduction factor
    if tailwater_head_ft > 0.0:
        s_ratio = tailwater_head_ft / head_ft
        k_sub = (1.0 - (s_ratio ** n_sub_exp)) ** 0.385
        q_capacity_cfs = q_free * max(k_sub, 0.0)
        is_submerged = True
    else:
        k_sub = 1.0
        q_capacity_cfs = q_free
        is_submerged = False

    if required_discharge_cfs > 0.0:
        dcr = required_discharge_cfs / q_capacity_cfs if q_capacity_cfs > 0 else float("inf")
        capacity_ok = dcr <= 1.0
    else:
        dcr = 0.0
        capacity_ok = q_capacity_cfs > 0.0

    freeboard_ok = (freeboard_ft >= min_freeboard_ft) if min_freeboard_ft > 0.0 else True
    passed = capacity_ok and freeboard_ok

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "q_capacity_cfs": round(q_capacity_cfs, 3),
        "q_unsubmerged_cfs": round(q_free, 3),
        "submergence_factor": round(k_sub, 4),
        "effective_length_ft": round(l_effective, 3),
        "is_submerged": is_submerged,
        "capacity_ok": capacity_ok,
        "freeboard_ok": freeboard_ok,
    }


def verify_open_channel_trapezoidal_flow(
    q_design_cfs: float,
    bottom_width_ft: float,
    flow_depth_ft: float,
    channel_slope_ft_per_ft: float,
    side_slope_z: float = 0.0,
    manning_n: float = 0.030,
    channel_total_depth_ft: float = 0.0,
    min_freeboard_ft: float = 0.0,
    max_velocity_fps: float = 8.0,
    min_velocity_fps: float = 1.5,
    permissible_shear_psf: float = 0.0,
    g_fps2: float = 32.174,
) -> Dict[str, Any]:
    """Civil/Water Resources trapezoidal and rectangular open channel conveyance, shear stress, and freeboard gate via Manning's equation."""
    scalars = [
        q_design_cfs,
        bottom_width_ft,
        flow_depth_ft,
        channel_slope_ft_per_ft,
        side_slope_z,
        manning_n,
        channel_total_depth_ft,
        min_freeboard_ft,
        max_velocity_fps,
        min_velocity_fps,
        permissible_shear_psf,
        g_fps2,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if q_design_cfs < 0:
        return {"pass": False, "reason": "negative_flow_input"}
    if (
        min(
            bottom_width_ft,
            flow_depth_ft,
            channel_slope_ft_per_ft,
            manning_n,
            max_velocity_fps,
            g_fps2,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_channel_slope"}
    if side_slope_z < 0:
        return {"pass": False, "reason": "negative_side_slope"}
    if min_velocity_fps < 0 or min_velocity_fps >= max_velocity_fps:
        return {"pass": False, "reason": "invalid_velocity_limits"}
    if min_freeboard_ft < 0 or permissible_shear_psf < 0:
        return {"pass": False, "reason": "negative_limit_parameter"}
    if channel_total_depth_ft > 0.0 and flow_depth_ft > channel_total_depth_ft:
        return {"pass": False, "reason": "flow_depth_exceeds_channel_depth"}

    # Cross-sectional hydraulic geometry for trapezoidal section (z:1 H:V)
    b = bottom_width_ft
    y = flow_depth_ft
    z = side_slope_z

    area_ft2 = (b + z * y) * y
    perimeter_ft = b + 2.0 * y * math.sqrt(1.0 + (z**2))
    hydraulic_radius_ft = area_ft2 / perimeter_ft
    top_width_ft = b + 2.0 * z * y
    hydraulic_depth_ft = area_ft2 / top_width_ft

    # Manning's equation for normal discharge capacity (US Customary units)
    v_normal_fps = (
        (1.486 / manning_n)
        * (hydraulic_radius_ft ** (2.0 / 3.0))
        * math.sqrt(channel_slope_ft_per_ft)
    )
    q_capacity_cfs = v_normal_fps * area_ft2

    # Froude number and flow regime classification
    froude = v_normal_fps / math.sqrt(g_fps2 * hydraulic_depth_ft)
    if abs(froude - 1.0) < 1e-4:
        flow_regime = "critical"
    elif froude < 1.0:
        flow_regime = "subcritical"
    else:
        flow_regime = "supercritical"

    # Boundary tractive shear stress on channel bed (psf): tau = gamma * y * S
    gamma_w = 62.4
    tau_bed_psf = gamma_w * y * channel_slope_ft_per_ft

    # Demand capacity evaluations
    flow_dcr = q_design_cfs / q_capacity_cfs if q_capacity_cfs > 0 else float("inf")
    capacity_ok = flow_dcr <= 1.0
    velocity_ok = min_velocity_fps <= v_normal_fps <= max_velocity_fps

    freeboard_ft = None
    freeboard_ok = True
    if channel_total_depth_ft > 0.0:
        freeboard_ft = channel_total_depth_ft - y
        if min_freeboard_ft > 0.0:
            freeboard_ok = freeboard_ft >= min_freeboard_ft

    shear_dcr = None
    shear_ok = True
    if permissible_shear_psf > 0.0:
        shear_dcr = tau_bed_psf / permissible_shear_psf
        shear_ok = shear_dcr <= 1.0

    dcr_candidates = [flow_dcr]
    if shear_dcr is not None:
        dcr_candidates.append(shear_dcr)
    gov_dcr = max(dcr_candidates)

    passed = capacity_ok and velocity_ok and freeboard_ok and shear_ok
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "q_capacity_cfs": round(q_capacity_cfs, 3),
        "v_normal_fps": round(v_normal_fps, 3),
        "flow_area_sqft": round(area_ft2, 3),
        "hydraulic_radius_ft": round(hydraulic_radius_ft, 3),
        "froude_number": round(froude, 3),
        "flow_regime": flow_regime,
        "shear_stress_psf": round(tau_bed_psf, 4),
        "freeboard_ft": round(freeboard_ft, 3) if freeboard_ft is not None else None,
        "capacity_ok": capacity_ok,
        "velocity_ok": velocity_ok,
        "freeboard_ok": freeboard_ok,
        "shear_ok": shear_ok,
    }


def verify_nfpa92_stairwell_pressurization(
    design_pressure_diff_in_wg: float,
    door_width_in: float,
    door_height_in: float,
    door_closer_force_lbf: float,
    fan_airflow_cfm: float,
    num_open_doors: int = 2,
    num_closed_doors: int = 10,
    open_door_velocity_fpm: float = 200.0,
    leakage_area_per_door_sqft: float = 0.25,
    latch_setback_in: float = 3.0,
    min_pressure_diff_in_wg: float = 0.10,
    max_door_opening_force_lbf: float = 30.0,
    safety_factor: float = 1.15,
) -> Dict[str, Any]:
    """NFPA 92 / IBC 909 & 1010 stairwell pressurization differential pressure, door opening force, and fan airflow gate."""
    scalars = [
        design_pressure_diff_in_wg,
        door_width_in,
        door_height_in,
        door_closer_force_lbf,
        fan_airflow_cfm,
        float(num_open_doors),
        float(num_closed_doors),
        open_door_velocity_fpm,
        leakage_area_per_door_sqft,
        latch_setback_in,
        min_pressure_diff_in_wg,
        max_door_opening_force_lbf,
        safety_factor,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            design_pressure_diff_in_wg,
            door_width_in,
            door_height_in,
            fan_airflow_cfm,
            min_pressure_diff_in_wg,
            max_door_opening_force_lbf,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_pressure"}
    if door_closer_force_lbf < 0 or latch_setback_in < 0 or open_door_velocity_fpm < 0 or leakage_area_per_door_sqft < 0:
        return {"pass": False, "reason": "negative_parameter_input"}
    if num_open_doors < 0 or num_closed_doors < 0:
        return {"pass": False, "reason": "negative_door_count"}
    if latch_setback_in >= door_width_in:
        return {"pass": False, "reason": "latch_setback_exceeds_door_width"}
    if door_closer_force_lbf >= max_door_opening_force_lbf:
        return {"pass": False, "reason": "closer_force_exceeds_allowable_opening_force"}
    if safety_factor < 1.0:
        return {"pass": False, "reason": "invalid_safety_factor"}

    # Door surface area in sqft
    door_area_sqft = (door_width_in * door_height_in) / 144.0

    # NFPA 92 Eq. 4.4.2.2 & IBC 909 door unlatching opening force (lbf)
    # 1.0 in. w.g. = 5.204 psf
    moment_arm_ratio = door_width_in / (2.0 * (door_width_in - latch_setback_in))
    f_pressure_lbf = 5.204 * door_area_sqft * design_pressure_diff_in_wg * moment_arm_ratio
    total_door_force_lbf = door_closer_force_lbf + f_pressure_lbf

    # Maximum allowable pressure differential based on 30 lbf door opening force limit (IBC 1010.1.3)
    allowable_pressure_force = max_door_opening_force_lbf - door_closer_force_lbf
    max_allowed_pressure_diff_in_wg = allowable_pressure_force / (5.204 * door_area_sqft * moment_arm_ratio)

    # Pressure differential compliance check
    pressure_ok = min_pressure_diff_in_wg <= design_pressure_diff_in_wg <= max_allowed_pressure_diff_in_wg
    door_force_ok = total_door_force_lbf <= max_door_opening_force_lbf

    # Airflow through open egress doors (NFPA 92 Section 4.4.3.4)
    q_open_doors_cfm = num_open_doors * door_area_sqft * open_door_velocity_fpm

    # Leakage airflow across closed doors at design pressure diff (Q = 2610 * A * sqrt(dP))
    q_leakage_doors_cfm = num_closed_doors * 2610.0 * leakage_area_per_door_sqft * math.sqrt(design_pressure_diff_in_wg)

    # Total required design airflow including safety/duct leakage allowance
    q_required_cfm = (q_open_doors_cfm + q_leakage_doors_cfm) * safety_factor

    airflow_dcr = q_required_cfm / fan_airflow_cfm if fan_airflow_cfm > 0 else float("inf")
    force_dcr = total_door_force_lbf / max_door_opening_force_lbf
    pressure_min_dcr = min_pressure_diff_in_wg / design_pressure_diff_in_wg

    gov_dcr = max(airflow_dcr, force_dcr, pressure_min_dcr)
    passed = pressure_ok and door_force_ok and (airflow_dcr <= 1.0)

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "total_door_force_lbf": round(total_door_force_lbf, 2),
        "max_allowed_door_force_lbf": round(max_door_opening_force_lbf, 2),
        "max_pressure_diff_in_wg": round(max_allowed_pressure_diff_in_wg, 4),
        "min_pressure_diff_in_wg": round(min_pressure_diff_in_wg, 4),
        "q_required_cfm": round(q_required_cfm, 1),
        "q_open_doors_cfm": round(q_open_doors_cfm, 1),
        "q_leakage_doors_cfm": round(q_leakage_doors_cfm, 1),
        "airflow_dcr": round(airflow_dcr, 4),
        "door_force_ok": door_force_ok,
        "pressure_ok": pressure_ok,
    }


def verify_cooling_tower_performance(
    cooling_water_flow_gpm: float,
    t_hot_water_in_f: float,
    t_cold_water_out_f: float,
    wet_bulb_temp_f: float,
    tower_capacity_tons: float,
    cycles_of_concentration: float = 4.0,
    min_approach_f: float = 5.0,
    drift_rate_pct: float = 0.005,
    max_drift_rate_pct: float = 0.010,
    max_makeup_gpm: float = 0.0,
) -> Dict[str, Any]:
    """ASHRAE Fundamentals & CTI cooling tower thermal heat rejection, approach, and water balance gate."""
    scalars = [
        cooling_water_flow_gpm,
        t_hot_water_in_f,
        t_cold_water_out_f,
        wet_bulb_temp_f,
        tower_capacity_tons,
        cycles_of_concentration,
        min_approach_f,
        drift_rate_pct,
        max_drift_rate_pct,
        max_makeup_gpm,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(cooling_water_flow_gpm, tower_capacity_tons, min_approach_f, max_drift_rate_pct)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_flow_or_capacity"}
    if drift_rate_pct < 0:
        return {"pass": False, "reason": "negative_drift_rate"}
    if max_makeup_gpm < 0:
        return {"pass": False, "reason": "negative_makeup_capacity"}
    if cycles_of_concentration <= 1.0:
        return {"pass": False, "reason": "invalid_cycles_of_concentration"}
    if t_hot_water_in_f <= t_cold_water_out_f:
        return {"pass": False, "reason": "hot_water_not_cooled"}
    if t_cold_water_out_f <= wet_bulb_temp_f:
        return {"pass": False, "reason": "cold_water_below_wet_bulb"}

    # Thermal range and approach to ambient wet-bulb
    range_f = t_hot_water_in_f - t_cold_water_out_f
    approach_f = t_cold_water_out_f - wet_bulb_temp_f
    approach_ok = approach_f >= min_approach_f

    # ASHRAE standard water heat rejection (500 = 8.33 lb/gal * 60 min/hr * 1.0 Btu/lb-F)
    q_rejected_btuh = 500.0 * cooling_water_flow_gpm * range_f
    # 1 Cooling tower ton = 15,000 Btu/hr (12,000 evaporator + 3,000 heat of compression)
    tons_required = q_rejected_btuh / 15000.0
    thermal_dcr = tons_required / tower_capacity_tons
    thermal_ok = thermal_dcr <= 1.0

    # ASHRAE/CTI water consumption equations
    # Evaporation rate: ~0.0008 gpm per gpm circulated per deg F range
    evaporation_gpm = 0.0008 * cooling_water_flow_gpm * range_f
    drift_gpm = cooling_water_flow_gpm * (drift_rate_pct / 100.0)
    drift_ok = drift_rate_pct <= max_drift_rate_pct

    # Dissolved solids mass balance for blowdown: B = E / (COC - 1) - D
    raw_blowdown = (evaporation_gpm / (cycles_of_concentration - 1.0)) - drift_gpm
    blowdown_gpm = max(raw_blowdown, 0.0)
    makeup_gpm = evaporation_gpm + drift_gpm + blowdown_gpm

    makeup_ok = True
    makeup_dcr = None
    if max_makeup_gpm > 0.0:
        makeup_dcr = makeup_gpm / max_makeup_gpm
        makeup_ok = makeup_dcr <= 1.0

    gov_dcr = max(thermal_dcr, makeup_dcr) if makeup_dcr is not None else thermal_dcr
    passed = thermal_ok and approach_ok and drift_ok and makeup_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "q_rejected_btuh": round(q_rejected_btuh, 2),
        "tons_required": round(tons_required, 2),
        "range_f": round(range_f, 2),
        "approach_f": round(approach_f, 2),
        "evaporation_gpm": round(evaporation_gpm, 3),
        "drift_gpm": round(drift_gpm, 4),
        "blowdown_gpm": round(blowdown_gpm, 3),
        "makeup_gpm": round(makeup_gpm, 3),
        "approach_ok": approach_ok,
        "thermal_ok": thermal_ok,
        "drift_ok": drift_ok,
        "makeup_ok": makeup_ok,
    }


def verify_ashrae_expansion_tank_sizing(
    system_volume_gal: float,
    t_initial_f: float,
    t_max_f: float,
    p_fill_psi: float,
    p_relief_psi: float,
    tank_volume_gal: float,
    pressure_margin_psi: float = 5.0,
    tank_acceptance_gal: float = 0.0,
    pipe_expansion_coeff: float = 6.5e-6,
    glycol_expansion_factor: float = 1.0,
    atmospheric_pressure_psi: float = 14.696,
) -> Dict[str, Any]:
    """ASHRAE Systems and Equipment Ch. 14 hydronic closed-loop expansion tank volume and acceptance check."""
    inputs = [
        system_volume_gal,
        t_initial_f,
        t_max_f,
        p_fill_psi,
        p_relief_psi,
        tank_volume_gal,
        pressure_margin_psi,
        tank_acceptance_gal,
        pipe_expansion_coeff,
        glycol_expansion_factor,
        atmospheric_pressure_psi,
    ]
    if any(math.isnan(val) for val in inputs):
        return {"pass": False, "reason": "nan_input"}
    if min(system_volume_gal, tank_volume_gal, atmospheric_pressure_psi) <= 0:
        return {"pass": False, "reason": "non_positive_volume_or_pressure"}
    if p_fill_psi < 0:
        return {"pass": False, "reason": "negative_fill_pressure"}
    if tank_acceptance_gal < 0 or pressure_margin_psi < 0 or pipe_expansion_coeff < 0:
        return {"pass": False, "reason": "negative_margin_or_allowance"}
    if glycol_expansion_factor < 1.0:
        return {"pass": False, "reason": "invalid_glycol_factor"}
    if t_initial_f < 32.0 or t_max_f > 250.0:
        return {"pass": False, "reason": "temperature_out_of_water_liquid_range"}
    if t_max_f <= t_initial_f:
        return {"pass": False, "reason": "max_temp_not_greater_than_initial"}

    p_max_psi = p_relief_psi - pressure_margin_psi
    if p_max_psi <= p_fill_psi:
        return {"pass": False, "reason": "max_pressure_not_greater_than_fill"}

    def calc_water_density(t_f: float) -> float:
        t_c = (t_f - 32.0) * (5.0 / 9.0)
        # Tilton & Taylor formulation for pure water density (kg/m3 to lb/ft3)
        num = ((t_c - 3.98) ** 2) * (t_c + 288.9414)
        den = 508929.2 * (t_c + 68.12963)
        rho_kg_m3 = 999.972 * (1.0 - (num / den))
        return rho_kg_m3 * 0.06242796

    rho_init = calc_water_density(t_initial_f)
    rho_max = calc_water_density(t_max_f)
    water_expansion_ratio = (rho_init / rho_max) - 1.0
    e_liquid = water_expansion_ratio * glycol_expansion_factor

    # Net expansion ratio accounting for 3D volumetric thermal expansion of piping
    delta_t = t_max_f - t_initial_f
    e_net = max(e_liquid - 3.0 * pipe_expansion_coeff * delta_t, 0.0)
    v_e_gal = system_volume_gal * e_net

    # Expansion tank pressure factor (PF)
    p1_abs = p_fill_psi + atmospheric_pressure_psi
    p2_abs = p_max_psi + atmospheric_pressure_psi
    pf = (p2_abs - p1_abs) / p2_abs

    v_t_req_gal = (v_e_gal / pf) if pf > 0 else float("inf")
    tank_vol_dcr = v_t_req_gal / tank_volume_gal if tank_volume_gal > 0 else float("inf")
    vol_ok = tank_vol_dcr <= 1.0

    acceptance_dcr = None
    acceptance_ok = True
    if tank_acceptance_gal > 0.0:
        acceptance_dcr = v_e_gal / tank_acceptance_gal
        acceptance_ok = acceptance_dcr <= 1.0

    gov_dcr = max(tank_vol_dcr, acceptance_dcr) if acceptance_dcr is not None else tank_vol_dcr
    passed = vol_ok and acceptance_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "v_tank_required_gal": round(v_t_req_gal, 2),
        "acceptance_volume_gal": round(v_e_gal, 2),
        "pressure_factor": round(pf, 4),
        "liquid_expansion_pct": round(e_liquid * 100.0, 3),
        "net_expansion_pct": round(e_net * 100.0, 3),
        "p_max_psi": round(p_max_psi, 2),
        "volume_ok": vol_ok,
        "acceptance_ok": acceptance_ok,
    }


def verify_nec_dwelling_service_load(
    service_rating_a: float,
    dwelling_area_sqft: float,
    appliance_ratings_va: Sequence[float],
    cooking_appliances_va: float = 0.0,
    clothes_dryer_va: float = 0.0,
    ac_cooling_va: float = 0.0,
    electric_heating_va: float = 0.0,
    small_appliance_branch_circuits: int = 2,
    laundry_branch_circuits: int = 1,
    service_voltage_v: float = 240.0,
    heating_demand_factor: float = 1.0,
    min_service_rating_a: float = 100.0,
) -> Dict[str, Any]:
    """NEC Article 220.82 Optional Method single-family dwelling service load calculation and service rating gate."""
    scalars = [
        service_rating_a,
        dwelling_area_sqft,
        cooking_appliances_va,
        clothes_dryer_va,
        ac_cooling_va,
        electric_heating_va,
        float(small_appliance_branch_circuits),
        float(laundry_branch_circuits),
        service_voltage_v,
        heating_demand_factor,
        min_service_rating_a,
    ]
    if any(math.isnan(val) for val in scalars) or any(math.isnan(a) for a in appliance_ratings_va):
        return {"pass": False, "reason": "nan_input"}
    if min(service_rating_a, dwelling_area_sqft, service_voltage_v, min_service_rating_a) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_rating"}
    if (
        cooking_appliances_va < 0
        or clothes_dryer_va < 0
        or ac_cooling_va < 0
        or electric_heating_va < 0
        or any(a < 0 for a in appliance_ratings_va)
    ):
        return {"pass": False, "reason": "negative_load_input"}
    if small_appliance_branch_circuits < 2:
        return {"pass": False, "reason": "insufficient_small_appliance_circuits"}
    if laundry_branch_circuits < 0:
        return {"pass": False, "reason": "negative_circuit_count"}
    if not (0.0 < heating_demand_factor <= 1.0):
        return {"pass": False, "reason": "invalid_heating_demand_factor"}

    # NEC 220.82(B) General Loads
    # 3 VA/sqft general lighting + receptacles (220.82(B)(1))
    general_lighting_va = 3.0 * dwelling_area_sqft
    # 1500 VA for each small-appliance and laundry circuit (220.82(B)(2))
    small_appl_va = 1500.0 * small_appliance_branch_circuits
    laundry_va = 1500.0 * laundry_branch_circuits
    fixed_appl_va = sum(appliance_ratings_va)

    va_gen_connected = (
        general_lighting_va
        + small_appl_va
        + laundry_va
        + fixed_appl_va
        + cooking_appliances_va
        + clothes_dryer_va
    )

    # 100% of first 10 kVA, 40% of remainder
    if va_gen_connected <= 10000.0:
        va_gen_demand = va_gen_connected
    else:
        va_gen_demand = 10000.0 + 0.40 * (va_gen_connected - 10000.0)

    # NEC 220.82(C) HVAC Loads (largest of non-coincident cooling vs heating)
    cooling_va = ac_cooling_va
    heating_va = electric_heating_va * heating_demand_factor

    if cooling_va == 0.0 and heating_va == 0.0:
        va_hvac_demand = 0.0
        gov_hvac = "none"
    elif cooling_va >= heating_va:
        va_hvac_demand = cooling_va
        gov_hvac = "cooling"
    else:
        va_hvac_demand = heating_va
        gov_hvac = "heating"

    va_total = va_gen_demand + va_hvac_demand
    i_calc = va_total / service_voltage_v
    dcr = i_calc / service_rating_a

    # NEC 230.79(C) minimum 100A service rating for single-family dwelling
    service_size_ok = service_rating_a >= min_service_rating_a
    capacity_ok = dcr <= 1.0
    passed = capacity_ok and service_size_ok

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "calculated_current_a": round(i_calc, 2),
        "total_demand_va": round(va_total, 2),
        "general_connected_va": round(va_gen_connected, 2),
        "general_demand_va": round(va_gen_demand, 2),
        "hvac_demand_va": round(va_hvac_demand, 2),
        "governing_hvac": gov_hvac,
        "capacity_ok": capacity_ok,
        "service_size_ok": service_size_ok,
    }


def verify_aashto_pavement_structural_number(
    design_esals: float,
    d1_surface_in: float,
    d2_base_in: float,
    d3_subbase_in: float = 0.0,
    a1_coeff: float = 0.44,
    a2_coeff: float = 0.14,
    a3_coeff: float = 0.11,
    m2_drainage: float = 1.0,
    m3_drainage: float = 1.0,
    subgrade_resilient_modulus_psi: float = 5000.0,
    reliability_pct: float = 90.0,
    overall_std_dev: float = 0.45,
    initial_psi: float = 4.2,
    terminal_psi: float = 2.5,
    zr_override: float = 0.0,
) -> Dict[str, Any]:
    """AASHTO (1993) flexible pavement structural number, layer thickness, and ESAL capacity gate."""
    scalars = [
        design_esals,
        d1_surface_in,
        d2_base_in,
        d3_subbase_in,
        a1_coeff,
        a2_coeff,
        a3_coeff,
        m2_drainage,
        m3_drainage,
        subgrade_resilient_modulus_psi,
        reliability_pct,
        overall_std_dev,
        initial_psi,
        terminal_psi,
        zr_override,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            design_esals,
            d1_surface_in,
            d2_base_in,
            a1_coeff,
            a2_coeff,
            m2_drainage,
            subgrade_resilient_modulus_psi,
            overall_std_dev,
            terminal_psi,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_traffic"}
    if d3_subbase_in < 0 or a3_coeff < 0 or m3_drainage < 0:
        return {"pass": False, "reason": "negative_subbase_parameter"}
    if initial_psi <= terminal_psi:
        return {"pass": False, "reason": "initial_serviceability_not_above_terminal"}
    if initial_psi > 5.0 or terminal_psi < 1.0:
        return {"pass": False, "reason": "serviceability_index_out_of_bounds"}

    # AASHTO Table 5.1 minimum layer thickness checks based on ESAL traffic level
    if design_esals <= 50000.0:
        min_d1, min_d2 = 1.0, 4.0
    elif design_esals <= 150000.0:
        min_d1, min_d2 = 2.0, 4.0
    elif design_esals <= 500000.0:
        min_d1, min_d2 = 2.5, 4.0
    elif design_esals <= 2000000.0:
        min_d1, min_d2 = 3.0, 6.0
    elif design_esals <= 7000000.0:
        min_d1, min_d2 = 3.5, 6.0
    else:
        min_d1, min_d2 = 4.0, 6.0

    thickness_ok = (d1_surface_in >= min_d1) and (d2_base_in >= min_d2)

    # AASHTO standard normal deviate (Z_R) lookup
    zr_table = {
        50.0: 0.000,
        70.0: -0.524,
        75.0: -0.674,
        80.0: -0.841,
        85.0: -1.036,
        90.0: -1.282,
        95.0: -1.645,
        98.0: -2.054,
        99.0: -2.326,
        99.9: -3.090,
    }
    if zr_override < 0.0:
        zr = zr_override
    elif reliability_pct in zr_table:
        zr = zr_table[reliability_pct]
    elif 50.0 <= reliability_pct <= 99.9:
        # Piecewise interpolation for unlisted reliability values
        sorted_keys = sorted(zr_table.keys())
        idx = next(i for i, k in enumerate(sorted_keys) if k >= reliability_pct)
        k_lower, k_upper = sorted_keys[idx - 1], sorted_keys[idx]
        frac = (reliability_pct - k_lower) / (k_upper - k_lower)
        zr = zr_table[k_lower] + frac * (zr_table[k_upper] - zr_table[k_lower])
    else:
        return {"pass": False, "reason": "invalid_reliability_percentage"}

    # AASHTO structural number formulation: SN = a1*D1 + a2*D2*m2 + a3*D3*m3
    sn = (
        (a1_coeff * d1_surface_in)
        + (a2_coeff * d2_base_in * m2_drainage)
        + (a3_coeff * d3_subbase_in * m3_drainage)
    )
    if sn <= 0:
        return {"pass": False, "reason": "zero_or_negative_structural_number"}

    # AASHTO 1993 flexible pavement design equation for allowable 18-kip ESALs (W18)
    delta_psi = initial_psi - terminal_psi
    term_sn = sn + 1.0
    term_psi_num = math.log10(delta_psi / (4.2 - 1.5))
    term_psi_den = 0.40 + (1094.0 / (term_sn ** 5.19))
    psi_term = term_psi_num / term_psi_den

    log_w18 = (
        zr * overall_std_dev
        + 9.36 * math.log10(term_sn)
        - 0.20
        + psi_term
        + 2.32 * math.log10(subgrade_resilient_modulus_psi)
        - 8.07
    )
    w18_allowable = 10.0 ** log_w18
    esal_dcr = design_esals / w18_allowable if w18_allowable > 0 else float("inf")
    traffic_ok = esal_dcr <= 1.0

    passed = traffic_ok and thickness_ok
    return {
        "pass": passed,
        "dcr": round(esal_dcr, 4),
        "structural_number": round(sn, 3),
        "w18_allowable_esals": round(w18_allowable, 1),
        "design_esals": round(design_esals, 1),
        "min_surface_thickness_in": min_d1,
        "min_base_thickness_in": min_d2,
        "traffic_ok": traffic_ok,
        "thickness_ok": thickness_ok,
    }


def verify_nfpa14_standpipe_demand(
    num_standpipes: int,
    available_flow_gpm: float,
    available_pressure_psi: float,
    most_remote_elevation_ft: float,
    piping_friction_loss_psi: float,
    system_class: str = "class_1",
    is_fully_sprinklered: bool = True,
    min_residual_pressure_override_psi: float = 0.0,
    max_system_pressure_psi: float = 175.0,
) -> Dict[str, Any]:
    """NFPA 14 (2019/2024) & IBC 905 standpipe hydraulic demand, residual pressure, and PRV check."""
    scalars = [
        float(num_standpipes),
        available_flow_gpm,
        available_pressure_psi,
        most_remote_elevation_ft,
        piping_friction_loss_psi,
        min_residual_pressure_override_psi,
        max_system_pressure_psi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if num_standpipes <= 0:
        return {"pass": False, "reason": "non_positive_standpipe_count"}
    if available_flow_gpm <= 0 or available_pressure_psi <= 0 or max_system_pressure_psi <= 0:
        return {"pass": False, "reason": "non_positive_supply_flow_or_pressure"}
    if most_remote_elevation_ft < 0 or piping_friction_loss_psi < 0:
        return {"pass": False, "reason": "negative_elevation_or_friction_loss"}
    if min_residual_pressure_override_psi < 0:
        return {"pass": False, "reason": "negative_residual_pressure"}

    cls_clean = system_class.strip().lower().replace(" ", "_").replace("-", "_")
    class_map = {
        "class_1": "class_1",
        "class_i": "class_1",
        "1": "class_1",
        "i": "class_1",
        "class_2": "class_2",
        "class_ii": "class_2",
        "2": "class_2",
        "ii": "class_2",
        "class_3": "class_3",
        "class_iii": "class_3",
        "3": "class_3",
        "iii": "class_3",
    }
    if cls_clean not in class_map:
        return {"pass": False, "reason": "invalid_system_class"}
    norm_class = class_map[cls_clean]

    # NFPA 14 Section 7.10.1 Minimum Flow Rates
    if norm_class in ("class_1", "class_3"):
        base_flow = 500.0
        add_flow = (num_standpipes - 1) * 250.0
        max_cap = 1000.0 if is_fully_sprinklered else 1250.0
        q_demand_gpm = min(base_flow + add_flow, max_cap)
        code_min_residual = 100.0
    else:  # class_2
        q_demand_gpm = 100.0
        code_min_residual = 65.0

    p_residual = (
        min_residual_pressure_override_psi
        if min_residual_pressure_override_psi > 0.0
        else code_min_residual
    )

    # Static elevation pressure (0.4335 psi/ft of water)
    p_elevation = 0.4335 * most_remote_elevation_ft
    p_demand = p_residual + p_elevation + piping_friction_loss_psi

    flow_dcr = q_demand_gpm / available_flow_gpm
    press_dcr = p_demand / available_pressure_psi
    gov_dcr = max(flow_dcr, press_dcr)

    # NFPA 14 Section 7.8.3 pressure regulating device requirement
    prv_required = p_demand > max_system_pressure_psi

    flow_ok = flow_dcr <= 1.0
    press_ok = press_dcr <= 1.0
    passed = flow_ok and press_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "flow_dcr": round(flow_dcr, 4),
        "pressure_dcr": round(press_dcr, 4),
        "q_demand_gpm": round(q_demand_gpm, 1),
        "p_demand_psi": round(p_demand, 2),
        "p_elevation_psi": round(p_elevation, 2),
        "min_residual_pressure_psi": round(p_residual, 1),
        "prv_required": prv_required,
        "system_class": norm_class,
        "flow_ok": flow_ok,
        "pressure_ok": press_ok,
    }


def verify_ashrae_fan_power_limitation(
    design_airflow_cfm: float,
    motor_nameplate_hp: float,
    fan_brake_horsepower: float = 0.0,
    system_type: str = "variable_volume",
    compliance_path: str = "nameplate_hp",
    pressure_drop_credit_in_wg: float = 0.0,
    duct_static_pressure_in_wg: float = 0.0,
    fan_efficiency_pct: float = 0.0,
) -> Dict[str, Any]:
    """ASHRAE 90.1 Section 6.5.3.1 Fan System Power Limitation and motor nameplate sizing gate."""
    scalars = [
        design_airflow_cfm,
        motor_nameplate_hp,
        fan_brake_horsepower,
        pressure_drop_credit_in_wg,
        duct_static_pressure_in_wg,
        fan_efficiency_pct,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if design_airflow_cfm <= 0 or motor_nameplate_hp <= 0:
        return {"pass": False, "reason": "non_positive_airflow_or_motor_hp"}
    if pressure_drop_credit_in_wg < 0:
        return {"pass": False, "reason": "negative_pressure_credit"}
    if duct_static_pressure_in_wg < 0:
        return {"pass": False, "reason": "negative_static_pressure"}
    if fan_efficiency_pct < 0 or fan_efficiency_pct > 100.0:
        return {"pass": False, "reason": "invalid_fan_efficiency"}

    # Resolve Fan Brake Horsepower (BHP)
    if fan_brake_horsepower > 0.0:
        eff_bhp = fan_brake_horsepower
    elif duct_static_pressure_in_wg > 0.0 and fan_efficiency_pct > 0.0:
        # Air Horsepower AHP = (CFM * TSP) / 6356; BHP = AHP / (eta / 100)
        eff_bhp = (design_airflow_cfm * duct_static_pressure_in_wg) / (
            6356.0 * (fan_efficiency_pct / 100.0)
        )
    else:
        return {"pass": False, "reason": "missing_fan_bhp_or_pressure_efficiency"}

    sys_clean = system_type.strip().lower().replace(" ", "_")
    type_map = {
        "variable_volume": "vav",
        "vav": "vav",
        "constant_volume": "cv",
        "cv": "cv",
    }
    if sys_clean not in type_map:
        return {"pass": False, "reason": "invalid_system_type"}
    norm_type = type_map[sys_clean]

    path_clean = compliance_path.strip().lower().replace(" ", "_")
    path_map = {
        "nameplate_hp": "nameplate_hp",
        "nameplate": "nameplate_hp",
        "fan_bhp": "fan_bhp",
        "bhp": "fan_bhp",
    }
    if path_clean not in path_map:
        return {"pass": False, "reason": "invalid_compliance_path"}
    norm_path = path_map[path_clean]

    # ASHRAE 90.1 Table 6.5.3.1-1 Pressure drop adjustment credit A = (CFM * PD) / 4131
    credit_a = (design_airflow_cfm * pressure_drop_credit_in_wg) / 4131.0

    # Table 6.5.3.1-1 maximum allowable fan system power coefficients
    if norm_path == "nameplate_hp":
        coeff = 0.00130 if norm_type == "vav" else 0.00094
        allowable_limit = (design_airflow_cfm * coeff) + credit_a
        demand_val = motor_nameplate_hp
    else:  # fan_bhp
        coeff = 0.00110 if norm_type == "vav" else 0.00082
        allowable_limit = (design_airflow_cfm * coeff) + credit_a
        demand_val = eff_bhp

    power_dcr = demand_val / allowable_limit if allowable_limit > 0 else float("inf")
    power_limit_ok = power_dcr <= 1.0

    # Motor continuous duty check: nameplate HP must equal or exceed fan BHP
    motor_capacity_ok = motor_nameplate_hp >= eff_bhp
    passed = power_limit_ok and motor_capacity_ok

    return {
        "pass": passed,
        "dcr": round(power_dcr, 4),
        "allowable_limit_hp": round(allowable_limit, 3),
        "fan_bhp": round(eff_bhp, 3),
        "motor_nameplate_hp": round(motor_nameplate_hp, 2),
        "pressure_credit_hp": round(credit_a, 3),
        "compliance_path": norm_path,
        "system_type": norm_type,
        "power_limit_ok": power_limit_ok,
        "motor_capacity_ok": motor_capacity_ok,
    }


def verify_aci_shear_friction(
    vu_kips: float,
    avf_in2: float,
    ac_in2: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    interface_type: str = "monolithic",
    concrete_type: str = "normalweight",
    compression_force_pc_kips: float = 0.0,
    tension_force_nu_kips: float = 0.0,
    reinforcement_angle_deg: float = 90.0,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """ACI 318-19 Section 22.9 LRFD shear friction design check across concrete interfaces."""
    scalars = [
        vu_kips,
        avf_in2,
        ac_in2,
        fc_psi,
        fy_psi,
        compression_force_pc_kips,
        tension_force_nu_kips,
        reinforcement_angle_deg,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_shear_load"}
    if avf_in2 < 0:
        return {"pass": False, "reason": "negative_reinforcement_area"}
    if min(ac_in2, fc_psi, fy_psi, phi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if compression_force_pc_kips < 0 or tension_force_nu_kips < 0:
        return {"pass": False, "reason": "negative_clamping_force"}
    if not (0.0 < reinforcement_angle_deg <= 90.0):
        return {"pass": False, "reason": "invalid_reinforcement_angle"}

    # Interface roughness classification and coefficient of friction mu (ACI 318-19 Table 22.9.4.2)
    itype = interface_type.strip().lower().replace(" ", "_").replace("-", "_")
    valid_interfaces = {
        "monolithic",
        "roughened",
        "hardened_roughened",
        "smooth",
        "non_roughened",
        "structural_steel",
    }
    if itype not in valid_interfaces:
        return {"pass": False, "reason": "invalid_interface_type"}

    # Concrete density modification factor lambda (ACI 318-19 Table 19.2.4.2)
    ctype = concrete_type.strip().lower().replace(" ", "_").replace("-", "_")
    lambda_map: Dict[str, float] = {
        "normalweight": 1.0,
        "sand_lightweight": 0.85,
        "all_lightweight": 0.75,
    }
    if ctype not in lambda_map:
        return {"pass": False, "reason": "invalid_concrete_type"}
    lambda_factor = lambda_map[ctype]

    if itype == "monolithic":
        mu = 1.4 * lambda_factor
    elif itype in ("roughened", "hardened_roughened"):
        mu = 1.0 * lambda_factor
    elif itype in ("smooth", "non_roughened"):
        mu = 0.6 * lambda_factor
    else:  # structural_steel
        mu = 0.7 * lambda_factor

    # ACI 318-19 Table 20.2.2.4b: maximum fy for shear friction design is 60,000 psi
    fy_effective_psi = min(fy_psi, 60000.0)

    # ACI 318-19 Section 22.9.4.4: net tension across shear plane reduces effective clamping steel force
    nu_lb = tension_force_nu_kips * 1000.0
    total_steel_cap_lb = avf_in2 * fy_effective_psi
    if nu_lb > total_steel_cap_lb:
        return {
            "pass": False,
            "reason": "tension_exceeds_reinforcement_capacity",
            "nu_kips": round(tension_force_nu_kips, 2),
            "steel_capacity_kips": round(total_steel_cap_lb / 1000.0, 2),
        }

    clamping_steel_lb = total_steel_cap_lb - nu_lb
    compression_lb = compression_force_pc_kips * 1000.0
    alpha_rad = math.radians(reinforcement_angle_deg)

    # ACI 318-19 Eq. 22.9.4.2 & 22.9.4.3 nominal shear friction strength
    if abs(reinforcement_angle_deg - 90.0) < 1e-4:
        vn_shear_friction_lb = (mu * clamping_steel_lb) + (mu * compression_lb)
    else:
        angle_multiplier = (mu * math.sin(alpha_rad)) + math.cos(alpha_rad)
        vn_shear_friction_lb = (clamping_steel_lb * angle_multiplier) + (mu * compression_lb)

    # ACI 318-19 Table 22.9.4.4 nominal shear strength upper bounds (Vn_max)
    if ctype == "normalweight" and itype in ("monolithic", "roughened", "hardened_roughened"):
        vn_max_lb = min(
            0.20 * fc_psi * ac_in2,
            (480.0 + 0.08 * fc_psi) * ac_in2,
            1600.0 * ac_in2,
        )
    else:
        vn_max_lb = min(
            0.20 * fc_psi * ac_in2,
            800.0 * ac_in2,
        )

    vn_lb = min(vn_shear_friction_lb, vn_max_lb)
    vn_kips = vn_lb / 1000.0
    phi_vn_kips = phi * vn_kips

    dcr = vu_kips / phi_vn_kips if phi_vn_kips > 0 else float("inf")
    limit_state = "shear_friction" if vn_shear_friction_lb <= vn_max_lb else "section_compression_limit"
    passed = (dcr <= 1.0) and (vn_lb > 0.0)

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_vn_kips": round(phi_vn_kips, 2),
        "vn_kips": round(vn_kips, 2),
        "vn_shear_friction_kips": round(vn_shear_friction_lb / 1000.0, 2),
        "vn_max_kips": round(vn_max_lb / 1000.0, 2),
        "friction_coeff_mu": round(mu, 3),
        "governing_limit_state": limit_state,
    }


def verify_aci_beam_deflection(
    span_ft: float,
    b_in: float,
    h_in: float,
    d_in: float,
    as_in2: float,
    fc_psi: float,
    m_dead_kip_ft: float,
    m_live_kip_ft: float,
    as_prime_in2: float = 0.0,
    time_factor_xi: float = 2.0,
    live_deflection_limit_ratio: float = 360.0,
    total_deflection_limit_ratio: float = 240.0,
    lambda_factor: float = 1.0,
    es_psi: float = 29000000.0,
) -> Dict[str, Any]:
    """ACI 318-19 Section 24.2 immediate and long-term deflection check for reinforced concrete beams."""
    scalars = [
        span_ft,
        b_in,
        h_in,
        d_in,
        as_in2,
        fc_psi,
        m_dead_kip_ft,
        m_live_kip_ft,
        as_prime_in2,
        time_factor_xi,
        live_deflection_limit_ratio,
        total_deflection_limit_ratio,
        lambda_factor,
        es_psi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            span_ft,
            b_in,
            h_in,
            d_in,
            as_in2,
            fc_psi,
            live_deflection_limit_ratio,
            total_deflection_limit_ratio,
            lambda_factor,
            es_psi,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if m_dead_kip_ft < 0 or m_live_kip_ft < 0:
        return {"pass": False, "reason": "negative_moment_input"}
    if as_prime_in2 < 0 or time_factor_xi < 0:
        return {"pass": False, "reason": "negative_steel_or_time_factor"}
    if d_in >= h_in:
        return {"pass": False, "reason": "effective_depth_exceeds_height"}

    # Modulus of elasticity of concrete (ACI 318-19 Section 19.2.2.1) and modular ratio n
    ec_psi = 57000.0 * math.sqrt(fc_psi)
    ec_ksi = ec_psi / 1000.0
    n = es_psi / ec_psi

    # Gross section properties and cracking moment Mcr (ACI 318-19 Section 24.2.3.5)
    ig_in4 = (b_in * (h_in**3)) / 12.0
    yt_in = h_in / 2.0
    fr_psi = 7.5 * lambda_factor * math.sqrt(fc_psi)
    m_cr_kip_in = (fr_psi * ig_in4 / yt_in) / 1000.0
    m_cr_kip_ft = m_cr_kip_in / 12.0
    m_23cr_kip_in = (2.0 / 3.0) * m_cr_kip_in

    # Cracked transformed section neutral axis depth c (in)
    d_prime_in = max(h_in - d_in, 1.5)
    a_quad = 0.5 * b_in
    b_quad = n * as_in2 + (n - 1.0) * as_prime_in2
    c_quad = -(n * as_in2 * d_in + (n - 1.0) * as_prime_in2 * d_prime_in)
    disc = (b_quad**2) - (4.0 * a_quad * c_quad)
    c_in = (-b_quad + math.sqrt(disc)) / (2.0 * a_quad)

    # Cracked moment of inertia Icr
    icr_in4 = (
        (b_in * (c_in**3)) / 3.0
        + n * as_in2 * ((d_in - c_in) ** 2)
        + (n - 1.0) * as_prime_in2 * ((c_in - d_prime_in) ** 2)
    )

    def calc_ie(m_kip_in: float) -> float:
        if m_kip_in <= m_23cr_kip_in:
            return ig_in4
        ratio_sq = (m_23cr_kip_in / m_kip_in) ** 2
        denom = 1.0 - ratio_sq * (1.0 - (icr_in4 / ig_in4))
        if denom <= 0:
            return ig_in4
        return min(icr_in4 / denom, ig_in4)

    span_in = span_ft * 12.0
    m_dead_in = m_dead_kip_ft * 12.0
    m_total_in = (m_dead_kip_ft + m_live_kip_ft) * 12.0

    # Immediate dead load deflection
    if m_dead_in > 0:
        ie_dead = calc_ie(m_dead_in)
        delta_dead_in = (5.0 * m_dead_in * (span_in**2)) / (48.0 * ec_ksi * ie_dead)
    else:
        ie_dead = ig_in4
        delta_dead_in = 0.0

    # Immediate total (dead + live) deflection
    if m_total_in > 0:
        ie_total = calc_ie(m_total_in)
        delta_total_imm_in = (5.0 * m_total_in * (span_in**2)) / (48.0 * ec_ksi * ie_total)
    else:
        ie_total = ig_in4
        delta_total_imm_in = 0.0

    # ACI 318-19 Section 24.2.3.1 immediate live load deflection
    delta_live_in = max(0.0, delta_total_imm_in - delta_dead_in)

    # Long-term creep and shrinkage factor lambda_delta (ACI 318-19 Section 24.2.4.1.1)
    rho_prime = as_prime_in2 / (b_in * d_in)
    lambda_delta = time_factor_xi / (1.0 + 50.0 * rho_prime)

    # Long-term total deflection (immediate live + sustained dead creep)
    delta_sustained_creep_in = lambda_delta * delta_dead_in
    delta_total_in = delta_live_in + delta_dead_in + delta_sustained_creep_in

    # Allowable deflection thresholds (ACI 318-19 Table 24.2.2)
    allowable_live_in = span_in / live_deflection_limit_ratio
    allowable_total_in = span_in / total_deflection_limit_ratio

    live_dcr = delta_live_in / allowable_live_in if allowable_live_in > 0 else float("inf")
    total_dcr = delta_total_in / allowable_total_in if allowable_total_in > 0 else float("inf")
    gov_dcr = max(live_dcr, total_dcr)

    passed = (live_dcr <= 1.0) and (total_dcr <= 1.0)
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "live_dcr": round(live_dcr, 4),
        "total_dcr": round(total_dcr, 4),
        "delta_live_in": round(delta_live_in, 4),
        "delta_total_in": round(delta_total_in, 4),
        "delta_dead_in": round(delta_dead_in, 4),
        "allowable_live_in": round(allowable_live_in, 3),
        "allowable_total_in": round(allowable_total_in, 3),
        "ie_effective_in4": round(ie_total, 1),
        "ig_gross_in4": round(ig_in4, 1),
        "icr_cracked_in4": round(icr_in4, 1),
        "m_cr_kip_ft": round(m_cr_kip_ft, 2),
        "lambda_delta": round(lambda_delta, 3),
        "live_deflection_ok": live_dcr <= 1.0,
        "total_deflection_ok": total_dcr <= 1.0,
    }


def verify_aci_corbel_capacity(
    vu_kips: float,
    shear_span_a_in: float,
    b_in: float,
    d_in: float,
    h_in: float,
    as_provided_in2: float,
    ah_provided_in2: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    nuc_kips: float = 0.0,
    min_nuc_ratio: float = 0.20,
    friction_coeff_mu: float = 1.4,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """ACI 318-19 Section 16.5 LRFD design capacity check for reinforced concrete corbels and brackets."""
    scalars = [
        vu_kips,
        shear_span_a_in,
        b_in,
        d_in,
        h_in,
        as_provided_in2,
        ah_provided_in2,
        fc_psi,
        fy_psi,
        nuc_kips,
        min_nuc_ratio,
        friction_coeff_mu,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(b_in, d_in, h_in, as_provided_in2, fc_psi, fy_psi, friction_coeff_mu, phi) <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if vu_kips < 0 or nuc_kips < 0 or shear_span_a_in < 0:
        return {"pass": False, "reason": "negative_load_or_span"}
    if ah_provided_in2 < 0 or min_nuc_ratio < 0:
        return {"pass": False, "reason": "negative_stirrup_or_ratio"}
    if d_in >= h_in:
        return {"pass": False, "reason": "effective_depth_exceeds_total_height"}

    # ACI 318-19 Section 16.5.1.1 shear span-to-depth ratio constraint (a / d <= 1.0)
    a_over_d = shear_span_a_in / d_in
    if a_over_d > 1.0:
        return {
            "pass": False,
            "reason": "shear_span_ratio_exceeds_unity",
            "a_over_d": round(a_over_d, 4),
        }

    # ACI 318-19 Section 16.5.2.4 minimum design horizontal tensile force Nuc >= 0.20 * Vu
    nuc_eff_kips = max(nuc_kips, min_nuc_ratio * vu_kips)

    # ACI 318-19 Section 16.5.2.1 maximum nominal shear strength upper bounds
    vn_max_lb = min(
        0.20 * fc_psi * b_in * d_in,
        (480.0 + 0.08 * fc_psi) * b_in * d_in,
        1600.0 * b_in * d_in,
    )
    phi_vn_max_kips = (phi * vn_max_lb) / 1000.0
    shear_dcr = vu_kips / phi_vn_max_kips if phi_vn_max_kips > 0 else float("inf")
    shear_section_ok = shear_dcr <= 1.0

    fc_ksi = fc_psi / 1000.0
    fy_ksi = min(fy_psi, 60000.0) / 1000.0

    # ACI 318-19 Section 16.5.4.3 shear-friction reinforcement Avf
    avf_in2 = vu_kips / (phi * friction_coeff_mu * fy_ksi)

    # ACI 318-19 Section 16.5.4.5 direct tensile reinforcement An
    an_in2 = nuc_eff_kips / (phi * fy_ksi)

    # ACI 318-19 Section 16.5.4.4 flexural reinforcement Af
    # Factored moment Mu at the face of support
    mu_kip_in = (vu_kips * shear_span_a_in) + (nuc_eff_kips * (h_in - d_in))
    mn_req_kip_in = mu_kip_in / phi

    term_flex = (2.0 * mn_req_kip_in) / (0.85 * fc_ksi * b_in * (d_in**2))
    if term_flex >= 1.0:
        return {"pass": False, "reason": "flexural_capacity_exceeded"}

    af_in2 = ((0.85 * fc_ksi * b_in * d_in) / fy_ksi) * (1.0 - math.sqrt(1.0 - term_flex))

    # ACI 318-19 Section 16.5.5.1 primary tension reinforcement Asc
    asc_strength_in2 = max(af_in2 + an_in2, (2.0 / 3.0) * avf_in2 + an_in2)
    asc_min_in2 = 0.04 * (fc_psi / fy_psi) * b_in * d_in
    asc_req_in2 = max(asc_strength_in2, asc_min_in2)

    asc_dcr = asc_req_in2 / as_provided_in2
    asc_ok = asc_dcr <= 1.0

    # ACI 318-19 Section 16.5.5.2 closed stirrups Ah parallel to Asc
    ah_req_in2 = 0.50 * (asc_req_in2 - an_in2)
    if ah_req_in2 > 0.0:
        ah_dcr = ah_req_in2 / ah_provided_in2 if ah_provided_in2 > 0 else float("inf")
        ah_ok = ah_dcr <= 1.0
    else:
        ah_dcr = 0.0
        ah_ok = True

    gov_dcr = max(shear_dcr, asc_dcr, ah_dcr)
    passed = shear_section_ok and asc_ok and ah_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "shear_dcr": round(shear_dcr, 4),
        "asc_dcr": round(asc_dcr, 4),
        "ah_dcr": round(ah_dcr, 4),
        "asc_req_in2": round(asc_req_in2, 3),
        "ah_req_in2": round(ah_req_in2, 3),
        "phi_vn_max_kips": round(phi_vn_max_kips, 2),
        "mu_kip_in": round(mu_kip_in, 2),
        "nuc_effective_kips": round(nuc_eff_kips, 2),
        "a_over_d": round(a_over_d, 4),
        "shear_section_ok": shear_section_ok,
        "asc_ok": asc_ok,
        "ah_ok": ah_ok,
    }


def verify_nec_transformer_overcurrent_protection(
    transformer_kva: float,
    primary_voltage_v: float,
    secondary_voltage_v: float,
    primary_ocpd_a: float,
    secondary_ocpd_a: float = 0.0,
    three_phase: bool = True,
    allow_next_standard_primary: bool = True,
    allow_next_standard_secondary: bool = True,
) -> Dict[str, Any]:
    """NEC Table 450.3(B) and Section 240.6(A) transformer overcurrent protection gate (<= 1000V)."""
    scalars = [
        transformer_kva,
        primary_voltage_v,
        secondary_voltage_v,
        primary_ocpd_a,
        secondary_ocpd_a,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(transformer_kva, primary_voltage_v, secondary_voltage_v, primary_ocpd_a) <= 0
    ):
        return {"pass": False, "reason": "non_positive_rating_or_voltage"}
    if secondary_ocpd_a < 0:
        return {"pass": False, "reason": "negative_secondary_ocpd"}

    standard_ratings = [
        15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
        110, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
        500, 600, 700, 800, 1000, 1200, 1600, 2000, 2500, 3000,
        4000, 5000, 6000,
    ]

    def get_next_standard(val: float) -> float:
        if val <= standard_ratings[0]:
            return float(standard_ratings[0])
        for r in standard_ratings:
            if r >= (val - 1e-6):
                return float(r)
        return val

    # Full load current calculations
    if three_phase:
        i1 = (transformer_kva * 1000.0) / (math.sqrt(3.0) * primary_voltage_v)
        i2 = (transformer_kva * 1000.0) / (math.sqrt(3.0) * secondary_voltage_v)
    else:
        i1 = (transformer_kva * 1000.0) / primary_voltage_v
        i2 = (transformer_kva * 1000.0) / secondary_voltage_v

    is_primary_only = secondary_ocpd_a <= 0.0
    if is_primary_only:
        scheme = "primary_only"
        if i1 >= 9.0:
            raw_max_primary = 1.25 * i1
            max_allowed_primary = (
                get_next_standard(raw_max_primary)
                if allow_next_standard_primary
                else raw_max_primary
            )
        elif i1 >= 2.0:
            max_allowed_primary = 1.67 * i1
        else:
            max_allowed_primary = 3.00 * i1

        max_allowed_secondary = None
        sec_dcr = None
        sec_ok = True
    else:
        scheme = "primary_and_secondary"
        raw_max_primary = 2.50 * i1
        max_allowed_primary = (
            get_next_standard(raw_max_primary)
            if allow_next_standard_primary
            else raw_max_primary
        )

        if i2 >= 9.0:
            raw_max_sec = 1.25 * i2
            max_allowed_secondary = (
                get_next_standard(raw_max_sec)
                if allow_next_standard_secondary
                else raw_max_sec
            )
        else:
            max_allowed_secondary = 1.67 * i2

        sec_dcr = secondary_ocpd_a / max_allowed_secondary if max_allowed_secondary > 0 else float("inf")
        sec_ok = secondary_ocpd_a <= max_allowed_secondary

    prim_dcr = primary_ocpd_a / max_allowed_primary if max_allowed_primary > 0 else float("inf")
    prim_ok = primary_ocpd_a <= max_allowed_primary

    gov_dcr = max(prim_dcr, sec_dcr) if sec_dcr is not None else prim_dcr
    passed = prim_ok and sec_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "primary_dcr": round(prim_dcr, 4),
        "secondary_dcr": round(sec_dcr, 4) if sec_dcr is not None else None,
        "primary_current_a": round(i1, 2),
        "secondary_current_a": round(i2, 2),
        "max_allowed_primary_a": round(max_allowed_primary, 2),
        "max_allowed_secondary_a": round(max_allowed_secondary, 2) if max_allowed_secondary is not None else None,
        "protection_scheme": scheme,
        "primary_ok": prim_ok,
        "secondary_ok": sec_ok,
    }


def verify_ite_signal_clearance_intervals(
    approach_speed_mph: float,
    intersection_width_ft: float,
    grade_pct: float = 0.0,
    provided_yellow_s: float = 0.0,
    provided_red_s: float = 0.0,
    reaction_time_s: float = 1.0,
    deceleration_rate_fps2: float = 10.0,
    vehicle_length_ft: float = 20.0,
    min_yellow_s: float = 3.0,
    max_yellow_s: float = 6.0,
    min_red_s: float = 1.0,
    cap_yellow_at_max: bool = True,
    pedestrian_crossing_distance_ft: float = 0.0,
    pedestrian_walking_speed_fps: float = 3.5,
    provided_ped_clearance_s: float = 0.0,
    g_fps2: float = 32.174,
) -> Dict[str, Any]:
    """ITE 2020 / MUTCD Section 4D.26 yellow change, all-red clearance, and pedestrian interval gate."""
    scalars = [
        approach_speed_mph,
        intersection_width_ft,
        grade_pct,
        provided_yellow_s,
        provided_red_s,
        reaction_time_s,
        deceleration_rate_fps2,
        vehicle_length_ft,
        min_yellow_s,
        max_yellow_s,
        min_red_s,
        pedestrian_crossing_distance_ft,
        pedestrian_walking_speed_fps,
        provided_ped_clearance_s,
        g_fps2,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            approach_speed_mph,
            intersection_width_ft,
            reaction_time_s,
            deceleration_rate_fps2,
            vehicle_length_ft,
            min_yellow_s,
            max_yellow_s,
            pedestrian_walking_speed_fps,
            g_fps2,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_speed_or_geometry"}
    if min_yellow_s >= max_yellow_s:
        return {"pass": False, "reason": "invalid_yellow_limits"}
    if (
        provided_yellow_s < 0
        or provided_red_s < 0
        or min_red_s < 0
        or pedestrian_crossing_distance_ft < 0
        or provided_ped_clearance_s < 0
    ):
        return {"pass": False, "reason": "negative_interval_or_distance"}

    # Approach velocity v (ft/s) = 1.467 * V (mph)
    v_fps = approach_speed_mph * (5280.0 / 3600.0)

    # Grade factor G as decimal (positive for upgrade, negative for downgrade)
    g_decimal = grade_pct / 100.0
    eff_decel = deceleration_rate_fps2 + (g_fps2 * g_decimal)
    if eff_decel <= 0.0:
        return {"pass": False, "reason": "excessive_downgrade_cannot_stop"}

    # ITE Yellow Change Interval formula: Y = t + v / [2 * (a + g * G)]
    y_calc = reaction_time_s + (v_fps / (2.0 * eff_decel))

    # MUTCD Section 4D.26 bounds (3.0s to 6.0s default)
    if y_calc < min_yellow_s:
        y_design = min_yellow_s
        excess_yellow = 0.0
    elif y_calc > max_yellow_s and cap_yellow_at_max:
        y_design = max_yellow_s
        excess_yellow = y_calc - max_yellow_s
    else:
        y_design = y_calc
        excess_yellow = 0.0

    # ITE Red Clearance Interval formula: R = (W + L) / v
    r_calc = (intersection_width_ft + vehicle_length_ft) / v_fps
    r_design = max(min_red_s, r_calc + excess_yellow)
    total_clearance_req_s = y_design + r_design

    # Timing verification gates
    yellow_ok = True
    yellow_dcr = 0.0
    if provided_yellow_s > 0.0:
        yellow_dcr = y_design / provided_yellow_s
        yellow_ok = yellow_dcr <= 1.0

    red_ok = True
    red_dcr = 0.0
    if provided_red_s > 0.0:
        red_dcr = r_design / provided_red_s
        red_ok = red_dcr <= 1.0

    total_dcr = 0.0
    if provided_yellow_s > 0.0 and provided_red_s > 0.0:
        total_provided = provided_yellow_s + provided_red_s
        total_dcr = total_clearance_req_s / total_provided

    # Optional Pedestrian Clearance Interval (MUTCD 4E.06): FDW = D_ped / S_ped
    ped_ok = True
    ped_dcr = None
    ped_clearance_req_s = None
    if pedestrian_crossing_distance_ft > 0.0:
        ped_clearance_req_s = pedestrian_crossing_distance_ft / pedestrian_walking_speed_fps
        if provided_ped_clearance_s > 0.0:
            ped_dcr = ped_clearance_req_s / provided_ped_clearance_s
            ped_ok = ped_dcr <= 1.0

    dcr_candidates = [yellow_dcr, red_dcr, total_dcr]
    if ped_dcr is not None:
        dcr_candidates.append(ped_dcr)
    gov_dcr = max(dcr_candidates)

    passed = yellow_ok and red_ok and ped_ok and (gov_dcr <= 1.0 if (provided_yellow_s > 0 or provided_red_s > 0) else True)

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "yellow_required_s": round(y_design, 2),
        "yellow_calculated_s": round(y_calc, 2),
        "red_required_s": round(r_design, 2),
        "red_calculated_s": round(r_calc, 2),
        "total_clearance_s": round(total_clearance_req_s, 2),
        "excess_yellow_transferred_s": round(excess_yellow, 2),
        "ped_clearance_req_s": round(ped_clearance_req_s, 2) if ped_clearance_req_s is not None else None,
        "approach_velocity_fps": round(v_fps, 2),
        "yellow_ok": yellow_ok,
        "red_ok": red_ok,
        "ped_ok": ped_ok,
    }


def verify_aci_anchor_bolt_tension(
    nu_kips: float,
    anchor_dia_in: float,
    hef_in: float,
    edge_distance_in: float,
    fc_psi: float,
    futa_ksi: float = 58.0,
    anchor_type: str = "cast_in_headed",
    is_cracked_concrete: bool = True,
    bearing_area_sqin: float = 0.0,
    phi_steel: float = 0.75,
    phi_concrete: float = 0.70,
) -> Dict[str, Any]:
    """ACI 318-19 Chapter 17 LRFD tensile strength check for anchors in concrete (steel rupture, breakout, pullout)."""
    scalars = [
        nu_kips,
        anchor_dia_in,
        hef_in,
        edge_distance_in,
        fc_psi,
        futa_ksi,
        bearing_area_sqin,
        phi_steel,
        phi_concrete,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if nu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(
            anchor_dia_in,
            hef_in,
            edge_distance_in,
            fc_psi,
            futa_ksi,
            phi_steel,
            phi_concrete,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if bearing_area_sqin < 0:
        return {"pass": False, "reason": "negative_bearing_area"}

    atype = anchor_type.strip().lower().replace(" ", "_").replace("-", "_")
    if atype not in ("cast_in_headed", "post_installed"):
        return {"pass": False, "reason": "invalid_anchor_type"}

    # 1. Steel strength in tension (ACI 318-19 Section 17.6.1)
    threads_per_inch: Dict[float, float] = {
        0.50: 13.0,
        0.625: 11.0,
        0.75: 10.0,
        0.875: 9.0,
        1.00: 8.0,
        1.125: 7.0,
        1.25: 7.0,
        1.50: 6.0,
    }
    if anchor_dia_in in threads_per_inch:
        n_t = threads_per_inch[anchor_dia_in]
        ase_n = (math.pi / 4.0) * ((anchor_dia_in - (0.9743 / n_t)) ** 2)
    else:
        ase_n = 0.75 * (math.pi * (anchor_dia_in**2)) / 4.0

    nsa_kips = ase_n * futa_ksi
    phi_nsa = phi_steel * nsa_kips

    # 2. Concrete breakout strength in tension (ACI 318-19 Section 17.6.2)
    kc = 24.0 if atype == "cast_in_headed" else 17.0
    lambda_a = 1.0
    nb_kips = (kc * lambda_a * math.sqrt(fc_psi) * (hef_in**1.5)) / 1000.0

    anco = 9.0 * (hef_in**2)
    c_crit = 1.5 * hef_in
    if edge_distance_in < c_crit:
        anc = (c_crit + edge_distance_in) * (3.0 * hef_in)
        psi_ed_n = 0.70 + 0.30 * (edge_distance_in / c_crit)
    else:
        anc = anco
        psi_ed_n = 1.0

    if is_cracked_concrete:
        psi_c_n = 1.0
    else:
        psi_c_n = 1.25 if atype == "cast_in_headed" else 1.40

    ncb_kips = (anc / anco) * psi_ed_n * psi_c_n * nb_kips
    phi_ncb = phi_concrete * ncb_kips

    # 3. Pullout strength in tension (ACI 318-19 Section 17.6.3)
    abr = bearing_area_sqin if bearing_area_sqin > 0.0 else (0.85 * (anchor_dia_in**2))
    np_lb = 8.0 * abr * fc_psi
    psi_c_p = 1.0 if is_cracked_concrete else 1.40
    npn_kips = (psi_c_p * np_lb) / 1000.0
    phi_npn = phi_concrete * npn_kips

    caps: Dict[str, float] = {
        "steel_tension": phi_nsa,
        "concrete_breakout": phi_ncb,
        "concrete_pullout": phi_npn,
    }
    gov_state = min(caps, key=caps.get)
    phi_nn_kips = caps[gov_state]

    dcr = nu_kips / phi_nn_kips if phi_nn_kips > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_nn_kips": round(phi_nn_kips, 2),
        "phi_nsa_kips": round(phi_nsa, 2),
        "phi_ncb_kips": round(phi_ncb, 2),
        "phi_npn_kips": round(phi_npn, 2),
        "nb_kips": round(nb_kips, 2),
        "anc_sqin": round(anc, 2),
        "governing_limit_state": gov_state,
    }


def verify_nfpa20_fire_pump_performance(
    rated_flow_gpm: float,
    rated_pressure_psi: float,
    churn_pressure_psi: float,
    flow_150_pressure_psi: float,
    demand_flow_gpm: float,
    demand_pressure_psi: float,
    suction_pressure_at_150_psi: float = 5.0,
    motor_nameplate_hp: float = 0.0,
    pump_efficiency_at_150_pct: float = 70.0,
    max_churn_pct: float = 140.0,
    min_150_head_pct: float = 65.0,
    min_suction_psi: float = 0.0,
) -> Dict[str, Any]:
    """NFPA 20 (2019/2022) centrifugal fire pump rating, head curve, demand point, suction, and motor gate."""
    scalars = [
        rated_flow_gpm,
        rated_pressure_psi,
        churn_pressure_psi,
        flow_150_pressure_psi,
        demand_flow_gpm,
        demand_pressure_psi,
        suction_pressure_at_150_psi,
        motor_nameplate_hp,
        pump_efficiency_at_150_pct,
        max_churn_pct,
        min_150_head_pct,
        min_suction_psi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(
            rated_flow_gpm,
            rated_pressure_psi,
            churn_pressure_psi,
            flow_150_pressure_psi,
            demand_flow_gpm,
            demand_pressure_psi,
            pump_efficiency_at_150_pct,
            max_churn_pct,
            min_150_head_pct,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_flow_pressure_or_efficiency"}
    if motor_nameplate_hp < 0:
        return {"pass": False, "reason": "negative_motor_hp"}
    if churn_pressure_psi < rated_pressure_psi:
        return {"pass": False, "reason": "churn_pressure_below_rated_pressure"}
    if flow_150_pressure_psi >= rated_pressure_psi:
        return {"pass": False, "reason": "peak_pressure_exceeds_rated_pressure"}

    # NFPA 20 Section 4.7.6 Churn (shutoff) net head limit: <= 140% of rated
    max_allowed_churn_psi = (max_churn_pct / 100.0) * rated_pressure_psi
    churn_ok = churn_pressure_psi <= max_allowed_churn_psi
    churn_dcr = churn_pressure_psi / max_allowed_churn_psi

    # NFPA 20 Section 4.7.6 Peak (150% flow) net head limit: >= 65% of rated
    min_allowed_150_psi = (min_150_head_pct / 100.0) * rated_pressure_psi
    head_150_ok = flow_150_pressure_psi >= min_allowed_150_psi
    head_150_dcr = min_allowed_150_psi / flow_150_pressure_psi if flow_150_pressure_psi > 0 else float("inf")

    # NFPA 20 Section 4.7.2 Maximum system demand cannot exceed 150% rated flow
    max_flow_capacity_gpm = 1.50 * rated_flow_gpm
    flow_dcr = demand_flow_gpm / max_flow_capacity_gpm
    flow_ok = flow_dcr <= 1.0

    # Fit quadratic pump curve P(Q) = A*Q^2 + B*Q + C through (0, P_churn), (Qr, Pr), (1.5*Qr, P_150)
    delta_1 = rated_pressure_psi - churn_pressure_psi
    delta_2 = flow_150_pressure_psi - churn_pressure_psi
    qr = rated_flow_gpm
    a_coeff = (delta_2 - 1.5 * delta_1) / (0.75 * (qr**2))
    b_coeff = (delta_1 - a_coeff * (qr**2)) / qr
    c_coeff = churn_pressure_psi

    # Available net pump discharge pressure at system demand flow
    p_available_psi = a_coeff * (demand_flow_gpm**2) + b_coeff * demand_flow_gpm + c_coeff
    press_dcr = demand_pressure_psi / p_available_psi if p_available_psi > 0 else float("inf")
    press_ok = press_dcr <= 1.0

    # NFPA 20 Section 4.16.1.1 Suction pressure at 150% flow >= 0 psi gauge
    suction_ok = suction_pressure_at_150_psi >= min_suction_psi

    # Motor nameplate sizing check at 150% peak flow power (BHP = Q * H / (1714 * eta))
    q_peak_gpm = 1.50 * rated_flow_gpm
    bhp_peak = (q_peak_gpm * flow_150_pressure_psi) / (
        1714.0 * (pump_efficiency_at_150_pct / 100.0)
    )
    if motor_nameplate_hp > 0.0:
        motor_dcr = bhp_peak / motor_nameplate_hp
        motor_ok = motor_dcr <= 1.0
    else:
        motor_dcr = None
        motor_ok = True

    dcr_candidates = [churn_dcr, head_150_dcr, flow_dcr, press_dcr]
    if motor_dcr is not None:
        dcr_candidates.append(motor_dcr)
    gov_dcr = max(dcr_candidates)

    passed = churn_ok and head_150_ok and flow_ok and press_ok and suction_ok and motor_ok
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "p_available_at_demand_psi": round(p_available_psi, 2),
        "pressure_dcr": round(press_dcr, 4),
        "flow_dcr": round(flow_dcr, 4),
        "churn_dcr": round(churn_dcr, 4),
        "head_150_dcr": round(head_150_dcr, 4),
        "bhp_peak": round(bhp_peak, 2),
        "motor_dcr": round(motor_dcr, 4) if motor_dcr is not None else None,
        "churn_ok": churn_ok,
        "head_150_ok": head_150_ok,
        "flow_ok": flow_ok,
        "pressure_ok": press_ok,
        "suction_ok": suction_ok,
        "motor_ok": motor_ok,
    }


def verify_aci_spread_footing(
    pu_kips: float,
    footing_length_ft: float,
    footing_width_ft: float,
    footing_thickness_in: float,
    effective_depth_d_in: float,
    column_width_in: float,
    column_depth_in: float,
    as_provided_in2: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    lambda_factor: float = 1.0,
    phi_shear: float = 0.75,
    phi_flexure: float = 0.90,
) -> Dict[str, Any]:
    """ACI 318-19 Section 13.2 & 13.3 structural spread footing one-way shear, flexural capacity, and minimum reinforcement check."""
    scalars = [
        pu_kips,
        footing_length_ft,
        footing_width_ft,
        footing_thickness_in,
        effective_depth_d_in,
        column_width_in,
        column_depth_in,
        as_provided_in2,
        fc_psi,
        fy_psi,
        lambda_factor,
        phi_shear,
        phi_flexure,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if pu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(
            footing_length_ft,
            footing_width_ft,
            footing_thickness_in,
            effective_depth_d_in,
            column_width_in,
            column_depth_in,
            as_provided_in2,
            fc_psi,
            fy_psi,
            lambda_factor,
            phi_shear,
            phi_flexure,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if effective_depth_d_in >= footing_thickness_in:
        return {"pass": False, "reason": "effective_depth_exceeds_thickness"}
    if column_width_in >= (footing_length_ft * 12.0) or column_depth_in >= (footing_width_ft * 12.0):
        return {"pass": False, "reason": "column_exceeds_footing_dimensions"}

    # Factored net soil bearing pressure (ksf)
    footing_area_sqft = footing_length_ft * footing_width_ft
    qu_ksf = pu_kips / footing_area_sqft

    # Cantilever projection from column face in the checked length direction (in)
    l_cant_in = (footing_length_ft * 12.0 - column_width_in) / 2.0
    bw_in = footing_width_ft * 12.0

    # 1. One-way (beam) shear at distance d from column face (ACI 318-19 Section 13.2.7.2)
    l_shear_in = max(0.0, l_cant_in - effective_depth_d_in)
    vu_kips = qu_ksf * footing_width_ft * (l_shear_in / 12.0)

    # ACI Table 22.5.5.1 simplified one-way shear strength without axial force
    sqrt_fc = math.sqrt(fc_psi)
    vc_kips = (2.0 * lambda_factor * sqrt_fc * bw_in * effective_depth_d_in) / 1000.0
    phi_vc_kips = phi_shear * vc_kips
    shear_dcr = vu_kips / phi_vc_kips if phi_vc_kips > 0 else float("inf")
    shear_ok = shear_dcr <= 1.0

    # 2. Flexural moment at column face (ACI 318-19 Section 13.2.6.2)
    l_cant_ft = l_cant_in / 12.0
    mu_kip_ft = qu_ksf * footing_width_ft * (l_cant_ft**2) / 2.0

    # Depth of equivalent rectangular compressive stress block (a)
    a_in = (as_provided_in2 * fy_psi) / (0.85 * fc_psi * bw_in)
    if a_in >= effective_depth_d_in:
        return {"pass": False, "reason": "compression_block_exceeds_depth"}

    mn_kip_in = as_provided_in2 * fy_psi * (effective_depth_d_in - a_in / 2.0) / 1000.0
    phi_mn_kip_ft = (phi_flexure * mn_kip_in) / 12.0
    flex_dcr = mu_kip_ft / phi_mn_kip_ft if phi_mn_kip_ft > 0 else float("inf")
    flex_ok = flex_dcr <= 1.0

    # 3. Minimum flexural / shrinkage-temperature reinforcement (ACI 318-19 Section 7.6.1.1 & 13.3.2.1)
    as_min_in2 = 0.0018 * bw_in * footing_thickness_in
    as_min_ok = as_provided_in2 >= as_min_in2
    as_min_dcr = as_min_in2 / as_provided_in2

    gov_dcr = max(shear_dcr, flex_dcr, as_min_dcr)
    passed = shear_ok and flex_ok and as_min_ok

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "shear_dcr": round(shear_dcr, 4),
        "flexure_dcr": round(flex_dcr, 4),
        "vu_kips": round(vu_kips, 2),
        "phi_vc_kips": round(phi_vc_kips, 2),
        "mu_kip_ft": round(mu_kip_ft, 2),
        "phi_mn_kip_ft": round(phi_mn_kip_ft, 2),
        "as_min_in2": round(as_min_in2, 3),
        "qu_ksf": round(qu_ksf, 4),
        "cantilever_in": round(l_cant_in, 2),
        "shear_ok": shear_ok,
        "flexure_ok": flex_ok,
        "as_min_ok": as_min_ok,
    }


def verify_ashrae_ahu_psychrometric_mixing(
    outdoor_airflow_cfm: float,
    return_airflow_cfm: float,
    t_outdoor_db_f: float,
    t_return_db_f: float,
    w_outdoor_lb_per_lb: float,
    w_return_lb_per_lb: float,
    t_leaving_db_f: float,
    w_leaving_lb_per_lb: float,
    coil_cooling_capacity_tons: float,
    coil_sensible_capacity_btuh: float = 0.0,
    min_oa_fraction: float = 0.0,
    min_mixed_temp_f: float = 40.0,
    elevation_ft: float = 0.0,
) -> Dict[str, Any]:
    """ASHRAE Fundamentals & 62.1 AHU economizer mixed-air psychrometrics, coil thermal duty, and freeze gate."""
    scalars = [
        outdoor_airflow_cfm,
        return_airflow_cfm,
        t_outdoor_db_f,
        t_return_db_f,
        w_outdoor_lb_per_lb,
        w_return_lb_per_lb,
        t_leaving_db_f,
        w_leaving_lb_per_lb,
        coil_cooling_capacity_tons,
        coil_sensible_capacity_btuh,
        min_oa_fraction,
        min_mixed_temp_f,
        elevation_ft,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if outdoor_airflow_cfm < 0 or return_airflow_cfm < 0:
        return {"pass": False, "reason": "negative_airflow"}
    total_cfm = outdoor_airflow_cfm + return_airflow_cfm
    if total_cfm <= 0:
        return {"pass": False, "reason": "zero_total_airflow"}
    if (
        w_outdoor_lb_per_lb < 0
        or w_return_lb_per_lb < 0
        or w_leaving_lb_per_lb < 0
    ):
        return {"pass": False, "reason": "negative_humidity_ratio"}
    if coil_cooling_capacity_tons <= 0:
        return {"pass": False, "reason": "non_positive_coil_capacity"}
    if coil_sensible_capacity_btuh < 0:
        return {"pass": False, "reason": "negative_sensible_capacity"}
    if not (0.0 <= min_oa_fraction <= 1.0):
        return {"pass": False, "reason": "invalid_oa_fraction_limit"}
    if elevation_ft < -1000.0 or elevation_ft > 30000.0:
        return {"pass": False, "reason": "invalid_elevation"}

    oa_fraction = outdoor_airflow_cfm / total_cfm
    oa_fraction_ok = oa_fraction >= (min_oa_fraction - 1e-6)

    t_mixed_db_f = oa_fraction * t_outdoor_db_f + (1.0 - oa_fraction) * t_return_db_f
    w_mixed = oa_fraction * w_outdoor_lb_per_lb + (1.0 - oa_fraction) * w_return_lb_per_lb

    if w_leaving_lb_per_lb > (w_mixed + 1e-6):
        return {"pass": False, "reason": "cooling_coil_adds_moisture"}
    if t_leaving_db_f >= t_mixed_db_f:
        return {"pass": False, "reason": "leaving_temp_not_below_mixed_temp"}

    p_atm_psia = 14.696 * ((1.0 - 0.0000068756 * elevation_ft) ** 5.2559)
    density_ratio = p_atm_psia / 14.696
    c_s = 1.08 * density_ratio
    c_l = 4840.0 * density_ratio
    c_t = 4.5 * density_ratio

    h_mixed = 0.240 * t_mixed_db_f + w_mixed * (1061.0 + 0.444 * t_mixed_db_f)
    h_leaving = 0.240 * t_leaving_db_f + w_leaving_lb_per_lb * (1061.0 + 0.444 * t_leaving_db_f)

    q_sensible_btuh = c_s * total_cfm * (t_mixed_db_f - t_leaving_db_f)
    q_latent_btuh = c_l * total_cfm * (w_mixed - w_leaving_lb_per_lb)
    q_total_btuh = c_t * total_cfm * (h_mixed - h_leaving)

    tons_required = q_total_btuh / 12000.0
    total_dcr = tons_required / coil_cooling_capacity_tons
    total_ok = total_dcr <= 1.0

    sensible_dcr = None
    sensible_ok = True
    if coil_sensible_capacity_btuh > 0.0:
        sensible_dcr = q_sensible_btuh / coil_sensible_capacity_btuh
        sensible_ok = sensible_dcr <= 1.0

    freeze_risk = t_mixed_db_f < min_mixed_temp_f
    freeze_ok = not freeze_risk

    dcr_candidates = [total_dcr]
    if sensible_dcr is not None:
        dcr_candidates.append(sensible_dcr)
    if min_oa_fraction > 0.0 and oa_fraction > 0.0:
        dcr_candidates.append(min_oa_fraction / oa_fraction)
    gov_dcr = max(dcr_candidates)

    passed = total_ok and sensible_ok and oa_fraction_ok and freeze_ok
    shr = q_sensible_btuh / (q_sensible_btuh + q_latent_btuh) if (q_sensible_btuh + q_latent_btuh) > 0 else 1.0

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "total_dcr": round(total_dcr, 4),
        "sensible_dcr": round(sensible_dcr, 4) if sensible_dcr is not None else None,
        "t_mixed_db_f": round(t_mixed_db_f, 2),
        "w_mixed_lb_per_lb": round(w_mixed, 5),
        "h_mixed_btu_per_lb": round(h_mixed, 2),
        "h_leaving_btu_per_lb": round(h_leaving, 2),
        "q_sensible_btuh": round(q_sensible_btuh, 1),
        "q_latent_btuh": round(q_latent_btuh, 1),
        "q_total_btuh": round(q_total_btuh, 1),
        "tons_required": round(tons_required, 2),
        "shr": round(shr, 4),
        "oa_fraction": round(oa_fraction, 4),
        "freeze_risk": freeze_risk,
        "oa_fraction_ok": oa_fraction_ok,
        "total_ok": total_ok,
        "sensible_ok": sensible_ok,
        "freeze_ok": freeze_ok,
    }


def verify_aisc_plate_girder_shear(
    vu_kips: float,
    d_in: float,
    tw_in: float,
    h_web_in: float,
    stiffener_spacing_a_in: float = 0.0,
    fy_ksi: float = 50.0,
    e_ksi: float = 29000.0,
    use_tension_field_action: bool = False,
    is_end_panel: bool = False,
    phi: float = 0.90,
) -> Dict[str, Any]:
    """AISC 360-16 Section G2 LRFD plate girder web shear buckling, stiffener spacing, and Tension Field Action check."""
    scalars = [
        vu_kips,
        d_in,
        tw_in,
        h_web_in,
        stiffener_spacing_a_in,
        fy_ksi,
        e_ksi,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if min(d_in, tw_in, h_web_in, fy_ksi, e_ksi, phi) <= 0:
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if stiffener_spacing_a_in < 0:
        return {"pass": False, "reason": "negative_stiffener_spacing"}
    if h_web_in > d_in:
        return {"pass": False, "reason": "web_depth_exceeds_total_depth"}

    # Web area Aw = d * tw (AISC 360-16 Section G2.1)
    aw_in2 = d_in * tw_in
    h_over_tw = h_web_in / tw_in

    is_stiffened = stiffener_spacing_a_in > 0.0
    if is_stiffened:
        a_over_h = stiffener_spacing_a_in / h_web_in
        # AISC 360-16 Section G2.1 web shear plate buckling coefficient kv
        if a_over_h <= 3.0:
            kv = 5.0 + 5.0 / (a_over_h**2)
        else:
            kv = 5.34
    else:
        a_over_h = 0.0
        kv = 5.34

    # Slenderness thresholds per AISC Section G2.1
    sqrt_term = math.sqrt((kv * e_ksi) / fy_ksi)
    limit_yield = 1.10 * sqrt_term
    limit_inelastic = 1.37 * sqrt_term

    # Web shear strength coefficient Cv1 without tension field action
    if h_over_tw <= limit_yield:
        cv1 = 1.0
        limit_state_base = "shear_yielding"
    elif h_over_tw <= limit_inelastic:
        cv1 = limit_yield / h_over_tw
        limit_state_base = "inelastic_shear_buckling"
    else:
        cv1 = (1.51 * kv * e_ksi) / ((h_over_tw**2) * fy_ksi)
        limit_state_base = "elastic_shear_buckling"

    # Tension Field Action (TFA) check per AISC 360-16 Section G2.2
    cv2 = None
    tfa_applied = False
    if use_tension_field_action:
        if is_end_panel:
            return {"pass": False, "reason": "tension_field_action_not_permitted_on_end_panel"}
        if not is_stiffened:
            return {"pass": False, "reason": "tension_field_action_requires_stiffeners"}
        if a_over_h > 3.0:
            return {"pass": False, "reason": "aspect_ratio_exceeds_tfa_limit"}

        if h_over_tw <= limit_yield:
            vn_kips = 0.60 * fy_ksi * aw_in2
            limit_state = "shear_yielding"
        else:
            cv2 = limit_yield / h_over_tw
            denom_tfa = 1.15 * math.sqrt(1.0 + (a_over_h**2))
            tfa_factor = cv2 + (1.0 - cv2) / denom_tfa
            vn_kips = 0.60 * fy_ksi * aw_in2 * tfa_factor
            limit_state = "tension_field_action"
            tfa_applied = True
    else:
        vn_kips = 0.60 * fy_ksi * aw_in2 * cv1
        limit_state = limit_state_base

    phi_vn_kips = phi * vn_kips
    dcr = vu_kips / phi_vn_kips if phi_vn_kips > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_vn_kips": round(phi_vn_kips, 2),
        "vn_kips": round(vn_kips, 2),
        "cv1": round(cv1, 4),
        "cv2": round(cv2, 4) if cv2 is not None else None,
        "kv": round(kv, 4),
        "h_over_tw": round(h_over_tw, 2),
        "governing_limit_state": limit_state,
        "is_stiffened": is_stiffened,
        "tfa_applied": tfa_applied,
    }


def verify_aci_concrete_shear_wall(
    vu_kips: float,
    hw_ft: float,
    lw_ft: float,
    tw_in: float,
    rho_t: float,
    rho_l: float,
    fc_psi: float,
    fy_psi: float = 60000.0,
    num_curtains: int = 2,
    is_special_seismic: bool = False,
    lambda_factor: float = 1.0,
    phi: float = 0.75,
) -> Dict[str, Any]:
    """ACI 318-19 Section 11.5 & 18.10 reinforced concrete shear wall in-plane shear capacity and detailing gate."""
    scalars = [
        vu_kips,
        hw_ft,
        lw_ft,
        tw_in,
        rho_t,
        rho_l,
        fc_psi,
        fy_psi,
        float(num_curtains),
        lambda_factor,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if vu_kips < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(hw_ft, lw_ft, tw_in, fc_psi, fy_psi, lambda_factor, phi) <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if rho_t < 0 or rho_l < 0:
        return {"pass": False, "reason": "negative_reinforcement_ratio"}
    if num_curtains not in (1, 2):
        return {"pass": False, "reason": "invalid_curtain_count"}

    # Cross-sectional gross area of concrete resisting shear (Acv)
    lw_in = lw_ft * 12.0
    acv_in2 = lw_in * tw_in
    hw_over_lw = hw_ft / lw_ft
    sqrt_fc = math.sqrt(fc_psi)

    # ACI 318-19 Section 18.10.4.1 concrete shear coefficient alpha_c
    if is_special_seismic or hw_over_lw <= 2.0:
        if hw_over_lw <= 1.5:
            alpha_c = 3.0
        elif hw_over_lw >= 2.0:
            alpha_c = 2.0
        else:
            alpha_c = 3.0 - 2.0 * (hw_over_lw - 1.5)
    else:
        alpha_c = 2.0

    # Concrete shear strength contribution (Vc)
    vc_lb = alpha_c * lambda_factor * sqrt_fc * acv_in2
    vc_kips = vc_lb / 1000.0

    # Transverse reinforcement shear strength contribution (Vs)
    vs_lb = rho_t * fy_psi * acv_in2
    vs_kips = vs_lb / 1000.0

    # Section maximum allowable nominal shear strength (ACI 11.5.4.2 & 18.10.4.4)
    vn_max_kips = (8.0 * sqrt_fc * acv_in2) / 1000.0
    vn_calc_kips = vc_kips + vs_kips
    vn_kips = min(vn_calc_kips, vn_max_kips)
    phi_vn_kips = phi * vn_kips

    shear_dcr = vu_kips / phi_vn_kips if phi_vn_kips > 0 else float("inf")
    shear_ok = shear_dcr <= 1.0

    # Minimum transverse reinforcement ratio rho_t (ACI 11.6.1 & 18.10.2.1)
    vc_threshold_kips = (0.5 * phi * 2.0 * lambda_factor * sqrt_fc * acv_in2) / 1000.0
    transverse_req = is_special_seismic or (vu_kips > vc_threshold_kips)
    rho_t_min = 0.0025 if transverse_req else 0.0020
    rho_t_ok = rho_t >= rho_t_min

    # Minimum longitudinal reinforcement ratio rho_l (ACI 11.6.2 & 18.10.4.3)
    if is_special_seismic and hw_over_lw <= 2.0:
        rho_l_min = max(0.0025, rho_t)
    elif transverse_req:
        rho_l_min = 0.0025
    else:
        rho_l_min = 0.0015
    rho_l_ok = rho_l >= rho_l_min

    # Two curtains required if Vu > 2 * phi * Acv * lambda * sqrt(f'c) or special wall with tw >= 10 in
    curtain_threshold_kips = (phi * 2.0 * lambda_factor * sqrt_fc * acv_in2) / 1000.0
    curtains_required = 2 if (vu_kips > curtain_threshold_kips or (is_special_seismic and tw_in >= 10.0)) else 1
    curtains_ok = num_curtains >= curtains_required

    passed = shear_ok and rho_t_ok and rho_l_ok and curtains_ok
    return {
        "pass": passed,
        "dcr": round(shear_dcr, 4),
        "phi_vn_kips": round(phi_vn_kips, 2),
        "vn_kips": round(vn_kips, 2),
        "vc_kips": round(vc_kips, 2),
        "vs_kips": round(vs_kips, 2),
        "alpha_c": round(alpha_c, 3),
        "hw_over_lw": round(hw_over_lw, 3),
        "rho_t_min": round(rho_t_min, 4),
        "rho_l_min": round(rho_l_min, 4),
        "curtains_required": curtains_required,
        "shear_ok": shear_ok,
        "rho_t_ok": rho_t_ok,
        "rho_l_ok": rho_l_ok,
        "curtains_ok": curtains_ok,
    }


def verify_nec_feeder_tap_rules(
    feeder_ocpd_a: float,
    tap_conductor_ampacity_a: float,
    tap_length_ft: float,
    termination_ocpd_a: float,
    calculated_load_a: float = 0.0,
    tap_rule: str = "auto",
    enclosed_in_raceway: bool = True,
) -> Dict[str, Any]:
    """NEC 240.21(B)(1) & 240.21(B)(2) 10-ft and 25-ft feeder tap rule compliance check."""
    scalars = [
        feeder_ocpd_a,
        tap_conductor_ampacity_a,
        tap_length_ft,
        termination_ocpd_a,
        calculated_load_a,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(feeder_ocpd_a, tap_conductor_ampacity_a, tap_length_ft, termination_ocpd_a)
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_rating_or_geometry"}
    if calculated_load_a < 0:
        return {"pass": False, "reason": "negative_load_input"}

    rule_clean = tap_rule.strip().lower().replace(" ", "_").replace("-", "_")
    valid_rules = {"auto", "10_foot", "10", "25_foot", "25"}
    if rule_clean not in valid_rules:
        return {"pass": False, "reason": "invalid_tap_rule"}

    if rule_clean in ("10_foot", "10"):
        rule = "10_foot"
    elif rule_clean in ("25_foot", "25"):
        rule = "25_foot"
    else:
        if tap_length_ft <= 10.0:
            rule = "10_foot"
        elif tap_length_ft <= 25.0:
            rule = "25_foot"
        else:
            return {"pass": False, "reason": "tap_length_exceeds_maximum_25ft"}

    if rule == "10_foot":
        if tap_length_ft > 10.0:
            return {"pass": False, "reason": "length_exceeds_10ft_for_rule"}
        min_ampacity_ratio = 0.10 * feeder_ocpd_a
        min_tap_ampacity = max(calculated_load_a, termination_ocpd_a, min_ampacity_ratio)
        length_dcr = tap_length_ft / 10.0
    else:
        if tap_length_ft > 25.0:
            return {"pass": False, "reason": "length_exceeds_25ft_for_rule"}
        min_ampacity_ratio = (1.0 / 3.0) * feeder_ocpd_a
        min_tap_ampacity = max(calculated_load_a, min_ampacity_ratio)
        length_dcr = tap_length_ft / 25.0

    ampacity_dcr = min_tap_ampacity / tap_conductor_ampacity_a
    ampacity_ok = tap_conductor_ampacity_a >= min_tap_ampacity

    termination_dcr = termination_ocpd_a / tap_conductor_ampacity_a
    termination_ok = termination_ocpd_a <= tap_conductor_ampacity_a

    enclosure_ok = enclosed_in_raceway
    gov_dcr = max(ampacity_dcr, termination_dcr, length_dcr)

    passed = ampacity_ok and termination_ok and enclosure_ok and (length_dcr <= 1.0)

    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "ampacity_dcr": round(ampacity_dcr, 4),
        "termination_dcr": round(termination_dcr, 4),
        "length_dcr": round(length_dcr, 4),
        "min_tap_ampacity_a": round(min_tap_ampacity, 2),
        "feeder_ocpd_ratio_ampacity_a": round(min_ampacity_ratio, 2),
        "tap_rule_applied": rule,
        "ampacity_ok": ampacity_ok,
        "termination_ok": termination_ok,
        "enclosure_ok": enclosure_ok,
    }


def verify_aisc_panel_zone_shear(
    dc_in: float,
    tw_in: float,
    bcf_in: float,
    tcf_in: float,
    db_in: float,
    ag_in2: float,
    vu_kips: float = 0.0,
    mu_beam_kip_ft: float = 0.0,
    v_col_kips: float = 0.0,
    pr_kips: float = 0.0,
    fy_ksi: float = 50.0,
    consider_deformation: bool = False,
    doubler_plate_t_in: float = 0.0,
    phi: float = 0.90,
) -> Dict[str, Any]:
    """AISC 360-16 Section J10.6 LRFD beam-column panel zone shear capacity and doubler plate check."""
    scalars = [
        dc_in,
        tw_in,
        bcf_in,
        tcf_in,
        db_in,
        ag_in2,
        vu_kips,
        mu_beam_kip_ft,
        v_col_kips,
        pr_kips,
        fy_ksi,
        doubler_plate_t_in,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if (
        min(dc_in, tw_in, bcf_in, tcf_in, db_in, ag_in2, fy_ksi, phi) <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if (
        vu_kips < 0
        or mu_beam_kip_ft < 0
        or v_col_kips < 0
        or pr_kips < 0
        or doubler_plate_t_in < 0
    ):
        return {"pass": False, "reason": "negative_load_or_dimension"}
    if 2.0 * tcf_in >= dc_in:
        return {"pass": False, "reason": "flanges_exceed_column_depth"}

    # Resolve panel zone shear demand force (Vu)
    if vu_kips > 0.0:
        v_demand_kips = vu_kips
    elif mu_beam_kip_ft > 0.0:
        v_demand_kips = max(0.0, (mu_beam_kip_ft * 12.0 / db_in) - v_col_kips)
    else:
        return {"pass": False, "reason": "zero_shear_demand"}

    # Column compressive strength Pc = Py = Fy * Ag
    pc_kips = fy_ksi * ag_in2
    if pr_kips >= pc_kips:
        return {"pass": False, "reason": "axial_load_exceeds_column_capacity"}

    pr_ratio = pr_kips / pc_kips
    t_eff = tw_in + doubler_plate_t_in

    # AISC 360-16 Section J10.6 nominal shear strength Rn
    if not consider_deformation:
        # 1. When effect of panel-zone deformation on frame stability is not considered
        if pr_ratio <= 0.40:
            rn_kips = 0.60 * fy_ksi * dc_in * t_eff
            gov_eq = "J10-9"
        else:
            rn_kips = 0.60 * fy_ksi * dc_in * t_eff * (1.4 - pr_ratio)
            gov_eq = "J10-10"
    else:
        # 2. When effect of panel-zone deformation on frame stability is considered
        flange_enhancement = (3.0 * bcf_in * (tcf_in**2)) / (db_in * dc_in * t_eff)
        rn_base = 0.60 * fy_ksi * dc_in * t_eff * (1.0 + flange_enhancement)
        if pr_ratio <= 0.75:
            rn_kips = rn_base
            gov_eq = "J10-11"
        else:
            rn_kips = rn_base * (1.9 - 1.2 * pr_ratio)
            gov_eq = "J10-12"

    phi_rn_kips = phi * rn_kips
    dcr = v_demand_kips / phi_rn_kips if phi_rn_kips > 0 else float("inf")
    passed = dcr <= 1.0
    doubler_required = not passed

    # Minimum doubler plate thickness to prevent local buckling (AISC J10.6 & F13.2)
    clear_web_depth_in = dc_in - 2.0 * tcf_in
    t_min_buckling_in = clear_web_depth_in / 50.0

    req_total_t_in = t_eff * dcr if dcr > 1.0 else t_eff
    min_doubler_thickness_in = max(req_total_t_in - tw_in, t_min_buckling_in) if doubler_required else 0.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_rn_kips": round(phi_rn_kips, 2),
        "rn_kips": round(rn_kips, 2),
        "v_demand_kips": round(v_demand_kips, 2),
        "pc_kips": round(pc_kips, 2),
        "pr_ratio": round(pr_ratio, 4),
        "governing_equation": gov_eq,
        "doubler_required": doubler_required,
        "min_doubler_thickness_in": round(min_doubler_thickness_in, 3) if doubler_required else None,
    }


def verify_nds_timber_beam_flexure_and_shear(
    m_max_kip_ft: float,
    v_max_kips: float,
    b_in: float,
    d_in: float,
    fb_psi: float,
    fv_psi: float,
    e_min_psi: float = 580000.0,
    e_modulus_psi: float = 1600000.0,
    unbraced_length_ft: float = 0.0,
    span_ft: float = 0.0,
    w_live_plf: float = 0.0,
    cd_duration: float = 1.0,
    cm_moisture: float = 1.0,
    ct_temp: float = 1.0,
    cf_size: float = 1.0,
    cr_repetitive: float = 1.0,
    deflection_limit_ratio: float = 360.0,
) -> Dict[str, Any]:
    """NDS 2018 Section 3.3 & 3.4 ASD timber beam flexural bending, shear, lateral stability (CL), and deflection check."""
    scalars = [
        m_max_kip_ft,
        v_max_kips,
        b_in,
        d_in,
        fb_psi,
        fv_psi,
        e_min_psi,
        e_modulus_psi,
        unbraced_length_ft,
        span_ft,
        w_live_plf,
        cd_duration,
        cm_moisture,
        ct_temp,
        cf_size,
        cr_repetitive,
        deflection_limit_ratio,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if m_max_kip_ft < 0 or v_max_kips < 0 or w_live_plf < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if unbraced_length_ft < 0 or span_ft < 0:
        return {"pass": False, "reason": "negative_geometry_input"}
    if (
        min(
            b_in,
            d_in,
            fb_psi,
            fv_psi,
            e_min_psi,
            e_modulus_psi,
            cd_duration,
            cm_moisture,
            ct_temp,
            cf_size,
            cr_repetitive,
            deflection_limit_ratio,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}

    # Actual rectangular cross-section properties
    area_in2 = b_in * d_in
    sx_in3 = (b_in * (d_in**2)) / 6.0
    ix_in4 = (b_in * (d_in**3)) / 12.0

    # Actual flexural and shear stresses
    m_lb_in = m_max_kip_ft * 12000.0
    fb_actual_psi = m_lb_in / sx_in3 if sx_in3 > 0 else float("inf")

    v_lb = v_max_kips * 1000.0
    fv_actual_psi = (1.5 * v_lb) / area_in2 if area_in2 > 0 else float("inf")

    # Reference design values adjusted by base modifiers (prior to CL)
    fb_star_psi = fb_psi * cd_duration * cm_moisture * ct_temp * cf_size * cr_repetitive
    emin_prime_psi = e_min_psi * cm_moisture * ct_temp

    # NDS 2018 Section 3.3.3 Beam Stability Factor (C_L)
    if unbraced_length_ft <= 0.0:
        cl = 1.0
        rb = 0.0
    else:
        lu_in = unbraced_length_ft * 12.0
        lu_over_d = lu_in / d_in
        if lu_over_d < 14.3:
            le_in = 2.06 * lu_in
        else:
            le_in = 1.63 * lu_in + 3.0 * d_in

        rb = math.sqrt((le_in * d_in) / (b_in**2))
        if rb > 50.0:
            return {
                "pass": False,
                "reason": "slenderness_rb_exceeds_50",
                "slenderness_rb": round(rb, 2),
            }

        # Critical buckling design value for bending
        fbe_psi = (1.20 * emin_prime_psi) / (rb**2)
        alpha = fbe_psi / fb_star_psi if fb_star_psi > 0 else 0.0
        term1 = (1.0 + alpha) / 1.9
        inside_sqrt = max((term1**2) - (alpha / 0.95), 0.0)
        cl = max(0.0, min(1.0, term1 - math.sqrt(inside_sqrt)))

    fb_prime_psi = fb_star_psi * cl
    flex_dcr = fb_actual_psi / fb_prime_psi if fb_prime_psi > 0 else float("inf")
    flex_ok = flex_dcr <= 1.0

    # NDS 2018 Section 3.4.2 adjusted shear design value F'v
    fv_prime_psi = fv_psi * cd_duration * cm_moisture * ct_temp
    shear_dcr = fv_actual_psi / fv_prime_psi if fv_prime_psi > 0 else float("inf")
    shear_ok = shear_dcr <= 1.0

    # Optional live load deflection check
    defl_dcr = None
    defl_ok = True
    delta_live_in = None
    allowable_defl_in = None
    if span_ft > 0.0 and w_live_plf > 0.0:
        l_in = span_ft * 12.0
        w_live_in = w_live_plf / 12.0
        e_prime_psi = e_modulus_psi * cm_moisture * ct_temp
        delta_live_in = (5.0 * w_live_in * (l_in**4)) / (384.0 * e_prime_psi * ix_in4)
        allowable_defl_in = l_in / deflection_limit_ratio
        defl_dcr = delta_live_in / allowable_defl_in if allowable_defl_in > 0 else float("inf")
        defl_ok = defl_dcr <= 1.0

    dcr_candidates = [flex_dcr, shear_dcr]
    if defl_dcr is not None:
        dcr_candidates.append(defl_dcr)
    gov_dcr = max(dcr_candidates)

    passed = flex_ok and shear_ok and defl_ok
    return {
        "pass": passed,
        "dcr": round(gov_dcr, 4),
        "flex_dcr": round(flex_dcr, 4),
        "shear_dcr": round(shear_dcr, 4),
        "defl_dcr": round(defl_dcr, 4) if defl_dcr is not None else None,
        "fb_actual_psi": round(fb_actual_psi, 1),
        "fb_prime_psi": round(fb_prime_psi, 1),
        "fv_actual_psi": round(fv_actual_psi, 1),
        "fv_prime_psi": round(fv_prime_psi, 1),
        "cl_factor": round(cl, 4),
        "slenderness_rb": round(rb, 2),
        "delta_live_in": round(delta_live_in, 4) if delta_live_in is not None else None,
        "allowable_defl_in": round(allowable_defl_in, 4) if allowable_defl_in is not None else None,
        "flex_ok": flex_ok,
        "shear_ok": shear_ok,
        "defl_ok": defl_ok,
    }


def verify_aisc_composite_beam_flexure(
    mu_kip_ft: float,
    span_ft: float,
    beam_spacing_ft: float,
    d_beam_in: float,
    tw_in: float,
    bf_in: float,
    tf_in: float,
    ag_in2: float,
    slab_thickness_total_in: float,
    deck_rib_height_in: float,
    fc_psi: float,
    num_shear_studs: int,
    stud_capacity_qn_kips: float,
    fy_ksi: float = 50.0,
    is_edge_beam: bool = False,
    effective_slab_width_override_in: float = 0.0,
    min_composite_pct: float = 25.0,
    phi: float = 0.90,
) -> Dict[str, Any]:
    """AISC 360-16 Chapter I LRFD composite steel beam flexural capacity and shear stud check."""
    scalars = [
        mu_kip_ft,
        span_ft,
        beam_spacing_ft,
        d_beam_in,
        tw_in,
        bf_in,
        tf_in,
        ag_in2,
        slab_thickness_total_in,
        deck_rib_height_in,
        fc_psi,
        float(num_shear_studs),
        stud_capacity_qn_kips,
        fy_ksi,
        effective_slab_width_override_in,
        min_composite_pct,
        phi,
    ]
    if any(math.isnan(val) for val in scalars):
        return {"pass": False, "reason": "nan_input"}
    if mu_kip_ft < 0:
        return {"pass": False, "reason": "negative_load_input"}
    if (
        min(
            span_ft,
            beam_spacing_ft,
            d_beam_in,
            tw_in,
            bf_in,
            tf_in,
            ag_in2,
            slab_thickness_total_in,
            fc_psi,
            stud_capacity_qn_kips,
            fy_ksi,
            phi,
        )
        <= 0
    ):
        return {"pass": False, "reason": "non_positive_geometry_or_material"}
    if deck_rib_height_in < 0 or num_shear_studs < 0 or min_composite_pct < 0:
        return {"pass": False, "reason": "negative_parameter_input"}
    if deck_rib_height_in >= slab_thickness_total_in:
        return {"pass": False, "reason": "deck_height_exceeds_total_slab_thickness"}
    if 2.0 * tf_in >= d_beam_in:
        return {"pass": False, "reason": "flanges_exceed_beam_depth"}

    # AISC 360-16 Section I3.1a effective slab width (b_eff)
    if effective_slab_width_override_in > 0.0:
        b_eff = effective_slab_width_override_in
    elif is_edge_beam:
        b_eff = min((span_ft * 12.0) / 8.0 + (beam_spacing_ft * 12.0) / 2.0, (span_ft * 12.0) / 4.0)
    else:
        b_eff = min(span_ft * 12.0 / 4.0, beam_spacing_ft * 12.0)

    # Concrete thickness above steel deck ribs
    t_c = slab_thickness_total_in - deck_rib_height_in
    fc_ksi = fc_psi / 1000.0

    # Component capacities for plastic stress distribution
    c_conc_kips = 0.85 * fc_ksi * b_eff * t_c
    t_steel_kips = fy_ksi * ag_in2
    v_studs_kips = num_shear_studs * stud_capacity_qn_kips

    # Compressive force transferred across composite interface (C)
    c_max_avail = min(c_conc_kips, t_steel_kips)
    c_kips = min(c_max_avail, v_studs_kips)

    # Degree of composite action check (AISC I3.2d.1 requires min 25%)
    composite_ratio = c_kips / c_max_avail if c_max_avail > 0 else 0.0
    composite_pct = composite_ratio * 100.0
    if composite_pct < min_composite_pct:
        return {
            "pass": False,
            "reason": "insufficient_composite_action",
            "composite_pct": round(composite_pct, 2),
            "min_composite_pct": round(min_composite_pct, 2),
        }

    # Concrete stress block depth (a) in slab above ribs
    a_in = c_kips / (0.85 * fc_ksi * b_eff)

    # Distance from top of steel flange to concrete compression centroid
    y_conc_in = deck_rib_height_in + t_c - (a_in / 2.0)

    # Steel internal force adjustment for plastic equilibrium: C + C_s = T_s
    c_s_kips = max(0.0, (t_steel_kips - c_kips) / 2.0)
    p_fy_kips = fy_ksi * bf_in * tf_in

    if c_s_kips == 0.0:
        # PNA lies in the concrete slab
        pna_loc = "in_slab"
        mn_steel_kip_in = t_steel_kips * (d_beam_in / 2.0)
    elif c_s_kips <= p_fy_kips:
        # PNA lies in the steel top flange
        pna_loc = "in_flange"
        y_f_in = c_s_kips / (fy_ksi * bf_in)
        mn_steel_kip_in = t_steel_kips * (d_beam_in / 2.0) - (2.0 * c_s_kips * (y_f_in / 2.0))
    else:
        # PNA lies in the steel web
        pna_loc = "in_web"
        c_web_kips = c_s_kips - p_fy_kips
        y_w_in = c_web_kips / (fy_ksi * tw_in)
        mn_steel_kip_in = (
            t_steel_kips * (d_beam_in / 2.0)
            - (2.0 * p_fy_kips * (tf_in / 2.0))
            - (2.0 * c_web_kips * (tf_in + y_w_in / 2.0))
        )

    # Nominal flexural strength: moments taken about top of steel beam
    mn_kip_in = (c_kips * y_conc_in) + mn_steel_kip_in
    mn_kip_ft = mn_kip_in / 12.0
    phi_mn_kip_ft = phi * mn_kip_ft

    dcr = mu_kip_ft / phi_mn_kip_ft if phi_mn_kip_ft > 0 else float("inf")
    passed = dcr <= 1.0

    return {
        "pass": passed,
        "dcr": round(dcr, 4),
        "phi_mn_kip_ft": round(phi_mn_kip_ft, 2),
        "mn_kip_ft": round(mn_kip_ft, 2),
        "pna_location": pna_loc,
        "composite_pct": round(composite_pct, 2),
        "c_kips": round(c_kips, 2),
        "c_conc_kips": round(c_conc_kips, 2),
        "t_steel_kips": round(t_steel_kips, 2),
        "effective_slab_width_in": round(b_eff, 2),
        "a_in": round(a_in, 3),
    }
