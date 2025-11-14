from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# --- Material properties (simplified first-order values for prototype) ---
MATERIALS = {
    "LFP":     {"density_g_cm3": 3.6, "capacity_mAh_g": 160, "nominal_voltage_V": 3.3, "expansion_pct_100SOC": 1.5},
    "NMC811":  {"density_g_cm3": 4.8, "capacity_mAh_g": 190, "nominal_voltage_V": 3.7, "expansion_pct_100SOC": 2.0},
    "Graphite":{"density_g_cm3": 2.26,"capacity_mAh_g": 350, "expansion_pct_100SOC": 8.0},
    "Si":      {"density_g_cm3": 2.33,"capacity_mAh_g": 3000,"expansion_pct_100SOC": 300.0},
    "Al":      {"density_g_cm3": 2.70},
    "Cu":      {"density_g_cm3": 8.96},
}
DEFAULT_FOIL_THK_UM = {"cathode_Al": 15.0, "anode_Cu": 8.0}
DEFAULT_SEPARATOR   = {"thickness_um": 20.0, "porosity": 0.45, "density_g_cm3": 0.9}

# --- Data classes ---
@dataclass
class ElectrodeSpec:
    material: str
    thickness_um: float
    porosity: float
    active_frac: float = 0.96
    density_g_cm3: Optional[float] = None
    capacity_mAh_g: Optional[float] = None
    silicon_frac: float = 0.0  # for graphite blends (0..1)

@dataclass
class CellDesignInput:
    cathode: ElectrodeSpec
    anode: ElectrodeSpec
    n_layers: int
    area_cm2: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    separator_thickness_um: float = DEFAULT_SEPARATOR["thickness_um"]
    separator_porosity: float = DEFAULT_SEPARATOR["porosity"]
    n_p_ratio: float = 1.10
    cathode_foil_um: float = DEFAULT_FOIL_THK_UM["cathode_Al"]
    anode_foil_um: float = DEFAULT_FOIL_THK_UM["anode_Cu"]
    format_hint: str = "pouch"
    electrolyte: Optional[str] = None
    ambient_C: float = 25.0

# --- Helpers ---
def _mat_prop(name, key, default=None):
    if name in MATERIALS and key in MATERIALS[name]:
        return MATERIALS[name][key]
    if default is None:
        raise ValueError(f"Missing {key} for {name}")
    return default

def _area(inp: CellDesignInput) -> float:
    if inp.area_cm2 is not None:
        return inp.area_cm2
    if inp.width_mm is not None and inp.height_mm is not None:
        return (inp.width_mm * inp.height_mm) / 100.0  # mm² → cm²
    raise ValueError("Provide area_cm2 or width_mm+height_mm.")

def areal_loading_mg_cm2(e: ElectrodeSpec) -> float:
    thk_cm = e.thickness_um * 1e-4         # µm → cm
    solid_vol = thk_cm * (1.0 - e.porosity)
    dens = e.density_g_cm3 if e.density_g_cm3 is not None else _mat_prop(e.material, "density_g_cm3")
    return solid_vol * dens * 1000.0       # g/cm² → mg/cm²

def _cap_mAh_g(e: ElectrodeSpec) -> float:
    base = e.capacity_mAh_g if e.capacity_mAh_g is not None else _mat_prop(e.material, "capacity_mAh_g")
    # mixing rule for graphite + silicon blends
    if e.silicon_frac > 0 and e.material.lower() in ["graphite", "c"]:
        si_cap = _mat_prop("Si", "capacity_mAh_g")
        base = e.silicon_frac * si_cap + (1 - e.silicon_frac) * base
    return base

def areal_capacity_mAh_cm2(e: ElectrodeSpec) -> float:
    return (areal_loading_mg_cm2(e) * e.active_frac / 1000.0) * _cap_mAh_g(e)

# --- Core electrochem estimates (isothermal baseline at ~25 °C) ---
def cell_capacity_Ah(inp: CellDesignInput, V_hint=None):
    area = _area(inp)
    c_areal = areal_capacity_mAh_cm2(inp.cathode)
    a_areal = areal_capacity_mAh_cm2(inp.anode) / inp.n_p_ratio
    limiting = min(c_areal, a_areal)
    total_mAh = limiting * area * inp.n_layers
    cap_Ah = total_mAh / 1000.0
    Vnom = V_hint if V_hint is not None else MATERIALS.get(inp.cathode.material, {}).get("nominal_voltage_V", 3.6)
    return cap_Ah, Vnom

def layer_thickness_cm(inp: CellDesignInput) -> float:
    return (inp.cathode.thickness_um + inp.anode.thickness_um +
            inp.separator_thickness_um + inp.cathode_foil_um + inp.anode_foil_um) * 1e-4

def stack_thickness_cm(inp: CellDesignInput) -> float:
    return layer_thickness_cm(inp) * inp.n_layers

def mass_estimate_g(inp: CellDesignInput) -> float:
    area = _area(inp)
    m_cath = areal_loading_mg_cm2(inp.cathode) * area / 1000.0
    m_an   = areal_loading_mg_cm2(inp.anode)   * area / 1000.0
    cfoil  = inp.cathode_foil_um * 1e-4 * area * MATERIALS["Al"]["density_g_cm3"]
    afoil  = inp.anode_foil_um   * 1e-4 * area * MATERIALS["Cu"]["density_g_cm3"]
    sep    = inp.separator_thickness_um * 1e-4 * area * DEFAULT_SEPARATOR["density_g_cm3"]
    return (m_cath + m_an + cfoil + afoil + sep) * inp.n_layers

def volume_estimate_cm3(inp: CellDesignInput) -> float:
    return stack_thickness_cm(inp) * _area(inp)

def energy_density_metrics(inp: CellDesignInput) -> Dict[str, float]:
    cap_Ah, Vnom = cell_capacity_Ah(inp)
    E_Wh = cap_Ah * Vnom
    m_g = mass_estimate_g(inp)
    vol_cm3 = volume_estimate_cm3(inp)
    whkg = E_Wh / (m_g / 1000.0) if m_g > 0 else float("nan")
    whL  = E_Wh / (vol_cm3 / 1000.0) if vol_cm3 > 0 else float("nan")
    return {"capacity_Ah": cap_Ah, "V_nom": Vnom, "E_Wh": E_Wh, "Wh_per_kg": whkg, "Wh_per_L": whL}

# --- Temperature effects (prototype heuristics) ---
def temperature_effects(ambient_C: float, cathode_mat: str) -> Dict[str, float]:
    """
    Heuristic factors relative to ~25°C baseline.
    - Capacity factor: low at cold, ~1 around 25–30°C, slight drop at high temp (due to side reactions / polarization relief tradeoffs)
    - Resistance factor: higher when cold; moderate decrease near 30–35°C; rises again with overheating.
    - Ideal window recommendation depends mildly on chemistry.
    """
    T = ambient_C
    # Capacity factor
    if T <= -10:
        cap_fac = 0.55
    elif -10 < T <= 0:
        cap_fac = 0.70
    elif 0 < T <= 10:
        cap_fac = 0.85
    elif 10 < T <= 30:
        cap_fac = 1.00
    elif 30 < T <= 40:
        cap_fac = 0.98
    elif 40 < T <= 50:
        cap_fac = 0.95
    else:  # >50
        cap_fac = 0.90

    # Resistance (lower is better) as a multiplier vs baseline R25
    if T <= -10:
        r_fac = 2.2
    elif -10 < T <= 0:
        r_fac = 1.8
    elif 0 < T <= 10:
        r_fac = 1.4
    elif 10 < T <= 30:
        r_fac = 1.0
    elif 30 < T <= 40:
        r_fac = 0.95
    elif 40 < T <= 50:
        r_fac = 1.10
    else:
        r_fac = 1.25

    # Recommend ideal band by chemistry (simple)
    if cathode_mat.upper().startswith("NMC"):
        ideal = (20, 30)   # tighter band for high-nickel
    else:  # LFP and others
        ideal = (20, 35)

    # Risk notes
    high_risk = T >= 45
    cold_risk = T <= 0

    return {
        "capacity_factor": cap_fac,
        "resistance_factor": r_fac,
        "ideal_low_C": ideal[0],
        "ideal_high_C": ideal[1],
        "high_temp_risk": high_risk,
        "cold_temp_risk": cold_risk
    }

# --- Mechanical / Thermal heuristics (prototype-grade) ---
def swelling_percent(inp: CellDesignInput) -> float:
    def exp_pct(e: ElectrodeSpec):
        base = MATERIALS.get(e.material, {}).get("expansion_pct_100SOC", 2.0)
        if e.material.lower() in ["graphite", "c"] and e.silicon_frac > 0:
            si = MATERIALS["Si"]["expansion_pct_100SOC"]
            base = (1 - e.silicon_frac) * base + e.silicon_frac * si
        return base

    c = inp.cathode.thickness_um
    a = inp.anode.thickness_um
    s = inp.separator_thickness_um
    period = c + a + s + inp.cathode_foil_um + inp.anode_foil_um
    if period <= 0:
        return 0.0
    return (c / period) * exp_pct(inp.cathode) + (a / period) * exp_pct(inp.anode)

def temperature_rise_estimate(inp: CellDesignInput, c_rate: float = 3.0) -> float:
    area_m2 = _area(inp) / 1e4                       # cm² → m²
    thickness_m = stack_thickness_cm(inp) / 100.0     # cm → m
    k_eff = 0.5                                       # effective thermal conductivity (W/m-K), rough
    theta = thickness_m / (k_eff * area_m2) if (k_eff * area_m2) > 0 else float('inf')
    cap_Ah, _ = cell_capacity_Ah(inp)
    I = c_rate * cap_Ah                                # A
    R_ohm = 0.02 * (3.0 / max(cap_Ah, 1e-6))          # rough scaling for internal resistance vs capacity
    qdot = (I ** 2) * R_ohm                           # W
    return qdot * theta                                # ΔT ≈ q * thermal resistance

def feasibility_flags(inp: CellDesignInput):
    swell = swelling_percent(inp)
    dT_3C = temperature_rise_estimate(inp, 3.0)

    def flag(v, ok, warn):
        return "✅ OK" if v <= ok else ("⚠ borderline" if v <= warn else "❌ risky")

    return {
        "swelling_flag": flag(swell, 5.0, 10.0),
        "thermal_flag_3C": flag(dT_3C, 5.0, 8.0),
    }

# --- AI-style suggestions (now temperature-aware) ---
def ai_suggestions(inp: CellDesignInput, temp_fx: Dict[str, float]):
    sugg = []
    swell = swelling_percent(inp)
    dT_3C = temperature_rise_estimate(inp, 3.0)
    metrics_25 = energy_density_metrics(inp)
    cap_25 = metrics_25["capacity_Ah"]

    # Chem-specific electrolyte hints
    if inp.cathode.material.upper() == "LFP":
        sugg.append("Electrolyte: EC:EMC 3:7 with 1M LiPF6; consider LiFSI salt for >40 °C operation.")
    elif "NMC" in inp.cathode.material.upper():
        sugg.append("Electrolyte: EC:EMC 3:7 or EC:DEC with 1M LiPF6; add 1–2% VC for cathode stability.")

    # Silicon blends / expansion
    if inp.anode.material.lower() == "graphite" and (swell > 6.0 or inp.anode.silicon_frac > 0):
        sugg.append("Add 5–10% FEC to stabilize SEI under anode expansion; review binder (CMC/SBR) for Si blends.")
    if inp.anode.silicon_frac >= 0.05:
        sugg.append("For Si blends: consider LiFSI and adjust anode thickness −5–10% if swelling is high.")

    # Thermal heuristics
    if dT_3C > 5.0:
        sugg.append("Thermal: reduce coating thickness ~10% or +0.02 porosity; consider ceramic-coated separator or cooling path improvement.")

    # Energy targeting
    if cap_25 < 2.5:
        sugg.append("Energy is low: increase #layers or thickness to hit target Ah.")
    else:
        sugg.append("If energy suffices, trim anode ~5% to reduce swelling risk.")

    # Rate loading nudges
    if areal_loading_mg_cm2(inp.cathode) > 12.0:
        sugg.append("Cathode >12 mg/cm²: lower thickness slightly for better rate capability.")
    if areal_loading_mg_cm2(inp.anode) > 10.0:
        sugg.append("Anode >10 mg/cm²: increase porosity a bit or thin coating for kinetics.")

    # Temperature-aware advisories
    T = inp.ambient_C
    if temp_fx["cold_temp_risk"]:
        sugg.append("Cold operation detected (≤0 °C): expect power loss and higher impedance; pre-heat cell or derate C-rate.")
    if temp_fx["high_temp_risk"]:
        sugg.append("High ambient (≥45 °C): elevated side reactions—use high-temp electrolyte, reduce charge rates, and ensure thermal management.")
    sugg.append(f"Ideal operating window for this chemistry: {temp_fx['ideal_low_C']}–{temp_fx['ideal_high_C']} °C for balanced power, life, and safety.")

    return sugg

# --- Orchestrator ---
def design_pouch(inp: CellDesignInput) -> Dict[str, Any]:
    # Baseline metrics at ~25 °C
    metrics = energy_density_metrics(inp)
    swell = swelling_percent(inp)
    dT1 = temperature_rise_estimate(inp, 1.0)
    dT3 = temperature_rise_estimate(inp, 3.0)
    flags = feasibility_flags(inp)

    # Temperature effects & adjusted metrics
    tfx = temperature_effects(inp.ambient_C, inp.cathode.material)
    cap_eff = metrics["capacity_Ah"] * tfx["capacity_factor"]
    # crude power proxy ~ 1/R, so use 1/r_fac to indicate relative power availability
    power_rel_vs_25 = 1.0 / tfx["resistance_factor"]

    out = {
        "inputs": {
            "area_cm2": _area(inp), "n_layers": inp.n_layers,
            "cathode": asdict(inp.cathode), "anode": asdict(inp.anode),
            "separator_thickness_um": inp.separator_thickness_um,
            "separator_porosity": inp.separator_porosity, "n_p_ratio": inp.n_p_ratio,
            "electrolyte": inp.electrolyte or "not specified",
            "ambient_C": inp.ambient_C
        },
        "electrochem": metrics,
        "electrochem_temp_adjusted": {
            "effective_capacity_Ah_at_ambient": cap_eff,
            "relative_power_vs_25C": power_rel_vs_25,
            "capacity_factor": tfx["capacity_factor"],
            "resistance_factor": tfx["resistance_factor"],
        },
        "mechanical": {
            "swelling_pct_100SOC": swell,
            "stack_thickness_cm": stack_thickness_cm(inp)
        },
        "thermal": {"deltaT_1C_C": dT1, "deltaT_3C_C": dT3},
        "feasibility": flags,
        "temperature_guidance": {
            "ambient_C": inp.ambient_C,
            "ideal_low_C": tfx["ideal_low_C"],
            "ideal_high_C": tfx["ideal_high_C"],
            "high_temp_risk": tfx["high_temp_risk"],
            "cold_temp_risk": tfx["cold_temp_risk"]
        },
        "ai_suggestions": ai_suggestions(inp, tfx),
    }
    return out
