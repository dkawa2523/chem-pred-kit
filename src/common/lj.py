from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ATM_IN_PA = 101325.0


class LJCalcError(ValueError):
    pass


def epsilon_over_k(Tc_K: Optional[float], Tb_K: Optional[float], method: str) -> Optional[float]:
    """
    Estimate epsilon/k [K].

    Implemented methods (aligned with chemicals.lennard_jones docs):
      - 'bird_critical': epsilon/k = 0.77 * Tc  (Bird, Stewart, Lightfoot)
      - 'bird_boiling':  epsilon/k = 1.15 * Tb (Bird, Stewart, Lightfoot)
      - 'flynn':         epsilon/k = 1.77 * Tc^(5/6) (Flynn; Stiel & Thodos paper reports it)
      - 'tee_gotoh_steward_1': epsilon/k = 0.7740 * Tc

    Notes:
      The coefficients come from CSP correlations documented in the 'chemicals' library docs.
    """
    method = method.lower()
    if method == "bird_critical":
        if Tc_K is None:
            return None
        return 0.77 * float(Tc_K)
    if method == "bird_boiling":
        if Tb_K is None:
            return None
        return 1.15 * float(Tb_K)
    if method == "flynn":
        if Tc_K is None:
            return None
        return 1.77 * (float(Tc_K) ** (5.0 / 6.0))
    if method == "tee_gotoh_steward_1":
        if Tc_K is None:
            return None
        return 0.7740 * float(Tc_K)
    raise LJCalcError(f"Unknown epsilon method: {method}")


def sigma_angstrom(Tc_K: Optional[float], Pc_Pa: Optional[float], method: str) -> Optional[float]:
    """
    Estimate sigma [Å].

    Implemented methods:
      - 'bird_critical': sigma = 2.44 * (Tc/Pc_atm)^(1/3)
        (Bird, Stewart, Lightfoot; original Pc units are atmospheres)

    For other sigma methods you may add correlations later.
    """
    method = method.lower()
    if method == "bird_critical":
        if Tc_K is None or Pc_Pa is None:
            return None
        Pc_atm = float(Pc_Pa) / ATM_IN_PA
        if Pc_atm <= 0:
            return None
        return 2.44 * ((float(Tc_K) / Pc_atm) ** (1.0 / 3.0))
    raise LJCalcError(f"Unknown sigma method: {method}")


def compute_lj(
    Tc_K: Optional[float],
    Pc_Pa: Optional[float],
    Tb_K: Optional[float],
    epsilon_method: str = "bird_critical",
    sigma_method: str = "bird_critical",
) -> Dict[str, Optional[float]]:
    """Compute LJ parameters as a dict."""
    eps = epsilon_over_k(Tc_K=Tc_K, Tb_K=Tb_K, method=epsilon_method)
    sig = sigma_angstrom(Tc_K=Tc_K, Pc_Pa=Pc_Pa, method=sigma_method)
    return {
        "lj_epsilon_over_k_K": eps,
        "lj_sigma_A": sig,
        "lj_epsilon_method": epsilon_method,
        "lj_sigma_method": sigma_method,
    }
