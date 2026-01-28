from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into element counts.

    Supports:
      - Simple Hill-like formulas: C6H6, Na2SO4, C10H24S2Sn
      - Parentheses with multipliers: (CH3)2CO, Ca(OH)2
      - Dot/hydrate separators: CuSO4·5H2O or CuSO4.5H2O

    Limitations:
      - Does not handle charge annotations like 'NH4+' explicitly (will ignore '+')
      - Does not handle isotopes like 'D' vs 'H' distinctly (treats as element if present)
    """
    if formula is None:
        return {}
    s = str(formula).strip()
    if not s:
        return {}

    # Normalize hydrate separators
    s = s.replace("·", ".")
    # Remove whitespace and charge signs
    s = re.sub(r"\s+", "", s)
    s = s.replace("+", "").replace("-", "")

    # Split on dot and sum parts
    parts = s.split(".")
    total: Dict[str, int] = defaultdict(int)
    for part in parts:
        if not part:
            continue
        # Handle leading multiplier like "5H2O"
        m = re.match(r"^(\d+)(.*)$", part)
        if m:
            mult = int(m.group(1))
            part = m.group(2)
        else:
            mult = 1
        counts = _parse_formula_with_parentheses(part)
        for el, n in counts.items():
            total[el] += mult * n
    return dict(total)


def _parse_formula_with_parentheses(s: str) -> Dict[str, int]:
    stack: List[Dict[str, int]] = [defaultdict(int)]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append(defaultdict(int))
            i += 1
        elif ch == ")":
            i += 1
            # Read multiplier
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            mult = int(num) if num else 1
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] += mult * n
        else:
            m = _ELEMENT_RE.match(s, i)
            if not m:
                # Unknown char; skip safely
                i += 1
                continue
            el = m.group(1)
            n_str = m.group(2)
            n = int(n_str) if n_str else 1
            stack[-1][el] += n
            i = m.end()
    return dict(stack[0])


def elements_string(counts: Dict[str, int]) -> str:
    """Return a canonical sorted element list string, e.g. 'Br,C,H,O'."""
    if not counts:
        return ""
    els = sorted(counts.keys())
    return ",".join(els)


def n_elements(counts: Dict[str, int]) -> int:
    return len(counts) if counts else 0


def get_elements_from_mol(mol) -> Dict[str, int]:
    """Get element counts from an RDKit Mol."""
    if mol is None:
        return {}
    counts: Dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return dict(counts)


def murcko_scaffold_smiles(mol) -> str:
    """Compute Murcko scaffold SMILES for scaffold split."""
    if MurckoScaffold is None or mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return ""
