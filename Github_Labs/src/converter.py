from typing import List, Dict, Tuple
import numpy as np
from .schemes import SCHEMES, Scheme

LetterScale = {
    # simple 4.0 mapping (extend if needed)
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D": 1.0, "F": 0.0
}

def normalize_grade(grade) -> float:
    """Accept letter grades or numeric GPA points and return numeric on 4.0 scale if letter, else pass-through float."""
    if isinstance(grade, (int, float)):
        return float(grade)
    if isinstance(grade, str):
        g = grade.strip().upper()
        if g in LetterScale:
            return LetterScale[g]
        # numeric-like strings are allowed
        try:
            return float(g)
        except ValueError:
            pass
    raise ValueError(f"Unsupported grade value: {grade!r}")

def weighted_gpa(courses: List[Dict], grade_scale_max: float = 4.0) -> float:
    """
    courses: [{"grade": "A" or 3.7, "credits": 3}, ...]
    grade_scale_max: maximum of the numeric grades you pass (e.g., 4.0 or 10.0)
    Returns GPA normalized to the provided grade_scale_max.
    """
    if not courses:
        raise ValueError("At least one course is required.")
    totals = []
    credits = []
    for c in courses:
        if "grade" not in c or "credits" not in c:
            raise ValueError("Each course needs 'grade' and 'credits'.")
        g = normalize_grade(c["grade"])
        cr = float(c["credits"])
        if cr <= 0:
            raise ValueError("Credits must be positive.")
        if g < 0 or g > grade_scale_max:
            raise ValueError(f"Grade {g} out of 0–{grade_scale_max}.")
        totals.append(g * cr)
        credits.append(cr)
    gpa = sum(totals) / sum(credits)
    return round(gpa, 3)

def convert_gpa_to_percentage(gpa: float, scheme_name: str = "us_4_linear") -> float:
    if scheme_name not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme_name}'. Available: {list(SCHEMES)}")
    scheme: Scheme = SCHEMES[scheme_name]
    return scheme.gpa_to_pct(gpa)

def batch_convert_gpas(gpas: List[float], scheme_name: str = "us_4_linear") -> List[float]:
    return [convert_gpa_to_percentage(g, scheme_name) for g in gpas]

def compute_transcript_percentage(
    courses: List[Dict],
    grade_scale_max: float,
    scheme_name: str
) -> Tuple[float, float]:
    """
    Returns (weighted_gpa, percentage) where GPA is on the same max scale you provided.
    If your letter mapping was on 4.0 but you want to convert as 10.0 CGPA, pass grade_scale_max=10.0 and give numeric grades accordingly.
    """
    gpa = weighted_gpa(courses, grade_scale_max=grade_scale_max)
    pct = convert_gpa_to_percentage(gpa, scheme_name=scheme_name)
    return gpa, pct