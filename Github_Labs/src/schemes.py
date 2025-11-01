from dataclasses import dataclass
from typing import Callable, Dict

@dataclass(frozen=True)
class Scheme:
    name: str
    max_gpa: float
    # Convert a GPA on this scheme to percentage
    gpa_to_pct: Callable[[float], float]

def _linear(max_gpa: float) -> Callable[[float], float]:
    def f(gpa: float) -> float:
        if gpa < 0 or gpa > max_gpa:
            raise ValueError(f"GPA must be within 0–{max_gpa}.")
        return round((gpa / max_gpa) * 100, 2)
    return f

def _affine(max_gpa: float, slope: float, intercept: float) -> Callable[[float], float]:
    # percentage = slope * gpa + intercept
    def f(gpa: float) -> float:
        if gpa < 0 or gpa > max_gpa:
            raise ValueError(f"GPA must be within 0–{max_gpa}.")
        return round(slope * gpa + intercept, 2)
    return f

SCHEMES: Dict[str, Scheme] = {
    "us_4_linear": Scheme(name="us_4_linear", max_gpa=4.0, gpa_to_pct=_linear(4.0)),
    "cgpa_10_linear": Scheme(name="cgpa_10_linear", max_gpa=10.0, gpa_to_pct=_linear(10.0)),
    "cgpa10_affine_9_5x": Scheme(name="cgpa10_affine_9_5x", max_gpa=10.0, gpa_to_pct=_affine(10.0, slope=9.5, intercept=0.0)),
}