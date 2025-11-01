import pytest
from src.converter import weighted_gpa, convert_gpa_to_percentage, batch_convert_gpas, compute_transcript_percentage

def test_weighted_gpa_basic():
    courses = [
        {"grade": "A", "credits": 3},
        {"grade": "B+", "credits": 3},
        {"grade": 3.0, "credits": 4},
    ]
    gpa = weighted_gpa(courses, grade_scale_max=4.0)
    assert 0 <= gpa <= 4.0
    assert round(gpa, 2) == round(((4.0*3)+(3.3*3)+(3.0*4))/(3+3+4), 2)

def test_convert_schemes():
    assert convert_gpa_to_percentage(4.0, "us_4_linear") == 100.0
    assert convert_gpa_to_percentage(10.0, "cgpa_10_linear") == 100.0
    assert convert_gpa_to_percentage(8.0, "cgpa10_affine_9_5x") == 76.0  # 9.5*8

def test_batch():
    out = batch_convert_gpas([4.0, 3.0], "us_4_linear")
    assert out == [100.0, 75.0]

def test_transcript_end_to_end():
    courses = [{"grade": "A-", "credits": 4}, {"grade": "B", "credits": 3}]
    gpa, pct = compute_transcript_percentage(courses, grade_scale_max=4.0, scheme_name="us_4_linear")
    assert 0 <= gpa <= 4.0
    assert 0 <= pct <= 100.0

def test_bad_inputs():
    with pytest.raises(ValueError):
        weighted_gpa([], 4.0)
    with pytest.raises(ValueError):
        weighted_gpa([{"grade": "A", "credits": 0}], 4.0)
    with pytest.raises(ValueError):
        weighted_gpa([{"grade": 5.0, "credits": 3}], 4.0)