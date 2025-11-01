README.md

MLOps Lab – GPA ➜ Percentage Converter

This lab implements a **GPA to Percentage Converter** system using **FastAPI**, **Python**, and **PyTest** — designed to demonstrate core MLOps concepts such as clean project structuring, testing, CI/CD integration, and API deployment.

Project Overview

This project converts **GPA values into percentages** across different grading schemes and supports:

-  **Single GPA Conversion**
-  **Batch Conversion for multiple GPAs**
-  **Weighted GPA Calculation from multiple courses**
-  **Support for multiple grading schemes (4.0 scale, 10-point CGPA, affine conversions)**
-  **REST API endpoints built with FastAPI**
-  **Automated Unit Tests (PyTest)**
-  **Continuous Integration via GitHub Actions**

## Folder Structure

```

Github_Labs/
├── src/
│   ├── **init**.py
│   ├── api.py                # FastAPI endpoints
│   ├── converter.py          # Core logic for GPA → Percentage conversion
│   ├── schemes.py            # Predefined grading schemes
│
├── test/
│   ├── test_api.py           # API endpoint unit tests
│   ├── test_converter.py     # Logic and function tests
│   └── conftest.py           # Ensures src/ is discoverable
│
├── workflows/
│   └── ci.yml                # GitHub Actions CI pipeline for automated testing
│
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation (this file)
└── .gitignore                # Ignore venvs, caches, models, etc.

````

Setup Instructions

1. Create & Activate Virtual Environment
**Windows (CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
````

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

Run Tests

To verify functionality and API correctness:

```bash
python -m pytest -q
```

All tests should pass 

Run the FastAPI Application

Start the server:

```bash
uvicorn src.api:app --reload
```

Then open the interactive docs:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example Requests

### **Single GPA Conversion**

**POST** `/convert`

```json
{
  "gpa": 3.8,
  "scheme": "us_4_linear"
}
```

**Response**

```json
{
  "percentage": 95.0
}
```

### **Batch GPA Conversion**

**POST** `/batch`

```json
{
  "gpas": [4.0, 3.0, 2.5],
  "scheme": "us_4_linear"
}
```

**Response**

```json
{
  "percentages": [100.0, 75.0, 62.5]
}
```

**Weighted GPA from Transcript**

**POST** `/transcript`

```json
{
  "courses": [
    {"grade": "A", "credits": 3},
    {"grade": "B+", "credits": 4},
    {"grade": 3.0, "credits": 3}
  ],
  "grade_scale_max": 4.0,
  "scheme": "us_4_linear"
}
```

**Response**

```json
{
  "weighted_gpa": 3.58,
  "percentage": 89.5
}
```

---

CI/CD Workflow (GitHub Actions)

Each push automatically triggers:

* Environment setup
* Dependency installation
* PyTest run for all modules

You can view CI results in your GitHub repository’s **Actions** tab.

Key Learnings

* How to structure a clean Python MLOps-style project
* How to write and run automated unit tests
* How to build a FastAPI microservice
* How to integrate CI/CD using GitHub Actions
* How to version control experiments and code effectively
