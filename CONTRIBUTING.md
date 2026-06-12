# Contributing to Semantic Movie Recommendation

Thank you for your interest in contributing! This guide will help you set up the project, understand the codebase, and submit high-quality contributions.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setting Up the Development Environment](#2-setting-up-the-development-environment)
3. [Project Structure](#3-project-structure)
4. [Code Style](#4-code-style)
5. [Adding New Features](#5-adding-new-features)
6. [Testing](#6-testing)
7. [Embedding New Movies](#7-embedding-new-movies)
8. [Reporting Bugs](#8-reporting-bugs)

---

## 1. Introduction

**Semantic Movie Recommendation** is a 2nd-year B.Tech NLP project that uses
[`sentence-transformers`](https://www.sbert.net/) to embed movie descriptions
and cosine similarity to surface semantically relevant recommendations.

Contributions that improve the recommendation quality, add new data, fix bugs,
or improve documentation are all welcome.

---

## 2. Setting Up the Development Environment

### Prerequisites

- Python **3.10** or later
- `git` installed
- A virtual environment tool (`venv` or `conda`)

### Steps

```bash
# 1. Fork this repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/Semantic-Movie-Recommendation.git
cd Semantic-Movie-Recommendation

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. (Optional) Install development extras
pip install pytest pytest-cov flake8 httpx

# 5. Verify everything works
pytest tests/ -v
```

### Running the API locally

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` to explore the interactive Swagger UI.

---

## 3. Project Structure

```
Semantic-Movie-Recommendation/
│
├── backend/                    # Core Python source code
│   ├── app.py                  # FastAPI REST API
│   ├── recommendation_engine.py
│   ├── embedding_service.py
│   ├── text_preprocessor.py
│   ├── generate_embeddings.py  # CLI script
│   └── evaluation.py
│
├── embeddings/                 # Data directory
│   └── movies_metadata.csv     # 250-movie dataset
│
├── notebooks/                  # Jupyter exploration notebooks
│   ├── 01_embedding_exploration.ipynb
│   └── 02_recommendation_evaluation.ipynb
│
├── tests/                      # Unit tests
│   ├── test_text_preprocessor.py
│   └── test_recommendation_engine.py
│
├── docs/                       # Documentation
│   └── api_reference.md
│
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── CONTRIBUTING.md             # This file
├── LICENSE
└── README.md
```

---

## 4. Code Style

We follow **PEP 8** with a maximum line length of **100 characters**.

### Key conventions

| Convention | Detail |
|---|---|
| **Line length** | Max 100 chars (`--max-line-length=100`) |
| **Type hints** | Required on all public functions and methods |
| **Docstrings** | Google-style, required for all public APIs |
| **Logging** | Use the `logging` module; no bare `print()` in library code |
| **Imports** | Standard library → third-party → local, separated by blank lines |
| **Naming** | `snake_case` for functions/variables, `PascalCase` for classes |

### Running the linter

```bash
flake8 backend/ tests/ --max-line-length=100 --ignore=E501,W503
```

Fix all flake8 errors before submitting a pull request. The CI pipeline
will reject any PR with linting errors.

### Pre-commit (recommended)

```bash
pip install pre-commit
pre-commit install
```

---

## 5. Adding New Features

### Branching strategy

Use the following prefixes for branch names:

```
feature/short-description     # New feature
fix/issue-number-description  # Bug fix
docs/what-you-changed         # Documentation updates
refactor/module-name          # Code refactoring
```

Example:
```bash
git checkout -b feature/genre-weighted-similarity
```

### Pull request process

1. **Fork** the repository and create a feature branch from `main`.
2. Write your code following the style guide above.
3. Add or update **unit tests** in `tests/`.
4. Run `pytest tests/ -v` and ensure all tests pass.
5. Run `flake8` and fix any warnings.
6. Submit a pull request with a clear description of:
   - What you changed and why
   - How it was tested
   - Any caveats or known limitations

### Merging policy

- PRs require at least one approving review.
- All CI checks must pass.
- Squash-merge preferred for feature branches.

---

## 6. Testing

### Running all tests

```bash
pytest tests/ -v
```

### Running a specific test file

```bash
pytest tests/test_text_preprocessor.py -v
```

### Running with coverage

```bash
pytest tests/ --cov=backend --cov-report=term-missing
```

### Writing new tests

- Place new test files in the `tests/` directory.
- Name test files `test_<module_name>.py`.
- Name test classes `Test<ClassOrFeatureName>`.
- Name test methods `test_<what_is_being_tested>`.
- Use `unittest.TestCase` as the base class.
- Mock external dependencies (file I/O, model downloads) using `unittest.mock`.

**Example:**

```python
class TestMyNewFeature(unittest.TestCase):
    def test_returns_expected_value(self) -> None:
        """Brief description of what is being verified."""
        result = my_function(input_value)
        self.assertEqual(result, expected_value)
```

---

## 7. Embedding New Movies

To add more movies to the recommendation system:

### Step 1 — Update the metadata CSV

Edit `embeddings/movies_metadata.csv` and append new rows. Ensure:
- `movie_id` is unique and incremented from the last existing ID.
- `genre` uses the format `"Genre1/Genre2"` (e.g. `"Sci-Fi/Thriller"`).
- `plot` is 1–3 sentences and surrounded by double quotes if it contains commas.
- `rating` is a float between `0.0` and `10.0`.

### Step 2 — Regenerate embeddings

Run the embedding script from the project root:

```bash
python backend/generate_embeddings.py \
    --metadata embeddings/movies_metadata.csv \
    --output_dir embeddings \
    --model_name all-MiniLM-L6-v2 \
    --batch_size 64
```

This saves:
- `embeddings/movie_embeddings.npy` — The embedding matrix
- `embeddings/movie_documents.txt` — One document per line

### Step 3 — Commit both files

```bash
git add embeddings/movies_metadata.csv embeddings/movie_documents.txt
# Note: movie_embeddings.npy is .gitignored (large binary)
git commit -m "data: add 20 new action movies to metadata"
```

> **Note:** The `.npy` embedding file is excluded from version control by
> `.gitignore`. Each developer regenerates it locally.

---

## 8. Reporting Bugs

If you encounter a bug, please open a GitHub Issue with:

1. **Title**: A concise one-line summary (e.g., `recommend() crashes when query is None`).
2. **Environment**: Python version, OS, installed package versions (`pip freeze`).
3. **Steps to reproduce**: Minimal code snippet or steps that trigger the issue.
4. **Expected behaviour**: What you expected to happen.
5. **Actual behaviour**: What actually happened (include the full traceback).
6. **Possible fix**: Optional — if you have an idea, share it.

### Issue labels

| Label | Meaning |
|---|---|
| `bug` | Confirmed bug |
| `enhancement` | New feature request |
| `documentation` | Docs improvement |
| `good first issue` | Good entry point for new contributors |
| `help wanted` | Extra attention needed |

---

*Thank you for helping make Semantic Movie Recommendation better!*
