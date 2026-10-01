<div align="center">

# camera-math-solver

**End-to-end computer vision and symbolic computation pipeline for solving printed and handwritten math problems directly from raw camera captures.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Computer Vision](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Symbolic Math](https://img.shields.io/badge/Math-SymPy-3B5526?style=flat-square)
![OCR](https://img.shields.io/badge/OCR-Tesseract_&_Paddle-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

</div>

---

## 📌 Overview

**camera-math-solver** is an autonomous mathematical reasoning pipeline that ingests camera images of mathematical questions (e.g. exams, worksheets), rectifies perspective distortion and orientation, extracts formulas and multiple-choice options into a canonical AST (`QuestionIR`), and independently computes solutions using symbolic computation (`SymPy`).

The system includes automated answer-key auditing that flags discrepancies between official answer keys and mathematical truth.

---

## ✨ Features

- **Automated Orientation & Rectification**: Evaluates 0/90/180/270-degree rotations using text coherence heuristics (Tesseract confidence scoring) without relying on unreliable EXIF metadata.
- **Perspective Distortion Correction**: Screen rectification to detect document borders, unwarp angles, and normalize lighting.
- **Canonical AST Representation**: Converts noisy raw OCR tokens into structured `QuestionIR` data structures containing question stems, expressions, and candidate options.
- **SymPy Symbolic Engine**: Solves algebraic, calculus, matrix, and arithmetic equations rigorously rather than relying on generative approximations.
- **Answer Key Auditor**: Compares symbolic solutions against provided answer keys to flag bad question keys, ambiguous wording, or missing correct options.

---

## 🏗️ Architecture

```
camera-math-solver/
├── main_pipeline.py         # End-to-end pipeline runner
├── image_preprocessing.py   # Automated orientation scoring & rotation
├── screen_rectifier.py      # Perspective warping and edge detection
├── ocr_ensemble.py          # Multi-model OCR coordinator (Tesseract / Paddle)
├── parser.py                # Mathematical AST parser and QuestionIR builder
├── solver_engine.py         # SymPy symbolic computation resolver
├── auditor.py               # Discrepancy detector against ground truth
└── models.py                # Pydantic data schemas and QuestionIR definition
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR installed locally:
  ```bash
  # macOS
  brew install tesseract
  ```

### Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/akularya6-del/camera-math-solver.git
   cd camera-math-solver
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install opencv-python sympy pytesseract pydantic pillow
   ```

3. **Run Pipeline**:
   ```bash
   python3 main_pipeline.py
   ```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
