#!/bin/bash
source .venv/bin/activate

echo "==================================="
echo "  CUET MATH SOLVER - DATA AUDITOR  "
echo "==================================="
python auditor.py

echo ""
echo "==================================="
echo "  CUET MATH SOLVER - E2E PIPELINE  "
echo "==================================="
python main_pipeline.py synthetic_q11.jpg --no-bank
