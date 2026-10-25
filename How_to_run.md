# BullStreet: Quickstart & Operations Guide

## 1. Quick Launch (Bloomberg Terminal Interface)

To launch the full Bloomberg Terminal web application with backend API and dual ML engines:

```bash
cd /Users/srinidhi/BullStreet
source .venv/bin/activate
python run_terminal.py
```
* **Terminal Interface**: Opens automatically at [http://127.0.0.1:8000](http://127.0.0.1:8000).
* **API Documentation**: Interactive Swagger docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 2. CLI Training & Inference

```bash
# Train dual models and generate performance benchmarks
python -m src.model_engine

# Run terminal predictions for a specific IPO
python train_and_predict.py --ipo IPO_0816
python train_and_predict.py --ipo "cargosol"
```

---

## 3. Rebuilding the Dataset Pipeline

If raw sources or cleaning rules are modified:
```bash
python main.py --stage clean --fetch-extended
python main.py --stage split
python main.py --stage report
python main.py --stage audit
pytest tests/test_pipeline.py -v
```

## Rules
- Close any `.csv` files in spreadsheet viewers before rebuilding.
- Always use `--fetch-extended` with `clean`.
- Never run `--stage collect` (raw datasets are frozen and preserved).