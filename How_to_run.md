# BullStreet: Commands

Run in PowerShell.

## Setup (first time only)


cd C:\Projects\BullStreet
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install xgboost shap matplotlib


## Every new terminal


cd C:\Projects\BullStreet
.venv\Scripts\Activate.ps1


## Rebuild the data


python main.py --stage clean --fetch-extended
python main.py --stage split
python main.py --stage report
python main.py --stage audit
pytest tests/test_pipeline.py -v


## Train and predict


python train_baseline.py
python train_and_predict.py


Look up one IPO:


python train_and_predict.py --ipo IPO_0816


## Rules

- Close the CSV files in Excel before rebuilding.
- Always use `--fetch-extended` with `clean`.
- Never run `--stage collect`.