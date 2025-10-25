# CBR Credit Scoring Demo (ReliefF + Weighted k-NN)

A small, explainable credit-scoring demo that uses **ReliefF** to learn feature
weights and a **Case-Based Reasoning / k-Nearest Neighbors (k-NN)** classifier
to predict credit class (1 = Good, 0 = Bad) from three attributes:

- `X1` = Income Level (`Low`, `Medium`, `High`)
- `X2` = Employment Type (`Salaried`, `Self_Employed`, `Contract`, `Unemployed`)
- `X3` = Loan Purpose (`Home`, `Auto`, `Education`, `Personal`, `Medical`)

## Repository structure
├─ 54_instance_data_knn.py # driver: builds weights, evaluates, interactive predict
├─ CBR_model.py # weighted k-NN / CBR classifier
├─ ReliefF.py # ReliefF feature weighting
├─ 54_instance_raw_data_credit_score_DEMO.xlsx # (optional) small demo dataset
├─ requirements.txt
└─ README.md

## Quick start

### 1) Create a virtual environment (recommended)
```bash
# Windows PowerShell / VS Code terminal
python -m venv .venv 
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

##Run the demo
python 54_instance_data_knn.py


