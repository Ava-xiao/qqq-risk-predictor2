# QQQ Weekly Large Drawdown Risk Predictor

Interactive risk predictor for QQQ using logistic regression.  
**Streamlit Demo**:https://qqq-risk-predictor2-inajp6gguba4u598esgxmw.streamlit.app/#eb641ad6

## Project Overview
- Predicts whether QQQ's next-week maximum drawdown will exceed -2.68% (20th percentile of historical drawdowns).
- Combines **technical indicators**, **Reddit sentiment (FinBERT)**, **macro factors (VIX, yield spread)**, and **interaction features** (Price‑Sentiment Divergence, Risk Resonance).
- Rolling‑window validation (100 weeks train, 1 week test, 57 predictions) + cost‑sensitive threshold optimization (FN cost=10, FP cost=1).
- Final model: **Logistic Regression (C=0.01)** achieves **Recall 86.7%**, AUC 0.625, and **reduces max drawdown from 21.46% to 2.27%** in backtest.

## Key Results
| Metric | Buy & Hold | Model Strategy | Improvement |
|--------|------------|----------------|--------------|
| Max Drawdown | -21.46% | -2.27% | **-19.19%** |
| Sharpe Ratio | 0.73 | 1.00 | +0.28 |
| Calmar Ratio | 0.70 | 2.53 | +1.82 |
| Annualised Volatility | 20.73% | 5.71% | -15.02% |

## Repository Structure
- `app.py` – Streamlit application
- `final_model.pkl`, `features.pkl`, `base_features.pkl` – model and artifacts
- `experiments/` – full Jupyter notebook `full_experiment_pipeline.ipynb` and data
- `requirements.txt` – dependencies

## Data Sources
- QQQ price & VIX: Yahoo Finance
- Treasury yields: FRED (DGS10, DGS2)
- Reddit sentiment: Custom crawler + FinBERT

## How to Run Locally
```bash
git clone https://github.com/Ava-xiao/qqq-risk-predictor2.git
cd qqq-risk-predictor2
pip install -r requirements.txt
streamlit run app.py
