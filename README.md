# Buffett Evaluator
🧠 AI-powered stock evaluator: discover which companies match Buffett's investment logic, explained by SHAP.

---

## 🔍 Obiettivo

Dato un set di **fondamentali finanziari**, la app:

* Classifica una stock come `Buffett-like` oppure `Avoid`
* Mostra **la probabilità della classificazione**
* Spiega **quali feature influenzano maggiormente** la decisione con SHAP
* Genera **una spiegazione in linguaggio naturale** tipo:
  “✅ AAPL è Buffett-like per: ROE elevato, PEG basso e margini solidi”

---

## 🌐 Demo online

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your_username/Buffett_Evaluator/main/app.py)

> ⚠️ Sostituisci `your_username` con il tuo GitHub user una volta deployata.

---

## 📸 Screenshot

![screenshot](https://your-screenshot-url.com/screenshot.png)

---

## 🧪 Dataset e modello

* ✅ Features prese da **Finviz** (es. P/E, ROE, P/B, EPS growth, Debt/Equity...)
* ✅ Pre-elaborazione con pandas
* ✅ Modello: `XGBoostClassifier` con hyperparameter tuning
* ✅ Oversampling minor class + explainability con SHAP

---

## ⚙️ Come usare la demo

1. Sposta gli slider nella sidebar sinistra per inserire i valori fondamentali della stock
2. Guarda la **predizione binaria** a destra (Buffett-like / Avoid)
3. Leggi la spiegazione testuale generata automaticamente
4. Apri il grafico SHAP per vedere le feature più influenti

---

## 🛠 Come eseguirla in locale

```bash
git clone https://github.com/your_username/Buffett_Evaluator.git
cd Buffett_Evaluator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Struttura del progetto

```
Buffett_Evaluator/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── buffett_model_balanced.pkl
├── data/
│   └── fundamentals_ready_binary.csv
└── src/
    ├── scrape_finviz.ipynb
    └── buffettclassification.ipynb
```

---

## 📘 Licenza

MIT License — open for improvement and contributions.

---

## ✨ Credits

Creato da [Dante Trabassi](https://github.com/DanteTrb) · Powered by SHAP + Streamlit + XGBoost
