import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# === CARICA MODELLO E DATI ===
model = joblib.load("models/buffett_model_balanced.pkl")
reference = pd.read_csv("data/fundamentals_ready_binary.csv")
X_ref = reference.drop(columns=["Ticker", "buffett_class", "buffett_binary"], errors="ignore")
medians = X_ref.median()
explainer = shap.Explainer(model, X_ref)

st.set_page_config(page_title="Buffett Evaluator", layout="wide")
st.title("🧠 Buffett Evaluator")
st.write("Valuta una stock in base ai principi fondamentali dello stile di investimento di Warren Buffett.")

# === FORM DI INPUT ===
st.sidebar.header("📥 Inserisci i dati fondamentali della stock")
input_data = {}
for feature in X_ref.columns:
    min_val = float(X_ref[feature].quantile(0.05))
    max_val = float(X_ref[feature].quantile(0.95))
    default = float(medians[feature])
    input_data[feature] = st.sidebar.slider(feature, min_val, max_val, default)

# === PREDIZIONE ===
x = pd.DataFrame([input_data])
pred = model.predict(x)[0]
proba = model.predict_proba(x)[0][1]

# === SPIEGAZIONE SHAP ===
shap_val = explainer(x)
contribs = shap_val[0].values

# Filtra contributi positivi
threshold = 0.02
pos_feats = [(f, contribs[i]) for i, f in enumerate(x.columns) if contribs[i] > threshold]
pos_feats = sorted(pos_feats, key=lambda x: -x[1])[:5]

# Frase spiegazione
if pred == 0:
    result_text = "🔴 La stock **non** è considerata Buffett-like."
    reason = "Motivo principale: mancano contributi positivi rilevanti."
elif not pos_feats:
    result_text = "🟡 La stock è Buffett-like, ma senza feature dominanti."
    reason = "Analisi incerta: nessuna feature supera la soglia di influenza."
else:
    result_text = "✅ La stock è **Buffett-like**."
    descr = [f"**{f}** elevato" if x[f].values[0] > medians[f] else f"**{f}** basso" for f, _ in pos_feats]
    reason = "Motivazione: " + ", ".join(descr)

# === OUTPUT ===
st.subheader("🧾 Risultato")
st.markdown(result_text)
st.write(f"Probabilità: {proba:.2f}")
st.markdown(reason)

# === SHAP PLOT ===
with st.expander("🔎 Visualizza contributi SHAP"):
    #st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.plots.bar(shap_val, max_display=10)
    st.pyplot(bbox_inches="tight")
