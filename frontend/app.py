"""
Streamlit Frontend — 4 pages:
  1. Prédiction individuelle
  2. Prédiction batch (upload CSV)
  3. Dashboard monitoring / drift
  4. Métriques MLflow
"""

import io
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config

cfg = load_config()
API_URL    = cfg["streamlit"]["api_base_url"]
MLFLOW_URI = cfg["streamlit"]["mlflow_tracking_uri"]

st.set_page_config(
    page_title="Loan Risk Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🏦 Loan Risk Monitor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction individuelle", "📂 Prédictions batch", "📊 Dashboard monitoring", "📈 Métriques MLflow"],
)

# ── API helpers ───────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ API non disponible. Lance `uvicorn api.main:app --reload`"
    except Exception as e:
        return None, f"❌ Erreur: {e}"


def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return None


# ── Sidebar API status ────────────────────────────────────────────────────────
health = api_health()
if health and health.get("status") == "ok":
    st.sidebar.success("✅ API connectée")
    st.sidebar.caption(f"Régression v{health.get('regression_model_version', '?')}")
    st.sidebar.caption(f"Classification v{health.get('classification_model_version', '?')}")
else:
    st.sidebar.error("🔴 API hors ligne")

st.sidebar.markdown("---")
st.sidebar.caption(f"MLflow: {MLFLOW_URI}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prédiction individuelle
# ══════════════════════════════════════════════════════════════════════════════

def render_risk_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Score", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f39c12"},
                {"range": [60, 100], "color": "#e74c3c"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0))
    return fig


def render_shap_chart(shap_features: list):
    if not shap_features:
        return None
    df_shap = pd.DataFrame(shap_features)
    df_shap = df_shap.sort_values("shap_value", key=abs, ascending=True)
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_shap["shap_value"]]
    fig = go.Figure(go.Bar(
        x=df_shap["shap_value"],
        y=df_shap["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="SHAP — Top contributeurs",
        xaxis_title="Impact sur le score",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


if page == "🔮 Prédiction individuelle":
    st.title("🔮 Prédiction individuelle")
    st.markdown("Remplis le formulaire pour obtenir une évaluation de risque et de décision d'approbation.")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Informations personnelles")
            Age                   = st.slider("Âge", 18, 80, 35)
            MaritalStatus         = st.selectbox("Statut marital", ["Single", "Married", "Divorced", "Widowed"])
            NumberOfDependents    = st.number_input("Nombre de dépendants", 0, 10, 0)
            EducationLevel        = st.selectbox("Niveau d'études", ["High School", "Bachelor's", "Master's", "Doctorate", "Associate"])
            EmploymentStatus      = st.selectbox("Statut emploi", ["Employed", "Self-Employed", "Unemployed"])
            Experience            = st.number_input("Expérience (ans)", 0.0, 50.0, 5.0)
            JobTenure             = st.number_input("Ancienneté poste (ans)", 0.0, 40.0, 3.0)
            HomeOwnershipStatus   = st.selectbox("Logement", ["Own", "Rent", "Mortgage", "Other"])

        with col2:
            st.subheader("Finances")
            AnnualIncome          = st.number_input("Revenu annuel ($)", 0.0, 500000.0, 60000.0, step=1000.0)
            MonthlyIncome         = AnnualIncome / 12
            SavingsAccountBalance = st.number_input("Épargne ($)", 0.0, 500000.0, 10000.0, step=500.0)
            CheckingAccountBalance= st.number_input("Compte courant ($)", 0.0, 100000.0, 2000.0, step=100.0)
            TotalAssets           = st.number_input("Actifs totaux ($)", 0.0, 2000000.0, 80000.0, step=1000.0)
            TotalLiabilities      = st.number_input("Passifs totaux ($)", 0.0, 1000000.0, 30000.0, step=1000.0)
            NetWorth              = TotalAssets - TotalLiabilities
            MonthlyDebtPayments   = st.number_input("Paiements dettes mensuels ($)", 0.0, 10000.0, 500.0)
            DebtToIncomeRatio     = st.slider("Ratio dette/revenu", 0.0, 2.0, 0.3, 0.01)
            TotalDebtToIncomeRatio= st.slider("Ratio total dette/revenu", 0.0, 5.0, 0.5, 0.01)

        with col3:
            st.subheader("Prêt & Crédit")
            LoanAmount            = st.number_input("Montant prêt ($)", 1000.0, 500000.0, 20000.0, step=500.0)
            LoanDuration          = st.slider("Durée prêt (mois)", 6, 360, 60)
            LoanPurpose           = st.selectbox("Objet du prêt", ["Home", "Auto", "Education", "Business", "Personal", "Debt Consolidation"])
            MonthlyLoanPayment    = st.number_input("Paiement mensuel prêt ($)", 0.0, 10000.0, 400.0)
            BaseInterestRate      = st.slider("Taux de base (%)", 0.0, 20.0, 3.5, 0.1)
            InterestRate          = st.slider("Taux appliqué (%)", 0.0, 30.0, 5.5, 0.1)
            CreditScore           = st.slider("Score de crédit", 300, 850, 680)
            CreditCardUtilizationRate = st.slider("Utilisation carte crédit", 0.0, 1.0, 0.3, 0.01)
            NumberOfOpenCreditLines = st.number_input("Lignes de crédit ouvertes", 0, 20, 3)
            NumberOfCreditInquiries = st.number_input("Demandes de crédit récentes", 0, 20, 1)
            BankruptcyHistory     = st.selectbox("Historique faillite", [0, 1])
            PreviousLoanDefaults  = st.number_input("Défauts de paiement passés", 0, 10, 0)
            PaymentHistory        = st.slider("Historique paiement (score)", 0.0, 10.0, 7.0, 0.1)
            LengthOfCreditHistory = st.number_input("Durée historique crédit (ans)", 0.0, 40.0, 5.0)
            UtilityBillsPaymentHistory = st.slider("Paiement factures utilitaires", 0.0, 1.0, 0.9, 0.01)

        submitted = st.form_submit_button("🚀 Analyser le dossier", use_container_width=True)

    if submitted:
        payload = {
            "Age": Age, "AnnualIncome": AnnualIncome, "CreditScore": CreditScore,
            "EmploymentStatus": EmploymentStatus, "EducationLevel": EducationLevel,
            "Experience": Experience, "LoanAmount": LoanAmount, "LoanDuration": LoanDuration,
            "MaritalStatus": MaritalStatus, "NumberOfDependents": NumberOfDependents,
            "HomeOwnershipStatus": HomeOwnershipStatus, "MonthlyDebtPayments": MonthlyDebtPayments,
            "CreditCardUtilizationRate": CreditCardUtilizationRate,
            "NumberOfOpenCreditLines": NumberOfOpenCreditLines,
            "NumberOfCreditInquiries": NumberOfCreditInquiries,
            "DebtToIncomeRatio": DebtToIncomeRatio, "BankruptcyHistory": BankruptcyHistory,
            "LoanPurpose": LoanPurpose, "PreviousLoanDefaults": PreviousLoanDefaults,
            "PaymentHistory": PaymentHistory, "LengthOfCreditHistory": LengthOfCreditHistory,
            "SavingsAccountBalance": SavingsAccountBalance,
            "CheckingAccountBalance": CheckingAccountBalance, "TotalAssets": TotalAssets,
            "TotalLiabilities": TotalLiabilities, "MonthlyIncome": MonthlyIncome,
            "UtilityBillsPaymentHistory": UtilityBillsPaymentHistory, "JobTenure": JobTenure,
            "NetWorth": NetWorth, "BaseInterestRate": BaseInterestRate,
            "InterestRate": InterestRate, "MonthlyLoanPayment": MonthlyLoanPayment,
            "TotalDebtToIncomeRatio": TotalDebtToIncomeRatio,
        }

        with st.spinner("Analyse en cours..."):
            risk_result, risk_err       = api_post("/predict/risk", payload)
            approval_result, approval_err = api_post("/predict/approval", payload)

        if risk_err:
            st.error(risk_err)
        else:
            st.markdown("---")
            col_a, col_b, col_c = st.columns([1, 1, 1])

            with col_a:
                st.plotly_chart(render_risk_gauge(risk_result["risk_score"]), use_container_width=True)
                level = risk_result["risk_level"]
                colors = {"Low": "green", "Medium": "orange", "High": "red"}
                st.markdown(f"<h3 style='text-align:center;color:{colors[level]}'>Risque {level}</h3>", unsafe_allow_html=True)

            with col_b:
                approved = approval_result["loan_approved"] if approval_result else None
                proba    = approval_result["approval_probability"] if approval_result else 0
                if approved is not None:
                    if approved:
                        st.success(f"✅ Prêt APPROUVÉ\nProbabilité : {proba:.1%}")
                    else:
                        st.error(f"❌ Prêt REFUSÉ\nProbabilité d'approbation : {proba:.1%}")
                    fig_proba = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Prob. approbation (%)"},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": "#2ecc71" if approved else "#e74c3c"},
                               "steps": [{"range": [0, 50], "color": "#fadbd8"}, {"range": [50, 100], "color": "#d5f5e3"}]},
                    ))
                    fig_proba.update_layout(height=250, margin=dict(t=30, b=0))
                    st.plotly_chart(fig_proba, use_container_width=True)

            with col_c:
                if risk_result.get("shap_top_features"):
                    shap_fig = render_shap_chart(risk_result["shap_top_features"])
                    if shap_fig:
                        st.plotly_chart(shap_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prédictions batch
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Prédictions batch":
    st.title("📂 Prédictions batch")
    st.markdown("Upload un fichier CSV pour prédire le risque et l'approbation sur plusieurs dossiers à la fois.")

    uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.subheader(f"Aperçu — {len(df_preview)} lignes")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("🚀 Lancer les prédictions batch", use_container_width=True):
            with st.spinner(f"Prédiction sur {len(df_preview)} dossiers..."):
                try:
                    uploaded.seek(0)
                    r = requests.post(
                        f"{API_URL}/predict/batch",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                        timeout=120,
                    )
                    r.raise_for_status()
                    result_df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
                    st.success(f"✅ {len(result_df)} prédictions générées")

                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    if "PredictedRiskScore" in result_df.columns:
                        col1.metric("RMSE Score moyen", f"{result_df['PredictedRiskScore'].mean():.1f}")
                        col2.metric("Score max", f"{result_df['PredictedRiskScore'].max():.1f}")
                    if "PredictedApproval" in result_df.columns:
                        approval_rate = result_df["PredictedApproval"].mean()
                        col3.metric("Taux approbation", f"{approval_rate:.1%}")
                        col4.metric("Dossiers approuvés", int(result_df["PredictedApproval"].sum()))

                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv_bytes = result_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        data=csv_bytes,
                        file_name=f"predictions_{uploaded.name}",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # Distribution charts
                    if "PredictedRiskScore" in result_df.columns:
                        fig1 = px.histogram(result_df, x="PredictedRiskScore", nbins=50,
                                           title="Distribution des RiskScores prédits",
                                           color_discrete_sequence=["steelblue"])
                        st.plotly_chart(fig1, use_container_width=True)

                    if "RiskLevel" in result_df.columns:
                        counts = result_df["RiskLevel"].value_counts()
                        fig2 = px.pie(values=counts.values, names=counts.index,
                                     title="Répartition des niveaux de risque",
                                     color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
                        st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur batch: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dashboard monitoring / drift
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard monitoring":
    st.title("📊 Dashboard Monitoring & Drift")

    # ── Check drift report files ──
    from pathlib import Path
    drift_dir = Path("reports/drift/")
    drift_files = sorted(drift_dir.glob("drift_summary_*.json"), reverse=True) if drift_dir.exists() else []

    if not drift_files:
        st.info("Aucun rapport de drift disponible. Lance `python src/drift_detection.py --data <path>` d'abord.")
    else:
        latest = drift_files[0]
        with open(latest) as f:
            drift_data = json.load(f)

        st.subheader(f"Dernier rapport — {drift_data['timestamp']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Features analysées", drift_data["n_features_checked"])
        col2.metric("Features driftées", drift_data["n_drifted_features"],
                    delta=f"PSI > {drift_data['psi_threshold']}", delta_color="inverse")
        col3.metric("PSI moyen", f"{drift_data['overall_mean_psi']:.4f}")
        retrain = drift_data["retrain_needed"]
        col4.metric("Réentraînement", "🔴 Requis" if retrain else "✅ Non requis")

        if drift_data["drifted_features"]:
            st.warning(f"⚠️ Features en drift : {', '.join(drift_data['drifted_features'])}")

        # PSI bar chart
        psi_df = pd.DataFrame([
            {"feature": k, "psi": v}
            for k, v in drift_data["feature_psi"].items()
        ]).sort_values("psi", ascending=False).head(20)

        fig = px.bar(
            psi_df, x="psi", y="feature", orientation="h",
            color="psi",
            color_continuous_scale=["green", "orange", "red"],
            range_color=[0, 0.4],
            title="PSI par feature (top 20)",
        )
        fig.add_vline(x=drift_data["psi_threshold"], line_dash="dash", line_color="red",
                      annotation_text=f"Seuil {drift_data['psi_threshold']}")
        st.plotly_chart(fig, use_container_width=True)

        # Historical drift trend
        if len(drift_files) > 1:
            history = []
            for f_path in drift_files[:20]:
                with open(f_path) as f:
                    d = json.load(f)
                history.append({"timestamp": d["timestamp"], "mean_psi": d["overall_mean_psi"],
                                 "n_drifted": d["n_drifted_features"]})
            hist_df = pd.DataFrame(history)
            fig2 = px.line(hist_df, x="timestamp", y="mean_psi",
                           title="Évolution du PSI moyen dans le temps", markers=True)
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Métriques MLflow
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Métriques MLflow":
    st.title("📈 Métriques MLflow")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        tab_reg, tab_clf = st.tabs(["📉 Régression", "🎯 Classification"])

        with tab_reg:
            st.subheader("Expérience : loan-risk-regression")
            try:
                runs = mlflow.search_runs(
                    experiment_names=[cfg["mlflow"]["experiment_regression"]],
                    order_by=["start_time DESC"],
                    max_results=20,
                )
                if runs.empty:
                    st.info("Aucun run trouvé. Lance un entraînement d'abord.")
                else:
                    cols_show = ["run_id", "start_time", "metrics.test_rmse", "metrics.test_mae", "metrics.test_r2"]
                    cols_show = [c for c in cols_show if c in runs.columns]
                    st.dataframe(runs[cols_show].rename(columns={
                        "metrics.test_rmse": "RMSE",
                        "metrics.test_mae": "MAE",
                        "metrics.test_r2": "R²",
                    }), use_container_width=True)

                    if "metrics.test_rmse" in runs.columns:
                        fig = px.line(runs.sort_values("start_time"),
                                     x="start_time", y="metrics.test_rmse",
                                     title="Évolution RMSE par run", markers=True)
                        fig.update_yaxes(title="RMSE (↓ mieux)")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de charger les runs: {e}")

        with tab_clf:
            st.subheader("Expérience : loan-approval-classification")
            try:
                runs_clf = mlflow.search_runs(
                    experiment_names=[cfg["mlflow"]["experiment_classification"]],
                    order_by=["start_time DESC"],
                    max_results=20,
                )
                if runs_clf.empty:
                    st.info("Aucun run trouvé.")
                else:
                    cols_show = ["run_id", "start_time", "metrics.test_roc_auc", "metrics.test_f1"]
                    cols_show = [c for c in cols_show if c in runs_clf.columns]
                    st.dataframe(runs_clf[cols_show].rename(columns={
                        "metrics.test_roc_auc": "ROC-AUC",
                        "metrics.test_f1": "F1",
                    }), use_container_width=True)

                    if "metrics.test_roc_auc" in runs_clf.columns:
                        fig2 = px.line(runs_clf.sort_values("start_time"),
                                      x="start_time", y="metrics.test_roc_auc",
                                      title="Évolution ROC-AUC par run", markers=True)
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de charger les runs: {e}")

        # ── Model Registry ──
        st.subheader("🗂️ Model Registry")
        for reg_name in [cfg["mlflow"]["model_registry_regression"], cfg["mlflow"]["model_registry_classification"]]:
            try:
                versions = client.search_model_versions(f"name='{reg_name}'")
                if versions:
                    df_reg = pd.DataFrame([{
                        "version": v.version,
                        "stage": v.current_stage,
                        "created": v.creation_timestamp,
                        "run_id": v.run_id[:8] + "...",
                    } for v in versions])
                    st.markdown(f"**{reg_name}**")
                    st.dataframe(df_reg, use_container_width=True)
            except Exception:
                st.caption(f"{reg_name}: aucun modèle enregistré")

    except Exception as e:
        st.error(f"Connexion MLflow impossible ({MLFLOW_URI}): {e}")
        st.info("Lance le serveur MLflow avec : `mlflow server --host 127.0.0.1 --port 5000`")