"""
Gera graficos para o README e documentacao do projecto.
Saida: pasta docs/imagens/
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
CORES = {
    "primaria": "#1B2838",
    "secundaria": "#2E86AB",
    "alerta": "#D64933",
    "sucesso": "#2E8B57",
    "neutro": "#6C757D",
    "fundo": "#F8F9FA",
}

OUTPUT_DIR = Path("../docs/imagens")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path("../dados/processed/hr_attrition_scored.parquet")
MODEL_FILE = Path("../models/modelo_attrition_optimizado.pkl")
IMPORTANCE_FILE = Path("../dados/processed/ml_feature_importance.csv")
COMPARACAO_FILE = Path("../dados/processed/ml_comparacao_modelos.csv")

FEATURES_RUIDO = ["TaxaDiaria", "TaxaHora", "TaxaMensal"]


def preparar_dados():
    """Carrega e prepara dados para os graficos."""
    df = pd.read_parquet(DATA_FILE)

    colunas_excluir = [c for c in df.columns if c.endswith("_Texto")]
    colunas_excluir += [c for c in df.columns if c.startswith("Faixa")]
    colunas_excluir += ["Saida", "RendimentoMensal_Log"]
    colunas_excluir += [c for c in FEATURES_RUIDO if c in df.columns]
    colunas_excluir += ["RiscoSaida_Prob", "RiscoSaida_Classe"]

    y = df["Saida"].copy()
    X = df.drop(columns=[c for c in colunas_excluir if c in df.columns])

    for col in X.select_dtypes(include="category").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return df, X, y


def grafico_matriz_confusao(X, y):
    """Matriz de confusao com cross-validation (resultados honestos)."""
    # Usa o mesmo pipeline do treino: SMOTE + XGBoost com os params do modelo
    modelo_base = joblib.load(MODEL_FILE)
    params = modelo_base.get_params()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(**{k: v for k, v in params.items() if k != "callbacks"})),
    ])

    # Predicoes out-of-fold: cada linha e prevista quando esta FORA do treino
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=["Ficou", "Saiu"],
                yticklabels=["Ficou", "Saiu"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao - XGBoost + SMOTE (Cross-Validation)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> matriz_confusao.png")


def grafico_roc_curve(X, y):
    """Curva ROC com cross-validation (resultados honestos)."""
    modelo_base = joblib.load(MODEL_FILE)
    params = modelo_base.get_params()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(**{k: v for k, v in params.items() if k != "callbacks"})),
    ])

    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=CORES["secundaria"], lw=2, label=f"XGBoost + SMOTE (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=CORES["neutro"], lw=1, linestyle="--", label="Aleatorio")
    ax.set_xlabel("Taxa Falsos Positivos")
    ax.set_ylabel("Taxa Verdadeiros Positivos")
    ax.set_title("Curva ROC (Cross-Validation)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curva_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> curva_roc.png")


def grafico_feature_importance():
    """Top 10 features mais importantes."""
    df_imp = pd.read_csv(IMPORTANCE_FILE).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_imp, x="Importancia_Pct", y="Feature", ax=ax, color=CORES["secundaria"])
    ax.set_xlabel("Importancia (%)")
    ax.set_ylabel("")
    ax.set_title("Top 10 Factores de Saida")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> feature_importance.png")


def grafico_comparacao_modelos():
    """Comparacao de metricas entre modelos."""
    df_comp = pd.read_csv(COMPARACAO_FILE)

    fig, ax = plt.subplots(figsize=(9, 5))
    metricas = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]
    x = np.arange(len(df_comp))
    width = 0.15

    for i, metrica in enumerate(metricas):
        ax.bar(x + i * width, df_comp[metrica], width, label=metrica)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_comp["Modelo"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Comparacao de Modelos")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparacao_modelos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> comparacao_modelos.png")


def grafico_shap_summary(X, y):
    """SHAP summary plot (beeswarm)."""
    modelo = joblib.load(MODEL_FILE)
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("SHAP - Impacto das Features na Predicao")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> shap_summary.png")


def grafico_distribuicao_risco(df):
    """Distribuicao da probabilidade de saida."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="RiscoSaida_Prob", hue="Saida", bins=30, ax=ax,
                 palette={0: CORES["sucesso"], 1: CORES["alerta"]}, alpha=0.7)
    ax.set_xlabel("Probabilidade de Saida (%)")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuicao do Risco por Classe Real")
    ax.legend(["Ficou", "Saiu"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribuicao_risco.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> distribuicao_risco.png")


def grafico_turnover_por_departamento(df):
    """Taxa de turnover por departamento."""
    turnover = df.groupby("Departamento")["Saida"].mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    turnover.plot(kind="barh", ax=ax, color=CORES["alerta"])
    ax.set_xlabel("Taxa de Turnover (%)")
    ax.set_ylabel("")
    ax.set_title("Turnover por Departamento")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "turnover_departamento.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> turnover_departamento.png")


if __name__ == "__main__":
    print("Gerando graficos...")
    df, X, y = preparar_dados()

    grafico_matriz_confusao(X, y)
    grafico_roc_curve(X, y)
    grafico_feature_importance()
    grafico_comparacao_modelos()
    grafico_shap_summary(X, y)
    grafico_distribuicao_risco(df)
    grafico_turnover_por_departamento(df)

    print(f"\nTodos os graficos em: {OUTPUT_DIR.resolve()}")
