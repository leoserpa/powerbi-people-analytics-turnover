"""
Pipeline ML Optimizado - Predicao de Turnover.

Melhorias sobre o baseline:
- Feature engineering (variaveis compostas)
- Remocao de features de ruido
- SMOTE para balancear classes
- Optuna para tuning de hiperparametros do XGBoost
- Comparacao antes/depois da optimizacao
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_FILE = Path("../dados/processed/hr_attrition_clean.parquet")
OUTPUT_DIR = Path("../dados/processed")
MODEL_DIR = Path("../models")

# Features que nao ajudam (correlacao ~0 com Attrition neste dataset)
FEATURES_RUIDO = ["TaxaDiaria", "TaxaHora", "TaxaMensal"]


def carregar_dados() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nao encontrado: {INPUT_FILE.resolve()}")
    df = pd.read_parquet(INPUT_FILE)
    log.info("Carregado: %d x %d", df.shape[0], df.shape[1])
    return df


def criar_features_compostas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variaveis que capturam relacoes entre colunas.

    Estas interacoes fazem sentido no contexto de RH:
    - Salario por nivel: quem ganha pouco para o seu nivel sai mais
    - Anos sem promocao / idade: estagnacao relativa
    - Satisfacao media: indicador geral de engagement
    """
    if "RendimentoMensal" in df.columns and "NivelCargo" in df.columns:
        # Quem ganha abaixo do esperado para o nivel tende a sair
        df["SalarioPorNivel"] = (df["RendimentoMensal"] / df["NivelCargo"].clip(lower=1)).astype(np.float32)

    if "AnosDesdeUltimaPromocao" in df.columns and "Idade" in df.columns:
        # Estagnacao relativa: muitos anos parado vs idade
        df["EstagnacaoRelativa"] = (df["AnosDesdeUltimaPromocao"] / df["Idade"].clip(lower=1)).astype(np.float32)

    if "AnosNaEmpresa" in df.columns and "AnosExperienciaTotal" in df.columns:
        # Proporcao da carreira nesta empresa (lealdade vs mobilidade)
        df["ProporcaoCarreiraEmpresa"] = (
            df["AnosNaEmpresa"] / df["AnosExperienciaTotal"].clip(lower=1)
        ).astype(np.float32)

    # Satisfacao media (indicador composto de engagement)
    cols_satisfacao = ["SatisfacaoAmbiente", "SatisfacaoCargo", "SatisfacaoRelacionamento"]
    cols_presentes = [c for c in cols_satisfacao if c in df.columns]
    if cols_presentes:
        df["SatisfacaoMedia"] = df[cols_presentes].mean(axis=1).astype(np.float32)

    if "EquilibrioVidaTrabalho" in df.columns and "HorasExtra" in df.columns:
        # Quem faz horas extra E tem mau equilibrio esta em risco
        df["PressaoTrabalho"] = (
            (df["HorasExtra"].astype(str) == "Sim").astype(np.int8) *
            (5 - df["EquilibrioVidaTrabalho"])
        ).astype(np.int8)

    log.info("Features compostas criadas")
    return df


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepara X e y com feature engineering e remocao de ruido."""
    # Remover colunas derivadas, labels, faixas e ruido
    colunas_excluir = [c for c in df.columns if c.endswith("_Texto")]
    colunas_excluir += [c for c in df.columns if c.startswith("Faixa")]
    colunas_excluir += ["Saida", "RendimentoMensal_Log"]
    colunas_excluir += [c for c in FEATURES_RUIDO if c in df.columns]

    y = df["Saida"].copy()
    X = df.drop(columns=[c for c in colunas_excluir if c in df.columns])

    # Codificar categoricas
    for col in X.select_dtypes(include="category").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()
    log.info("Features: %d (removidas %d de ruido, adicionadas compostas)",
             len(feature_names), len(FEATURES_RUIDO))
    return X, y, feature_names


def avaliar_baseline(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Avalia 4 modelos SEM SMOTE e SEM tuning (para comparacao)."""
    modelos = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            verbose=-1, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []

    for nome, modelo in modelos.items():
        y_pred = cross_val_predict(modelo, X, y, cv=cv, method="predict")
        y_proba = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
        resultados.append({
            "Modelo": nome,
            "Accuracy": round(accuracy_score(y, y_pred), 4),
            "Precision": round(precision_score(y, y_pred), 4),
            "Recall": round(recall_score(y, y_pred), 4),
            "F1_Score": round(f1_score(y, y_pred), 4),
            "AUC_ROC": round(roc_auc_score(y, y_proba), 4),
        })

    return pd.DataFrame(resultados)


def tunar_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 80) -> dict:
    """Usa Optuna para encontrar os melhores hiperparametros do XGBoost com SMOTE."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 8.0),
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": -1,
        }

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(**params)),
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    log.info("Optuna: %d trials com SMOTE + XGBoost...", n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info("Melhor AUC encontrado: %.4f", study.best_value)
    log.info("Melhores params: %s", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in study.best_params.items()})
    return study.best_params


def treinar_modelo_final(best_params: dict, X: pd.DataFrame, y: pd.Series):
    """Treina o modelo final com SMOTE + melhores hiperparametros."""
    best_params["eval_metric"] = "logloss"
    best_params["verbosity"] = 0
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    # Aplica SMOTE no dataset completo para treino final
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    log.info("SMOTE: %d -> %d amostras (classe 1: %d -> %d)",
             len(y), len(y_res), y.sum(), y_res.sum())

    modelo = XGBClassifier(**best_params)
    modelo.fit(X_res, y_res)
    log.info("Modelo final treinado")
    return modelo


def avaliar_modelo_final(modelo, X: pd.DataFrame, y: pd.Series, best_params: dict) -> pd.DataFrame:
    """Avalia o modelo optimizado com cross-validation (SMOTE dentro do CV)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_params_clean = {k: v for k, v in best_params.items()
                         if k not in ["eval_metric", "verbosity", "random_state", "n_jobs"]}
    best_params_clean["eval_metric"] = "logloss"
    best_params_clean["verbosity"] = 0
    best_params_clean["random_state"] = 42
    best_params_clean["n_jobs"] = -1

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(**best_params_clean)),
    ])

    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    resultado = {
        "Modelo": "XGBoost + SMOTE + Optuna",
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall": round(recall_score(y, y_pred), 4),
        "F1_Score": round(f1_score(y, y_pred), 4),
        "AUC_ROC": round(roc_auc_score(y, y_proba), 4),
    }
    return pd.DataFrame([resultado])


def calcular_feature_importance(modelo, feature_names: list[str]) -> pd.DataFrame:
    importancias = modelo.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importancia": importancias,
    }).sort_values("Importancia", ascending=False).reset_index(drop=True)
    df_imp["Importancia_Pct"] = (df_imp["Importancia"] / df_imp["Importancia"].sum() * 100).round(2)
    log.info("Top 5: %s", df_imp.head(5)["Feature"].tolist())
    return df_imp


def calcular_shap_values(modelo, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    log.info("SHAP calculado: %d x %d", *df_shap.shape)
    return df_shap


def gerar_scoring(modelo, X: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    probabilidades = modelo.predict_proba(X)[:, 1]
    df_score = df_original.copy()
    df_score["RiscoSaida_Prob"] = (probabilidades * 100).round(1)

    # Thresholds fixos baseados em logica de negocio:
    # Alto:  >= 40% (risco concreto, accao urgente)
    # Medio: 20-40% (monitorar, accao preventiva)
    # Baixo: < 20%  (sem accao imediata)
    df_score["RiscoSaida_Classe"] = pd.cut(
        df_score["RiscoSaida_Prob"],
        bins=[0, 20, 40, 100],
        labels=["Baixo", "Medio", "Alto"],
        include_lowest=True,
    ).astype("category")

    log.info("Scoring: %.1f%% alto, %.1f%% medio, %.1f%% baixo",
             (df_score["RiscoSaida_Classe"] == "Alto").mean() * 100,
             (df_score["RiscoSaida_Classe"] == "Medio").mean() * 100,
             (df_score["RiscoSaida_Classe"] == "Baixo").mean() * 100)
    return df_score


def exportar_tudo(df_comparacao, df_importance, df_shap, df_scoring, modelo) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df_comparacao.to_csv(OUTPUT_DIR / "ml_comparacao_modelos.csv", index=False, sep=";")
    df_importance.to_csv(OUTPUT_DIR / "ml_feature_importance.csv", index=False, sep=";")
    df_shap.to_csv(OUTPUT_DIR / "ml_shap_values.csv", index=False, sep=";")
    df_scoring.to_parquet(OUTPUT_DIR / "hr_attrition_scored.parquet", index=False)
    df_scoring.to_csv(OUTPUT_DIR / "hr_attrition_scored.csv", index=False, sep=";")
    joblib.dump(modelo, MODEL_DIR / "modelo_attrition_optimizado.pkl")
    log.info("Tudo exportado")


def executar_pipeline_ml() -> None:
    df = carregar_dados()

    # Feature engineering
    df = criar_features_compostas(df)

    X, y, feature_names = preparar_features(df)

    # Baseline (para comparacao)
    log.info("Avaliando baseline...")
    df_baseline = avaliar_baseline(X, y)
    log.info("Baseline AUC: %.4f", df_baseline["AUC_ROC"].iloc[0])

    # Tuning com Optuna + SMOTE
    best_params = tunar_xgboost(X, y, n_trials=80)

    # Avaliar modelo optimizado com CV
    df_optimizado = avaliar_modelo_final(modelo=None, X=X, y=y, best_params=best_params)

    # Tabela comparativa
    df_comparacao = pd.concat([df_baseline, df_optimizado], ignore_index=True)
    print("\n" + df_comparacao.to_string(index=False) + "\n")

    # Treinar modelo final no dataset completo
    modelo_final = treinar_modelo_final(best_params, X, y)

    # Artefactos para Power BI
    df_importance = calcular_feature_importance(modelo_final, feature_names)
    df_shap = calcular_shap_values(modelo_final, X, feature_names)
    df_scoring = gerar_scoring(modelo_final, X, df)

    exportar_tudo(df_comparacao, df_importance, df_shap, df_scoring, modelo_final)

    melhoria_auc = df_optimizado["AUC_ROC"].iloc[0] - df_baseline["AUC_ROC"].max()
    print(f"{'=' * 50}")
    print(f"  Baseline AUC:   {df_baseline['AUC_ROC'].max():.4f} (melhor dos 4)")
    print(f"  Optimizado AUC: {df_optimizado['AUC_ROC'].iloc[0]:.4f}")
    print(f"  Melhoria:       +{melhoria_auc:.4f}")
    print(f"  Alto risco:     {(df_scoring['RiscoSaida_Classe'] == 'Alto').sum()} colaboradores")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    try:
        executar_pipeline_ml()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    except Exception as e:
        log.exception("Falha: %s", e)
        sys.exit(1)
