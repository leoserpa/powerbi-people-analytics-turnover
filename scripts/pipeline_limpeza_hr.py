"""
Limpeza e saneamento do dataset IBM HR Attrition para Power BI e ML.
1470 registos, 35 colunas originais. Saida em Parquet e CSV.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("../dados/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUTPUT_PARQUET = Path("../dados/processed/hr_attrition_clean.parquet")
OUTPUT_CSV = Path("../dados/processed/hr_attrition_clean.csv")

# Estas 3 colunas tem o mesmo valor em todas as linhas (variancia zero).
# Nao discriminam nada, so ocupam espaco.
COLUNAS_VARIANCIA_ZERO = ["EmployeeCount", "Over18", "StandardHours"]

# Chave primaria. Util no Power BI para DISTINCTCOUNT, mas em ML causa overfitting.
COLUNA_ID = "EmployeeNumber"

# Texto repetitivo gasta menos RAM como tipo category do pandas.
COLUNAS_CATEGORICAS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

# Valores pequenos (max ~60). Cabem em int8 (-128 a 127).
COLUNAS_INT8 = [
    "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TrainingTimesLastYear", "WorkLifeBalance",
]

# Anos de experiencia/empresa. Max ~40, int16 chega e sobra.
COLUNAS_INT16 = [
    "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Valores monetarios (DailyRate ate ~1500, MonthlyIncome ate ~20k).
COLUNAS_INT32 = ["DailyRate", "HourlyRate", "MonthlyIncome", "MonthlyRate"]

# Mapas para converter escalas numericas em texto legivel no Power BI.
# A coluna numerica original fica intacta para correlacoes.
MAPA_SATISFACAO = {1: "Baixo", 2: "Medio", 3: "Alto", 4: "Muito Alto"}
MAPA_BALANCE = {1: "Mau", 2: "Regular", 3: "Bom", 4: "Excelente"}
MAPA_PERFORMANCE = {1: "Baixo", 2: "Abaixo da Media", 3: "Excelente", 4: "Excecional"}
MAPA_EDUCACAO = {
    1: "Abaixo da Licenciatura", 2: "Licenciatura",
    3: "Mestrado", 4: "Doutoramento", 5: "Pos-Doc",
}

VARIAVEIS_SATISFACAO = [
    "EnvironmentSatisfaction", "JobSatisfaction",
    "JobInvolvement", "RelationshipSatisfaction",
]

# Traducao das colunas para portugues (Power BI consome nomes em PT).
# Aplicada no final, depois de todas as transformacoes.
TRADUCAO_COLUNAS = {
    "Age": "Idade",
    "Attrition": "Saida",
    "BusinessTravel": "FrequenciaViagem",
    "DailyRate": "TaxaDiaria",
    "Department": "Departamento",
    "DistanceFromHome": "DistanciaCasa",
    "Education": "Escolaridade",
    "EducationField": "AreaFormacao",
    "EnvironmentSatisfaction": "SatisfacaoAmbiente",
    "Gender": "Genero",
    "HourlyRate": "TaxaHora",
    "JobInvolvement": "EnvolvimentoCargo",
    "JobLevel": "NivelCargo",
    "JobRole": "Cargo",
    "JobSatisfaction": "SatisfacaoCargo",
    "MaritalStatus": "EstadoCivil",
    "MonthlyIncome": "RendimentoMensal",
    "MonthlyRate": "TaxaMensal",
    "NumCompaniesWorked": "NumEmpresasAnteriores",
    "OverTime": "HorasExtra",
    "PercentSalaryHike": "PercentAumentoSalarial",
    "PerformanceRating": "AvaliacaoDesempenho",
    "RelationshipSatisfaction": "SatisfacaoRelacionamento",
    "StockOptionLevel": "NivelOpcaoAccoes",
    "TotalWorkingYears": "AnosExperienciaTotal",
    "TrainingTimesLastYear": "FormacaoUltimoAno",
    "WorkLifeBalance": "EquilibrioVidaTrabalho",
    "YearsAtCompany": "AnosNaEmpresa",
    "YearsInCurrentRole": "AnosNoCargoActual",
    "YearsSinceLastPromotion": "AnosDesdeUltimaPromocao",
    "YearsWithCurrManager": "AnosComGestorActual",
    # Labels ordinais
    "EnvironmentSatisfaction_Label": "SatisfacaoAmbiente_Texto",
    "JobSatisfaction_Label": "SatisfacaoCargo_Texto",
    "JobInvolvement_Label": "EnvolvimentoCargo_Texto",
    "RelationshipSatisfaction_Label": "SatisfacaoRelacionamento_Texto",
    "WorkLifeBalance_Label": "EquilibrioVidaTrabalho_Texto",
    "PerformanceRating_Label": "AvaliacaoDesempenho_Texto",
    "Education_Label": "Escolaridade_Texto",
    # Coluna transformada
    "MonthlyIncome_Log": "RendimentoMensal_Log",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """Le o CSV. Falha se o ficheiro nao existir."""
    if not caminho.exists():
        raise FileNotFoundError(f"Ficheiro nao encontrado: {caminho.resolve()}")
    df = pd.read_csv(caminho)
    log.info("Carregado: %d linhas x %d colunas", df.shape[0], df.shape[1])
    return df


def verificar_qualidade(df: pd.DataFrame) -> None:
    """Conta nulos e duplicados. Nao altera dados."""
    nulos = df.isnull().sum().sum()
    duplicados = df.duplicated().sum()
    log.info("Nulos: %d | Duplicados: %d", nulos, duplicados)
    if nulos > 0:
        log.warning("Colunas com nulos:\n%s", df.isnull().sum()[df.isnull().sum() > 0])


def remover_variancia_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina colunas onde todos os valores sao iguais (Var=0)."""
    presentes = [c for c in COLUNAS_VARIANCIA_ZERO if c in df.columns]
    for col in presentes:
        log.info("  %s: %d valor unico -> fora", col, df[col].nunique())
    return df.drop(columns=presentes)


def remover_identificador(df: pd.DataFrame) -> pd.DataFrame:
    """Retira o ID unico para evitar overfitting em modelos."""
    if COLUNA_ID in df.columns:
        df = df.drop(columns=[COLUNA_ID])
        log.info("%s removida", COLUNA_ID)
    return df


def aplicar_downcasting(df: pd.DataFrame) -> pd.DataFrame:
    """Reduz tipos numericos e converte texto repetitivo para category."""
    for col in COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in COLUNAS_INT8:
        if col not in df.columns:
            continue
        # Seguranca: se algum valor nao couber em int8, usa int16 em vez de corromper
        if df[col].max() > 127 or df[col].min() < -128:
            log.warning("%s nao cabe em int8 (max=%d), usando int16", col, df[col].max())
            df[col] = df[col].astype(np.int16)
        else:
            df[col] = df[col].astype(np.int8)

    for col in COLUNAS_INT16:
        if col in df.columns:
            df[col] = df[col].astype(np.int16)

    for col in COLUNAS_INT32:
        if col in df.columns:
            df[col] = df[col].astype(np.int32)

    log.info("Downcasting aplicado")
    return df


def criar_labels_ordinais(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia escalas numericas (1-4, 1-5) para texto ordenado.

    Usa CategoricalDtype com ordered=True para que graficos
    respeitem a ordem natural (Baixo < Medio < Alto < Muito Alto)
    sem precisar de sort manual no Power BI.
    """
    tipo_sat = pd.CategoricalDtype(["Baixo", "Medio", "Alto", "Muito Alto"], ordered=True)
    tipo_bal = pd.CategoricalDtype(["Mau", "Regular", "Bom", "Excelente"], ordered=True)
    tipo_perf = pd.CategoricalDtype(
        ["Baixo", "Abaixo da Media", "Excelente", "Excecional"], ordered=True
    )
    tipo_edu = pd.CategoricalDtype(
        ["Abaixo da Licenciatura", "Licenciatura", "Mestrado", "Doutoramento", "Pos-Doc"],
        ordered=True,
    )

    for col in VARIAVEIS_SATISFACAO:
        if col in df.columns:
            df[f"{col}_Label"] = df[col].map(MAPA_SATISFACAO).astype(tipo_sat)

    if "WorkLifeBalance" in df.columns:
        df["WorkLifeBalance_Label"] = df["WorkLifeBalance"].map(MAPA_BALANCE).astype(tipo_bal)

    if "PerformanceRating" in df.columns:
        df["PerformanceRating_Label"] = (
            df["PerformanceRating"].map(MAPA_PERFORMANCE).astype(tipo_perf)
        )

    if "Education" in df.columns:
        df["Education_Label"] = df["Education"].map(MAPA_EDUCACAO).astype(tipo_edu)

    log.info("Labels ordinais criadas")
    return df


def tratar_assimetria(df: pd.DataFrame) -> pd.DataFrame:
    """log1p em MonthlyIncome para reduzir skewness.

    Em RH, salarios altos e colaboradores estagnados que saem sao os eventos
    mais caros. Remover outliers distorce a realidade, por isso ficam intactos.
    A transformacao so serve para modelos sensiveis a distribuicoes normais.
    """
    if "MonthlyIncome" not in df.columns:
        return df

    skew_antes = df["MonthlyIncome"].skew()
    df["MonthlyIncome_Log"] = np.log1p(df["MonthlyIncome"]).astype(np.float32)
    skew_depois = df["MonthlyIncome_Log"].skew()

    log.info("MonthlyIncome skew: %.3f -> %.3f", skew_antes, skew_depois)
    return df


def codificar_target(df: pd.DataFrame) -> pd.DataFrame:
    """Yes/No -> 1/0. Permite calcular taxa de turnover por media simples."""
    if "Attrition" not in df.columns:
        return df

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(np.int8)
    log.info("Attrition binaria. Turnover: %.1f%%", df["Attrition"].mean() * 100)
    return df


def traduzir_valores_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """Traduz os valores dentro das colunas categoricas para portugues.

    No Power BI, slicers e legendas mostram estes valores directamente.
    Se ficarem em ingles, o dashboard fica bilingue e confuso.
    """
    traducoes = {
        "BusinessTravel": {
            "Non-Travel": "Sem Viagem",
            "Travel_Rarely": "Viaja Raramente",
            "Travel_Frequently": "Viaja Frequentemente",
        },
        "Department": {
            "Sales": "Vendas",
            "Research & Development": "Investigacao e Desenvolvimento",
            "Human Resources": "Recursos Humanos",
        },
        "EducationField": {
            "Life Sciences": "Ciencias da Vida",
            "Medical": "Medicina",
            "Marketing": "Marketing",
            "Technical Degree": "Curso Tecnico",
            "Human Resources": "Recursos Humanos",
            "Other": "Outro",
        },
        "Gender": {
            "Male": "Masculino",
            "Female": "Feminino",
        },
        "MaritalStatus": {
            "Single": "Solteiro(a)",
            "Married": "Casado(a)",
            "Divorced": "Divorciado(a)",
        },
        "OverTime": {
            "Yes": "Sim",
            "No": "Nao",
        },
        "JobRole": {
            "Sales Executive": "Executivo de Vendas",
            "Research Scientist": "Investigador",
            "Laboratory Technician": "Tecnico de Laboratorio",
            "Manufacturing Director": "Director de Producao",
            "Healthcare Representative": "Representante de Saude",
            "Manager": "Gestor",
            "Sales Representative": "Representante de Vendas",
            "Research Director": "Director de Investigacao",
            "Human Resources": "Recursos Humanos",
        },
    }

    for col, mapa in traducoes.items():
        if col in df.columns:
            df[col] = df[col].map(mapa).fillna(df[col]).astype("category")

    log.info("Valores categoricos traduzidos para PT")
    return df


def criar_faixas_powerbi(df: pd.DataFrame) -> pd.DataFrame:
    """Cria faixas pre-calculadas para evitar bins dinamicos no Power BI.

    Bins no Power BI sao lentos e nao permitem ordenacao personalizada.
    Melhor trazer ja calculados do Python.
    """
    if "Age" in df.columns:
        bins_idade = [17, 25, 35, 45, 55, 65]
        labels_idade = ["18-25", "26-35", "36-45", "46-55", "56-65"]
        df["FaixaEtaria"] = pd.cut(
            df["Age"], bins=bins_idade, labels=labels_idade
        ).astype("category")

    if "MonthlyIncome" in df.columns:
        bins_salario = [0, 3000, 6000, 10000, 15000, 25000]
        labels_salario = ["Ate 3k", "3k-6k", "6k-10k", "10k-15k", "Acima 15k"]
        df["FaixaSalarial"] = pd.cut(
            df["MonthlyIncome"], bins=bins_salario, labels=labels_salario
        ).astype("category")

    if "DistanceFromHome" in df.columns:
        bins_dist = [0, 5, 10, 20, 30]
        labels_dist = ["Perto (0-5)", "Medio (6-10)", "Longe (11-20)", "Muito Longe (21+)"]
        df["FaixaDistancia"] = pd.cut(
            df["DistanceFromHome"], bins=bins_dist, labels=labels_dist
        ).astype("category")

    if "YearsAtCompany" in df.columns:
        bins_anos = [-1, 2, 5, 10, 20, 50]
        labels_anos = ["0-2 anos", "3-5 anos", "6-10 anos", "11-20 anos", "20+ anos"]
        df["FaixaAntiguidade"] = pd.cut(
            df["YearsAtCompany"], bins=bins_anos, labels=labels_anos
        ).astype("category")

    log.info("Faixas para Power BI criadas")
    return df


def traduzir_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas de ingles para portugues.

    So renomeia as que existem no DataFrame (ignora as que ja foram removidas).
    Facilita leitura no Power BI por utilizadores que nao dominam ingles.
    """
    mapa_presente = {k: v for k, v in TRADUCAO_COLUNAS.items() if k in df.columns}
    df = df.rename(columns=mapa_presente)
    log.info("Colunas traduzidas: %d de %d", len(mapa_presente), len(df.columns))
    return df


def exportar(df: pd.DataFrame) -> None:
    """Grava Parquet (comprimido, colunar) e CSV (compatibilidade universal)."""
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Gravado: %s, %s", OUTPUT_PARQUET, OUTPUT_CSV)


def imprimir_sumario(df: pd.DataFrame, mem_antes: int) -> None:
    """Mostra reducao de memoria e estrutura final."""
    mem_final = df.memory_usage(deep=True).sum()
    reducao = (1 - mem_final / mem_antes) * 100

    print(f"\n{'=' * 50}")
    print(f"  Registos: {df.shape[0]} | Colunas: {df.shape[1]}")
    print(f"  RAM: {mem_antes / 1024:.0f} KB -> {mem_final / 1024:.0f} KB ({reducao:.0f}% menos)")
    print(f"{'=' * 50}\n")
    df.info()


def executar_pipeline() -> pd.DataFrame:
    """Corre o pipeline completo de limpeza."""
    df = carregar_dataset(INPUT_FILE)
    mem_antes = df.memory_usage(deep=True).sum()

    verificar_qualidade(df)
    df = remover_variancia_zero(df)
    df = remover_identificador(df)
    df = aplicar_downcasting(df)
    df = criar_labels_ordinais(df)
    df = tratar_assimetria(df)
    df = codificar_target(df)
    df = traduzir_valores_categoricos(df)
    df = criar_faixas_powerbi(df)
    df = traduzir_colunas(df)
    exportar(df)
    imprimir_sumario(df, mem_antes)

    return df


if __name__ == "__main__":
    try:
        executar_pipeline()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    except Exception as e:
        log.exception("Falha no pipeline: %s", e)
        sys.exit(1)
