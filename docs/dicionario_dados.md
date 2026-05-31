# Dicionário de Dados - HR Attrition

Tabela `processed`: 1.470 registros e 52 colunas (originais traduzidas + features
de ML + colunas de scoring e calculadas).
Fonte original: IBM HR Analytics Employee Attrition & Performance (Kaggle).

## Sumário

- [Variável Alvo](#variável-alvo)
- [Demográficas](#demográficas)
- [Cargo e Departamento](#cargo-e-departamento)
- [Compensação](#compensação)
- [Satisfação e Desempenho](#satisfação-e-desempenho)
- [Formação](#formação)
- [Features Compostas (ML)](#features-compostas-ml)
- [Scoring ML](#scoring-ml)
- [Ficheiros de Suporte (Power BI)](#ficheiros-de-suporte-power-bi)
- [Notas de Modelagem](#notas-de-modelagem)

## Variável Alvo

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Saida | int8 | 1 = colaborador saiu, 0 = ficou. Taxa: 16,1% |

## Demográficas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Idade | int8 | Idade do colaborador (18-60) |
| Genero | category | Masculino / Feminino |
| EstadoCivil | category | Solteiro(a) / Casado(a) / Divorciado(a) |
| DistanciaCasa | int8 | Km entre casa e trabalho (1-29) |
| FaixaEtaria | category | 18-25, 26-35, 36-45, 46-55, 56-65 |
| FaixaDistancia | category | Perto / Médio / Longe / Muito Longe |

## Cargo e Departamento

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Departamento | category | Vendas / Investigação e Desenvolvimento / Recursos Humanos |
| Cargo | category | 9 funções traduzidas para PT |
| NivelCargo | int8 | Hierarquia 1-5 (1=júnior, 5=diretor) |
| AnosNaEmpresa | int16 | Antiguidade na empresa |
| AnosNoCargoActual | int16 | Tempo no cargo atual |
| AnosComGestorActual | int16 | Tempo com o mesmo gestor |
| AnosDesdeUltimaPromocao | int16 | Estagnação na carreira |
| AnosExperienciaTotal | int16 | Experiência profissional total |
| NumEmpresasAnteriores | int8 | Quantas empresas antes desta |
| FaixaAntiguidade | category | 0-2, 3-5, 6-10, 11-20, 20+ anos |
| FrequenciaViagem | category | Sem Viagem / Viaja Raramente / Viaja Frequentemente |

## Compensação

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| RendimentoMensal | int32 | Salário mensal bruto |
| TaxaDiaria | int32 | Taxa diária (uso interno) |
| TaxaHora | int32 | Taxa por hora |
| TaxaMensal | int32 | Taxa mensal (uso interno) |
| PercentAumentoSalarial | int8 | Último aumento salarial (%) |
| NivelOpcaoAccoes | int8 | Stock options (0-3) |
| HorasExtra | category | Sim / Não |
| FaixaSalarial | category | Até 3k, 3k-6k, 6k-10k, 10k-15k, Acima 15k |

## Satisfação e Desempenho

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| SatisfacaoAmbiente | int8 | 1-4 (escala numérica) |
| SatisfacaoCargo | int8 | 1-4 |
| EnvolvimentoCargo | int8 | 1-4 |
| SatisfacaoRelacionamento | int8 | 1-4 |
| EquilibrioVidaTrabalho | int8 | 1-4 |
| AvaliacaoDesempenho | int8 | 1-4 (neste dataset só 3 e 4) |
| SatisfacaoAmbiente_Texto | category | Baixo / Médio / Alto / Muito Alto |
| SatisfacaoCargo_Texto | category | Baixo / Médio / Alto / Muito Alto |
| EnvolvimentoCargo_Texto | category | Baixo / Médio / Alto / Muito Alto |
| SatisfacaoRelacionamento_Texto | category | Baixo / Médio / Alto / Muito Alto |
| EquilibrioVidaTrabalho_Texto | category | Mau / Regular / Bom / Excelente |
| AvaliacaoDesempenho_Texto | category | Baixo / Abaixo da Média / Excelente / Excecional |

## Formação

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Escolaridade | int8 | 1-5 (escala numérica) |
| Escolaridade_Texto | category | Abaixo da Licenciatura até Pós-Doc |
| AreaFormacao | category | Ciências da Vida, Medicina, Marketing, Curso Técnico, RH, Outro |
| FormacaoUltimoAno | int8 | Sessões de formação no último ano |

## Features Compostas (ML)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| SalarioPorNivel | float32 | RendimentoMensal / NivelCargo (sub-remuneração) |
| EstagnacaoRelativa | float32 | AnosDesdeUltimaPromocao / Idade |
| ProporcaoCarreiraEmpresa | float32 | AnosNaEmpresa / AnosExperienciaTotal |
| SatisfacaoMedia | float32 | Média das 3 satisfações (ambiente, cargo, relacionamento) |
| PressaoTrabalho | int8 | HorasExtra * (5 - EquilibrioVidaTrabalho) |

## Scoring ML

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| RiscoSaida_Prob | float64 | Probabilidade de saída prevista pelo modelo (0-100%) |
| RiscoSaida_Classe | category | Baixo / Médio / Alto (thresholds fixos sobre RiscoSaida_Prob) |
| RiscoSaida_Pct | float64 | Coluna calculada: probabilidade em formato decimal (0-1) para formatação em % no Power BI |
| Risco_Texto | string | Coluna calculada: rótulo de risco usado nos visuais |
| RendimentoMensal_Log | float32 | log1p do salário (para modelos sensíveis à normalidade) |

Classificação de risco (thresholds de negócio aplicados no pipeline de scoring):

| Classe | Regra | Colaboradores |
|--------|-------|---------------|
| Alto | RiscoSaida_Prob >= 40% | 312 |
| Médio | RiscoSaida_Prob entre 20% e 40% | 155 |
| Baixo | RiscoSaida_Prob < 20% | 1.003 |

## Ficheiros de Suporte (Power BI)

| Ficheiro / Tabela | Descrição |
|----------|-----------|
| ml_comparacao_modelos | Métricas dos 5 modelos testados (Modelo, Accuracy, Precision, Recall, F1_Score, AUC_ROC) |
| ml_feature_importance | Importância de cada feature (Feature, Importancia, Importancia_Pct) |
| ml_shap_values.csv | SHAP values por colaborador (explicabilidade individual) |

> Nota: os CSVs em `dados/processed/` usam `;` como separador (locale PT). Para ler
> em pandas: `pd.read_csv(ficheiro, sep=";")`.

## Notas de Modelagem

- **Validação**: cross-validation 5-fold estratificado; SMOTE aplicado dentro do
  pipeline de CV (sem vazamento para os folds de teste).
- **Modelo final**: XGBoost + SMOTE + Optuna (80 trials). AUC-ROC 0.819, recall 60,8%.
- **RiscoSaida_Prob é in-sample**: gerado pelo modelo treinado no dataset completo,
  portanto otimista; as métricas honestas estão na tabela ml_comparacao_modelos.
- **Features de ruído removidas do treino**: TaxaDiaria, TaxaHora, TaxaMensal
  (correlação ~0 com a saída). Permanecem na tabela para referência.
