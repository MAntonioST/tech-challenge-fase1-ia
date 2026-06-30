# 🩺 Tech Challenge - Diagnóstico de Câncer de Mama

> **Fase 1 - IA Foundations**  
> **FIAP - Pós-Graduação em IA para DEVs**  
> **Autor:** Marco Antonio Teixeira

---

##  Descrição

Este projeto implementa um sistema de **Machine Learning** para apoiar o diagnóstico de câncer de mama, classificando tumores como **malignos** ou **benignos**. A solução utiliza o dataset público **Breast Cancer Wisconsin** e aplica técnicas de classificação supervisionada, análise exploratória, pré-processamento com `sklearn.pipeline` e explicabilidade com **SHAP**.

 **Nota importante:** este sistema é uma ferramenta de apoio à decisão médica. O diagnóstico final sempre deve ser confirmado por um profissional de saúde qualificado.

---

##  Objetivo

Construir a base de um sistema de IA para processamento de dados médicos relacionados à saúde feminina, aplicando fundamentos de:
- Análise Exploratória de Dados (EDA)
- Pré-processamento de dados
- Modelagem preditiva com múltiplos algoritmos
- Avaliação com métricas adequadas
- Explicabilidade de modelos (Feature Importance + SHAP)

---

##  Dataset

**Breast Cancer Wisconsin (Diagnostic)**  
- Origem: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Disponível via: `scikit-learn.datasets.load_breast_cancer`
- Registros: 569 amostras
- Features: 30 características numéricas extraídas de imagens de tumores
- Target: 0 = Maligno, 1 = Benigno

Também disponível no Kaggle: [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

---

##  Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/MAntonioST/tech-challenge-fase1-ia.git
cd tech-challenge-fase1-ia/diagnostico-cancer


# Crie e ative o ambiente virtual
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# nstale as dependências
pip install -r requirements.txt

# Execute o pipeline
python main.py


## Estrutura do Projeto

diagnostico-cancer/
├── main.py                 # Script principal (pipeline completo)
├── requirements.txt        # Dependências do projeto
├── outputs/                # Gráficos e métricas gerados
│   ├── correlation_heatmap.png
│   ├── class_distribution.png
│   ├── shap_beeswarm.png
│   ├── shap_feature_importance.png
│   └── metrics_comparison.csv
└── README.md


##  Tecnologias Utilizadas

- **Python 3.12+** — Linguagem principal
- **scikit-learn 1.3+** — Pré-processamento, modelos clássicos e métricas
- **XGBoost 2.0+** — Gradient Boosting
- **SHAP 0.43+** — Explicabilidade (SHapley Additive exPlanations)
- **pandas 2.0+** — Manipulação e análise de dados
- **matplotlib 3.7+** — Visualização de dados
- **seaborn 0.12+** — Visualização estatística
