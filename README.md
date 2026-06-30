# 🩺 Tech Challenge - Diagnóstico de Câncer de Mama

> **Fase 1 - IA Foundations**  
> **FIAP - Pós-Graduação em IA para DEVs**  
> **Autor:** Marco

---

## 📋 Descrição

Este projeto implementa um sistema de **Machine Learning** para apoiar o diagnóstico de câncer de mama, classificando tumores como **malignos** ou **benignos**. A solução utiliza o dataset público **Breast Cancer Wisconsin** e aplica técnicas de classificação supervisionada, análise exploratória, pré-processamento com `sklearn.pipeline` e explicabilidade com **SHAP**.

> ⚠️ **Nota importante:** este sistema é uma ferramenta de apoio à decisão médica. O diagnóstico final sempre deve ser confirmado por um profissional de saúde qualificado.

---

## 🎯 Objetivo

Construir a base de um sistema de IA para processamento de dados médicos relacionados à saúde feminina, aplicando fundamentos de:
- Análise Exploratória de Dados (EDA)
- Pré-processamento de dados
- Modelagem preditiva com múltiplos algoritmos
- Avaliação com métricas adequadas
- Explicabilidade de modelos (Feature Importance + SHAP)

---

## 📦 Dataset

**Breast Cancer Wisconsin (Diagnostic)**  
- Origem: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Disponível via: `scikit-learn.datasets.load_breast_cancer`
- Registros: 569 amostras
- Features: 30 características numéricas extraídas de imagens de tumores
- Target: 0 = Maligno, 1 = Benigno

Também disponível no Kaggle: [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

---

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd diagnostico-cancer
