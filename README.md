# Tech Challenge - Fase 1: Diagnóstico de Câncer de Mama com IA

Este repositório contém a solução para o Tech Challenge da Fase 1 da Pós-Graduação em IA para Devs (FIAP). O objetivo do projeto é aplicar técnicas de Machine Learning para classificar diagnósticos de câncer de mama (Maligno ou Benigno) utilizando dados estruturados.

## 📊 Sobre o Dataset
Foi utilizado o dataset público **Breast Cancer Wisconsin**, disponibilizado nativamente pela biblioteca `scikit-learn` (originalmente do UCI Machine Learning Repository). 
- **Características:** 569 amostras e 30 variáveis numéricas (como raio, textura, perímetro e área do tumor).
- **Target:** `0` (Maligno) e `1` (Benigno).
- **Qualidade:** O dataset não possui valores nulos, dispensando técnicas complexas de imputação de dados.

## ⚙️ Tecnologias Utilizadas
- **Linguagem:** Python 3
- **Manipulação de Dados:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Regressão Logística e Árvore de Decisão)
- **Explicabilidade:** SHAP

## 🚀 Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
   cd NOME_DO_REPOSITORIO
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou venv\Scripts\activate no Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn shap
   ```

4. **Execute o pipeline principal:**
   ```bash
   python tech_challenge_fase1.py
   ```
   *Nota: O script irá pausar a execução para exibir os gráficos de análise e explicabilidade. Feche a janela de cada gráfico para que o código continue rodando.*

## 📈 Resultados Obtidos
Foram treinados dois modelos preditivos com os seguintes resultados na base de teste (20% dos dados):

- **Regressão Logística:**
  - Acurácia: 97.37%
  - Recall: 98.59%
  - F1-Score: 97.90%

- **Árvore de Decisão:**
  - Acurácia: 94.74%
  - Recall: 95.77%
  - F1-Score: 95.77%

A Regressão Logística apresentou o melhor desempenho geral. A explicabilidade do modelo foi garantida através da extração de *Feature Importance* e da geração de gráficos *SHAP values*, permitindo entender o impacto de cada variável na decisão final do algoritmo.