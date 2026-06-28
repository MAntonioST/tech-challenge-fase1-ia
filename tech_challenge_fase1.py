import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import shap

print("="*50)
print(" INICIANDO O PIPELINE DE MACHINE LEARNING")
print("="*50)

# ==========================================
# 1. CARREGAMENTO E EXPLORAÇÃO DE DADOS
# ==========================================
print("\n[1/5] Carregando a base de dados (Breast Cancer Wisconsin)...")
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

# Adicionando a coluna alvo (Target: 0 = Maligno, 1 = Benigno)
df['diagnostico'] = cancer_data.target

print("\n--- Primeiras linhas do Dataset ---")
print(df.head())

print("\n--- Informações Gerais (Verificando dados nulos) ---")
df.info()

print("\n--- Estatísticas Descritivas ---")
print(df.describe())

# ==========================================
# 2. VISUALIZAÇÃO E CORRELAÇÃO
# ==========================================
print("\n[2/5] Gerando gráficos de distribuição e correlação...")
print("-> ATENÇÃO: Feche a janela do gráfico para o código continuar rodando!")

# Gráfico 1: Distribuição dos diagnósticos
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnostico', hue='diagnostico', data=df, palette='Set2', legend=False)
plt.title('Distribuição dos Diagnósticos (0 = Maligno, 1 = Benigno)')
plt.show() # O terminal vai pausar aqui até você fechar a janela

# Gráfico 2: Mapa de Calor de Correlação (Exigência do PDF)
# Vamos pegar apenas as 10 primeiras colunas para o gráfico não ficar ilegível
plt.figure(figsize=(10, 8))
colunas_selecionadas = list(cancer_data.feature_names[:10]) + ['diagnostico']
sns.heatmap(df[colunas_selecionadas].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlação (Top 10 Variáveis)')
plt.show() # O terminal vai pausar aqui novamente

# ==========================================
# 3. PRÉ-PROCESSAMENTO
# ==========================================
print("\n[3/5] Realizando o pré-processamento (Divisão e Padronização)...")

# Separar Variáveis (X) e Alvo (y)
X = df.drop('diagnostico', axis=1)
y = df['diagnostico']

# Dividir em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização (Colocar todos os dados na mesma escala)
# Isso é crucial para a Regressão Logística funcionar bem
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODELAGEM E AVALIAÇÃO
# ==========================================
print("\n[4/5] Treinando os modelos e extraindo métricas...")

# Modelo 1: Regressão Logística
modelo_lr = LogisticRegression(random_state=42)
modelo_lr.fit(X_train_scaled, y_train)
y_pred_lr = modelo_lr.predict(X_test_scaled)

print("\n--- Resultados: Regressão Logística ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")

# Modelo 2: Árvore de Decisão
modelo_dt = DecisionTreeClassifier(random_state=42, max_depth=5)
modelo_dt.fit(X_train_scaled, y_train)
y_pred_dt = modelo_dt.predict(X_test_scaled)

print("\n--- Resultados: Árvore de Decisão ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")

# ==========================================
# 5. EXPLICABILIDADE (FEATURE IMPORTANCE E SHAP)
# ==========================================
print("\n[5/5] Gerando explicabilidade do modelo (Feature Importance e SHAP)...")
print("-> ATENÇÃO: Feche as janelas dos gráficos para finalizar o script!")

# Feature Importance (Árvore de Decisão)
importancias = pd.Series(modelo_dt.feature_importances_, index=X.columns)
importancias_top10 = importancias.nlargest(10)

plt.figure(figsize=(8, 5))
importancias_top10.plot(kind='barh', color='teal')
plt.title('Top 10 Variáveis Mais Importantes (Árvore de Decisão)')
plt.gca().invert_yaxis()
plt.show()

# SHAP Values (Exigência do PDF)
# Usando o TreeExplainer para a Árvore de Decisão
explainer = shap.TreeExplainer(modelo_dt)
shap_values = explainer.shap_values(X_test_scaled)

# Plotando o resumo do SHAP
plt.figure()
plt.title('Impacto das Variáveis nas Previsões (SHAP)')
# O SHAP summary_plot já chama o matplotlib por baixo dos panos
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
plt.show()

print("\n" + "="*50)
print(" PIPELINE CONCLUÍDO COM SUCESSO!")
print("="*50)