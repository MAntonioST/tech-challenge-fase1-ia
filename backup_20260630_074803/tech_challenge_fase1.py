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

print("=" * 50)
print(" INICIANDO O PIPELINE DE MACHINE LEARNING")
print("=" * 50)

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

# Gráfico 1: Distribuição dos diagnósticos (CORRIGIDO: removido hue redundante)
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnostico', data=df, palette='Set2')
plt.title('Distribuição dos Diagnósticos (0 = Maligno, 1 = Benigno)')
plt.xlabel('Diagnóstico')
plt.ylabel('Quantidade de Casos')
plt.show()

# Gráfico 2: Mapa de Calor de Correlação
plt.figure(figsize=(10, 8))
colunas_selecionadas = list(cancer_data.feature_names[:10]) + ['diagnostico']
sns.heatmap(df[colunas_selecionadas].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlação (Top 10 Variáveis)')
plt.show()

# ==========================================
# 3. PRÉ-PROCESSAMENTO
# ==========================================
print("\n[3/5] Realizando o pré-processamento (Divisão e Padronização)...")

# Separar Variáveis (X) e Alvo (y)
X = df.drop('diagnostico', axis=1)
y = df['diagnostico']

# Dividir em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter arrays escalados de volta para DataFrame (preserva nomes das colunas para o SHAP)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# ==========================================
# 4. MODELAGEM E AVALIAÇÃO
# ==========================================
print("\n[4/5] Treinando os modelos e extraindo métricas...")

# Modelo 1: Regressão Logística
modelo_lr = LogisticRegression(random_state=42, max_iter=10000)
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
print("\n[5/5] Gerando explicabilidade do modelo...")

# --- Feature Importance ---
importancias = pd.Series(modelo_dt.feature_importances_, index=X.columns)
importancias_top10 = importancias.nlargest(10)

plt.figure(figsize=(10, 6))
bars = importancias_top10.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Variáveis Mais Importantes\n(Árvore de Decisão)', fontsize=14, fontweight='bold')
plt.xlabel('Importância Relativa', fontsize=12)
plt.ylabel('Variável', fontsize=12)
plt.gca().invert_yaxis()

# Adiciona valores nas barras
for i, v in enumerate(importancias_top10.values):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# --- SHAP Values ---
print("\nGerando gráficos SHAP...")

explainer = shap.TreeExplainer(modelo_dt)
shap_values = explainer.shap_values(X_test_scaled_df)

if isinstance(shap_values, list):
    shap_classe = shap_values[1]
else:
    shap_classe = shap_values[:, :, 1]

shap_explicacao = shap.Explanation(
    values=shap_classe,
    data=X_test_scaled_df.values,
    feature_names=X.columns.tolist()
)

# ===== GRÁFICO 1: SHAP Beeswarm (principal) =====
# Cria a figura ANTES com o tamanho desejado
plt.figure(figsize=(14, 10))

shap.plots.beeswarm(
    shap_explicacao,
    show=False,
    max_display=10,
    plot_size=None,           
    color_bar_label="Valor da Variável"
)

# Agora ajustamos os elementos do gráfico
ax = plt.gca()
ax.set_title(
    'Impacto das Variáveis na Previsão de Câncer de Mama\n'
    '(SHAP Beeswarm Plot)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.set_xlabel('Valor SHAP (Impacto na Previsão)', fontsize=12)
ax.set_ylabel('Variáveis (ordenadas por importância)', fontsize=12)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Texto explicativo na parte inferior
plt.figtext(
    0.5, 0.02,
    '● Azul: Valores baixos da variável    ● Vermelho: Valores altos da variável\n'
    '← SHAP negativo (tende a Benigno) | SHAP positivo (tende a Maligno) →',
    ha='center',
    fontsize=11,
    style='italic',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
)


plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()

# ===== GRÁFICO 2: SHAP Bar Plot (versão simplificada para slides) =====
print("\nGerando gráfico de barras SHAP...")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_classe,
    X_test_scaled_df,
    plot_type='bar',
    show=False,
    max_display=10,
    plot_size=None               
)

ax2 = plt.gca()
ax2.set_title(
    'Importância Média das Variáveis (SHAP)\n'
    'Quanto maior a barra, mais a variável influencia a previsão',
    fontsize=14,
    fontweight='bold',
    pad=15
)
ax2.set_xlabel('Importância SHAP (|valor médio|)', fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(" PIPELINE CONCLUÍDO COM SUCESSO!")
print("=" * 50)
