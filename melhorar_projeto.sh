#!/bin/bash

# ==============================================================================
# SCRIPT DE MELHORIA DO PROJETO
# Tech Challenge - Fase 1: IA Foundations
# FIAP - Pós-Graduação em IA para DEVs
#
# Autor: Marco
#
# Objetivo: Transformar o projeto monolítico atual em uma estrutura modular,
# profissional e alinhada aos requisitos do edital.
# ==============================================================================

# -------------------------------
# CONFIGURAÇÕES E CORES
# -------------------------------

# Define cores para deixar os logs mais legíveis no terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Diretório base do projeto
PROJECT_DIR=$(pwd)

# Nome do script atual (evita apagar ele mesmo)
SCRIPT_NAME=$(basename "$0")

# -------------------------------
# FUNÇÃO: Mensagens de log
# -------------------------------
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# -------------------------------
# 1. VERIFICAÇÃO DO DIRETÓRIO
# -------------------------------
log "Diretório atual: $PROJECT_DIR"

# Verifica se existe algum arquivo Python antigo no diretório
if [ -z "$(find . -maxdepth 1 -name '*.py' -type f)" ]; then
    warn "Nenhum arquivo .py encontrado no diretório raiz."
    warn "Certifique-se de executar este script na raiz do projeto 'diagnostico-cancer'."
    read -p "Deseja continuar mesmo assim? (s/n): " confirm
    if [ "$confirm" != "s" ]; then
        log "Execução cancelada pelo usuário."
        exit 0
    fi
else
    log "Arquivos Python encontrados. Prosseguindo..."
fi

# -------------------------------
# 2. BACKUP DOS ARQUIVOS ANTIGOS
# -------------------------------
log "Criando backup dos arquivos antigos..."

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Move os arquivos .py antigos para o backup, exceto o próprio script
find . -maxdepth 1 -name "*.py" -type f ! -name "$SCRIPT_NAME" -exec mv {} "$BACKUP_DIR/" \;

# Se houver um README.md antigo, também faz backup
if [ -f "README.md" ]; then
    cp "README.md" "$BACKUP_DIR/README.md.bak"
    warn "README.md existente foi copiado para $BACKUP_DIR/README.md.bak"
fi

success "Backup criado em: $BACKUP_DIR"

# -------------------------------
# 3. CRIAÇÃO DA ESTRUTURA DE PASTAS
# -------------------------------
log "Criando estrutura de diretórios..."

# Cria as pastas do projeto (src/, data/, outputs/)
mkdir -p src data outputs

# 'data/' será para datasets locais (mesmo que vazia por enquanto)
# 'outputs/' será para gráficos e resultados gerados
# 'src/' será o pacote com os módulos do pipeline

success "Pastas criadas: src, data, outputs"

# -------------------------------
# 4. ARQUIVO requirements.txt
# -------------------------------
log "Criando requirements.txt..."

cat > requirements.txt << 'EOF'
# ==============================================================================
# DEPENDÊNCIAS DO PROJETO
# Tech Challenge - Diagnóstico de Câncer de Mama
# ==============================================================================
#
# Instalação: pip install -r requirements.txt
# ==============================================================================

# Manipulação de dados
pandas>=2.0.0
numpy>=1.24.0

# Visualização de dados
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0

# Modelos avançados (boosting)
xgboost>=2.0.0

# Explicabilidade (SHAP)
shap>=0.43.0
EOF

success "requirements.txt criado"

# -------------------------------
# 5. ARQUIVOS DO PACOTE src/
# -------------------------------
log "Criando módulos Python..."

# -----------------------------
# 5.1 src/__init__.py
# -----------------------------
cat > src/__init__.py << 'EOF'
# ==============================================================================
# PACOTE src/
# Módulos do pipeline de Machine Learning para diagnóstico de câncer de mama.
# ==============================================================================
#
# Estrutura:
#   - data_loader.py    : Carregamento e exploração inicial dos dados
#   - eda.py            : Análise exploratória e visualizações
#   - preprocessing.py  : Pré-processamento (split, padronização, pipeline)
#   - modeling.py       : Treinamento de modelos de classificação
#   - evaluation.py     : Métricas, matriz de confusão e comparação
#   - explainability.py : Feature importance e SHAP
# ==============================================================================

"""
Tech Challenge - Diagnóstico de Câncer de Mama.

Fase 1 - IA Foundations - FIAP.

Autor: Marco.
"""
EOF

# -----------------------------
# 5.2 src/data_loader.py
# -----------------------------
cat > src/data_loader.py << 'EOF'
# ==============================================================================
# MÓDULO data_loader.py
# Responsável por carregar o dataset e realizar a exploração inicial.
# ==============================================================================

import logging
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Cria um logger para este módulo (mensagens serão exibidas no console)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """
    Carrega o dataset Breast Cancer Wisconsin do scikit-learn.

    O dataset contém características extraídas de imagens digitalizadas de
    massas mamárias. Cada linha representa um paciente e a coluna 'diagnostico'
    indica se o tumor é maligno (0) ou benigno (1).

    Returns:
        pd.DataFrame: DataFrame com as features e a coluna alvo 'diagnostico'.
    """
    logger.info("Carregando dataset Breast Cancer Wisconsin...")

    cancer_data = load_breast_cancer()
    df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

    # Adiciona a coluna alvo (target): 0 = Maligno, 1 = Benigno
    df['diagnostico'] = cancer_data.target

    logger.info(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Exibe informações exploratórias sobre o dataset.

    Args:
        df: DataFrame com os dados carregados.
    """
    logger.info("=== EXPLORAÇÃO INICIAL DOS DADOS ===")

    # Conta quantos casos são malignos e benignos
    classe_counts = df['diagnostico'].value_counts().sort_index()
    maligno = classe_counts[0]
    benigno = classe_counts[1]

    logger.info(f"Distribuição: {maligno} malignos (0) | {benigno} benignos (1)")
    logger.info(f"Proporção de benignos: {benigno / len(df):.1%}")

    # Verifica valores nulos (dataset do sklearn não possui, mas é boa prática)
    nulos = df.isnull().sum().sum()
    logger.info(f"Valores nulos: {nulos}")

    logger.info("=" * 50)
EOF

# -----------------------------
# 5.3 src/eda.py
# -----------------------------
cat > src/eda.py << 'EOF'
# ==============================================================================
# MÓDULO eda.py
# Análise Exploratória de Dados (EDA).
# Gera visualizações e salva automaticamente em outputs/.
# ==============================================================================

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Pasta onde os gráficos serão salvos
OUTPUT_DIR = Path("outputs")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Gera e salva o gráfico de distribuição das classes (maligno vs benigno).

    Args:
        df: DataFrame com coluna 'diagnostico'.
    """
    logger.info("Gerando gráfico de distribuição de classes...")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        x='diagnostico',
        data=df,
        palette=['#e74c3c', '#2ecc71'],  # Vermelho para maligno, verde para benigno
        edgecolor='black',
        linewidth=1.2
    )

    # Adiciona rótulos nas barras
    for i, (classe, label) in enumerate([(0, 'Maligno'), (1, 'Benigno')]):
        count = (df['diagnostico'] == classe).sum()
        ax.text(i, count + 2, str(count), ha='center', fontsize=14, fontweight='bold')
        ax.text(i, count / 2, label, ha='center', fontsize=12, color='white', fontweight='bold')

    plt.title('Distribuição dos Diagnósticos\nBreast Cancer Wisconsin Dataset', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Diagnóstico', fontsize=12)
    plt.ylabel('Quantidade de Casos', fontsize=12)
    plt.xticks([0, 1], ['0 - Maligno', '1 - Benigno'])

    # Linha de referência com a média
    media = len(df) / 2
    plt.axhline(y=media, color='gray', linestyle='--', alpha=0.5, label=f'Média: {media:.0f}')
    plt.legend()

    _save_and_close('distribuicao_classes.png')
    logger.info("Gráfico de distribuição salvo.")


def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 12) -> None:
    """
    Gera e salva o mapa de calor de correlação das principais features.

    Args:
        df: DataFrame completo.
        top_n: Quantidade de variáveis a exibir (default: 12).
    """
    logger.info("Gerando mapa de calor de correlação...")

    # Seleciona as primeiras 'top_n' features + o target
    feature_cols = [c for c in df.columns if c != 'diagnostico']
    selected = feature_cols[:top_n] + ['diagnostico']

    plt.figure(figsize=(12, 9))
    corr_matrix = df[selected].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdBu_r',
        center=0,
        fmt='.2f',
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlação'},
        annot_kws={'size': 9}
    )

    plt.title(f'Mapa de Correlação - Top {top_n} Variáveis', fontsize=16, fontweight='bold', pad=20)

    _save_and_close('heatmap_correlacao.png')
    logger.info("Heatmap de correlação salvo.")


def generate_all_plots(df: pd.DataFrame) -> None:
    """
    Gera todos os gráficos de EDA.

    Args:
        df: DataFrame com os dados.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)  # Cria a pasta se não existir
    plot_class_distribution(df)
    plot_correlation_heatmap(df)


def _save_and_close(filename: str, dpi: int = 150) -> None:
    """
    Salva o gráfico atual como PNG e fecha a figura.

    Args:
        filename: Nome do arquivo de imagem.
        dpi: Resolução da imagem.
    """
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
EOF

# -----------------------------
# 5.4 src/preprocessing.py
# -----------------------------
cat > src/preprocessing.py << 'EOF'
# ==============================================================================
# MÓDULO preprocessing.py
# Responsável pelo pré-processamento dos dados.
# Utiliza sklearn.pipeline.Pipeline conforme solicitado no edital.
# ==============================================================================

import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa features/target, divide treino/teste e aplica pipeline de padronização.

    Args:
        df: DataFrame com as features e a coluna alvo 'diagnostico'.
        test_size: Proporção do conjunto de teste (default: 20%).
        random_state: Semente para reprodutibilidade (default: 42).

    Returns:
        Tuple (X_train, X_test, y_train, y_test) processados e padronizados.
    """
    logger.info("Iniciando pré-processamento...")

    # Separa variáveis independentes (X) e variável alvo (y)
    X = df.drop('diagnostico', axis=1)
    y = df['diagnostico']

    # Divisão estratificada: mantém a proporção de classes em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Split: {len(X_train)} treino | {len(X_test)} teste (estratificado)")

    # Identifica colunas numéricas (dataset é todo numérico, mas deixamos genérico)
    numeric_features = X.columns.tolist()

    # ColumnTransformer permite aplicar transformações diferentes por tipo de coluna
    # Aqui aplicamos StandardScaler em todas as colunas numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )

    # Pipeline do sklearn: sequência organizada de transformações
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Aplica fit no treino e transforma ambos os conjuntos
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Converte de volta para DataFrame para preservar nomes das colunas
    X_train_final = pd.DataFrame(X_train_processed, columns=numeric_features)
    X_test_final = pd.DataFrame(X_test_processed, columns=numeric_features)

    logger.info("Pipeline de pré-processamento concluído (ColumnTransformer + StandardScaler)")
    return X_train_final, X_test_final, y_train, y_test
EOF

# -----------------------------
# 5.5 src/modeling.py
# -----------------------------
cat > src/modeling.py << 'EOF'
# ==============================================================================
# MÓDULO modeling.py
# Treinamento de múltiplos modelos de classificação.
# Edital pede "duas ou mais técnicas"; usamos quatro para robustez.
# ==============================================================================

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Semente global para garantir resultados reprodutíveis
RANDOM_STATE = 42


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Treina quatro modelos de classificação e retorna um dicionário.

    Modelos:
        1. Regressão Logística (baseline linear)
        2. Árvore de Decisão (interpretável)
        3. Random Forest (ensemble por bagging)
        4. XGBoost (ensemble por gradient boosting - estado da arte)

    Args:
        X_train: Features de treino padronizadas.
        y_train: Target de treino.

    Returns:
        Dict: {nome_modelo: modelo_treinado}.
    """
    logger.info("=" * 50)
    logger.info("INICIANDO TREINAMENTO DOS MODELOS")
    logger.info("=" * 50)

    models = {}

    # 1. Regressão Logística: modelo linear, rápido e interpretável
    logger.info("Treinando Regressão Logística...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, solver='lbfgs')
    lr.fit(X_train, y_train)
    models['Regressão Logística'] = lr
    logger.info("  ✓ Concluído")

    # 2. Árvore de Decisão: modelo baseado em regras, fácil de visualizar
    logger.info("Treinando Árvore de Decisão...")
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt.fit(X_train, y_train)
    models['Árvore de Decisão'] = dt
    logger.info("  ✓ Concluído")

    # 3. Random Forest: ensemble que combina várias árvores (bagging)
    logger.info("Treinando Random Forest...")
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1  # Usa todos os núcleos do processador
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    logger.info("  ✓ Concluído")

    # 4. XGBoost: gradient boosting, geralmente o modelo mais performático
    logger.info("Treinando XGBoost...")
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    logger.info("  ✓ Concluído")

    logger.info(f"Total de modelos treinados: {len(models)}")
    return models
EOF

# -----------------------------
# 5.6 src/evaluation.py
# -----------------------------
cat > src/evaluation.py << 'EOF'
# ==============================================================================
# MÓDULO evaluation.py
# Avaliação dos modelos: métricas, matriz de confusão e comparação visual.
# ==============================================================================

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")


def evaluate_all(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Calcula métricas para todos os modelos e retorna tabela comparativa.

    Métricas:
        - Acurácia
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC (se o modelo tiver predict_proba)

    Args:
        models: Dicionário {nome: modelo_treinado}.
        X_test: Features de teste.
        y_test: Target de teste.

    Returns:
        pd.DataFrame: Tabela comparativa de métricas (índice = nome do modelo).
    """
    logger.info("=" * 50)
    logger.info("AVALIAÇÃO DOS MODELOS")
    logger.info("=" * 50)

    results = []

    for nome, modelo in models.items():
        y_pred = modelo.predict(X_test)

        # Probabilidades para ROC-AUC (só se o modelo suportar)
        y_proba = (
            modelo.predict_proba(X_test)[:, 1]
            if hasattr(modelo, 'predict_proba')
            else None
        )

        metrics = {
            'Modelo': nome,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
        }

        if y_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)

        results.append(metrics)

        # Exibe resultados individualmente
        logger.info(f"\n--- {nome} ---")
        for k, v in metrics.items():
            if k != 'Modelo':
                logger.info(f"  {k}: {v:.4f}")

        # Classification report detalhado
        logger.info(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    df_results = pd.DataFrame(results).set_index('Modelo').round(4)

    # Identifica o melhor modelo pelo F1-Score
    best_model = df_results['F1-Score'].idxmax()
    best_f1 = df_results['F1-Score'].max()
    logger.info(f"\n🏆 Melhor modelo: {best_model} (F1-Score: {best_f1:.4f})")

    return df_results


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Gera e salva matriz de confusão para todos os modelos, lado a lado.

    Args:
        models: Dicionário {nome: modelo_treinado}.
        X_test: Features de teste.
        y_test: Target de teste.
    """
    logger.info("Gerando matrizes de confusão...")

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (nome, modelo) in zip(axes, models.items()):
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Maligno (0)', 'Benigno (1)'],
            yticklabels=['Maligno (0)', 'Benigno (1)'],
            ax=ax,
            cbar=False,
            annot_kws={'size': 16, 'fontweight': 'bold'}
        )

        ax.set_title(nome, fontsize=12, fontweight='bold')
        ax.set_xlabel('Previsto', fontsize=10)
        ax.set_ylabel('Real', fontsize=10)

    plt.suptitle('Matrizes de Confusão - Todos os Modelos', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_and_close('matrizes_confusao.png')


def plot_metrics_comparison(df_results: pd.DataFrame) -> None:
    """
    Gera gráfico de barras comparando as métricas entre os modelos.

    Args:
        df_results: DataFrame com métricas (índice = nome do modelo).
    """
    logger.info("Gerando gráfico comparativo de métricas...")

    fig, ax = plt.subplots(figsize=(10, 6))

    df_results.plot(
        kind='bar',
        ax=ax,
        edgecolor='black',
        linewidth=0.8,
        colormap='Set2'
    )

    ax.set_title('Comparação de Métricas entre Modelos', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Valor da Métrica', fontsize=12)
    ax.set_ylim(0.85, 1.01)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Adiciona valores nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

    # Linha de referência 95%
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] - 0.3, 0.952, '95%', color='red', fontsize=9, alpha=0.7)

    plt.xticks(rotation=0)
    plt.tight_layout()

    _save_and_close('comparacao_metricas.png')
    logger.info("Gráfico comparativo salvo.")


def _save_and_close(filename: str, dpi: int = 150) -> None:
    """
    Salva o gráfico atual e fecha a figura.

    Args:
        filename: Nome do arquivo.
        dpi: Resolução da imagem.
    """
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
EOF

# -----------------------------
# 5.7 src/explainability.py
# -----------------------------
cat > src/explainability.py << 'EOF'
# ==============================================================================
# MÓDULO explainability.py
# Explicabilidade dos modelos: Feature Importance e SHAP.
# Ajuda a interpretar quais variáveis mais influenciam a previsão.
# ==============================================================================

import logging
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs")


def plot_feature_importance(
    model: Any,
    feature_names: pd.Index,
    model_name: str,
    top_n: int = 15
) -> None:
    """
    Gera e salva gráfico de Feature Importance do modelo.

    Args:
        model: Modelo treinado (precisa ter atributo feature_importances_).
        feature_names: Nomes das features.
        model_name: Nome do modelo (usado no título).
        top_n: Número de features a exibir.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Modelo {model_name} não possui feature_importances_")
        return

    logger.info(f"Gerando Feature Importance - {model_name}...")

    importancias = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importancias)))

    ax.barh(
        range(len(importancias)),
        importancias.values,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_yticks(range(len(importancias)))
    ax.set_yticklabels(importancias.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importância Relativa', fontsize=12)
    ax.set_title(f'Top {top_n} Variáveis Mais Importantes\\n{model_name}', fontsize=14, fontweight='bold', pad=15)

    # Valores nas barras
    for i, val in enumerate(importancias.values):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.lower().replace(' ', '_').replace('ç', 'c').replace('á', 'a')
    _save_and_close(f'feature_importance_{safe_name}.png')
    logger.info(f"Feature importance de {model_name} salvo.")


def plot_shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str,
    max_display: int = 12
) -> None:
    """
    Gera e salva gráficos SHAP (beeswarm e bar plot) para o modelo.

    Args:
        model: Modelo treinado.
        X_test: Features de teste (DataFrame).
        model_name: Nome do modelo.
        max_display: Máximo de features exibidas.
    """
    logger.info(f"Gerando análise SHAP - {model_name}...")

    try:
        # TreeExplainer para modelos baseados em árvores
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP pode retornar lista (classificação binária) ou array 3D
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Classe positiva (benigno = 1)
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values

        # ===== Gráfico 1: SHAP Beeswarm =====
        fig, ax = plt.subplots(figsize=(14, 10))

        shap_explanation = shap.Explanation(
            values=shap_vals,
            data=X_test.values,
            feature_names=X_test.columns.tolist()
        )

        shap.plots.beeswarm(
            shap_explanation,
            show=False,
            max_display=max_display,
            plot_size=None
        )

        ax = plt.gca()
        ax.set_title(f'Impacto das Variáveis na Previsão\\n{model_name} - SHAP Beeswarm', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Valor SHAP (Impacto na Previsão)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        plt.tight_layout()
        safe_name = model_name.lower().replace(' ', '_').replace('ç', 'c').replace('á', 'a')
        _save_and_close(f'shap_beeswarm_{safe_name}.png')

        # ===== Gráfico 2: SHAP Bar Plot =====
        fig, ax = plt.subplots(figsize=(10, 6))

        shap.summary_plot(
            shap_vals,
            X_test,
            plot_type='bar',
            show=False,
            max_display=max_display,
            plot_size=None
        )

        ax = plt.gca()
        ax.set_title(f'Importância Média das Variáveis (SHAP)\\n{model_name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Média |SHAP| (impacto absoluto)', fontsize=12)

        plt.tight_layout()
        _save_and_close(f'shap_bar_{safe_name}.png')

        logger.info(f"Análise SHAP de {model_name} concluída.")

    except Exception as e:
        logger.error(f"Erro na análise SHAP de {model_name}: {e}")
        logger.warning("Pulando análise SHAP para este modelo...")
EOF

# -------------------------------
# 6. ARQUIVO main.py (ORQUESTRADOR)
# -------------------------------
log "Criando main.py..."

cat > main.py << 'EOF'
# ==============================================================================
# main.py
# Script principal que orquestra todo o pipeline de Machine Learning.
# ==============================================================================
#
# Fluxo:
#   1. Carrega dados
#   2. Gera EDA
#   3. Pré-processa
#   4. Treina modelos
#   5. Avalia modelos
#   6. Explica com Feature Importance e SHAP
#
# Uso: python main.py
# ==============================================================================

import logging
import sys
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para importar o pacote src
sys.path.insert(0, str(Path(__file__).parent))

# Configuração global do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Importa os módulos do projeto
from src.data_loader import load_data, explore_data
from src.eda import generate_all_plots
from src.preprocessing import prepare_data
from src.modeling import train_all_models
from src.evaluation import evaluate_all, plot_confusion_matrices, plot_metrics_comparison
from src.explainability import plot_feature_importance, plot_shap_analysis


def main():
    """
    Pipeline principal para classificação de câncer de mama.
    Executa todas as etapas de forma sequencial e segura.
    """
    # Cria diretórios necessários
    Path("outputs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("  TECH CHALLENGE - DIAGNÓSTICO DE CÂNCER DE MAMA")
    logger.info("  Fase 1 - IA Foundations - FIAP")
    logger.info("  Autor: Marco")
    logger.info("=" * 60)

    try:
        # ==========================================
        # ETAPA 1: Carregamento dos dados
        # ==========================================
        logger.info("\\n📂 ETAPA 1/6: Carregamento dos Dados")
        df = load_data()
        explore_data(df)

        # ==========================================
        # ETAPA 2: Análise Exploratória (EDA)
        # ==========================================
        logger.info("\\n📊 ETAPA 2/6: Análise Exploratória")
        generate_all_plots(df)

        # ==========================================
        # ETAPA 3: Pré-processamento
        # ==========================================
        logger.info("\\n🔧 ETAPA 3/6: Pré-processamento")
        X_train, X_test, y_train, y_test = prepare_data(df)

        # ==========================================
        # ETAPA 4: Modelagem
        # ==========================================
        logger.info("\\n🤖 ETAPA 4/6: Treinamento dos Modelos")
        models = train_all_models(X_train, y_train)

        # ==========================================
        # ETAPA 5: Avaliação
        # ==========================================
        logger.info("\\n📈 ETAPA 5/6: Avaliação dos Modelos")
        df_results = evaluate_all(models, X_test, y_test)

        # Salva tabela comparativa em CSV
        df_results.to_csv("outputs/resultados_modelos.csv")
        logger.info("Tabela de resultados salva em outputs/resultados_modelos.csv")

        # Gera gráficos de avaliação
        plot_confusion_matrices(models, X_test, y_test)
        plot_metrics_comparison(df_results)

        # ==========================================
        # ETAPA 6: Explicabilidade
        # ==========================================
        logger.info("\\n🔍 ETAPA 6/6: Explicabilidade (SHAP)")

        # Seleciona o melhor modelo (maior F1-Score)
        best_model_name = df_results['F1-Score'].idxmax()
        best_model = models[best_model_name]
        logger.info(f"Modelo selecionado para explicabilidade: {best_model_name}")

        plot_feature_importance(best_model, X_train.columns, best_model_name)
        plot_shap_analysis(best_model, X_test, best_model_name)

        # ==========================================
        # RESUMO FINAL
        # ==========================================
        logger.info("\\n" + "=" * 60)
        logger.info("  ✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 60)
        logger.info(f"\\n🏆 Melhor modelo: {best_model_name}")
        logger.info(f"📊 F1-Score: {df_results.loc[best_model_name, 'F1-Score']:.4f}")
        logger.info("📁 Gráficos salvos em: outputs/")
        logger.info("📄 Resultados: outputs/resultados_modelos.csv")

    except Exception as e:
        logger.error(f"❌ Erro durante a execução: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
EOF

# -------------------------------
# 7. ARQUIVO .gitignore
# -------------------------------
log "Criando .gitignore..."

cat > .gitignore << 'EOF'
# ==============================================================================
# .gitignore
# Arquivos e pastas que não devem ser versionados no Git.
# ==============================================================================

# Python cache
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/
.eggs/

# Ambiente virtual
venv/
env/
.venv/

# Dados (podem ser grandes)
data/*.csv
data/*.parquet

# Resultados gerados (serão recriados ao executar main.py)
outputs/*.png
outputs/*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Sistema operacional
.DS_Store
Thumbs.db
EOF

# -------------------------------
# 8. ARQUIVO README.md
# -------------------------------
log "Criando README.md..."

cat > README.md << 'EOF'
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
