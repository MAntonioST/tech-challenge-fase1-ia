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
