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
