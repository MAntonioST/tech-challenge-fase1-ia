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
    hue='diagnostico',
    data=df,
    palette=['#c0392b', '#27ae60'],
    legend=False
)

    # Personaliza os rótulos do eixo X
    ax.set_xticklabels(['0 - Maligno', '1 - Benigno'])



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
