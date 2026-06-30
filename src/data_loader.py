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
