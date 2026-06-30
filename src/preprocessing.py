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
