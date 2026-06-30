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
