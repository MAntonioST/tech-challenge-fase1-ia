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
