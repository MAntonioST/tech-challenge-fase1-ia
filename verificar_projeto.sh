#!/bin/bash

# =============================================================================
# SCRIPT DE VERIFICAÇÃO - Tech Challenge FIAP
# Autor: MyHub Nitro
# Uso: ./verificar_projeto.sh
# =============================================================================

# Cores para saída
VERDE='\033[0;32m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
AZUL='\033[0;34m'
CINZA='\033[0;37m'
RESET='\033[0m'

# Contadores
OK=0
AVISO=0
ERRO=0

# Diretório atual
DIR_ATUAL=$(pwd)
echo -e "${AZUL}"
echo "=================================================================="
echo "  AUDITORIA DO PROJETO - Tech Challenge FIAP"
echo "  Diretório: $DIR_ATUAL"
echo "================================================================"
echo -e "${RESET}"

# Função para verificar existência de arquivo/pasta
verificar() {
    if [ -e "$1" ]; then
        echo -e "  ${VERDE}✅ OK:${RESET} $2"
        OK=$((OK + 1))
        return 0
    else
        echo -e "  ${VERMELHO}❌ FALTANDO:${RESET} $2"
        ERRO=$((ERRO + 1))
        return 1
    fi
}

# Função para verificar conteúdo em arquivo
verificar_conteudo() {
    arquivo="$1"
    padrao="$2"
    descricao="$3"
    
    if [ -f "$arquivo" ]; then
        if grep -q -i "$padrao" "$arquivo"; then
            echo -e "  ${VERDE}✅ OK:${RESET} $descricao"
            OK=$((OK + 1))
            return 0
        else
            echo -e "  ${AMARELO}⚠️  AVISO:${RESET} $descricao (não encontrado)"
            AVISO=$((AVISO + 1))
            return 1
        fi
    fi
    return 1
}

# =============================================================================
# 1. ESTRUTURA DE ARQUIVOS
# =============================================================================
echo -e "\n${AZUL}📁 1. ESTRUTURA DE ARQUIVOS OBRIGATÓRIOS${RESET}"

verificar "README.md" "Arquivo README.md"
verificar "requirements.txt" "Arquivo requirements.txt"
verificar "*.py" "Scripts Python (.py)"

# Verifica se há arquivos .py
python_files=$(find . -maxdepth 2 -name "*.py" -type f | wc -l)
if [ "$python_files" -gt 0 ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Encontrados $python_files arquivos Python"
    OK=$((OK + 1))
else
    echo -e "  ${VERMELHO}❌ FALTANDO:${RESET} Nenhum arquivo Python encontrado"
    ERRO=$((ERRO + 1))
fi

# Verifica Dockerfile (opcional)
if [ -f "Dockerfile" ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Dockerfile presente (opcional)"
    OK=$((OK + 1))
else
    echo -e "  ${CINZA}ℹ️  INFO:${RESET} Dockerfile não encontrado (opcional, não obrigatório)${RESET}"
fi

# Verifica pasta data/
if [ -d "data" ]; then
    data_files=$(find data -type f | wc -l)
    echo -e "  ${VERDE}✅ OK:${RESET} Pasta data/ encontrada ($data_files arquivo(s))"
    OK=$((OK + 1))
else
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Pasta data/ não encontrada. Considere criar uma para o dataset"
    AVISO=$((AVISO + 1))
fi

# Verifica pasta outputs/
if [ -d "outputs" ] || [ -d "img" ] || [ -d "figures" ] || [ -d "results" ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Pasta para saídas encontrada (outputs/img/figures/results)"
    OK=$((OK + 1))
else
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Pasta outputs/ não encontrada. Recomendado para gráficos"
    AVISO=$((AVISO + 1))
fi

# =============================================================================
# 2. CONTEÚDO DO README
# =============================================================================
echo -e "\n${AZUL}📄 2. CONTEÚDO DO README.md${RESET}"

if [ -f "README.md" ]; then
    verificar_conteudo "README.md" "descrição\|descrição\|sobre\|objetivo" "Descrição do projeto"
    verificar_conteudo "README.md" "executar\|rodar\|instalação\|install\|pip install" "Instruções de execução"
    verificar_conteudo "README.md" "dataset\|dados\|kaggle" "Link ou menção ao dataset"
    verificar_conteudo "README.md" "resultado\|acurácia\|accuracy\|recall\|f1" "Resultados obtidos"
    verificar_conteudo "README.md" "autor\|integrante\|grupo\|membro" "Autores do projeto"
fi

# =============================================================================
# 3. DEPENDÊNCIAS
# =============================================================================
echo -e "\n${AZUL}📦 3. DEPENDÊNCIAS (requirements.txt)${RESET}"

if [ -f "requirements.txt" ]; then
    verificar_conteudo "requirements.txt" "pandas" "pandas"
    verificar_conteudo "requirements.txt" "numpy" "numpy"
    verificar_conteudo "requirements.txt" "scikit-learn\|sklearn" "scikit-learn"
    verificar_conteudo "requirements.txt" "xgboost" "xgboost"
    verificar_conteudo "requirements.txt" "shap" "shap"
    verificar_conteudo "requirements.txt" "matplotlib" "matplotlib"
    verificar_conteudo "requirements.txt" "seaborn" "seaborn"
fi

# =============================================================================
# 4. ANÁLISE DOS CÓDIGOS PYTHON
# =============================================================================
echo -e "\n${AZUL}🐍 4. ANÁLISE DOS CÓDIGOS PYTHON${RESET}"

# Junta todos os .py em um texto temporário para análise
all_py_content=$(mktemp)
find . -maxdepth 3 -name "*.py" -type f -exec cat {} + > "$all_py_content" 2>/dev/null

if [ -s "$all_py_content" ]; then
    verificar_conteudo "$all_py_content" "train_test_split" "Separação treino/teste"
    verificar_conteudo "$all_py_content" "RandomForest" "Modelo Random Forest"
    verificar_conteudo "$all_py_content" "XGBClassifier\|XGBoost" "Modelo XGBoost"
    verificar_conteudo "$all_py_content" "accuracy_score" "Métrica accuracy"
    verificar_conteudo "$all_py_content" "recall_score" "Métrica recall"
    verificar_conteudo "$all_py_content" "f1_score" "Métrica F1-score"
    verificar_conteudo "$all_py_content" "Pipeline\|ColumnTransformer" "Pipeline de pré-processamento"
    verificar_conteudo "$all_py_content" "StandardScaler\|MinMaxScaler" "Normalização/escala"
    verificar_conteudo "$all_py_content" "feature_importances_" "Feature importance"
    verificar_conteudo "$all_py_content" "shap" "SHAP para explicabilidade"
    verificar_conteudo "$all_py_content" "confusion_matrix" "Matriz de confusão"
    verificar_conteudo "$all_py_content" "classification_report" "Classification report"
    verificar_conteudo "$all_py_content" "matplotlib\|seaborn" "Visualização de dados"
    verificar_conteudo "$all_py_content" "describe\|info\|isnull\|shape" "Análise exploratória de dados"
else
    echo -e "  ${VERMELHO}❌ ERRO:${RESET} Nenhum código Python encontrado para análise"
    ERRO=$((ERRO + 1))
fi

# =============================================================================
# 5. VERIFICAÇÃO DE GRÁFICOS
# =============================================================================
echo -e "\n${AZUL}📊 5. VERIFICAÇÃO DE GRÁFICOS GERADOS${RESET}"

png_files=$(find . -maxdepth 3 -name "*.png" -type f | wc -l)
jpg_files=$(find . -maxdepth 3 -name "*.jpg" -type f | wc -l)
total_imgs=$((png_files + jpg_files))

if [ "$total_imgs" -ge 5 ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Encontradas $total_imgs imagens (PNG/JPG) — ótimo para o relatório!"
    OK=$((OK + 1))
elif [ "$total_imgs" -gt 0 ]; then
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Apenas $total_imgs imagem(s). O edital pede gráficos e análises"
    AVISO=$((AVISO + 1))
else
    echo -e "  ${VERMELHO}❌ FALTANDO:${RESET} Nenhuma imagem/gráfico encontrado"
    ERRO=$((ERRO + 1))
fi

# Lista imagens encontradas
if [ "$total_imgs" -gt 0 ]; then
    echo -e "  ${CINZA}Imagens encontradas:${RESET}"
    find . -maxdepth 3 \( -name "*.png" -o -name "*.jpg" \) -type f | sed 's|^./|    - |'
fi

# =============================================================================
# 6. VERIFICAÇÃO DO DATASET
# =============================================================================
echo -e "\n${AZUL}🗃️ 6. VERIFICAÇÃO DO DATASET${RESET}"

csv_files=$(find . -maxdepth 3 -name "*.csv" -type f | wc -l)
if [ "$csv_files" -gt 0 ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Encontrado(s) $csv_files arquivo(s) CSV"
    OK=$((OK + 1))
else
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Nenhum arquivo CSV encontrado. Verifique se há link no README"
    AVISO=$((AVISO + 1))
fi

# Verifica link do dataset no README
if [ -f "README.md" ]; then
    if grep -q -E "https?://(www\.)?kaggle\.com|https?://archive\.ics\.uci\.edu|https?://github\.com" README.md; then
        echo -e "  ${VERDE}✅ OK:${RESET} Link para dataset externo encontrado no README"
        OK=$((OK + 1))
    else
        echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Nenhum link de dataset externo claro no README"
        AVISO=$((AVISO + 1))
    fi
fi

# =============================================================================
# 7. BOAS PRÁTICAS DE CÓDIGO
# =============================================================================
echo -e "\n${AZUL}🧹 7. BOAS PRÁTICAS DE CÓDIGO${RESET}"

verificar_conteudo "$all_py_content" "def " "Uso de funções (organização do código)"
verificar_conteudo "$all_py_content" "import" "Imports organizados"
verificar_conteudo "$all_py_content" "# " "Comentários no código"

# Verifica se há código comentado antigo (bad smell)
linhas_comentadas=$(grep -c "^[[:space:]]*#.*" $all_py_content 2>/dev/null || echo 0)
if [ "$linhas_comentadas" -gt 10 ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} Código bem comentado ($linhas_comentadas linhas comentadas)"
    OK=$((OK + 1))
elif [ "$linhas_comentadas" -gt 0 ]; then
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Poucos comentários ($linhas_comentadas linhas). Recomendo adicionar mais"
    AVISO=$((AVISO + 1))
else
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} Código sem comentários. Adicione explicações"
    AVISO=$((AVISO + 1))
fi

# Verifica arquivo .gitignore
if [ -f ".gitignore" ]; then
    echo -e "  ${VERDE}✅ OK:${RESET} .gitignore presente"
    OK=$((OK + 1))
else
    echo -e "  ${AMARELO}⚠️  AVISO:${RESET} .gitignore não encontrado. Recomendado para evitar subir venv/dados desnecessários"
    AVISO=$((AVISO + 1))
fi

# Limpa arquivo temporário
rm -f "$all_py_content"

# =============================================================================
# 8. RESUMO FINAL
# =============================================================================
echo -e "\n${AZUL}==================================================================${RESET}"
echo -e "${AZUL}  📊 RESUMO DA AUDITORIA${RESET}"
echo -e "${AZUL}==================================================================${RESET}"
echo -e "  ${VERDE}✅ OK:        $OK${RESET}"
echo -e "  ${AMARELO}⚠️  AVISOS:    $AVISO${RESET}"
echo -e "  ${VERMELHO}❌ FALTANDO:  $ERRO${RESET}"
echo ""

if [ $ERRO -eq 0 ] && [ $AVISO -eq 0 ]; then
    echo -e "  ${VERDE}🎉 PARABÉNS! Seu projeto está 100% alinhado com o Tech Challenge!${RESET}"
elif [ $ERRO -eq 0 ]; then
    echo -e "  ${AMARELO}👍 BOM TRABALHO! Seu projeto está quase pronto, mas tem $AVISO aviso(s) para revisar.${RESET}"
else
    echo -e "  ${VERMELHO}⚠️  ATENÇÃO! Seu projeto tem $ERRO erro(s) crítico(s) e $AVISO aviso(s). Corrija antes da entrega.${RESET}"
fi

echo -e "\n${CINZA}Dica: execute 'python -m py_compile src/*.py' para verificar erros de sintaxe.${RESET}\n"
