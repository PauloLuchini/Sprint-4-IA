from pymongo import MongoClient
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from joblib import dump
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter
import joblib  
import pandas as pd
app = Flask(__name__)
CORS(app)
# Carregar seus modelos
model1 = joblib.load('modelos/decision_tree_model.joblib')
model2 = joblib.load('modelos/random_forest_model.joblib')
model3 = joblib.load('modelos/svc_model.joblib')
# Adicione mais modelos se necessário

@app.route('/prever', methods=['POST'])
def prever():
    dados = request.json
    client = MongoClient("mongodb://localhost:27017/")
    db = client['HiperpersonalizacaoMarketing']

    # Extraindo dados das coleções e convertendo para DataFrames
    empresas_df = pd.DataFrame(list(db.Empresas.find()))
    comportamento_df = pd.DataFrame(list(db.ComportamentoNegocios.find()))
    desempenho_df = pd.DataFrame(list(db.DesempenhoFinanceiro.find()))
    tendencias_df = pd.DataFrame(list(db.TendenciasGastos.find()))

    # Removendo a coluna `_id` de cada DataFrame para evitar conflitos durante o merge
    empresas_df = empresas_df.drop(columns=["_id"])
    comportamento_df = comportamento_df.drop(columns=["_id"])
    desempenho_df = desempenho_df.drop(columns=["_id"])
    tendencias_df = tendencias_df.drop(columns=["_id"])

    # Realizando o merge dos DataFrames no campo `empresa_id`
    merged_df = empresas_df.merge(comportamento_df, on="empresa_id") \
                        .merge(desempenho_df, on="empresa_id") \
                        .merge(tendencias_df, on="empresa_id")

    # Removendo campos desnecessários para o modelo
    merged_df = merged_df.drop(columns=["empresa_id","nome", "site", "responsavel_contato", "responsavel_orcamento","usorecursos_especificos","motivo_investimento","retorno_esperado","responsavel_orcamento","ano","proposta_negocio","data_fundacao","tamanho","localizacao_geografica","tipo_empresa","area_mais_lucrativa","area_mais_custosa","categoria_gasto","valor_gasto","prioridade_investimento","contrato_assinado","data_fundacao","tempo_como_cliente","potencial_crescimento", "motivo_interesse","lucro_liquido", "despesas", "investimentos","margem_lucro","avaliacao_risco","variacao_percentual","projecao_crescimento", "nivel_satisfacao", "frequencia_interacao","periodo","interesse_novos_servicos"])

    print(merged_df.columns.tolist())




        # Aplicando OneHotEncoder nas variáveis categóricas
    categorical_features = [
        "setor"
    ]

        # Definindo variável alvo
    target = "feedback_servicos_produtos"
    if target in categorical_features:
        categorical_features.remove(target)

        # Convertendo colunas categóricas para string
    for feature in categorical_features:
        merged_df[feature] = merged_df[feature].astype(str)

        # Instancia o OneHotEncoder
    onehotencoder = OneHotEncoder(drop='first', sparse_output=False)
    categorical_encoded = onehotencoder.fit_transform(merged_df[categorical_features])

        # Cria DataFrame para as variáveis categóricas codificadas
    encoded_columns = onehotencoder.get_feature_names_out(categorical_features)
    categorical_df = pd.DataFrame(categorical_encoded, columns=encoded_columns, index=merged_df.index)

        # Concatenando o DataFrame codificado com o restante dos dados
    merged_df = pd.concat([merged_df.drop(columns=categorical_features), categorical_df], axis=1)

        # Normalizando as variáveis numéricas
    numerical_features = [
            "numero_funcionarios", "faturamento_anual", "receita_anual", 
            "crescimento_anual", 
    ]

    for feature in numerical_features:
        # Remover sinal de porcentagem se houver e converter para float
        merged_df[feature] = merged_df[feature].replace({r"%": ""}, regex=True).astype(float) / 100 if merged_df[feature].dtype == 'object' else merged_df[feature].astype(float)


    # Normalizando as variáveis numéricas
    scaler = StandardScaler()
    merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])



    variance_threshold = VarianceThreshold(threshold=0.05)
    X_reduced = variance_threshold.fit_transform(merged_df.drop(columns=[target]))

    selected_columns = merged_df.drop(columns=[target]).columns[variance_threshold.get_support()]
    X_reduced_df = pd.DataFrame(X_reduced, columns=selected_columns)

    X_reduced_df_clean = X_reduced_df.fillna(X_reduced_df.mean())

    pca = PCA(n_components=0.95)  
    X_pca = pca.fit_transform(X_reduced_df_clean)

    # Criando DataFrame com os componentes principais
    X_pca_df = pd.DataFrame(X_pca)
    y = merged_df[target]

    # Verificar e garantir que não haja valores nulos ou strings
    X_pca_df = X_pca_df.apply(pd.to_numeric, errors='coerce').fillna(0)



    # Dividindo os dados
    X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.3, random_state=42)



    # Listando algoritmos a serem testados
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVC": SVC(random_state=42)
    }


    min_class_samples = y_train.value_counts().min()

    if min_class_samples > 1:
        cv_folds = max(2, min(5, min_class_samples))
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        print("Usando KFold devido ao número insuficiente de amostras por classe.")
        kf = KFold(n_splits=2, shuffle=True, random_state=42)

    lista_predicao = []
    # Avaliando cada modelo com busca de hiperparâmetros
    for model_name, model in models.items():
        print(f"\nAvaliando modelo: {model_name}")

        if model_name == "Decision Tree":
            param_grid = {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif model_name == "SVC":
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf']
            }

        grid_search = GridSearchCV(model, param_grid, cv=skf if min_class_samples > 1 else kf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Melhores hiperparâmetros para {model_name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Avaliando o melhor modelo
        y_pred = best_model.predict(X_test)
        lista_predicao += y_pred.tolist()
    contagem = Counter(lista_predicao)
    mais_comum = contagem.most_common(1)[0] 
    print(mais_comum)
    return jsonify({'previsao':mais_comum[0]})

if __name__ == '__main__':
    app.run(debug=True)
