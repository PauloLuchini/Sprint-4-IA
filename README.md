# Hiperpersonalização de Marketing - Previsão de Feedback

Este projeto é uma aplicação web que utiliza modelos de aprendizado de máquina para prever o feedback de empresas com base em dados específicos. A aplicação é dividida em um front-end em HTML e um back-end em Python, utilizando Flask e MongoDB para gerenciar dados.

## Estrutura do Projeto

/HiperpersonalizacaoMarketing │ ├── /modelos │ ├── decision_tree_model.joblib │ ├── random_forest_model.joblib │ └── svc_model.joblib │ ├── app.py └── index.html


- **/modelos**: contém os modelos treinados de machine learning salvos em formato `.joblib`.
- **app.py**: código do servidor Flask que gerencia a lógica do back-end e a previsão de feedback.
- **index.html**: arquivo HTML que contém a interface do usuário para enviar dados e receber previsões.

## Tecnologias Utilizadas

- Python
  - Flask: Framework para desenvolvimento web.
  - scikit-learn: Biblioteca para machine learning.
  - Pandas: Biblioteca para manipulação de dados.
  - pymongo: Biblioteca para interagir com o MongoDB.
  - joblib: Biblioteca para salvar e carregar modelos.
  
- HTML/CSS: Para a estrutura e estilo da interface do usuário.

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu_usuario/HiperpersonalizacaoMarketing.git
   cd HiperpersonalizacaoMarketing
Instale as dependências do Python:



pip install flask pymongo scikit-learn pandas joblib flask-cors

Certifique-se de ter o MongoDB instalado e em execução em localhost:27017.

Coloque os modelos treinados na pasta /modelos.

## Como Executar

Inicie o servidor Flask:

python app.py

Abra o index.html em um navegador.

Preencha os dados necessários no formulário e clique em "Enviar para Previsão". A previsão será exibida na tela.

## Integrantes
RM: 550373 - Leonardo Yuuki Nakazone

RM: 99119 - Leonardo Blanco Pérez Ribeiro

RM: 98082 - Paulo Henrique Luchini Ferreira

RM: 97999 - Gustavo Moreira Gonçalves

RM: 552184 - Daniel Soares Delfin
