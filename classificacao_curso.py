import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

# Dataframe obtido a partir do csv
# Troca do valor do resultado para um valor categorico binario (sim e nao)
df = pd.read_csv('curso.csv')

# Acessando colunas especificadas do dataframe
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

# Converte colunas categoricas para colunas binarias
Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df # So ha a coluna binaria

# Get os valores reais
X = Xdummies_df.values
Y = Ydummies_df.values

# Divisao padrao de dados para treino e execucao
PORCENTAGEM_TREINO = 0.8
PORCENTAGEM_TESTE = 0.1
TAMANHO_TREINO = int(PORCENTAGEM_TREINO * len(Y))
TAMANHO_TESTE = int(PORCENTAGEM_TESTE * len(Y))
TAMANHO_VALIDACAO = int(len(Y) - TAMANHO_TREINO - TAMANHO_TESTE)

# Separacao dos dados
X_treino = X[:TAMANHO_TREINO]
Y_treino = Y[:TAMANHO_TREINO]

X_teste = X[TAMANHO_TREINO:TAMANHO_TREINO + TAMANHO_TESTE]
Y_teste = Y[TAMANHO_TREINO:TAMANHO_TREINO + TAMANHO_TESTE]

# Criacao da 3º fase do processo
# Alem da fase de treino e a teste
# adicionamos a fase de validacao com o algoritmo 
# melhor entre as opcoes
X_validacao = X[TAMANHO_TREINO + TAMANHO_TESTE:]
Y_validacao = Y[TAMANHO_TREINO + TAMANHO_TESTE:]


def fit_and_predict(nome, 
                    modelo, 
                    treino_dados, 
                    treino_resultado, 
                    teste_dados, 
                    teste_resultado):
                    
  modelo.fit(treino_dados, treino_resultado)
  resultado = modelo.predict(teste_dados)

  acertos = resultado == teste_resultado

  total_acertos = sum(acertos)
  total_elementos = len(teste_dados)

  taxa_acerto = 100.0 * total_acertos / total_elementos

  msg = 'Taxa de acerto do algoritmo {0}: {1}'.format(nome, taxa_acerto)

  print(msg)
  return taxa_acerto

def teste_real(modelo, validacao_treino, validacao_teste):
  resultado = modelo.predict(validacao_treino)
  acertos = resultado == validacao_teste

  total_acertos = sum(acertos)
  total_de_elementos = len(validacao_teste)

  taxa_acerto = 100.0 * total_acertos / total_de_elementos

  msg = 'Taxa de acerto do vencedor entre os dois \
algoritmos no mundo real: {0}'.format(taxa_acerto)
  print(msg)

# Algoritmo 1
modelo_multinomalNB = MultinomialNB()
resultado_multinomalNB = fit_and_predict('MultinomialNB',
                                          modelo_multinomalNB,
                                          X_treino, 
                                          Y_treino, 
                                          X_teste, 
                                          Y_teste) #82.0%

# Algoritmo 2
modelo_AdaBoost = AdaBoostClassifier()
resultado_adaBoost = fit_and_predict('AdaBoostClassifier', 
                                      modelo_AdaBoost, 
                                      X_treino, 
                                      Y_treino, 
                                      X_teste, 
                                      Y_teste) #85.0%

# Qual eh o melhor?
if resultado_multinomalNB > resultado_adaBoost:
  vencedor = modelo_multinomalNB
else:
  vencedor = modelo_multinomalNB

# Rodada de validacao
teste_real(vencedor, X_validacao, Y_validacao) #82.0

acerto_base = max(Counter(Y).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(Y)

print('Taxa de acerto base: %f' % taxa_de_acerto_base) #83.2%