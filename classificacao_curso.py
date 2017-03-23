import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Modelo utilizado na predicao
modelo = MultinomialNB()
# Dataframe obtido a partir do csv
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
PORCENTAGEM_TREINO = 0.9
TAMANHO_TREINO = int(PORCENTAGEM_TREINO * len(Y))
TAMANHO_TESTE = int(len(Y) - TAMANHO_TREINO)

# Separacao dos dados
X_treino = X[:TAMANHO_TREINO]
Y_treino = Y[:TAMANHO_TREINO]

X_teste = X[-TAMANHO_TESTE:]
Y_teste = Y[-TAMANHO_TESTE:]

# Treino do modelo
modelo.fit(X_treino, Y_treino)
# Predicao
resultado = modelo.predict(X_teste)

# Compara o resultado com os resultados de test
# ex: (result = [1, 1, 0]) - (resultados de test = [1, 1, 0])
# ex: [0, 0, 0]
diferencas = resultado - Y_teste

# lista todos os acertos
acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(X_teste)

taxa_acerto = 100.0 * total_acertos / total_elementos

print(taxa_acerto) # 82%
print(total_elementos) # 100