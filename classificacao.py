from data import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

X_treino = X[:90]
Y_treino = Y[:90]

X_teste = X[-9:]
Y_teste = Y[-9:]

modelo = MultinomialNB()
modelo.fit(X_treino, Y_treino)

resultado = modelo.predict(X_teste)

diferencas = resultado - Y_teste

acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(X_teste)

taxa_acertos = 100.0 * total_acertos / total_elementos

print(taxa_acertos)


