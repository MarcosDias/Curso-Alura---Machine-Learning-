import csv

def carregar_acessos():
    """ Carrega dados CSV"""
    X = []
    Y = []

    with open('acesso.csv', 'r') as csv_file:
        leitor = csv.reader(csv_file)

        next(leitor)

        for home, como_funciona, contato, comprou in leitor:
            X.append([int(home), int(como_funciona), int(contato)])
            Y.append(int(comprou))

        return X, Y
