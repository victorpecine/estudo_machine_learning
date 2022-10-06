from sklearn.svm import LinearSVC


# Características
# O pelo é longo?
# A perna é curta?
# Late?

porco1 = [0, 1, 0]

porco2 = [0, 1, 1]

porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]

cachorro2 = [1, 0, 1]

cachorro3 = [1, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

classes = [1, 1, 1, 0, 0, 0]
# porco = 1
# cachorro = 0

modelo = LinearSVC().fit(dados, classes)


animal = []

pelo_longo = int(input('O pelo é longo?\nNão [0] / Sim [1]\n'))

perna_curta = int(input('A perna é curta?\nNão [0] / Sim [1]\n'))

latido = int(input('Late?\nNão [0] / Sim [1]\n'))

animal.append(pelo_longo)
animal.append(perna_curta)
animal.append(latido)

animal_predito = modelo.predict([animal])

if animal_predito == 0:
    print('Cachorro')
else:
    print('Porco')
