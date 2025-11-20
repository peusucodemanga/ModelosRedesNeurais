import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

X = np.array([[0, 0], 
              [0, 1],
              [1, 0], 
              [1, 1]])

y = np.array([[0], 
              [1], 
              [1], 
              [0]])

np.random.seed(1)

pesos_entrada_oculta = 2 * np.random.random((2, 4)) - 1
pesos_oculta_saida = 2 * np.random.random((4, 1)) - 1

taxa_aprendizado = 0.1
epocas = 10000

for i in range(epocas):
    #essa é a camda de entrada/indo pra oculta
    entrada_oculta = np.dot(X, pesos_entrada_oculta)
    saida_oculta = sigmoide(entrada_oculta)

    #essa é a cmad indo da oculta pra saida comum
    #lembrar que da pra fazer com np.dot pq ele faz o produto vet automaticamente usando essa funcao do numpy
    entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
    saida_prevista = sigmoide(entrada_saida)

    #calculo do erro comum :D
    erro = y - saida_prevista
    
    #
    delta_saida = erro * derivada_sigmoide(saida_prevista)
    erro_camada_oculta = delta_saida.dot(pesos_oculta_saida.T)
    delta_oculta = erro_camada_oculta * derivada_sigmoide(saida_oculta)

    #
    pesos_oculta_saida += saida_oculta.T.dot(delta_saida) * taxa_aprendizado
    pesos_entrada_oculta += X.T.dot(delta_oculta) * taxa_aprendizado

print("Saída final após treino:")
print(saida_prevista)
print("\n")
#aqui ele arredonda automaticmanete, lembrar de usar 2,4 e 4,1 nos pesos pq usando 2,1 2,1 n tava convergindo 
print(np.round(saida_prevista))