import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivadaSigmoide(x):
    return x * (1 - x)


def gerarAleat(selet):
            pesosEntradaOculta = 2 * np.random.random((2, 4)) - 1
            pesosOcultaSaida = 2 * np.random.random((4, 1)) - 1
            biasEntrada = 2 * np.random.random((2, 4)) - 1
            biasSaida = 2 * np.random.random((4, 1)) - 1
            return pesosEntradaOculta, pesosOcultaSaida, biasEntrada, biasSaida



def main():
    X = np.array([[0, 0], 
                [0, 1],
                [1, 0], 
                [1, 1]])

    y = np.array([[0], 
                [1], 
                [1], 
                [0]])

    np.random.seed(1)

    pesosEntradaOculta, pesosOcultaSaida, biasEntrada, biasSaida = gerarAleat(2)


    taxaAprendizado = 0.1
    epocas = 10000

    for i in range(epocas):
        #essa é a camda de entrada/indo pra oculta
        entradaOculta = np.dot(X, pesosEntradaOculta)
        saidaOculta = sigmoide(entradaOculta)

        #essa é a cmad indo da oculta pra saida comum
        #lembrar que da pra fazer com np.dot pq ele faz o produto vet automaticamente usando essa funcao do numpy
        entradaSaida = np.dot(saidaOculta, pesosOcultaSaida)
        saidaPrevista = sigmoide(entradaSaida)

        #calculo do erro comum :D
        erro = y - saidaPrevista
        
        #calc dos erros (?)
        deltaSaida = erro * derivadaSigmoide(saidaPrevista)
        erroCamadaOculta = deltaSaida.dot(pesosOcultaSaida.T)
        deltaOculta = erroCamadaOculta * derivadaSigmoide(saidaOculta)

        #atualizacao Pesos
        pesosOcultaSaida += saidaOculta.T.dot(deltaSaida) * taxaAprendizado
        pesosEntradaOculta += X.T.dot(deltaOculta) * taxaAprendizado


        
        #lembrar do bias :(
        #print(pesosOcultaSaida)
        #print(pesosEntradaOculta)


    print("Saida final apos treino:D \n")
    print(saidaPrevista)
    print("\n\n")
    #aqui ele arredonda automaticmanete, lembrar de usar 2,4 e 4,1 nos pesos pq usando 2,1 2,1 n tava convergindo 
    print(np.round(saidaPrevista))
    #MULTI LAYER PERCEPTRON 
if __name__ == "__main__":
    main()