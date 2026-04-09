import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivadaSigmoide(X):
    return X * (1 - X)

def inicializar_parametros():
    #Gera um array de numeros aleatorios tendo o primeiro sendo o "x" e o segundo sendo o "y"
    pesosEntradaOculta = 2 * np.random.random((2, 4)) - 1
    pesosOcultaSaida = 2 * np.random.random((4, 1)) - 1
    
    # Mesma coisa aqui 
    biasEntrada = 2 * np.random.random((1, 4)) - 1
    biasSaida = 2 * np.random.random((1, 1)) - 1
    #(2, 4, 1)
    return pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida

def forward(X, pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida):
    #s1 e o SOMATORIO e a1 é o calculo da sigmoide usando esse somatorio
    s1 = np.dot(X, pesosEntradaOculta) + biasEntrada
    a1 = sigmoide(s1)
    
    #mesma coisa com o s2
    s2 = np.dot(a1, pesosOcultaSaida) + biasSaida
    a2 = sigmoide(s2)
    
    return a1, a2

def backpropagation(X, y, a1, a2, pesosOcultaSaida):
    erro_saida = y - a2
    delta_saida = erro_saida * derivadaSigmoide(a2)
    
    erro_oculta = delta_saida.dot(pesosOcultaSaida.T)
    delta_oculta = erro_oculta * derivadaSigmoide(a1)
    
    return delta_oculta, delta_saida

def atualizar_pesos(X, a1, delta_oculta, delta_saida, pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida, taxa):
    #atualização dos pesos no geral, nao tem muito o que dizer :D
    pesosOcultaSaida += a1.T.dot(delta_saida) * taxa
    pesosEntradaOculta += X.T.dot(delta_oculta) * taxa
    
    #esse keepdims e pra manter a dimensão do array apos a operação da soma
    biasSaida += np.sum(delta_saida, axis=0, keepdims=True) * taxa
    biasEntrada += np.sum(delta_oculta, axis=0, keepdims=True) * taxa
    
    return pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida

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
    
    epocas = 10000
    taxa_aprendizado = 0.1
    
    pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida = inicializar_parametros()

    for i in range(epocas):
        a1, a2 = forward(X, pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida)
        
        delta_oculta, delta_saida = backpropagation(X, y, a1, a2, pesosOcultaSaida)
        
        pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida = atualizar_pesos(
            X, a1, delta_oculta, delta_saida, 
            pesosEntradaOculta, biasEntrada, pesosOcultaSaida, biasSaida, taxa_aprendizado
        )

    print("Saida final com Bias:\n")
    print(a2)
    print("\nArredondado:\n", np.round(a2))
#custo e fronteira de separação
        

if __name__ == "__main__":
    main()