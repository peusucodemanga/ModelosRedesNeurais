import numpy as np

def adaline(andInput):
    a = [1, 1, 0, 0]
    b = [0, 1, 0, 1]
    d = [0, 1, 0, 0]

    weights = [0.7, 0.7]
    bias = 0
    lr = 0.1
    epochs = 0
    tolerancia = 0.001
    max_epochs = 1000

    #funcao sigmoide normal
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    #derivada da sigmoide, sendo x a função sigmoide normal
    def sigmoideDerivada(x):
        return x * (1 - x)

    while epochs < max_epochs:
        for i in range(4): 
            soma = a[i]*weights[0] + b[i]*weights[1] + bias
            y = sigmoide(soma)
            erro = d[i] - y
            eqm = erro**2
            weights[0] += lr * erro * sigmoideDerivada(y) * a[i]
            weights[1] += lr * erro * sigmoideDerivada(y) * b[i]
            bias += lr * erro * sigmoideDerivada(y)
        epochs += 1

        #para antes de realizar todas as epocas usando o erro quadratico medio
        if eqm < tolerancia:
            break
    

    print(f"Treinamento concluído em {epochs} épocas")
    print(f"Pesos finais: {weights}")
    print(f"Bias final: {bias}\n")

    for i in range(4):
        soma = andInput[i][0]*weights[0] + andInput[i][1]*weights[1] + bias
        out = sigmoide(soma)
        saida_binaria = 1 if out >= 0.5 else 0
        print(f"{andInput[i]} = {saida_binaria} (saída real: {out:.3f})")

andInput = np.array([
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1]
])

adaline(andInput)
