import numpy as np

andInput = np.array([
    [1,0],
    [1,1],
    [0,0],
    [0,1]
])

def adaline(andInput):
    a = [1,1,0,0]
    b = [1,0,1,0]
    x = [1,0,0,0]


    weights = [0.7, 0.7] #pesos iniciais
    bias = 0 #bias inicial
    lr = 0.01  #learning rate
    epochs = 0 #epoca inicial
    
    while epochs <= 1000:  
        i = 0
        while i < 4:
            soma = a[i]*weights[0] + b[i]*weights[1] + bias
            erro = x[i] - soma
            weights[0] += lr * erro * a[i]
            weights[1] += lr * erro * b[i]
            bias += lr * erro
            i += 1
        epochs += 1
    
    for i in range(4):
        soma = np.dot(andInput[i], weights) + bias
        out = 1 if soma >= 0.5 else 0  # discretiza só para mostrar
        print(f"{andInput[i]} = {out}")
    
    print('\nPesos obtidos:', weights)
    print('Bias obtido:', bias)

adaline(andInput)
