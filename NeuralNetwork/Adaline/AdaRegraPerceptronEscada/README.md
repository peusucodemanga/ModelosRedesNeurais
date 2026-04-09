# Rede neural: Adaline
Dando continuidade para os tipos diferentes de redes neurais, a rede Adaline ou a rede "Adaptive Linear Neuron" é a próxima na cronologia após a rede Perceptron, que serviu como base para a construção dela.

### Diferança Adaline VS Perceptron:
Essa rede neural tem sua principal diferença na separação linear encontrada após o aprendizado do neurônio. Nela, após os estágios de aprendizado ela acha a melhor reta que separa os pontos enquanto que na rede Perceptron, ele pode achar diversas retas diferentes que também podem separar esses mesmos pontos.

Essa diferença ocorre por causa da ordem em que a Adaline opera e pelas duas principais diferenças entre elas duas, o modo com a função de ativação funciona e o seu método de aprendizagem

Vale destacar que a rede Adaline foi feita especificamente para resolver problemas de regressão linear, em que era preciso oferecer resultados diferentes de 1 e 0 e isso só foi possível após a sua função de ativação linear.
### Método de aprendizagem

A atualização dos pesos é feita de forma parecida com a fórmula do Perceptron, porém o erro aqui é tratado de forma continua e não binária, então o neurônio de certa forma se preocupa em achar a otimização da melhor reta possível para o problema apresentado. 

Então, no final o resultado obtido através do erro com o número contínuo diretamente.

