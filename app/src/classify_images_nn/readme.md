# COMPUTER VISION - NEURAL NETWORK

## Important Topics

Here are some of the key topics I will be covering during this study project:

```
01. Coleta e Preparação de Dados
02. Pré-processamento de Dados
03. Escolha da Arquitetura da Rede Neural
04. Inicialização de Parâmetros
05. Feedforward (Propagação Direta)
06. Cálculo da Função de Ativação
07. Cálculo da Função de Custo
08. Retropropagação (Backpropagation)
09. Treinamento da Rede Neural
10. Avaliação do Modelo
11. Ajuste e Otimização
12. Implantação
```

## Base

1. Inicialização:
   Inicialize a estrutura da rede neural, incluindo o número de camadas, o número de neurônios em cada camada, os pesos iniciais e as funções de ativação.

2. Forward Pass (Propagação Direta):
   Alimente os dados de entrada na camada de entrada.
   Para cada camada oculta e a camada de saída:
   Calcule a soma ponderada das entradas multiplicadas pelos pesos.
   Aplique a função de ativação à soma ponderada para obter as saídas da camada.

3. Cálculo do Erro:
   Compare as saídas da rede com os valores alvo para calcular o erro. Isso é feito usando uma função de erro, como o erro quadrático médio (MSE).

4. Backward Pass (Retropropagação):
   Propague o erro da camada de saída para as camadas anteriores.
   Atualize os pesos em cada camada para reduzir o erro. Isso é feito usando algoritmos de otimização, como o Gradiente Descendente.
   Ajuste os pesos com base nas derivadas dos erros em relação aos pesos (gradiente) usando a regra da cadeia.

5. Repetição:
   Repita as etapas 2 a 4 para várias épocas de treinamento até que o erro atinja um valor aceitável ou até que a rede convirja.

6. Predição:
   Use a rede neural treinada para fazer previsões em novos dados não vistos.

- Bias: additional neuron to improve the performance

### MEAN SQUARE ERROR (MSE)

- MSE = (1/n) . SUM(fi - yi)^2

### ROOT MEAN SQUARE ERROR (RMSE)

- RMSE = ((1/n) . SUM(fi - yi)^2)^1/2

### PARAMS

- Learning Rate
- Batch Size
- Epochs
