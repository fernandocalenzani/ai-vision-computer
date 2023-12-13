import numpy as np

"""
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

"""


import numpy as np
import sympy as sp

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
layers = [
    np.array([
        [-0.424, 0.358],
        [-0.740, -0.577],
        [-0.961, -0.469]
    ]),
    np.array([-0.017, -0.893, 0.148])
]
Yexp = [0, 1, 1, 0]
Y = []

err = 100
Err = []

u = sp.symbols('u')
f = 1 / (1 + sp.exp(-u))
moment = 1


def calc_delta(err, u_value):
    dydx = sp.diff(f, u)
    return err*dydx.subs(u, u_value)


def calc_delta_hide_layer(dydx_f, w, dOut):
    return dydx_f * w * dOut


def f_sum(inputs, weights):
    outputs = inputs
    for layer_weights in weights:
        summ = np.dot(layer_weights, outputs)
        outputs = np.vectorize(lambda x: f.subs(u, x))(summ)
    return outputs


for i in range(0, len(X), 1):
    y = f_sum(X[i], layers)
    err = y - Yexp[i]

    delta = calc_delta(err, y)
    delta_hide = calc_delta_hide_layer()
    print(delta, err, y)

    Err.append(err)
    Y.append(y)

""" print(Y)
print(Err) """
