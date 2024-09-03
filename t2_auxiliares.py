"""
Funções auxiliares usadas na Tarefa 1
"""
import torch
import torchvision
import pi
import matplotlib.pyplot as plt
import random
import math

def ola_auxiliar():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente esteja configurado corretamente no Google Colab.    
  """
  print('Olá do t2_auxiliares.py!')


def gerar_dados_de_exemplo(
    num_amostras=5,
    tamanho_entrada=4,
    tamanho_oculta=10,
    num_categorias=3,
    dtype=torch.float32,
    device='cuda'):
  """
  Gera dados de exemplo para usar ao desenvolver uma rede de duas camadas.

  Entrada:
  - num_amostras: Inteiro N indicando o tamanho do conjunto de dados
  - tamanho_entrada: Inteiro D indicando a dimensão dos dados de entrada
  - tamanho_oculta: Inteiro H indicando o número de unidades ocultas no modelo
  - num_classes: Inteiro C indicando o número de categorias
  - dtype: tipo de dados do torch para todos os dados retornados
  - device: dispositivo no qual os tensores de saída irão residir 

  Retorno: Uma tupla de:
  - exemplo_X: tensor de dtype `dtype` e shape (N, D) contendo os dados
  - exemplo_y: tensor de dtype int64 e shape (N,) contendo rótulos, onde cada 
    elemento é um inteiro no intervalo [0, C)
  - params: Um dicionário de parâmetros do modelo de exemplos, com chaves:
    - 'W1': tensor de dtype `dtype` e shape (D, H) fornecendo pesos da 1a camada
    - 'b1': tensor de dtype `dtype` e shape (H,) fornecendo vieses da 1a camada
    - 'W2': tensor de dtype `dtype` e shape (H, C) fornecendo pesos da 2a camada
    - 'b2': tensor de dtype `dtype` e shape (C,) fornecendo vieses da 2a camada
  """
  N = num_amostras
  D = tamanho_entrada
  H = tamanho_oculta
  C = num_categorias

  # Define a semente aleatória para experimentos repetíveis.
  pi.redefinir_semente(0)

  # Gera alguns parâmetros aleatórios, armazenando-os em um dicionário
  params = {}
  params['W1'] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
  params['b1'] = torch.zeros(H, device=device, dtype=dtype)
  params['W2'] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)
  params['b2'] = torch.zeros(C, device=device, dtype=dtype)

  # Gera algumas amostras e rótulos aleatórios
  exemplo_X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
  exemplo_y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

  return exemplo_X, exemplo_y, params


################# Visualizações #################

def plotar_estats(dic_estat):
  # Plota a função de perda e as acurácias de treinamento / validação
  plt.subplot(1, 2, 1)
  plt.plot(dic_estat['historico_perda'], 'o')
  plt.title('Histórico de Perda')
  plt.xlabel('Iteração')
  plt.ylabel('Perda')

  plt.subplot(1, 2, 2)
  plt.plot(dic_estat['historico_acc_treinamento'], 'o-', label='treinamento')
  plt.plot(dic_estat['historico_acc_validacao'], 'o-', label='validacao')
  plt.title('Histórico de Acurácia de Classificação')
  plt.xlabel('Época')
  plt.ylabel('Acurácia de Classificação')
  plt.legend()

  plt.gcf().set_size_inches(14, 4)
  plt.show()


def visualizar_grade(Xs, lim_sup=255.0, padding=1):
  """
  Remodele um tensor 4D de dados de imagem em uma grade para facilitar a visualização.

  Entradas:
  - Xs: Dados de shape (N, H, W, C)
  - lim_sup: A grade de saída terá valores escalados para o intervalo [0, lim_sup]
  - padding: O número de pixels em branco entre os elementos da grade
  """
  (N, H, W, C) = Xs.shape
  # print(Xs.shape)
  tamanho_grade = int(math.ceil(math.sqrt(N)))
  altura_grade = H * tamanho_grade + padding * (tamanho_grade - 1)
  largura_grade = W * tamanho_grade + padding * (tamanho_grade - 1)
  grade = torch.zeros((altura_grade, largura_grade, C), device=Xs.device)
  proximo_indice = 0
  y0, y1 = 0, H
  for y in range(tamanho_grade):
    x0, x1 = 0, W
    for x in range(tamanho_grade):
      if proximo_indice < N:
        img = Xs[proximo_indice]
        inf, sup = torch.min(img), torch.max(img)
        grade[y0:y1, x0:x1] = lim_sup * (img - inf) / (sup - inf)
        proximo_indice += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  return grade


# Visualizar os pesos da rede
def mostrar_pesos_da_rede(rede):
  W1 = rede['W1']
  W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
  plt.imshow(visualizar_grade(W1, padding=3).type(torch.uint8).cpu())
  plt.gca().axis('off')
  plt.show()


def plotar_curvas_de_acuracia(dic_estat):
  plt.subplot(1, 2, 1)
  for key, single_stats in dic_estat.items():
    plt.plot(single_stats['historico_acc_treinamento'], label=str(key))
  plt.title('Histórico de Acurácia de Treinamento')
  plt.xlabel('Época')
  plt.ylabel('Acurácia de Classificação')

  plt.subplot(1, 2, 2)
  for key, single_stats in dic_estat.items():
    plt.plot(single_stats['historico_acc_validacao'], label=str(key))
  plt.title('Histórico de Acurácia de Validação')
  plt.xlabel('Época')
  plt.ylabel('Acurácia de Classificação')
  plt.legend()

  plt.gcf().set_size_inches(14, 5)
  plt.show()
