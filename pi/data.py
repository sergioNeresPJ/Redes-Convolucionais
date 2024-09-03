import os
import random
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10

import pi


def _extract_tensors(dset, num=None, x_dtype=torch.float32):
  """
  Extract the data and labels from a CIFAR10 dataset object and convert them to
  tensors.

  Input:
  - dset: A torchvision.datasets.CIFAR10 object
  - num: Optional. If provided, the number of samples to keep.
  - x_dtype: Optional. data type of the input image

  Returns:
  - x: `x_dtype` tensor of shape (N, 3, 32, 32)
  - y: int64 tensor of shape (N,)
  """
  x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
  y = torch.tensor(dset.targets, dtype=torch.int64)
  if num is not None:
    if num <= 0 or num > x.shape[0]:
      raise ValueError('Invalid value num=%d; must be in the range [0, %d]'
                       % (num, x.shape[0]))
    x = x[:num].clone()
    y = y[:num].clone()
  return x, y


def cifar10(num_train=None, num_test=None, x_dtype=torch.float32):
  """
  Return the CIFAR10 dataset, automatically downloading it if necessary.
  This function can also subsample the dataset.

  Inputs:
  - num_train: [Optional] How many samples to keep from the training set.
    If not provided, then keep the entire training set.
  - num_test: [Optional] How many samples to keep from the test set.
    If not provided, then keep the entire test set.
  - x_dtype: [Optional] Data type of the input image

  Returns:
  - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
  - y_train: int64 tensor of shape (num_train, 3, 32, 32)
  - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
  - y_test: int64 tensor of shape (num_test, 3, 32, 32)
  """
  download = not os.path.isdir('cifar-10-batches-py')
  dset_train = CIFAR10(root='.', download=download, train=True)
  dset_test = CIFAR10(root='.', train=False)
  x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
  x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)
 
  return x_train, y_train, x_test, y_test


def preprocessar_cifar10(
    cuda=True,
    exibir_amostras=True,
    truque_vies=False,
    achatado=True, 
    taxa_validacao=0.2,
    dtype=torch.float32):
  """
  Retorna uma versão pré-processada do conjunto de dados CIFAR10, automaticamente
  baixando se necessário. Realizamos as seguintes etapas:

  (0) [Opcional] Visualiza algumas imagens do conjunto de dados
  (1) Normaliza os dados subtraindo a média
  (2) [Opcional] Remodela cada imagem de forma (3, 32, 32) em um vetor de forma (3072,)
  (3) [Opcional] Truque do viés: adiciona uma dimensão extra de uns aos dados
  (4) Elabora um conjunto de validação do conjunto de treinamento

  Entrada:
  - cuda: se verdadeiro, move todo o conjunto de dados para a GPU
  - taxa_validacao: Real no intervalo (0, 1) indicando a fração do conjunto de 
    treinamento para reservar para validação
  - truque_vies: Booleano dizendo se deve ou não aplicar o truque do viés
  - achatado: Booleano dizendo se deve ou não remodelar cada imagem em um vetor
  - exibir_amostras: Booleano dizendo se deve ou não visualizar amostras de dados
  - dtype: opcional, tipo de dados da imagem de entrada X

  Retorna um dicionário com as seguintes chaves:
  - 'X_treino': tensor de dtype `dtype` e shape (N_treino, D) contendo imagens de treino
  - 'X_val': tensor de dtype `dtype` e shape (N_val, D) contendo imagens de validação
  - 'X_teste': tensor de dtype `dtype` e shape (N_teste, D) contendo imagens de teste
  - 'y_treino': tensor de dtype int64 e shape (N_treino,) contendo rótulos de treino
  - 'y_val':  tensor de dtype int64 e shape (N_val,) contendo rótulos de validação
  - 'y_teste':  tensor de dtype int64 e shape (N_teste,) contendo rótulos de teste

  N_treino, N_val e N_teste são os números de amostras nos conjuntos de treinamento, 
  validação e teste, respectivamente. Os valores precisos de N_treino e N_val são 
  determinados pelo parâmetro taxa_validacao. D é a dimensão dos dados de imagem; 
  se truque_vies for False e achatado for True, então D = 32 * 32 * 3 = 3072;
  se truque_vies for True e achatado for True, então D = 1 + 32 * 32 * 3 = 3073.    
  """
  X_treino, y_treino, X_teste, y_teste = cifar10(x_dtype=dtype)

  # Move os dados para a GPU
  if cuda:
    X_treino = X_treino.cuda()
    y_treino = y_treino.cuda()
    X_teste = X_teste.cuda()
    y_teste = y_teste.cuda()

  # 0. Visualiza algumas amostras do conjunto de dados.
  if exibir_amostras:
    classes = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    amostras_por_classe = 12
    amostras = []
    pi.redefinir_semente(0)
    for y, cls in enumerate(classes):
        plt.text(-4, 34 * y + 18, cls, ha='right')
        idxs, = (y_treino == y).nonzero(as_tuple=True)
        for i in range(amostras_por_classe):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            amostras.append(X_treino[idx])
    img = torchvision.utils.make_grid(amostras, nrow=amostras_por_classe)
    plt.imshow(pi.tensor_to_image(img))
    plt.axis('off')
    plt.show()

  # 1. Normaliza os dados: subtrai o RGB médio (média zero)
  imagem_media = X_treino.mean(dim=(0, 2, 3), keepdim=True)
  X_treino -= imagem_media
  X_teste -= imagem_media

  # 2. Remodela os dados de imagem em linhas
  if achatado:
    X_treino = X_treino.reshape(X_treino.shape[0], -1)
    X_teste = X_teste.reshape(X_teste.shape[0], -1)

  # 3. Adiciona dimensão do viés e transforma em colunas
  if truque_vies:
    treino_uns = torch.ones(X_treino.shape[0], 1, device=X_treino.device)
    X_treino = torch.cat([X_treino, treino_uns], dim=1)
    teste_uns = torch.ones(X_teste.shape[0], 1, device=X_teste.device)
    X_teste = torch.cat([X_teste, teste_uns], dim=1)

  # 4. Pega o conjunto de validação do conjunto de treinamento
  # Nota: Não deve ser retirado do conjunto de teste
  # Para permumação aleatória, você pode usar torch.randperm ou torch.randint
  # Mas, para esta tarefa, usamos o fatiamento.
  num_treinamento = int( X_treino.shape[0] * (1.0 - taxa_validacao) )
  num_validacao = X_treino.shape[0] - num_treinamento

  # retorna o conjunto de dados
  dic_dados = {}
  dic_dados['X_val'] = X_treino[num_treinamento:num_treinamento + num_validacao]
  dic_dados['y_val'] = y_treino[num_treinamento:num_treinamento + num_validacao]
  dic_dados['X_treino'] = X_treino[0:num_treinamento]
  dic_dados['y_treino'] = y_treino[0:num_treinamento]

  dic_dados['X_teste'] = X_teste
  dic_dados['y_teste'] = y_teste
  return dic_dados
