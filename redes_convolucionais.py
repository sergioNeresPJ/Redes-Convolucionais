"""
Implementa redes convolucionais no PyTorch.
AVISO: você NÃO DEVE usar ".to()" ou ".cuda()" em cada bloco de implementação.
"""
import torch
import random
from pi import Solucionador
from redes_totalmente_conectadas import *

def ola_redes_convolucionais():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente esteja configurado corretamente no Google Colab.    
  """
  print('Olá do redes_convolucionais.py!')


class RedeConvTresCamadas(torch.nn.Module):
  """
  Uma rede convolucional de três camadas com a seguinte arquitetura:
  conv - relu - max pooling 2x2 - linear - relu - linear
  A rede opera em mini-lotes de dados que têm shape (N, C, H, W)
  consistindo em N imagens, cada uma com altura H e largura W e com C
  canais de entrada.
  """

  def __init__(self, dims_entrada=(3, 32, 32), num_filtros=32, tamanho_filtro=5,
               dim_oculta=100, num_classes=10, escala_peso=1e-3):
    """
    Inicializa a nova rede.
    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Número de filtros a serem usados na camada de convolução
    - tamanho_filtro: Largura/altura dos filtros a serem usados na camada de convolução
    - dim_oculta: Número de unidades a serem usadas na camada oculta totalmente conectada
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos.
    """
    super().__init__()

    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)
    
    self.escala_peso = escala_peso

    ########################################################################
    # TODO: Inicialize pesos, vieses para a rede convolucional de três     #
    # camadas. Os pesos devem ser inicializados a partir de uma Gaussiana  #
    # centrada em 0,0 com desvio padrão igual a escala_peso; vieses devem  #
    # ser inicializados com zero.                                          #
    #                                                                      #
    # IMPORTANTE: Para esta tarefa, você pode assumir que o preenchimento  #
    # e o passo da primeira camada de convolução são escolhidos para que   #
    # **a largura e a altura da entrada sejam preservada**.                #
    ########################################################################
    C, H, W = dims_entrada
    # Camada de convolução
    padding = (tamanho_filtro - 1) // 2
    self.conv1 = torch.nn.Conv2d(
      in_channels = C, 
      out_channels = num_filtros,
      kernel_size = tamanho_filtro, 
      padding = padding)
        
    # Inicializando pesos e vieses da camada convolucional
    torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=escala_peso)
    torch.nn.init.zeros_(self.conv1.bias)
        
    # Dimensões após o max pooling
    H_pool = H // 2
    W_pool = W // 2
        
    # Camada linear oculta
    self.fc1 = torch.nn.Linear(num_filtros * H_pool * W_pool, dim_oculta)
    torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=escala_peso)
    torch.nn.init.zeros_(self.fc1.bias)
        
    # Camada linear de saída
    self.fc2 = torch.nn.Linear(dim_oculta, num_classes)
    torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=escala_peso)
    torch.nn.init.zeros_(self.fc2.bias)
    ########################################################################
    #                           FIM DO SEU CODIGO                          #
    ########################################################################

    self.reset_parameters()

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    # Calcule o passo para frente
    pontuacoes = None
    ########################################################################
    # TODO: Implemente o passo para frente para rede convolucional de três #
    # camadas, calculando as pontuações de classe para X e armazenando-as  #
    # na variável pontuações.                                              #
    ########################################################################
    
    # Aplica a convolução
    X_conv = self.conv1(X)  

    # Aplica a função de ativação ReLU
    X_conv_relu = torch.nn.functional.relu(X_conv)

    # Max Pooling 2x2
    X_pool = torch.nn.functional.max_pool2d(
      X_conv_relu, kernel_size=2, stride=2)
    
    # Achatar (flatten) a saída da camada de pooling
    X_pool_flat = torch.flatten(X_pool, start_dim=1)

    # Camada linear oculta + ReLU
    X_lin = self.fc1(X_pool_flat)
    X_lin_relu = torch.nn.functional.relu(X_lin)

    # Camada linear final para calcular as pontuações de classe
    pontuacoes = self.fc2(X_lin_relu)

    ########################################################################
    #                           FIM DO SEU CODIGO                          #
    ########################################################################    
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for param in self.parameters():
      if isinstance(param, torch.nn.Conv2d) or isinstance(param, torch.nn.Linear):
        torch.nn.init.normal_(param.weight, std=self.escala_peso)
        torch.nn.init.zeros_(param.bias)


class RedeConvProfunda(torch.nn.Module):
  """
  Uma rede neural convolucional com um número arbitrário de camadas de 
  convolução no estilo da rede VGG. Todas as camadas de convolução usarão 
  filtro de tamanho 3 e preenchimento de 1 para preservar o tamanho do mapa 
  de ativação, e todas as camadas de agrupamento serão camadas de agrupamento 
  por máximo com campos receptivos de 2x2 e um passo de 2 para reduzir pela 
  metade o tamanho do mapa de ativação.

  A rede terá a seguinte arquitetura:

  {conv - [normlote?] - relu - [agrup?]} x (L - 1) - linear

  Cada estrutura {...} é uma "camada macro" que consiste em uma camada de 
  convolução, uma camada de normalização de lote opcional, uma não linearidade 
  ReLU e uma camada de agrupamento opcional. Depois de L-1 dessas macrocamadas, 
  uma única camada totalmente conectada é usada para prever pontuações de classe.

  A rede opera em minilotes de dados que possuem shape (N, C, H, W) consistindo 
  de N imagens, cada uma com altura H e largura W e com C canais de entrada.
  """
  def __init__(self, dims_entrada=(3, 32, 32),
               num_filtros=[8, 8, 8, 8, 8],
               agrups_max=[0, 1, 2, 3, 4],
               normlote=False,
               num_classes=10, escala_peso=1e-3):
    """
    Inicializa uma nova rede.

    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Lista de comprimento (L - 1) contendo o número de filtros
      de convolução para usar em cada macrocamada.
    - agrups_max: Lista de inteiros contendo os índices (começando em zero) das 
      macrocamadas que devem ter agrupamento por máximo.
    - normlote: Booleano dizendo se normalização do lote deve ou não ser 
      incluída em cada macrocamada.
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos, ou a string "kaiming" para usar a inicialização Kaiming.
    """
    super().__init__()

    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)
    
    self.num_camadas = len(num_filtros)+1
    self.escala_peso = escala_peso
    self.agrups_max = agrups_max
    self.normlote = normlote

    #######################################################################
    # TODO: Inicialize os parâmetros para o RedeConvProfunda.             #
    #                                                                     #
    # Pesos para camadas de convolução e totalmente conectadas devem ser  #
    # inicializados de acordo com escala_peso. Os vieses devem ser        #
    # inicializados com zero. Parâmetros de escala (gamma) e deslocamento #
    # (beta) de camadas de normalização de lote devem ser inicializados   #
    # com um e zero, respectivamente.                                     #
    #######################################################################
    self.camadas = torch.nn.ModuleList()
    self.flag = False # flag que verifica se indices de agrup_max foram adaptados
    C, H, W = dims_entrada
    tamanho_filtro = 3

    H_pool = H
    W_pool = W

    in_channels = C

    is_kaiming = self.escala_peso == 'kaiming'

    for i, num_filtros_camada in enumerate(num_filtros):
      # Camadas de convolução
      padding = (tamanho_filtro - 1) // 2
      cam_conv = torch.nn.Conv2d(
        in_channels = in_channels, 
        out_channels = num_filtros_camada,
        kernel_size = tamanho_filtro, 
        padding = padding)
      
      # Inicializando pesos e vieses da camada convolucional
      if is_kaiming is True:
        torch.nn.init.kaiming_normal_(cam_conv.weight, mode='fan_in', nonlinearity='relu')
      else:
        torch.nn.init.normal_(cam_conv.weight, mean=0.0, std=escala_peso)
      torch.nn.init.zeros_(cam_conv.bias)

      self.camadas.append(cam_conv)
      in_channels = num_filtros_camada

      if normlote is True:
        # Adiciona a camada de normalização de lote
        norm_lote = torch.nn.BatchNorm2d(num_filtros_camada)
        torch.nn.init.zeros_(norm_lote.bias)
        self.camadas.append(norm_lote)

      if i in agrups_max:
        H_pool = H_pool // 2
        W_pool = W_pool // 2
    tam_entrada = num_filtros[-1] * H_pool * W_pool
        
    # Camada linear de saída
    camada_saida = torch.nn.Linear(tam_entrada, num_classes)
    if is_kaiming is True:
      torch.nn.init.kaiming_normal_(camada_saida.weight, mode='fan_in', nonlinearity='relu')
    else:
      torch.nn.init.normal_(camada_saida.weight, mean=0.0, std=escala_peso)
    torch.nn.init.zeros_(camada_saida.bias)
    self.camadas.append(camada_saida)

    #######################################################################
    #                           FIM DO SEU CODIGO                         #
    #######################################################################

    # Verifique se obtivemos o número correto de parâmetros
    if not self.normlote:
      params_por_camada_macro = 2  # peso e viés
    else:
      params_por_camada_macro = 4  # peso, viés, escala, deslocamento
    num_params = params_por_camada_macro * len(num_filtros) + 2
    msg = 'self.parameters() tem o número errado de ' \
          'elementos. Obteve %d; esperava %d'
    msg = msg % (len(list(self.parameters())), num_params)
    assert len(list(self.parameters())) == num_params, msg
    
    self.reset_parameters()

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    # Calcule o passo para frente
    pontuacoes = None
    ##################################################################
    # TODO: Implemente o passo para frente para a RedeConvProfunda,  #
    # calculando as pontuações de classe para X e armazenando-as na  #
    # variável pontuacoes.                                           #
    ##################################################################
    out = X
    if self.normlote is True and self.flag is False:
      self.agrups_max = [x * 2 for x in self.agrups_max]
      self.flag = True

    for i, camada in enumerate(self.camadas):
      #print(f"Shape out na camada {i}: {out.shape}")
      if isinstance(camada, torch.nn.Conv2d):
        out = camada(out)
        if self.normlote and i < len(self.camadas) - 1 and isinstance(self.camadas[i+1], torch.nn.BatchNorm2d):
          out = self.camadas[i+1](out)  # Aplicar normalização de lote
        out = torch.nn.functional.relu(out)
        if i in self.agrups_max:
          out = torch.nn.functional.max_pool2d(
            out, kernel_size=2, stride=2)
    out_flat = out.view(out.size(0), -1)
    pontuacoes = self.camadas[-1](out_flat)

    ##################################################################
    #                        FIM DO SEU CODIGO                       #
    ##################################################################    
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for nome, camada in self.named_modules():
      if isinstance(camada, torch.nn.Conv2d) or isinstance(camada, torch.nn.Linear):
        if isinstance(self.escala_peso, str) and self.escala_peso == "kaiming":
          ############################################################################
          # TODO: Inicializa os pesos das camadas de convolução e lineares usando o  #
          # método de Kaiming.                                                       #
          ############################################################################
          # Substitua a comando "pass" pelo seu código
          torch.nn.init.kaiming_normal_(camada.weight, mode='fan_in', nonlinearity='relu')
          ############################################################################
          #                             FIM DO SEU CODIGO                            #
          ############################################################################          
        else:
          torch.nn.init.normal_(camada.weight, std=self.escala_peso)
        torch.nn.init.zeros_(camada.bias)


def encontrar_parametros_sobreajuste():
  taxa_aprendizagem = 1e-5  # Tente com este!
  escala_peso = 2e-3   # Tente com este!
  ###############################################################
  # TODO: Altere escala_peso e taxa_aprendizagem para que seu   #
  # modelo atinja 100% de acurácia de treinamento em 30 épocas. #
  ############################################################### 
  taxa_aprendizagem = 5e-4
  escala_peso = 2e-1
  ##############################################################
  #                        FIM DO SEU CODIGO                    #
  ###############################################################
  return escala_peso, taxa_aprendizagem


def criar_instancia_solucionador_convolucional(dic_dados, device):
  modelo = None
  solucionador = None
  #########################################################
  # TODO: Treine a melhor RedeConvProfunda possível na    #
  # CIFAR-10 em 60 segundos.                              #
  #########################################################
  num_epocas = 1
  taxa_aprendizagem = 0.001291464786063588
  reg = 0.00022331923735628278
  escala_peso = 'kaiming'
  tam_lote = 150
  num_filtros = ([128] * 2) + ([256] * 1) + ([512] * 2) + ([256] * 1)
  agr_idx = [0, 2, 3, 4, 5]
  normlote = True
  dims_entrada = dic_dados['X_treino'].shape[1:]
  modelo = RedeConvProfunda(
    dims_entrada=dims_entrada,
    num_classes=10,
    num_filtros= num_filtros,
    agrups_max= agr_idx,
    normlote = normlote,
    escala_peso=escala_peso)
  dados_de_treinamento = ConjuntoDeDados(dic_dados['X_treino'], dic_dados['y_treino'])
  dados_de_validacao = ConjuntoDeDados(dic_dados['X_val'], dic_dados['y_val'])
  dados = {
    'treinamento': dados_de_treinamento,
    'validacao': dados_de_validacao,
  }
  funcao_de_perda = perda_softmax
  otimizador = torch.optim.Adam(
    modelo.parameters(), 
    lr=taxa_aprendizagem, 
    weight_decay=reg)
  solucionador = Solucionador(
    modelo=modelo,
    funcao_de_perda=funcao_de_perda,
    otimizador=otimizador,
    dados=dados,
    num_epocas=num_epocas,
    tamanho_lote = tam_lote,
    device=device,
  )
  #########################################################
  #                  FIM DO SEU CODIGO                    #
  #########################################################
  return solucionador
