import pickle
import time

import torch


class Solucionador(object):
  """
  Um Solucionador encapsula toda a lógica necessária para treinar modelos 
  de classificação. O Solucionador executa a descida de gradiente estocástica 
  usando diferentes otimizadores.
  O Solucionador aceita dados de treinamento e validação e rótulos para 
  que possa verificar periodicamente a acurácia de classificação nos dados 
  de treinamento e de validação para ficar atento ao sobreajuste.
  Para treinar um modelo, você primeiro construirá uma instância do Solucionador, 
  passando o modelo, função de perda, otimizador, escalonador, conjunto de dados 
  e várias opções (taxa de aprendizagem, tamanho do lote, etc) para o construtor. 
  Em seguida, você chamará o método treinar() para executar o processo de 
  otimização e treinar o modelo.
  Depois que o método treinar() retornar, seu modelo conterá os parâmetros
  que tiveram o melhor desempenho no conjunto de validação ao longo do treinamento.
  Além disso, a variável de instância solucionador.historico_perda conterá uma 
  lista de todas as perdas encontradas durante o treinamento e as variáveis de 
  instância solucionador.historico_acc_treinamento e solucionador.historico_acc_validacao
  serão listas de acurácia do modelo nos conjuntos de treinamento e validação 
  a cada época. O uso de exemplo pode ser semelhante a este:
  dados = {
    'treinamento': # conjunto de dados e rótulos de treinamento
    'validacao': # conjunto de dados e rótulos de validação
  }
  modelo = MeuModelo()
  funcao_de_perda = MinhaFuncaoDePerda()
  otimizador = MeuOtimizador(modelo.parameters(), lr=taxa_aprendizagem, weight_decay=reg)
  escalonador = MeuEscalonador(otimizador, decaimento_taxa_aprendizagem)
  solucionador = Solucionador (
          modelo, 
          funcao_de_perda, 
          otimizador, 
          dados,
          escalonador, 
          num_epocas=10, 
          tamanho_lote=100,
          imprime_cada=100,
          device='cuda')
  solucionador.treinar()
  """

  def __init__(self, modelo, funcao_de_perda, otimizador, dados, **kwargs):
    """
    Constrói uma nova instância do Solucionador.
    Argumentos necessários:
    - modelo: um modelo do PyTorch
    - funcao_de_perda: uma função de perda do PyTorch
    - otimizador: um otimizador do PyTorch
    - dados: um dicionário de dados de treinamento e validação contendo:
      'treinamento': conjunto de dados do PyTorch contendo dados e rótulos de treinamento
      'validacao': conjunto de dados do PyTorch contendo dados e rótulos de validação
    Argumentos opcionais:
    - escalonador: Se não for None, usa um escalonador do PyTorch para reduzir 
      a taxa de aprendizado.
    - tamanho_lote: tamanho dos mini-lotes usados para calcular a perda durante 
      o treinamento.
    - num_epocas: o número de épocas a serem executadas durante o treinamento.
    - imprime_cada: Inteiro; as perdas de treinamento serão impressas a cada
      imprime_cada iterações.
    - imprime_acc_cada: imprimiremos a acurácia a cada imprime_acc_cada épocas.
    - verbose: Boolean; se definido como falso, nenhuma saída será impressa
      durante o treinamento.
    - num_amostras_treinamento: número de amostras de treinamento usadas para verificar 
      a acurácia de treinamento; o padrão é 1000; defina como None para usar todo o 
      conjunto de treinamento.
    - num_amostras_validacao: Número de amostras de validação a serem usadas para verificar 
      a acurácia de validação; o padrão é None, que usa todo o conjunto de validação.
    - nome_ponto_checagem: Se não for None, salve os pontos de checagem do modelo 
      aqui a cada época.
    """
    self.treinamento = dados["treinamento"]
    self.validacao = dados["validacao"]

    self.modelo = modelo
    self.funcao_de_perda = funcao_de_perda
    self.otimizador = otimizador
    self.otimizador_state_dict = self.otimizador.state_dict()
        
    # Descompacte argumentos de palavras-chave
    self.escalonador = kwargs.pop("escalonador", None)
    if self.escalonador is not None:
      self.escalonador_state_dict = self.escalonador.state_dict()
    self.tamanho_lote = kwargs.pop("tamanho_lote", 100)
    self.num_epocas = kwargs.pop("num_epocas", 10)
    self.num_amostras_treinamento = kwargs.pop("num_amostras_treinamento", 1000)
    self.num_amostras_validacao = kwargs.pop("num_amostras_validacao", None)

    self.device = kwargs.pop("device", "cpu")

    self.nome_ponto_checagem = kwargs.pop("nome_ponto_checagem", None)
    self.imprime_cada = kwargs.pop("imprime_cada", 10)
    self.imprime_acc_cada = kwargs.pop("imprime_acc_cada", 1)
    self.verbose = kwargs.pop("verbose", True)

    # Lança um erro se houver argumentos de palavra-chave extras
    if len(kwargs) > 0:
      extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
      raise ValueError("Argumentos desconhecidos %s" % extra)

    self._redefinir()

  def _redefinir(self):
    """
    Configura algumas variáveis de depuração do processo de otimização. 
    Não chame isso manualmente.
    """
    # Configure algumas variáveis de depuração do processo de otimizaçã
    self.epoca = 0
    self.melhor_acc_validacao = 0
    self.melhores_parametros = {}
    self.historico_perda = []
    self.historico_acc_treinamento = []
    self.historico_acc_validacao = []
    self.modelo.to(self.device)

    # Faz uma cópia detalhada dos parâmetros do otimizador e escalonador
    self.otimizador.load_state_dict(self.otimizador_state_dict)
    if self.escalonador is not None:
      self.escalonador.load_state_dict(self.escalonador_state_dict)

  def _passo(self, iterador, carregador):
    """
    Executa um passo de treinamento. Isso é chamado de train() e não deve
    ser chamado manualmente.
    """
    # Cria um minilote de dados de treinamento
    try:
      X_lote, y_lote = next(iterador)
    except StopIteration:
      iterador = iter(carregador)
      X_lote, y_lote = next(iterador)

    # Configura o modelo para o modo de treinamento
    self.modelo.train()

    pontuacoes = self.modelo(X_lote.to(self.device))                  # calcula a saída do modelo
    perda = self.funcao_de_perda(pontuacoes, y_lote.to(self.device))  # calcula a perda

    self.otimizador.zero_grad()  # limpa os gradientes anteriores
    perda.backward()             # calcula o gradiente de todos os parâmetros com relação a perda

    self.otimizador.step()       # atualiza o modelo usando o gradiente calculado 

    # Retorna a perda
    return perda

  def _salvar_ponto_de_checagem(self):
    if self.nome_ponto_checagem is None:
      return
    ponto_checagem = {
      "modelo": self.modelo.state_dict(),
      "otimizador": self.otimizador.state_dict(),
      "escalonador": self.escalonador.state_dict() if self.escalonador is not None else None,
      "tamanho_lote": self.tamanho_lote,
      "num_amostras_treinamento": self.num_amostras_treinamento,
      "num_amostras_validacao": self.num_amostras_validacao,
      "epoca": self.epoca,
      "historico_perda": self.historico_perda,
      "historico_acc_treinamento": self.historico_acc_treinamento,
      "historico_acc_validacao": self.historico_acc_validacao,
    }
    nome_do_arquivo = "%s_epoca_%d.pkl" % (self.checkpoint_name, self.epoch)
    if self.verbose:
      print('Salvando ponto de checagem em "%s"' % nome_do_arquivo)
    with open(nome_do_arquivo, "wb") as f:
      pickle.dump(ponto_checagem, f)

  def verificar_acuracia(self, conjunto_de_dados, num_amostras=None, tamanho_lote=100):
    """
    Verifica a acurácia do modelo nos dados fornecidos.
     Entrada:
     - conjunto: Conjunto de dados do PyTorch contendo dados e rótulos
     - num_amostras: se não for None, subamostra os dados e apenas avalia 
       o modelo em num_amostras pontos de dados.
     - tamanho_lote: divide X e y em lotes deste tamanho para evitar o uso
       muita memória.
     Retorna:
     - acc: escalar fornecendo a fração de instâncias que foram corretamente
       classificados pelo modelo.
    """

    # Talvez subamostra os dados
    N = len(conjunto_de_dados)
    if num_amostras is not None and N > num_amostras:
      indices = torch.randperm(N, device=self.device)[:num_amostras]
      N = num_amostras
      amostrador = torch.utils.data.SubsetRandomSampler(indices)
    else:
      amostrador = None
    carregador = torch.utils.data.DataLoader(dataset=conjunto_de_dados, batch_size=tamanho_lote, shuffle=False, sampler=amostrador)

    # Configura o modelo para o modo de avaliação
    self.modelo.eval()

    # Calcule predições em lotes
    acc = 0
    for i, (dados, rotulos) in enumerate(carregador):
      pontuacoes = self.modelo(dados)
      predicoes = torch.argmax(pontuacoes, dim=1)
      acc += (predicoes == rotulos).to(torch.float).mean()

    return acc.item() / len(carregador)

  def treinar(self, limite_de_tempo=None, retornar_melhores_parametros=True):
    """
    Executa o processo de otimização para treinar o modelo.
    """
    # Define um carregador de lotes para o conjuntos de dados
    carregador = torch.utils.data.DataLoader(dataset=self.treinamento, batch_size=self.tamanho_lote, shuffle=True)
    iterador = iter(carregador)

    iteracoes_por_epoca = len(carregador)
    num_iteracoes = self.num_epocas * iteracoes_por_epoca
    tempo_anterior = tempo_inicio = time.time()

    for t in range(num_iteracoes):
            
      tempo_atual = time.time()
      if (limite_de_tempo is not None) and (t > 0):
        tempo_seguinte = tempo_atual - tempo_anterior
        if tempo_atual - tempo_inicio + tempo_seguinte > limite_de_tempo:
          print(
              "(Tempo %.2f segundos; Iteração %d / %d) perda: %f"
              % (
                  tempo_atual - tempo_inicio,
                  t,
                  num_iteracoes,
                  self.historico_perda[-1],
              )
          )
          print("Fim do treinamento; a próxima iteração excederá o limite de tempo.")
          break
      tempo_anterior = tempo_atual

      # Executa um passo do treinamento e retorna a perda correspondente
      perda = self._passo(iterador, carregador)
      self.historico_perda.append(perda.item())

      # Talvez imprime a perda de treinamento
      if self.verbose and t % self.imprime_cada == 0:
        print(
            "(Tempo %.2f segundos; Iteração %d / %d) perda: %f"
            % (
                time.time() - tempo_inicio,
                t + 1,
                num_iteracoes,
                self.historico_perda[-1],
            )
        )

      # No final de cada época, incrementa o contador de épocas e diminui
      # a taxa de aprendizagem.
      final_epoca = (t + 1) % iteracoes_por_epoca == 0
      if final_epoca:
        self.epoca += 1
        if self.escalonador is not None:
          self.escalonador.step()

      # Verifica a acurácia de treinamento e validação na primeira iteração, 
      # na última, e no final de cada época.
      with torch.no_grad():
        primeira_it = t == 0
        ultima_it = t == num_iteracoes - 1
        if primeira_it or ultima_it or final_epoca:              
          acc_treinamento = self.verificar_acuracia(
            self.treinamento, num_amostras=self.num_amostras_treinamento
          )
          acc_validacao = self.verificar_acuracia(
            self.validacao, num_amostras=self.num_amostras_validacao
          )
          self.historico_acc_treinamento.append(acc_treinamento)
          self.historico_acc_validacao.append(acc_validacao)
          self._salvar_ponto_de_checagem()

          if self.verbose and self.epoca % self.imprime_acc_cada == 0:
            print(
                "(Época %d / %d) acurácia de treinamento: %.2f%%; acurácia de validação: %.2f%%"
                % (self.epoca, self.num_epocas, float(100.0 * acc_treinamento), float(100.0 * acc_validacao))
            )

          # Acompanha o melhor modelo
          if acc_validacao > self.melhor_acc_validacao:
            self.melhor_acc_validacao = acc_validacao
            self.melhores_parametros = self.modelo.state_dict()

    # No final do treinamento, troque os melhores parâmetros no modelo
    if retornar_melhores_parametros:
      self.modelo.load_state_dict(self.melhores_parametros)
        
    return {
      'historico_perda': self.historico_perda,
      'historico_acc_treinamento': self.historico_acc_treinamento,
      'historico_acc_validacao': self.historico_acc_validacao,
    }
