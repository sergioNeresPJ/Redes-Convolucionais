import random
import torch

"""
Utilitários gerais para ajudar na implementação
"""

def redefinir_semente(numero):
  """
  Redefine a semente aleatória para o número específico

  Entrada:
  - numero: Um número semente para usar
  """
  random.seed(numero)
  torch.manual_seed(numero)
  return
