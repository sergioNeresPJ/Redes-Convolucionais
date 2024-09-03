import random
import torch
import pi

""" Utilitários para computação e verificação de gradientes. """


def verificar_gradiente_esparso(f, x, gradiente_analitico, num_verificacoes=10, h=1e-7):
  """
  Função útil para realizar verificação de gradiente numérica. Usamos a 
  fórmula da diferença centrada para calcular uma derivada numérica:
  
  f'(x) =~ (f(x + h) - f(x - h)) / (2h)

  Em vez de calcular um gradiente numérico completo, amostramos esparsamente 
  algumas dimensões ao longo das quais computar derivadas numéricas.

  Entradas:
  - f: Uma função que recebe a um tensor do torch e retorna um escalar do torch 
  - x: Um tensor do torch contendo o ponto no qual avaliar o gradiente numérico
  - gradiente_analitico: Um tensor do torch contendo o gradiente analítico de f em x
  - num_verificacoes: O número de dimensões ao longo das quais verificar
  - h: Tamanho do passo para calcular derivadas numéricas
  """
  # fixa semente aleatória ems 0
  pi.redefinir_semente(0)
  for i in range(num_verificacoes):
    
    ix = tuple([random.randrange(m) for m in x.shape])
    
    valori = x[ix].item()
    x[ix] = valori + h # incrementa de h
    fxph = f(x).item() # avalia f(x + h)
    x[ix] = valori - h # decrementa de h
    fxmh = f(x).item() # avalia f(x - h)
    x[ix] = valori     # restaura

    grad_numerico   = (fxph - fxmh) / (2 * h)
    grad_analitico  = gradiente_analitico[ix]
    erro_rel_acima  = abs(grad_numerico - grad_analitico)
    erro_rel_abaixo = (abs(grad_numerico) + abs(grad_analitico) + 1e-12)
    erro_rel = erro_rel_acima / erro_rel_abaixo
    msg = 'numérico: %f analítico: %f, erro relativo: %e'
    print(msg % (grad_numerico, grad_analitico, erro_rel))


def calcular_gradiente_numerico(f, x, h=1e-7):
  """ 
  Calcula o gradiente numérico de f em x usando uma aproximação de 
  diferenças finitas. Usamos a diferença centrada:

  df    f(x + h) - f(x - h)
  -- ~= -------------------
  dx           2 * h
  
  Entrada:
  - f: Uma função que recebe a um tensor do torch e retorna um escalar do torch 
  - x: Um tensor do torch contendo o ponto no qual calcular o gradiente numérico
  - h: epsilon usado no cálculo da diferença finita

  Retorno:
  - grad: Um tensor de mesmo shape que x contendo o gradiente de f em x
  """ 
  achatado_x = x.contiguous().flatten()
  grad = torch.zeros_like(x)
  achatado_grad = grad.flatten()

  # itera sobre todos os índices em x
  for i in range(achatado_x.shape[0]):
    valori = achatado_x[i].item() # Salva o valor original
    achatado_x[i] = valori + h    # Incrementa de h
    fxph = f(x).item()            # Avalia f(x + h)
    achatado_x[i] = valori - h    # Decrementa de h
    fxmh = f(x).item()            # Avalia f(x - h)
    achatado_x[i] = valori        # Restaura o valor original

    # calcula a derivada parcial com a fórmula centrada
    achatado_grad[i] = (fxph - fxmh) / (2 * h)

  # Observe que como achatado_grad era apenas uma referência a grad, 
  # podemos apenas retornar o objeto no shape de x retornando grad
  return grad


def erro_rel(x, y, eps=1e-10):
  """
  Calcula o erro relativo entre um par de tensores x e y,
  que é definido como:

                         max_i |x_i - y_i]|
  erro_rel(x, y) = -------------------------------
                   max_i |x_i| + max_i |y_i| + eps

  Entrada:
  - x, y: Tensores de mesmo shape
  - eps: Constante positiva pequena para estabilidade numérica

  Retorno:
  - erro_rel: Escalar indicando o erro relativo entre x e y
  """
  """ retorna o erro relativo entre x e y """
  acima = (x - y).abs().max().item()
  abaixo = (x.abs() + y.abs()).clamp(min=eps).max().item()
  return acima / abaixo
