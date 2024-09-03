import os
import zipfile

_T2_FILES = [
    'redes_totalmente_conectadas.py',
    'redes_totalmente_conectadas.ipynb',
    'redes_convolucionais.py',
    'redes_convolucionais.ipynb',
    'melhor_sobreajuste_rede_cinco_camadas.pt',
    'melhor_rede_duas_camadas.pt',
    'um_minuto_rede_convolucional.pt',
    'sobreajuste_rede_convolucional.pt',
]


def make_t2_submission(assignment_path, uniquename=None, uniqueid=None):
    _make_submission(assignment_path, _T2_FILES, 'T2', uniquename, uniqueid)


def _make_submission(
    assignment_path,
    file_list,
    assignment_no,
    uniquename=None,
    uniqueid=None):
  if uniquename is None or uniqueid is None:
    uniquename, uniqueid = _get_user_info()
  zip_path = '{}_{}_{}.zip'.format(uniquename, uniqueid, assignment_no)
  zip_path = os.path.join(assignment_path, zip_path)
  print('Gravando arquivo zip em: ', zip_path)
  with zipfile.ZipFile(zip_path, 'w') as zf:
    for filename in file_list:
      in_path = os.path.join(assignment_path, filename)
      if not os.path.isfile(in_path):
        raise ValueError('Não foi possível encontrar o arquivo "%s"' % filename)
      zf.write(in_path, filename)


def _get_user_info():
  if uniquename is None:
    uniquename = input('Digite seu primeiro nome (por exemplo, jurandy): ')
  if uniqueid is None:
    uniqueid = input('Digite seu RA (por exemplo, 12345678):')
  return uniquename, uniqueid
