
import numpy as np
from core import Variable


def _dot_var(v, verbose=False):
  dot_var = '{} [label="{}", color=orange, style=filled]\n'
  
  name = '' if v.name is None else v.name
  if verbose and v.data is not None:
    if v.name is not None:
      name += ': '
    name += str(v.shape) + ' ' + str(v.dtype)
  return dot_var.format(id(v), name)

def _dot_func(f):
  dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
  txt = dot_func.format(id(f), f.__class__.__name__)
  
  dot_edge = '{} -> {}\n'
  for x in f.inputs:
    txt += dot_edge.format(id(x), id(f))
  for y in f.outputs:
    txt += dot_edge.format(id(f), id(y())) # y는 약한 참조
  return txt

def get_dot_graph(output, verbose=True):
  txt = ''
  funcs = []
  seen_set = set()
  
  def add_func(f):
    if f not in seen_set:
      funcs.append(f)
      # funcs.sort(key=lambda x: x.generation)
      seen_set.add(f)
      
  add_func(output.creator)
  txt += _dot_var(output, verbose)
  while funcs:
    func = funcs.pop()
    txt += _dot_func(func)
    for x in func.inputs:
      txt += _dot_var(x, verbose)
      
      if x.creator is not None:
        add_func(x.creator)
  return 'digraph g {\n' + txt + '}'

x = Variable(np.random.randn(2,3))
x.name = 'x'
print(_dot_var(x))
print(_dot_var(x, verbose=True))

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1
txt = _dot_func(y.creator)
print(txt)