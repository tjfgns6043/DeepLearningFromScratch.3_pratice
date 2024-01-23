from step02 import Function
import numpy as np

class Exp(Function):
  def forward(self, x):
    return np.exp(x)