import numpy as np
import weakref
import contextlib

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 함수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1)
    
    def cleargrad(self):
       self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
           if f not in seen_set:
              funcs.append(f)
              seen_set.add(f)
              funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys) 
            if not isinstance(gxs, tuple): 
              gxs = (gxs,)
              
            for x, gx in zip(f.inputs, gxs): 
                if x.grad is None:
                  x.grad = gx
                else:
                   x.grad = x.grad + gx

                if x.creator is not None:
                  add_func(x.creator)
            if not retain_grad:
               for y in f.outputs:
                  y().grad = None # y는 약한 참조


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): 
          ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 해당 부분
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] # 해당 부분
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data # 수정 전:  x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)

class Add(Function):
  def forward(self, x0, x1):
    y = x0 + x1
    return y
  
  def backward(self, gy):
    return gy, gy

def add(x0, x1):
  return Add()(x0, x1) 

class Config:
   enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
       yield
    finally:
       setattr(Config, name, old_value)

def no_grad():
   return using_config('enable_backprop', False)

with no_grad():
   x = Variable(np.array(2.0))
   y = square(x)
