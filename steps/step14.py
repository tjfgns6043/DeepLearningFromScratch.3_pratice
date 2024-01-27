import numpy as np



class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def cleargrad(self):
       self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys) 
            if not isinstance(gxs, tuple): 
              gxs = (gxs,)
              
            for x, gx in zip(f.inputs, gxs): 
                if x.grad is None:
                  x.grad = gx
                else:
                   x.grad = x.grad + gx

                if x.creator is not None:
                  funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
          ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        

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

x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)


# 첫 번째 계산
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)


# 두 번쨰 계산(같은 x를 사용하여 다른 계산을 수행)
x.cleargrad() # 미분값 초기화
y = add(add(x, x), x)
y.backward()
print(x.grad)