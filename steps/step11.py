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

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

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
        x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)

class Add(Function):
  def forward(self, x0, x1):
    y = x0 + x1
    return y

def add(x0, x1):
  return Add()(x0, x1) 
  
# xs = [Variable(np.array(2)), Variable(np.array(3))] #리스트로 준비
# f = Add()
# ys = f(xs)
# y = ys[0]

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1) # Add 클래스 생성 과정이 감춰짐
print(y.data)