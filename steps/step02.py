from step01 import Variable

class Function:
  def __call__(self, input):
    x = input.data # 데이터를 꺼낸다.
    y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다.
    output = Variable(y) # Variable 형태로 되돌린다.
    return output
  def forward(self, x):
    raise NotImplemetedError()


class Square(Function):
  def forward(self, x):
    return x ** 2
  