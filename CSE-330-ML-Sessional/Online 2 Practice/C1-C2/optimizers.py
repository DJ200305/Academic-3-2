class Adam:
    def __init__(self,params,lr,row1=0.9,row2=0.999,delta=1e-8):
        self.params = list(params)
        self.lr = lr
        self.row1 = row1
        self.row2 = row2
        self.delta = delta
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.r = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0
    def step(self):
        self.t += 1
        for i,param in enumerate(self.params):
            if param.grad is None:
               continue
            self.v[i] = self.row1*self.v[i] + (1-self.row1)*param.grad
            self.r[i] = self.row2*self.r[i] + (1-self.row2)*param.grad*param.grad
            self.v[i] = self.v[i] / (1-self.row1**self.t)
            self.r[i] = self.r[i] / (1-self.row2**self.t)

            del_theta = -self.lr*(self.v[i]) /(torch.sqrt(self.r[i]) + self.delta)
            param.data += del_theta
    def zero_grad(self):
        for param in self.params:
          if param.grad is not None:
            param.grad.zero_() 

class SGD:
    def __init__(self, params, lr):
       self.params = list(params)
       self.lr = lr
    def step(self):
       for param in self.params:
          # w <-w-lr * grad
          param.data-= self.lr * param.grad
    def zero_grad(self):

       for param in self.params:
          if param.grad is not None:
             param.grad.zero_()

class SGD_Momentum:
    def __init__(self, params,momentum, lr):
       self.params = list(params)
       self.momentum = momentum
       self.lr = lr
       self.v = [torch.zeros_like(p.data) for p in self.params]
    def step(self):
       for i,param in enumerate((self.params)):
           if param.grad is None:
              continue
           self.v[i] = self.momentum*self.v[i]-self.lr*param.grad
           param.data += self.v[i]
    def zero_grad(self):

       for param in self.params:
          if param.grad is not None:
             param.grad.zero_()                         