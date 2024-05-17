import torch


class OpinionDynamics:
    def __init__(self, alpha, eps, X, N, h):
        self.__alpha = alpha
        self.__eps = eps
        self.__X = X
        self.__N = N
        self.__h = h

    def __phi(self, x, y):
        dist = torch.minimum(torch.abs(x - y), torch.tensor(self.__eps, dtype=torch.float32))
        res = torch.exp(1 - self.__eps**2 / (1e-6 + self.__eps**2 - dist**2))
        return res

    def __r(self, x):
        interaction_sum = torch.sum(torch.stack([self.__phi(x, y) for y in self.__X]))
        return 1 / (interaction_sum / self.__N)

    def __b(self, x, m):
        interaction = torch.sum(torch.stack([self.__phi(x, y) for y in self.__X]) * self.__X * m) / self.__N
        return -self.__alpha * x + self.__alpha * self.__r(x) * interaction

    def __g(self, x, q1, q2, m):
        p1 = torch.minimum(q1, torch.tensor(0.0))
        p2 = torch.maximum(q2, torch.tensor(0.0))
        b = self.__b(x, m)
        b1 = torch.minimum(b, torch.tensor(0.0))
        b2 = torch.maximum(b, torch.tensor(0.0))
        return (p1 ** 2) / 2 + (p2 ** 2) / 2 + b1 * q1 + b2 * q2

    def hamilton(self, x, m, grad_u):
        q1 = torch.roll(grad_u, -1) - grad_u
        q2 = grad_u - torch.roll(grad_u, 1)
        Hamiltonian = torch.stack([self.__g(xi, q1i, q2i, m) for xi, q1i, q2i in zip(self.__X, q1, q2)], dim=0)
        return Hamiltonian

