import torch


def partial_t(net, t, x):
    t.requires_grad = True
    val = net(t, x)
    grad = torch.autograd.grad(val.sum(), t, create_graph=True)[0]
    return grad


def partial_x(net, t, x):
    x.requires_grad = True
    val = net(t, x)
    grad = torch.autograd.grad(val.sum(), x, create_graph=True)[0]
    return grad


def laplacian(net, t, x):
    x.requires_grad = True
    val = net(t, x)
    grad_x = torch.autograd.grad(val.sum(), x, create_graph=True)[0]
    grad_grad_x = torch.autograd.grad(grad_x.sum(), x, create_graph=True)[0]
    return grad_grad_x
