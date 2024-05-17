import torch

from src.utils.partials import partial_t, partial_x, laplacian
from src.utils.conditions import initial_m


def fp_loss(m_net, t, x, nu, u_net, H):
    dt_m = partial_t(m_net, t, x)
    lap_m = laplacian(m_net, t, x)
    grad_u = partial_x(u_net, t, x)
    m = m_net(t, x)
    val = H(x, m, grad_u)
    grad_H = torch.autograd.grad(val.sum(), grad_u, create_graph=True)[0]
    div_term = torch.autograd.grad((m * grad_H).sum(), x, create_graph=True)[0]
    loss = torch.mean((dt_m - nu * lap_m - div_term) ** 2)
    return loss


def fp_cond_loss(m_net, x):
    rho_init = m_net(torch.zeros_like(x), x)
    loss = torch.mean((rho_init - initial_m(x)) ** 2)
    return loss
