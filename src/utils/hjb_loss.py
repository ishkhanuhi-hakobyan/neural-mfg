import torch

from src.utils.partials import partial_t, partial_x, laplacian
from src.utils.conditions import terminal_u


def hjb_loss(u_net, m_net, H, t, x, nu):
    dt_u = partial_t(u_net, t, x)
    lap_u = laplacian(u_net, t, x)
    grad_u = partial_x(u_net, t, x)
    m = m_net(t, x)
    H_val = H(x, m, grad_u)
    loss = torch.mean((dt_u + nu * lap_u - H_val) ** 2)
    return loss


def hjb_cond_loss(u_net, m_net, T, x):
    t = torch.full_like(x, T)
    u_T = u_net(t, x)
    m_T = m_net(t, x)
    g_val = terminal_u(x, m_T)
    loss = torch.mean((u_T - g_val) ** 2)
    return loss
