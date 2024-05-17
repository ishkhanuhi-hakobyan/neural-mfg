from scipy.stats import truncnorm
import torch


def initial_m(x, d=1):
    return (1 / 2 * torch.pi ** (d / 2)) * torch.exp(-torch.norm(x, dim=-1) ** 2 / 2)


def truncated_normal(size, mean=0, std=1, lower=-5, upper=5):
    return truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs(size)


# Define the initial distribution function for rho_0
def initial_rho(x, mean=0, std=1, lower=-5, upper=5):
    return torch.tensor(truncated_normal(len(x), mean, std, lower, upper), dtype=torch.float32)


def terminal_u(x, d=1):
    return (x - 2) ** 2
