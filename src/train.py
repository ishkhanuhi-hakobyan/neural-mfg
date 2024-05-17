import torch
import wandb

import torch.optim as optim

from tqdm import tqdm

from src.utils import hjb_loss, hjb_cond_loss, fp_loss, fp_cond_loss
from src.models import DensityNetwork, OpinionDynamics, ValueNetwork

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train():
    wandb.init(project="opinion-dynamics", entity="ishkhanuhi-hakobyan")
    # Example parameters (these should be set according to your specific problem)
    alpha = 0.5
    eps = 0.8
    N = 100
    h = 0.1
    nu = 1
    T = 1.0

    # Instantiate the networks
    u_net = ValueNetwork()
    m_net = DensityNetwork()

    wandb.watch(u_net, log="all")
    wandb.watch(m_net, log="all")

    # Define the optimizer
    optimizer_phi = optim.Adam(u_net.parameters(), lr=1e-4, weight_decay=1e-3)
    optimizer_rho = optim.Adam(m_net.parameters(), lr=1e-4, weight_decay=1e-3)

    # Training loop
    num_iterations = 1000
    batch_size = 128
    save_interval = 10
    for iteration in tqdm(range(num_iterations), total=num_iterations):
        # Sample batch from E and Omega
        t = torch.linspace(0, 1, batch_size).unsqueeze(1)
        x = torch.linspace(-5, 5, batch_size).unsqueeze(1)
        opinion_dynamics = OpinionDynamics(alpha, eps, x, N, h)
        # Compute HJB loss
        L_HJB = hjb_loss(u_net, m_net, opinion_dynamics.hamilton, t, x, nu)
        L_HJB_cond = hjb_cond_loss(u_net, m_net, T, x)

        # Total HJB Loss
        L_HJB_total = L_HJB + L_HJB_cond

        # Backpropagation for HJB loss
        optimizer_phi.zero_grad()
        optimizer_rho.zero_grad()
        L_HJB_total.backward()
        optimizer_phi.step()
        optimizer_rho.step()

        # Sample new batch for FP loss
        # t = torch.rand(batch_size, 1)
        # x = torch.randn(batch_size, 1)

        # opinion_dynamics = OpinionDynamics(alpha, eps, x, N, h)
        # Compute FP loss
        L_FP = fp_loss(m_net, t, x, nu, u_net, opinion_dynamics.hamilton)
        L_FP_cond = fp_cond_loss(m_net, x)

        # Total FP Loss
        L_FP_total = L_FP + L_FP_cond

        # Backpropagation for FP loss
        optimizer_phi.zero_grad()
        optimizer_rho.zero_grad()
        L_FP_total.backward()
        optimizer_phi.step()
        optimizer_rho.step()

        wandb.log({"HJB Loss": L_HJB_total.item(), "FP Loss": L_FP_total.item(), "Iteration": iteration})

        if iteration % 2 == 0:
            print(f"Iteration {iteration}, HJB Loss: {L_HJB_total.item()}, FP Loss: {L_FP_total.item()}")

        if iteration % save_interval == 0:
            torch.save({
                'u_net_state_dict': u_net.state_dict(),
                'm_net_state_dict': m_net.state_dict(),
                'optimizer_phi_state_dict': optimizer_phi.state_dict(),
                'optimizer_rho_state_dict': optimizer_rho.state_dict(),
                'iteration': iteration
            }, f"model_checkpoint_{iteration}.pth")

    torch.save({
        'u_net_state_dict': u_net.state_dict(),
        'm_net_state_dict': m_net.state_dict(),
        'optimizer_phi_state_dict': optimizer_phi.state_dict(),
        'optimizer_rho_state_dict': optimizer_rho.state_dict(),
        'iteration': num_iterations
    }, "model_final.pth")
    wandb.finish()
