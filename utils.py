import numpy as np
from tqdm import tqdm
import torch


class Transfer():
    def __init__(self, betas, betas_it, basis, basis_it, y, y_it, K=75):
        # Coefficients learned on the French source data
        self.betas = betas.copy()
        self.betas_vec = betas['betas_france'].to_numpy()
        # Coefficients learned on the Italian source data
        self.betas_it = betas_it.copy()
        self.betas_it_vec = betas_it['betas_italy'].to_numpy()
        # Spline basis for the French source data
        self.basis = basis.to_numpy()[:, 1:]
        # Spline basis for the Italian source data
        self.basis_it = basis_it.to_numpy()[:, 1:]
        # Number of iterations
        self.K = K
        #  Target data (French)
        self.y = y
        #  Target data (Italian)
        self.y_it = y_it
        #  Scale parameter (should be around 4)
        self.rho = y.sum() / y_it.sum()

    def compute_alpha(self, t, country='fr'):
        if country == 'fr':
            basis = self.basis
        else:
            basis = self.basis_it
        lam_max = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).max()
        lam_min = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).min()
        return .4/(lam_max + lam_min + 1e-8)

    def grad_loss(self, t, country='fr'):
        self.alpha = self.compute_alpha(t, country)
        if country == 'fr':
            y = self.y
            beta = self.betas_vec
            basis = self.basis
        else:
            y = self.y_it
            beta = self.betas_it_vec
            basis = self.basis_it
        beta_te = torch.tensor(beta, requires_grad=True)
        y_te = torch.tensor(y[:-t])
        basis_te = torch.tensor(basis[:-t].T)
        # Compute the loss
        loss = torch.sum((y_te - beta_te@basis_te)**2)
        # Compute the gradient
        loss.backward()
        # Return the gradient to numpy
        return self.alpha*beta_te.grad.numpy()

    def gam_ft(self):
        beta_t = self.betas_vec
        for k in range(self.K):
            beta_t = beta_t - self.grad_loss(k)
        return beta_t@self.basis.T

    # def gam_ft(self, type='test1'):
    #     if type == 'test1':
    #         l = len(test1)
    #     elif type == 'test2':
    #         l = len(test2)
    #     tuning = []
    #     for t in tqdm(range(l)):
    #         tuning.append(self.gam_ft_t(t))
    #     return tuning

    def gam_delta(self):
        # NOT WORKING
        beta_t_it = self.betas_it_vec
        for t in range(self.K):
            beta_t_it = beta_t_it - self.grad_loss(t, country='it')
        delta_t = beta_t_it - self.betas_it_vec
        beta_t = self.betas_vec + self.rho*delta_t
        return beta_t.T@self.basis
