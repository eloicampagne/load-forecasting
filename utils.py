import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim


class Transfer():
    def __init__(self, betas, betas_it, basis, basis_it, y, y_it, K=75):
        # Coefficients learned on the French source data
        # self.betas: pandas dataframe
        self.betas = betas.copy()
        # self.betas_vec: numpy array
        self.betas_vec = betas['betas_france'].to_numpy()
        self.betas_vec_0 = self.betas_vec.copy()
        # self.betas_list: list of betas at time t
        self.betas_list = []
        # Coefficients learned on the Italian source data
        self.betas_it = betas_it.copy()
        self.betas_it_vec = betas_it['betas_italy'].to_numpy()
        self.betas_it_vec_0 = self.betas_it_vec.copy()
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
        """
        Input: 
            t: time
            country: 'fr' or 'it'

        Returns:
            alpha: step size at time t
        """
        if country == 'fr':
            basis = self.basis
        else:
            basis = self.basis_it
        lam_max = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).max()
        lam_min = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).min()
        return .4/(lam_max + lam_min + 1e-8)

    def gam_ft_t(self, t,country='fr'):
        """
        Computes GAM finetuning at time t
        Input:
            t: time
            country: 'fr' or 'it'
        Updates:
            self.betas_vec: coefficients at time t
            self.betas_list: list of betas
        """
        self.alpha = self.compute_alpha(t, country)
        if country == 'fr':
            y = self.y
            beta = self.betas_vec_0
            basis = self.basis
        else:
            y = self.y_it
            beta = self.betas_it_vec_0
            basis = self.basis_it

        # Convert all the numpy arrays to torch tensors
        beta_te = torch.tensor(beta, requires_grad=True)
        y_te = torch.tensor(y[:t], requires_grad=False)
        basis_te = torch.tensor(basis[:t].T, requires_grad=False)

        # Define the optimizer (SGD)
        # With one parameter it amounts to compute the vanilla gradient descent
        optimizer = torch.optim.SGD([beta_te], lr=self.alpha)

        for k in range(self.K):
            # Zero the parameter gradients	
            optimizer.zero_grad()
            # Compute the loss
            loss = torch.sum((y_te - beta_te@basis_te)**2)
            # Compute the gradient
            loss.backward()
            # Backpropagate the gradient
            optimizer.step()
        
        # Convert the torch tensor to numpy array
        beta_t = beta_te.detach().numpy()
        # Update
        self.betas_vec = beta_t
        self.betas_list.append(beta_t)

    def gam_ft(self):
        """
        GAM finetuning for all t
        Returns:
            tuning: list of predictions at time t
        """
        l = len(self.y)
        tuning = []
        for t in tqdm(range(l)):
            # Compute the finetuning at time t
            self.gam_ft_t(t)
            # Compute the prediction at time t
            tuning.append(self.betas_vec@self.basis[t].T)
        return tuning

    def gam_delta(self):
        # NOT WORKING
        beta_t_it = self.betas_it_vec
        for t in range(self.K):
            beta_t_it = beta_t_it - self.grad_loss(t, country='it')
        delta_t = beta_t_it - self.betas_it_vec
        beta_t = self.betas_vec + self.rho*delta_t
        return beta_t.T@self.basis
