import numpy as np
import pandas as pd

class Transfer():
    def __init__(self, betas, betas_it, basis, basis_it, y, y_it, K=30):
        # Coefficients learned on the French source data
        self.betas = betas.copy()
        self.betas_vec = np.array([betas.to_numpy()[t][1] for t in range(len(betas))])
        # Coefficients learned on the Italian source data
        self.betas_it = betas_it.copy()
        self.betas_it_vec = np.array([betas_it.to_numpy()[t][1] for t in range(len(betas_it))])
        # Spline basis for the French source data
        self.basis = basis[:,1:]
        # Spline basis for the Italian source data
        self.basis_it = basis_it[:,1:]
        # Number of iterations
        self.K = K
        # Target data (French)
        self.y = y
        # Target data (Italian)
        self.y_it = y_it
        # Scale parameter (should be around 4)
        self.rho = y.sum() / y_it.sum()

    def compute_alpha(self, country='fr'):
        if country == 'fr':
            basis = self.basis
        else:
            basis = self.basis_it
        lam_max = np.linalg.eig(basis.T@basis)[0].max()
        lam_min = np.linalg.eig(basis.T@basis)[0].min()
        return .4/(lam_max + lam_min)
    
    def grad_loss(self, t, country='fr'):
        self.alpha = self.compute_alpha(country)
        if country == 'fr':
            y = self.y
            beta = self.betas_vec
            basis = self.basis
        else:
            y = self.y_it
            beta = self.betas_it_vec
            basis = self.basis_it
        obj = 0
        for s in range(t-1):
            obj += self.alpha*(-2)*(y[s] - beta.T@basis[s])*basis[s]
        return obj
    
    def gam_ft(self):
        beta_t = self.betas_vec
        for t in range(self.K):
            beta_t = beta_t - self.grad_loss(t)
        return beta_t@self.basis.T
    
    def gam_delta(self):
        beta_t_it = self.betas_it_vec
        for t in range(self.K):
            beta_t_it = beta_t_it - self.grad_loss(t, country='it')
        self.betas_it['betas_italy'] = beta_t_it
        df_same = pd.merge(self.betas, self.betas_it)
        delta_t = df_same['betas_italy'].to_numpy() - df_same['betas_france'].to_numpy()
        df_same['betas_france'] += self.rho*delta_t
        self.betas.update(df_same['betas_france'])
        return self.betas['betas_france'].to_numpy()@self.basis.T