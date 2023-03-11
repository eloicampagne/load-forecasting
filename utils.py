import numpy as np
from tqdm import tqdm
import torch

class Transfer():
    def __init__(self, betas, betas_it, basis, basis_it, y, y_it, K=75):
        '''
        Performs the transfer learning algorithm.
 
        Input:
            betas : pd.DataFrame, coefficients learned on the French source data
            betas_it : pd.DataFrame, coefficients learned on the Italian source data
            basis : pd.DataFrame, spline basis for the French source data
            basis_it : pd.DataFrame, spline basis for the Italian source data
            y : np.array, target data (French)
            y_it : np.array, target data (Italian)
            K : int, number of iterations for the gradient descent in the finetuning step
        '''
        self.betas = betas.copy()
        self.betas_vec = betas['betas_france'].to_numpy()
        self.betas_vec_0 = self.betas_vec.copy()
        self.betas_list = []
        self.betas_it = betas_it.copy()
        self.betas_it_vec = betas_it['betas_italy'].to_numpy()
        self.betas_it_vec_0 = self.betas_it_vec.copy()
        self.basis = basis.to_numpy()[:, 1:]
        self.basis_it = basis_it.to_numpy()[:, 1:]
        self.K = K
        self.y = y
        self.y_it = y_it

        #  Scale parameter (should be around 4)
        self.rho = y.sum() / y_it.sum()

    def compute_alpha(self, t, country='fr'):
        """
        Computes the step size alpha at time t
        Input: 
            t: int, time
            country: string, 'fr' or 'it'

        Returns:
            alpha: float, step size at time t
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
        Computes GAM finetuning at time t. It finds the best coefficients beta_t to predict up to y_t.
        Input:
            t: float, time
            country: string, 'fr' or 'it'
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
            tuning: list,  list of predictions at each time t given by self.y
        """
        l = len(self.y)
        tuning = []
        for t in tqdm(range(l)):
            # Compute the finetuning at time t
            self.gam_ft_t(t)
            # Compute the prediction at time t
            tuning.append(self.betas_vec@self.basis[t].T)
        return tuning

    def gam_delta_t(self,t):
        '''
        Compute the delta coefficients for the transfer learning algorithm
        '''
        # NOT WORKING
        beta_t_it = self.betas_it_vec
        for t in range(self.K):
            beta_t_it = beta_t_it - self.grad_loss(t, country='it')
        delta_t = beta_t_it - self.betas_it_vec
        beta_t = self.betas_vec + self.rho*delta_t
        return beta_t.T@self.basis

def mape(y_true, y_pred):
    '''
    Computes the mean absolute percentage error

    Input:
        y_true: np.array, true values
        y_pred: np.array, predicted values

    Returns:
        mape: float, mean absolute percentage error
    '''
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100