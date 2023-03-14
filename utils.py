import numpy as np
from tqdm import tqdm
import torch


class Transfer():

    def __init__(self, betas_fr, betas_it, basis_fr, basis_it, y_fr, y_it, rho=None, K=75):
        '''
        Performs the transfer learning algorithm.
 
        Input:
            betas : pd.DataFrame, coefficients learned on the French source data. Index is the name of the coefficient, column is the value of the coefficient
            betas_it : pd.DataFrame, coefficients learned on the Italian source data. Index is the name of the coefficient, column is the value of the coefficient
            basis : pd.DataFrame, spline basis for the French source data. Index is the time, columns are the values of each spline basis coefficient for each time
            basis_it : pd.DataFrame, spline basis for the Italian source data. Index is the time, columns are the values of each spline basis coefficient for each time
            y : np.array, target data (French)
            y_it : np.array, target data (Italian)
            K : int, number of iterations for the gradient descent in the finetuning step
        '''

        common_idx, fr_to_it, it_to_fr = np.intersect1d(
            betas_it.index, betas_fr.index, return_indices=True)
        self.fr_to_it = fr_to_it
        self.it_to_fr = it_to_fr

        self.betas_fr_0 = betas_fr.copy()['betas_france'].to_numpy()
        self.betas_fr_t = self.betas_fr_0.copy()

        self.betas_it_0 = betas_it.copy()['betas_italy'].to_numpy()
        self.betas_it_t = self.betas_it_0.copy()

        self.basis_fr = basis_fr.to_numpy()
        self.basis_it = basis_it.to_numpy()

        self.K = K

        self.y_fr = y_fr
        self.y_it = y_it

        if rho is None:
            #  Scale parameter (should be around 4)
            print("Data leakage warning: computing rho on the target data.")
            self.rho = y_fr.sum() / y_it.sum()
        else:
            self.rho = rho

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
            basis = self.basis_fr
        else:
            basis = self.basis_it
        lam_max = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).max()
        lam_min = np.real(np.linalg.eig(basis[:t].T@basis[:t])[0]).min()
        return .4/(lam_max + lam_min + 1e-8)

    def gradient_descent(self, beta, y, basis, t):
        """
        Computes the gradient descent at time t given the initial coefficients beta.
        Input:
            beta: np.array, initial coefficients
            y: np.array, target data
            basis: np.array, spline basis
            t: int, time
        Returns:
            beta_t: np.array, coefficients at time t
        """
        #  Convert all the numpy arrays to torch tensors
        beta_te = torch.tensor(beta, requires_grad=True)
        # No data leakage. We only use the data up to time t
        y_te = torch.tensor(y[:t], requires_grad=False)
        basis_te = torch.tensor(basis[:t].T, requires_grad=False)

        #  Define the optimizer (SGD)
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

        #  Convert the torch tensor to numpy array
        beta_t = beta_te.detach().numpy()

        return beta_t

    def gam_ft_t(self, t, country='fr'):
        """
        Computes GAM finetuning at time t.
        It finds the best coefficients beta_t to predict up to y_t.
        Corresponds to steps 1 and 2 of GAM-finetuned and GAM-delta.

        Input:
            t: float, time
            country: string, 'fr' or 'it'
        Updates:
            self.betas_t: coefficients at time t of the country
            self.betas_list: list of betas at each time t of the country
        Returns:
            beta_t: coefficients at time t of the country
        """
        self.alpha = self.compute_alpha(t, country)
        if country == 'fr':
            y = self.y_fr
            beta = self.betas_fr_0
            basis = self.basis_fr
        else:
            y = self.y_it
            beta = self.betas_it_0
            basis = self.basis_it

        beta_t = self.gradient_descent(beta, y, basis, t)
        return beta_t

    def gam_ft(self, country='fr'):
        """
        GAM finetuning for all t.
        Corresponds to steps 3 of GAM-finetuned.

        Input:
            country: string, 'fr' or 'it'
        Returns:
            tuning: list,  list of predictions at each time t given by self.y
        """

        if country == 'fr':
            l = len(self.y_fr)
            basis = self.basis_fr
        else:
            l = len(self.y_it)
            basis = self.basis_it

        tuning = []
        for t in tqdm(range(l)):
            #  Compute the finetuning at time t
            beta_t = self.gam_ft_t(t, country=country)
            #  Compute the prediction at time t
            tuning.append(beta_t@basis[t].T)
        return tuning

    def gam_delta_t(self, t, country_orig='it'):
        '''
        Compute the delta coefficients and the beta coefficient for the transfer learning algorithm.
        Corresponds to steps 1, 2, 3 of GAM-delta.

        Input:
            t: int, time
            country_orig: string, 'fr' or 'it'. Country of the source data from which we want to transfer the knowledge
        Returns:
            beta_array_dst: np.array, coefficients of the destination country
        '''
        beta_src_t = self.gam_ft_t(t, country=country_orig)

        if country_orig == 'fr':
            beta_0_src = self.betas_fr_0
            country_dest = 'it'
            beta_0_dst = self.betas_it_0
        else:
            beta_0_src = self.betas_it_0
            country_dest = 'fr'
            beta_0_dst = self.betas_fr_0

        beta_dst_t = np.copy(beta_0_dst)
        delta_t = beta_src_t - beta_0_src

        if country_dest == 'fr':
            beta_dst_t[self.it_to_fr] += self.rho*delta_t[self.fr_to_it]
        else:
            beta_dst_t[self.fr_to_it] += self.rho*delta_t[self.it_to_fr]

        return beta_dst_t

    def gam_delta(self, country_orig='it', country_dst='fr'):
        '''
        Compute the prediction of the transfer learning algorithm, ie step 4 of GAM-delta.

        Input:
            country_orig: string, 'fr' or 'it'. Country of the source data from which we want to transfer the knowledge
            country_dst: string, 'fr' or 'it'. Country of the target data to which we want to transfer the knowledge
        Returns:
            tuning : np.array, prediction of the transfer learning algorithm
        '''
        # beta_array_dst = self.gam_delta_fit(country_orig)
        if country_dst == 'fr':
            l = len(self.y_fr)
            basis = self.basis_fr
        else:
            l = len(self.y_it)
            basis = self.basis_it

        tuning = []
        for t in tqdm(range(l)):
            #  Compute the prediction at time t
            beta_t = self.gam_delta_t(t, country_orig=country_orig)
            tuning.append(beta_t@basis[t].T)
        return tuning

    def gam_delta_ft_t(self, t, country_src='it', country_dst='fr'):
        '''
        Performs the finetuning of the transfer learning algorithm at time t.

        Input:
            t: int, time
            country_src: string, 'fr' or 'it'. Country of the source data from which we want to transfer the knowledge
            country_dst: string, 'fr' or 'it'. Country of the target data to which we want to transfer the knowledge
        Returns:
            beta_t: np.array, coefficients of the destination country at time t
        '''

        if country_dst == 'fr':
            y = self.y_fr
            basis = self.basis_fr
        else:
            y = self.y_it
            basis = self.basis_it

        beta_t = self.gam_delta_t(t, country_orig=country_src)
        beta_t = self.gradient_descent(beta_t.copy(), y, basis, t)
        return beta_t

    def gam_delta_ft(self, country_src='it', country_dst='fr'):
        '''
        Performs the prediction of the finetuning of the transfer learning algorithm for all time t.

        Input:
            country_src: string, 'fr' or 'it'. Country of the source data from which we want to transfer the knowledge
            country_dst: string, 'fr' or 'it'. Country of the target data to which we want to transfer the knowledge
        Returns:
            tuning : np.array, prediction of the delta transfer learning algorithm
        '''
        if country_dst == 'fr':
            l = len(self.y_fr)
            basis = self.basis_fr
        else:
            l = len(self.y_it)
            basis = self.basis_it

        tuning = []
        for t in tqdm(range(l)):
            #  Compute the prediction at time t
            beta_t = self.gam_delta_ft_t(
                t, country_src=country_src, country_dst=country_dst)
            tuning.append(beta_t@basis[t].T)
        return tuning


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

def rmse(y_true, y_pred):
    '''
    Computes the root mean squared error

    Input:
        y_true: np.array, true values
        y_pred: np.array, predicted values

    Returns:
        rmse: float, root mean squared error
    '''
    return np.sqrt(np.mean((y_true - y_pred)**2))