import numpy as np
from scipy.optimize import minimize_scalar

class OASIS:
    def __init__(self, g, smin=None, lambda_=None):
        self.g = g
        self.smin = smin
        self.lambda_ = lambda_

    def fit(self, y, sigma=None):
        """Infer spikes from fluorescence trace.
        
        Parameters
        ----------
        y : array_like
            Fluorescence trace
        sigma : float, optional
            Noise standard deviation. Required if lambda_ is None
            
        Returns
        -------
        c : array_like
            Denoised calcium trace
        s : array_like
            Binary spike train (0: no spike, 1: spike)
        """
        y = np.asarray(y, dtype=np.float64)  # Use float64 for better precision
        T = len(y)
        
        # Initialize pools
        pools = []  # Each pool is (value, weight, time, length)
        
        # Initialize solution
        c = np.zeros(T, dtype=np.float64)
        s = np.zeros(T, dtype=np.float64)
        
        # If lambda_ not provided, initialize it to 0
        lambda_ = float(self.lambda_) if self.lambda_ is not None else 0.0
        
        # Calculate mu from lambda_
        mu = lambda_ * (1 - self.g + np.zeros(T, dtype=np.float64))
        mu[-1] = lambda_  # Last time point has different mu
        
        # Initialize first pool with first data point
        pools.append((float(y[0] - mu[0]), 1.0, 0, 1))
        
        # Process each time point
        for t in range(1, T):
            # Add new pool
            pools.append((float(y[t] - mu[t]), 1.0, t, 1))
            
            # Merge pools if necessary
            while len(pools) > 1:
                i = len(pools) - 2  # Second to last pool
                if self.smin is None:
                    # L1 penalty version
                    if pools[i+1][0] >= self.g**pools[i][3] * pools[i][0]:
                        break
                else:
                    # Hard threshold version
                    if pools[i+1][0] >= self.g**pools[i][3] * pools[i][0] + self.smin:
                        break
                
                # Merge pools i and i+1
                vi = pools[i][0]
                wi = pools[i][1]
                li = pools[i][3]
                
                vip1 = pools[i+1][0]
                wip1 = pools[i+1][1]
                
                # New pool parameters with careful handling of numerical precision
                g_power = self.g**li
                g_power_2 = self.g**(2*li)
                w_new = wi + g_power_2 * wip1
                v_new = (wi*vi + g_power*wip1*vip1) / w_new
                t_new = pools[i][2]
                l_new = li + pools[i+1][3]
                
                # Replace pools i and i+1 with merged pool
                pools[i] = (float(v_new), float(w_new), t_new, l_new)
                pools.pop(i+1)
        
        # Construct solution
        for v, w, t, l in pools:
            v = max(0, v)  # Enforce non-negativity
            for tau in range(l):
                c[t+tau] = v * (self.g**tau)
        
        # Calculate continuous spike train with non-negativity constraint
        s_continuous = np.zeros(T, dtype=np.float64)
        s_continuous[1:] = np.maximum(0, c[1:] - self.g * c[:-1])
        s_continuous[0] = max(0, c[0])
        
        # Convert to binary spikes using threshold
        threshold = 0.1 * s_continuous.max() if s_continuous.max() > 0 else 0
        s = (s_continuous > threshold).astype(np.float64)
        
        # If lambda_ was not provided, optimize it
        if self.lambda_ is None and sigma is not None:
            # Function to minimize: distance between RSS and sigma^2 * T
            def objective(lambda_):
                self.lambda_ = float(lambda_)
                c_new, _ = self.fit(y)
                RSS = np.sum((y - c_new)**2)
                return (RSS - sigma**2 * T)**2
            
            # Find optimal lambda_
            # Use a reasonable upper bound based on data scale
            max_lambda = 10.0 * np.abs(y).max()
            res = minimize_scalar(objective, bounds=(0, max_lambda), method='bounded')
            self.lambda_ = float(res.x)
            
            # Rerun with optimal lambda_
            c, s = self.fit(y)
        
        return c, s 