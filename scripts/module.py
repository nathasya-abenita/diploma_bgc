import matplotlib.pyplot as plt
import numpy as np

class HumanForestModel:
    '''
    Apply Euler method to solve the variables
    N(t) : human population
    R(t) : area coverage of forest
    '''
    def __init__ (self, dt, r, r_, a0, beta, Rc, 
                  u, gamma,
                  N0, R0, t_bounds):
        self.dt = dt        # timestamp
        self.r = r          # human growth rate
        self.r_ = r_        # forest renewability
        self.a0 = a0        # technological parameter
        self.beta = beta    # human population carrying capacity
        self.Rc = Rc        # forest carrying capacity

        self.u = u          # human intervention
        self.gamma = gamma  # climate stress to forest

        # initial conditions
        self.N0 = N0       
        self.R0 = R0

        # time
        self.t_bounds = t_bounds
        duration = t_bounds[1] - t_bounds[0]
        self.n_iter = int(duration / dt) + 1

    def update_N (self, N, R):
        dndt = self.r * N * (1 - N / (self.beta * R))
        return dndt * self.dt + N
    
    def update_R (self, N, R):
        drdt = self.r_ * (1 - R / self.Rc)   \
               - self.a0 * N * R             \
               + (self.u - self.gamma) * R
        return drdt * self.dt + R
    
    def solve(self):
        # Initialize arrays
        N_list = [self.N0]
        R_list = [self.R0]
        t_list = [self.t_bounds[0]]

        # Perform numerical integration
        for i in range (self.n_iter):
            # Pick old values
            N_old = N_list[i]
            R_old = R_list[i]

            # Compute new step
            N_new = self.update_N(N_old, R_old) 
            R_new = self.update_R(N_old, R_old)

            # Save new values
            N_list.append(N_new)
            R_list.append(R_new)
            t_list.append(t_list[i] + self.dt)

        return N_list, R_list, t_list
    
if __name__ == '__main__':
    # Define parameters
    dt = 1
    r  = 0.01               # years**{-1}
    r_ = 0.001              # years**{-1}
    a0 = 1e-12              # years**{-1}
    beta = 700
    Rc = 6e7                # km**2
    u = 0
    gamma = 0
    N0 = 6e9
    R0 = 4e7
    t_bounds = [2000, 2500]

    # Call 
    model = HumanForestModel(dt, r, r_, a0, beta, Rc, u, gamma, N0, R0, t_bounds)
    N_list, R_list, t_list = model.solve()

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(t_list, N_list)
    axs[0].set_ylabel(r'N(t)'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t_list, R_list)
    axs[1].set_ylabel(r'R(t)'); axs[0].set_xlabel(r'$t$')
    plt.show()
