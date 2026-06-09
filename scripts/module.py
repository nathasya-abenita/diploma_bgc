import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_ensemble (t_list, ens, ax, label=None):
    # Compute statistics
    ens = np.array(ens)
    mean = np.mean(ens, axis=0)
    q025 = np.quantile(ens, 0.025, axis=0)
    q975 = np.quantile(ens, 0.975, axis=0)

    # Plot
    ax.plot(t_list, mean, label=label)
    ax.fill_between(t_list, q025, q975, alpha=0.15, linewidth=0)
    return ax

def plot_collapse_ci(ax, gamma_list, clps_ens_list,
                     lower_q=0.025, upper_q=0.975, outfile='collapse_time.csv'):

    gamma = np.asarray(gamma_list)

    # Compute confidence interval
    mean_ct = np.array([np.mean(ct) for ct in clps_ens_list])
    q_low = np.array([np.quantile(ct, lower_q) for ct in clps_ens_list])
    q_high = np.array([np.quantile(ct, upper_q) for ct in clps_ens_list])

    # Export values as CSV file
    df = pd.DataFrame({'gamma': gamma, 'q_low': q_low, 
                       'mean_ct': mean_ct, 'q_high': q_high})
    df.to_csv(outfile)

    # Plot confidence interval
    ax.plot(gamma, mean_ct, marker="o", label="ensemble mean")

    ax.fill_between(
        gamma,
        q_low,
        q_high,
        alpha=0.3,
        label=f"{100*(upper_q-lower_q):.0f}% interval"
    )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("collapse time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax

class HumanForestModel:
    '''
    Apply Euler method to solve the variables
    N(t) : human population
    R(t) : area coverage of forest
    '''
    def __init__ (self, dt, r, r_, a0, beta, Rc, gamma,
                  N0, R0, t_bounds):
        self.dt = dt        # timestamp
        self.r = r          # human growth rate
        self.r_ = r_        # forest renewability
        self.a0 = a0        # technological parameter
        self.beta = beta    # human population carrying capacity
        self.Rc = Rc        # forest carrying capacity
        self.gamma = gamma  # climate response to ENSO

        # initial conditions
        self.N0 = N0       
        self.R0 = R0

        # time
        self.t_bounds = t_bounds
        self.t1 = t_bounds[0]
        self.t2 = t_bounds[1]
        duration = t_bounds[1] - t_bounds[0]
        self.n_iter = int(duration / dt) + 1

    def generate_enso (self, enso_file=r'./data/nino34r_det.csv', plot=False):
        # Open data and keep values after pre-industrial
        df = pd.read_csv(enso_file).set_index(keys='time')
        df = df.rename(columns={'Nino3.4r': 'enso'})['1950':'2025']

        # Prepare historical values intersecting with t_bounds
        enso_hist = df.loc[str(self.t1): '2025', "enso"].to_list()
        enso_future = self.bootstrap_future_enso(
            df['enso'],
            block_years=5
        )
        enso_sim = enso_hist + enso_future

        if plot:
            fig, ax = plt.subplots()
            ax.plot(enso_sim)
        return enso_sim
                

    def update_N (self, N, R):
        dndt = self.r * N * (1 - N / (self.beta * R))
        return dndt * self.dt + N
    
    def update_R (self, N, R, enso):
        enso = max([0, enso])

        drdt = self.r_ * R * (1 - R / self.Rc)   \
               - self.a0 * N * R \
               - self.gamma * enso * R
        
        return drdt * self.dt + R
    
    def solve(self, enso_list):
        # Initialize arrays
        N_list = [self.N0]
        R_list = [self.R0]
        t_list = [self.t_bounds[0]]

        # Perform numerical integration
        for i in range (self.n_iter):
            # Pick ENSO index
            enso = enso_list[i]

            # Pick old values
            N_old = N_list[i]
            R_old = R_list[i]

            # Compute new step
            N_new = self.update_N(N_old, R_old) 
            R_new = self.update_R(N_old, R_old, enso)

            # Save new values
            N_list.append(N_new)
            R_list.append(R_new)
            t_list.append(t_list[i] + self.dt)

        # Find collapse time
        time_idx = np.argmax(N_list)
        collapse_time = t_list[time_idx]

        return N_list, R_list, t_list, collapse_time
    
    def bootstrap_future_enso(self, enso_series, block_years=5):

        rng = np.random.default_rng()

        target_months = (self.t2 - 2025) * 12 + 1
        block_size = block_years * 12
        values = enso_series.values

        synthetic = []
        while len(synthetic) < target_months:
            start = rng.integers(0, len(values) - block_size)
            block = values[start:start + block_size]
            synthetic.extend(block)

        synthetic = synthetic[:target_months]
        return synthetic
    
if __name__ == '__main__':
    # Define parameters [Global case]
    # dt = 1/12               # year
    # r  = 0.01               # years**{-1}
    # r_ = 0.001              # years**{-1}
    # a0 = 1e-12              # years**{-1}
    # beta = 700
    # Rc = 6e7                # km**2
    # N0 = 6e9
    # R0 = 4e7
    # gamma = 0               # years**{-1}
    # t_bounds = [2000, 2500]

    # Define parameters [Sumba case]
    dt = 1/12               # year
    r  = 0.01               # years**{-1}
    r_ = 0.001              # years**{-1}
    a0 = 0.14 / 20 / 686e3 #1.8e-2 / 20 / 686e3      # years**{-1}

    Rc = 10_910                # km**2
    N0 = 780_000
    R0 = 5_023
    Nc = 500 * Rc #1.2 * N0 #Rc * 100
    beta = Nc/Rc
    gamma = 0             # years**{-1}
    t_bounds = [2020, 2200]

    # Call 
    model = HumanForestModel(dt, r, r_, a0, beta, Rc, gamma, N0, R0, t_bounds)
    enso_list = model.generate_enso()
    N_list, R_list, t_list, col_time = model.solve(enso_list)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(t_list, N_list, label='climate non-responsive')
    axs[0].set_ylabel(r'N(t)'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t_list, R_list)
    axs[1].set_ylabel(r'R(t)'); axs[0].set_xlabel(r'$t$')
    
    # Call
    gamma = 0.05
    model = HumanForestModel(dt, r, r_, a0, beta, Rc, gamma, N0, R0, t_bounds)
    N_list, R_list, t_list, col_time = model.solve(enso_list)
    axs[0].plot(t_list, N_list, label='climate responsive')
    axs[1].plot(t_list, R_list)
    axs[0].legend()
    plt.show()
