# Import packages
import numpy as np 
import control 
from scipy import signal
from scipy import fft
import pathlib
from matplotlib import pyplot as plt 
from itertools import combinations

# Plotting parameters
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Read in all input-output (IO) data
script_dir = pathlib.Path(__file__).parent
path = script_dir / 'SINE_SWEEP_DATA'
all_files = sorted(path.glob("*.csv"))
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
data = np.array(data)

# get the different combinations
def training_sets_combinations(data):
    """
    input: number of datasets 
    ouptut: combinations of training datasets (array)
    example: in our case (4 datasets)
        [0, 1, 2]
        [0, 1, 3]
        [0, 2, 3]
        [1, 2, 3]

        the remaining dataset is the test one 
    """
    # shape of the data
    shape = np.shape(data)
    n_x = shape[0] # 4
    n_y = shape[1] # 2000
    n_z = shape[2] # 3

    # array of indices of datasets
    n_datasets = np.arange(n_x)
    comb = list(combinations(n_datasets, n_x - 1))
    for i in comb:
        comb_array = np.array(comb)
    return comb_array 

combs = training_sets_combinations(data)
print(training_sets_combinations(data))



# get the averaged cross spectral densities from the datasets 
def averaged_training_CSD(combs, data):
    """
    input: combinations of training datasets, data
    output: average CSD for each training dataset combination (Puy/Puu) = P_d --> P_avg
    example: in our case (4 datasets)
        comb 0: [0, 1, 2] --> P_avg_0
        comb 1: [0, 1, 3] --> P_avg_1
        comb 2: [0, 2, 3] --> P_avg_2
        comb 3: [1, 2, 3] --> P_avg_3

        the remaining dataset is the test one 
    """
    # Dimensions
    # assume each dataset has the same number of datapoints 
    data_shape = np.shape(data)
    no_datasets = data_shape[0] - 1
    data_read = data[0, :, :] # just pick the first dataset to define stuff
    t = data_read[:, 0]
    dt = t[1] - t[0]
    N = t.size
    # Nyquist frequency
    # Extract input and output
    u_raw = data_read[:, 1]  # V, volts
    y_raw = data_read[:, 2]  # LPM, liters per minute
    f, Puy = signal.csd(u_raw, y_raw, fs=1/dt, window='hann')
    f, Puu = signal.csd(u_raw, u_raw, fs=1/dt, window='hann')
    # so f is our Nyquist frequency

    # define the final matrix with the CSD evaluated at each frequency
    # in our case, CSD_mat should be 129x4
    CSD_mat = np.zeros((len(f), len(combs)), dtype=complex)
    k = 0
    # loop through each combination
    for comb in combs:
        # initialize the column matrix (ths is gonna be the nth column of CSD_mat) 
        P_avg = np.zeros((len(f), ), dtype=complex)
        P_holder = np.zeros((len(f), no_datasets), dtype=complex) # holds the CSD of each dataset of this combination
        # loop through each dataset of this combination
        l = 0
        for d in comb:
            training_dataset_d = data[d, :, :]
            # input/output data 
            u = training_dataset_d[:, 1].ravel()  # V, volts
            y = training_dataset_d[:, 2].ravel()  # LPM, liters per minute

            # # normalize the data
            u_max = np.max(u)
            u = u / u_max
            y_max = np.max(y)
            y = y / y_max

            # Puy and Puu
            f, Puy = signal.csd(u, y, fs=1/dt, window='hann')
            f, Puu = signal.csd(u, u, fs=1/dt, window='hann')
            for i in range(len(f)):
                P_holder[i, l] = Puy[i] / Puu[i] # CSD_d
            l += 1
        # compute the average of the three datasets
        for i in range(len(f)):
            P_avg[i] = np.mean(P_holder[i])
       
        CSD_mat[:,k] = P_avg
        k += 1

    return CSD_mat, f

CSD_mat, f = averaged_training_CSD(combs, data)
print(averaged_training_CSD(combs, data))

# Limit to 8 Hz
f_max = 8
idx = f <= f_max
f = f[idx]
CSD_mat = CSD_mat[idx, :]


# LS fit function 
def LS_fit(m, n, CSD_mat, f):
    """
    input: m, n (order of the numerator and denominator), CSD_mat (fx4) array of the frequency 
    response data for each combination of training data sets, f (all frequencies)
    output: 1D array of the transfer functions corresponding to each of the columns of CSD_mat
    """
    omega = 2 * np.pi * f  # rad/s
    N_w, N_comb = CSD_mat.shape
    tf_list = []

    # Loop over each column of CSD_mat (each training combination)
    for col in range(N_comb):
        G_col = CSD_mat[:, col]
        G_real = np.real(G_col)
        G_imag = np.imag(G_col)

        # Construct A and b matrices
        A = np.zeros((N_w, m + n + 1), dtype=complex)
        b_vals = np.zeros((N_w, 1), dtype=complex)

        n_mat = np.zeros((N_w, m + 1), dtype=complex)
        d_mat = np.zeros((N_w, n), dtype=complex)

        # construct b 
        for i in range(n):
            b_vals[i] = ((omega[i] ** 2) * G_real[i]) + ((omega[i] ** 2) * G_imag[i])

        # decompose into real and imag parts
        b = np.vstack([np.real(b_vals), 
                       np.imag(b_vals)]).reshape(-1,1)
        
        # numerator terms (n_mat)
        for k in range(m + 1):
            for i_freq in range(N_w):
                n_mat[i_freq, k] = -(omega[i_freq])**k

        # denominator terms (d_mat)
        for l in range(n):
            for i_freq in range(N_w):
                d_mat[i_freq, l] = (G_real[i_freq] + G_imag[i_freq])*(omega[i_freq])**l

        # real/imag block stacking
        n_block = np.vstack([np.real(n_mat), 
                             np.imag(n_mat)])
        
        d_block = np.vstack([np.real(d_mat), 
                             np.imag(d_mat)])
        
        A = np.hstack([n_block, d_block])

        # Solve least squares Ax = b
        x = np.linalg.solve(A.T @ A, A.T @ b)

        # Construct transfer function for this combination
        num = x[:m + 1].flatten()
        den = np.concatenate(([1], x[m + 1:].flatten()))
        G_est = control.tf(num, den)

        tf_list.append(G_est)

    return tf_list

TFs = LS_fit(1, 1, CSD_mat, f)
print("TFs are", TFs)


def plotting(f, CSD_mat, TFs):
    """
    Plots measured averaged frequency responses (CSD_mat columns) 
    and their corresponding LS-fitted transfer functions (TFs).
    """
    omega = 2 * np.pi * f
    N_f, N_comb = CSD_mat.shape

    # Limit to 8 Hz
    f_mask = f <= 8
    f_plot = f[f_mask]
    omega_plot = omega[f_mask]
    CSD_plot = CSD_mat[f_mask, :]

    # Subplots setup
    fig, axs = plt.subplots(2, N_comb, figsize=(5 * N_comb, 8), squeeze=False)
    fig.suptitle("Averaged Frequency Responses and LS Fits", fontsize=16, fontweight="bold")

    for i in range(N_comb):
        G_exp = CSD_plot[:, i]

        # --- Experimental data ---
        mag_exp = 20 * np.log10(np.abs(G_exp))
        phase_exp = np.angle(G_exp, deg=True)

        # --- Fitted TF data ---
        mag_fit, phase_fit, omega_fit = control.frequency_response(TFs[i], omega_plot)
        mag_fit = np.squeeze(mag_fit)
        phase_fit = np.squeeze(phase_fit)
        mag_fit_dB = 20 * np.log10(mag_fit)
        phase_fit_deg = np.degrees(phase_fit)

        # --- Magnitude plot ---
        axs[0, i].semilogx(f_plot, mag_exp, 'b', label='Measured FRF')
        axs[0, i].semilogx(f_plot, mag_fit_dB, 'r--', label='LS Fit')
        axs[0, i].set_title(f"Combination {i+1}")
        axs[0, i].set_ylabel("Magnitude (dB)")
        axs[0, i].grid(True, which='both', linestyle='--', alpha=0.6)
        axs[0, i].legend()

        # --- Phase plot ---
        axs[1, i].semilogx(f_plot, phase_exp, 'b', label='Measured FRF')
        axs[1, i].semilogx(f_plot, phase_fit_deg, 'r--', label='LS Fit')
        axs[1, i].set_xlabel("Frequency (Hz)")
        axs[1, i].set_ylabel("Phase (deg)")
        axs[1, i].grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plotting(f, CSD_mat, TFs)












    