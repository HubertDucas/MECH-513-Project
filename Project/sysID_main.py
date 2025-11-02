######## MECH 513 PROJECT - FRANCIS MAZIADE, HUBERT DUCAS ########

# Feedback control will be used to change the water flow rate 
# given the changing motor temperatures in real time. By doing 
# that, the total energy consumed by the pumps will be reduced 
# when water is pumped “as needed” rather than “all the time”. 

# In the first part, we'll system ID our nominal plant and define 
# uncertainty bounds

# Import packages
import numpy as np 
import control 
from scipy import signal
from scipy import fft
import pathlib
from matplotlib import pyplot as plt 

# Data (the datasets have the form: time (s), input voltage (V), volume flow rate (LPM))
# Pump does not pump if voltage < ~1.5V

############ Handout Code - Extracting data ############

"""MECH 513 sample code.

J R Forbes, 2025/10/13

This code loads the data.
"""


# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

# Read in all input-output (IO) data
# path = pathlib.Path(r"C:\Users\LENOVO\Documents\McGill\Term 7\MECH 513\Project\SINE_SWEEP_DATA")
script_dir = pathlib.Path(__file__).parent
path = script_dir / 'SINE_SWEEP_DATA'
all_files = sorted(path.glob("*.csv"))
# print(path)
# all_files.sort()
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
# print(np.shape(data))


# Load a dataset (choose one of the four sets)
k_train = 2
data_read = data[k_train, :, :]

# # Read data
# data_read = np.loadtxt('sine_sweep_data_1_0.csv',
#                         dtype=float,
#                         delimiter=',',
#                         skiprows=1,)


# Extract time
t = data_read[:, 0]
N = t.size
T = t[1] - t[0]

# Extract input and output
u_raw = data_read[:, 1]  # V, volts
y_raw = data_read[:, 2]  # LPM, liters per minute


######## plot psd 
f_ps, psd = signal.periodogram(y_raw, fs=1 / T, scaling='spectrum')

# Plot the PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f_ps, psd)  
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('PSD raw input')
plt.grid(True)
plt.show()
###########

# Plot data time dmaine
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$u(t)$ (V)')
ax[1].set_ylabel(r'$y(t)$ (LPM)')
# Plot data
ax[0].plot(t, u_raw, label='input', color='C0')
ax[1].plot(t, y_raw, label='output', color='C1')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='best')
fig.tight_layout()

# This ell variable will allow you to save a plot with the number ell in the plot name
# ell = k_train
# fig.savefig('test_plot_%s.pdf' % ell)

# Show plots
plt.show()

############ Handout Code Ends ############





############ Sstem ID code ############


# First, check the conditioning of the dataset, we want a well-conditionned problem 
def conditioning(data):
    cond = np.linalg.cond(data)
    return cond

print("The condition number of the data set", conditioning(data_read))

# Want a function that takes in the orders of the num and denom, and a 
# dataset and from that compptes the sysID'd transfer fucntion
def sysIDed_TF(m, n, data):
    """
    We want a function that takes in:
        m = order of the numerator 
        n = order of the denominator 
        data = dataset of interest 

    and outputs:
        approsimate transfer fuction for this order + error metrics 
        so we cna compare this order with other orders
    """

    ############ Admin ############
    # Convert rad / s to Hz
    radpersec2Hz = 1 / 2 / np.pi
    Hz2radpersec = 1 * 2 * np.pi

    # time, input, output
    t = data[:, 0].ravel()
    u = data[:, 1].ravel() # input: voltage of the pump's motor
    y = data[:, 2].ravel() # output: volume flow rate 

    # dt
    dt = t[1] - t[0]

    ############ FFTs ############
    # Chirp frequency bounds (need to figure out how we pick these?)
    chirp_freq_low = 0.1  # Hz, min frequency
    chirp_freq_high = 100  # Hz, max frequency

    # Single-sided FFT of input
    u_fft = fft.rfft(u, n=N) / N  # same units as u
    u_mag = np.abs(u_fft)  # compute the magnitude of each u_fft
    u_mag[1:] = 2 * u_mag[1:]  # multiply all mag's by 2, but the zero frequency
    u_phase = np.angle(u_fft, deg=False)  # compute the angle

    f = fft.rfftfreq(N, d=dt)  # the frequencies in Hz
    N_f_max = np.searchsorted(f, chirp_freq_high)  # find index of max frequency (chirp higher bound)
    omega = f * 2 * np.pi   # the frequencies in rad/s
    w = omega[:N_f_max]

    # Recompute the FFT with the correct scaling
    u_FFT = np.zeros(N_f_max, dtype=complex)
    for i in range(N_f_max):
        u_FFT[i] = u_mag[i] * np.cos(u_phase[i]) + 1j * u_mag[i] * np.sin(u_phase[i])


    # FFT of ouptut 
    y_fft = fft.rfft(y, n=N) / N # same units as y
    y_mag = np.abs(y_fft) # compute the magnitude of each y_fft
    y_mag[1:] = 2 * y_mag[1:] # multiply all magnitutes by 2, but the zero frequency
    y_phase = np.angle(y_fft, deg=False)  # compute the angle

    y_FFT = np.zeros(N_f_max, dtype=complex)
    for i in range(N_f_max):
        y_FFT[i] = y_mag[i] * np.cos(y_phase[i]) + 1j * y_mag[i] * np.sin(y_phase[i])


    ############ Plots ############

    # Golden ratio
    gr = (1 + np.sqrt(5)) / 2

    # Figure height
    height = 11 / 2.54  # cm

    # Find index of chirp_freq_low
    f_differences = np.abs(f - chirp_freq_low)
    N_f_min = f_differences.argmin()

    # Plot FFT of input and output
    fig, ax = plt.subplots(figsize=(height * gr, height))
    ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(u_FFT[N_f_min:N_f_max]), '.', color='C0', label=r'$|u(j\omega)|$')
    ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(y_FFT[N_f_min:N_f_max]), '.', color='C1', label=r'$|y(j\omega)|$')
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'Magnitude (dB)')
    ax.legend(loc='lower left')
    fig.tight_layout()
    plt.show()
    # fig.savefig(f'figs/IO_freq_resp.pdf')

    # # Plot FFT of output
    # fig, ax = plt.subplots(figsize=(height * gr, height))
    # ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(y_FFT[N_f_min:N_f_max]), '.', color='C0', label=r'$|y(j\omega)|$')
    # ax.semilogx(f[N_f_min:N_f_max], 20 * np.log10(y_mag[N_f_min:N_f_max]), '.', color='C1', label=r'$|y(j\omega)|$')
    # ax.set_xlabel(r'$\omega$ (Hz)')
    # ax.set_ylabel(r'Magnitude (dB)')
    # ax.legend(loc='lower left')
    # fig.tight_layout()
    # plt.show()


    ############ TF ############
    # initialize the transfer function 
    G = np.zeros(N_f_max, dtype=complex) # array of size equal to the number of TFs (imaginary numbers)
    G_mag = np.zeros(N_f_max,)
    G_phase = np.zeros(N_f_max)

    # initialize the num and denom arrays
    num = np.zeros(N_f_max, dtype=complex)
    denom = np.zeros(N_f_max, dtype=complex)

    # Puy and Puu
    f, Puy = signal.csd(u, y, fs=1/dt, window='hann')
    f, Puu = signal.csd(u, u, fs=1/dt, window='hann')
    # print(Puy)
    # print(Puu)

    # need to figure this out, Puy, Puu and G don't have the same size instead
    # I think we only want to keep the frequencies where the system is actually 
    # excited by the input so we keep the chirp lower and higher boundaries
    mask = (f >= chirp_freq_low) & (f <= chirp_freq_high)
    f = f[mask]
    Puy = Puy[mask]
    Puu = Puu[mask]
    N_f_max = len(f)

    # Puy = Puy[N_f_min:N_f_max]
    # Puu = Puu[N_f_min:N_f_max]

    for i in range(N_f_max):
        # Compute G at each frequency
        G[i] = Puy[i] / Puu[i]

        # num = np.mean(np.conj(u_FFT[i]) * y_FFT[i])

        # # compute the denom
        # denom = np.mean(np.abs(u_FFT[i])**2)

        # G[i] = num/denom
        
        # Compute magnitude and phase of G.
        G_mag[i] = np.sqrt((np.real(G[i]))**2 + (np.imag(G[i]))**2)  # absolute
        G_phase[i] = np.arctan2(np.imag(G[i]), np.real(G[i]))  # rad

    G_mag_dB = 20 * np.log10(G_mag)
    G_phase_deg = G_phase * 360 / 2 / np.pi
    # G_phase_deg = np.unwrap(G_phase_deg, period = 360)  # unwrap the phase

    ############ Plots ############
    # Plot Bode plot
    fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
    # Magnitude plot
    axes[0].semilogx(f[N_f_min:N_f_max], G_mag_dB[N_f_min:N_f_max], '.', color='C3', label='System ID')
    axes[0].set_yticks(np.arange(-80, 20, 20))
    axes[0].set_xlabel(r'$\omega$ (Hz)')
    axes[0].set_ylabel(r'Magnitude (dB)')
    axes[0].legend(loc='best')
    # Phase plot
    # axes[1].semilogx(f_shared_Hz, phase_G_deg, color='C2', label='True')
    axes[1].semilogx(f[N_f_min:N_f_max], G_phase_deg[N_f_min:N_f_max], '.', color='C3', label='System ID')
    axes[1].set_yticks(np.arange(-90, 210, 30))
    axes[1].set_xlabel(r'$\omega$ (Hz)')
    axes[1].set_ylabel(r'Phase (deg)')
    # fig.savefig(f'figs/system_ID_freq_resp.pdf')

    # Plot show
    plt.show()


    ############ Save frequency response data ############
    
    data_write = np.block([[f[N_f_min:N_f_max]], [np.real(G[N_f_min:N_f_max])], [np.imag(G[N_f_min:N_f_max])]]).T
    np.savetxt('freq_resp_data.csv',
            data_write,
            fmt='%.8f',
            delimiter=',',
            header='f (Hz), Re{G(jw)}, Im{G(jw)}')

    ############ Fit ############

    # Read data from frequency response dtaa 
    data_read = np.loadtxt('freq_resp_data.csv',
                            dtype=float,
                            delimiter=',',
                            skiprows=1,)

    f = data_read[:, 0].ravel()
    omega = f * 2 * np.pi   # the frequencies in rad/s
    G_real = data_read[:, 1].ravel()
    G_imag = data_read[:, 2].ravel()

    N_w = f.shape[0]
    G = np.zeros(N_w, dtype=complex)
    for i in range(N_w):
        G[i] = G_real[i] + 1j * G_imag[i]

    # Compute freq response
    N_f_max = f.shape[0]
    G_mag = np.zeros(N_f_max,)
    G_phase = np.zeros(N_f_max)
    for i in range(N_f_max):
        G_mag[i] = np.sqrt((np.real(G[i]))**2 + (np.imag(G[i]))**2)  # absolute
        G_phase[i] = np.arctan2(np.imag(G[i]), np.real(G[i]))  # rad

    N_w = f.shape[0]


    # Construct A and b matrices
    A = np.zeros((N_w, m + n + 1), dtype=complex)
    b_vals = np.zeros((N_w, 1), dtype=complex)

    # define the generalized n and d matricces 
    # CAREFUL, order m is used to construct n and d

    n_mat = np.zeros((N_w, m+1))
    d_mat = np.zeros((N_w, n))

    # construct b 
    for i in range(n):
        b_vals[i] = ((omega[i] ** 2) * G_real[i]) + ((omega[i] ** 2) * G_imag[i])

    b = np.vstack([np.real(b_vals), 
                   np.imag(b_vals)]).reshape(-1,1) # decompose into real and imag parts
    
    # order loop
    for k in range(m+1): # following Fit slide 7/20 from course notes
        # data looop
        for i_freq in range(N_w):
            n_mat[i_freq, k] = -(omega[i_freq])**k
    
    for l in range(n):
        for i_freq in range(N_w):
            d_mat[i_freq, l] = (G_real[i_freq] + G_imag[i_freq])*(omega[i_freq])**l
    
    n_block = np.vstack([np.real(n_mat), 
                         np.imag(n_mat)])
    
    d_block = np.vstack([np.real(d_mat), 
                         np.imag(d_mat)])
    
    # print(n_block.shape)  # (2*N_w, m+1)
    # print(d_block.shape)  # (2*N_w, n)
    # print(b.shape)        # (2*N_w, 1)
    
    A = np.hstack([n_block, d_block])


    # Solve Ax = b problem for x
    x = np.linalg.solve(A.T @ A, A.T @ b)

    # Extract transfer function.
    num = x[:m+1].flatten()          # numerator coefficients
    den = np.concatenate(([1], x[m+1:].flatten()))  # 1 for first s^something
    G_est = control.tf(num, den)
    # print("Original transfer function: ", P, "\n")
    print("Estimated transfer function: ", G_est)


    ############ Error metrics ############


    return G_est #, G, perc_VAF, FIT, RMSE or whatever is needed


G = sysIDed_TF(1, 3, data=data_read)
print(G)


