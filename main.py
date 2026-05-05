import numpy as np
import math
# Module to display T/F matrix
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
# Module for Unit testing
import pytest
# Module for convolutionnal decoder
import sk_dsp_comm.fec_conv as fec

my_data = np.genfromtxt('./tfMatrix.csv', delimiter=';')
mat_complex = my_data[:,0::2] +1j*my_data[:,1::2]

# print(mat_complex.shape)
symbols_per_frame, N = mat_complex.shape

Nre = 624

tfMatrix_short = np.hstack((mat_complex[:, :Nre//2], mat_complex[:, -Nre//2:]))

def powerDistributionGraph(Z):
    """
    Draw the power distribution graph
    """
    fig, ax = plt.subplots()
    cs = ax.contourf(np.linspace(0, len(Z[0]), len(Z[0])), np.linspace(0, len(Z), len(Z)), np.abs(Z)**2)
    cbar = fig.colorbar(cs)
    ax.set_title('Distribution de puissance')
    ax.set_xlabel('Fréquence sous porteuse')
    ax.set_ylabel('Symboles')



# --- Removing PSCH and SSCH channels
qamMatrix = tfMatrix_short[2:,:]
                             

powerDistributionGraph(qamMatrix)
plt.show()

