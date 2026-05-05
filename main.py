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
                             

# powerDistributionGraph(qamMatrix)
# plt.show()

# 2.2.1 BPSK decoding
# 2.2.1.1
qamSeq = qamMatrix[0]
# print(qamSeq) # Montre du BPSK

# 2.2.1.2
def bpsk_demod(qamSeq):
    out = []
    for s in qamSeq:
        if s.real > 0:
            out.append(1)
        else:
            out.append(0)
    return out

bitSeq = bpsk_demod(qamSeq)

# print(bitSeq) # Montre la séquence binaire décodée

def test_bpsk():
    # BPSK decoding test
    assert bpsk_demod(np.array([1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0])) == [1, 1, 1, 0]
    assert bpsk_demod(np.array([1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0])) == [1, 1, 0, 1, 1, 1, 0, 1]
    assert bpsk_demod(np.array([1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0])) == [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    assert bpsk_demod(np.array([-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0,-1.0+1j*0.0])) == [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    assert bpsk_demod(np.array([-1.2+1j*-0.2,-0.9+1j*-0.3,-1.1+1j*0.1,-1.0+1j*-0.0,-0.8+1j*0.2,-1.1+1j*-0.0,1.0+1j*0.2,-1.0+1j*0.0,-1.0+1j*0.1,1.2+1j*0.1,1.1+1j*-0.1,-1.0+1j*-0.1,1.1+1j*-0.1,-1.0+1j*0.2,-0.8+1j*-0.1,-1.0+1j*0.1])) == [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]


test_bpsk()