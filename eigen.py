import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy import interpolate
from scipy.constants import m_p, c, e
import matplotlib.pyplot as plt

n_turns = 8192

#SPS parametrs
alpha = 0.
R = 6911/(2.*np.pi)
Q = 20.13
Q_s = 0.017
[Q_frac, Q_int] = np.modf(Q)
Q_frac = np.round(Q_frac, 2)
beta = R/Q
gamma = (1.+alpha**2)/beta
mu = 2.*np.pi*Q
intensity = 1e11
damper_gain = 0.01
'''
h=4620 is the number of buckets of 5ns 
(Harmonic number) 4620=5*924 
Fully filled machine: 924 bunches spaced by 25ns
Real filling scheme: 192=4*48 bunches (4 bunch trains separated by 225 ns, bunches within a train 25 ns) 
'''
# Import Wake Table (post LS2, Q20 optics) [V/pC/mm]
wakes = pd.read_csv('WallWake_CZ_Q20_SPS.wake','\t', header=None, float_precision=np.float64)

# Factor for obtaining wake kick
gamma_rel = 27.7 # Relativistic mass factor
b_sq = 1 - (1/(gamma_rel**2))
E_0 = m_p*(c**2)
C = intensity*e**2/(E_0*b_sq*gamma_rel) # Factor defined by N. Mounet

z = c * wakes[0] * 1e-9 # [ns] in table. Converted to [m]
dip_X = C * wakes[1] * 1e15 # Factor 1e15 to convert [V/pC/mm] to [V/C/m]
dip_Y = C * wakes[2] * 1e15
quad_X = C *  wakes[3] * 1e15
quad_Y = C * wakes[4] * 1e15

# Filling scheme
#z_new = np.arange(0, 23.1e-6 * c, 25e-9 * c) # Completely filled SPS
n_bunch_train = 4
bunch_train = np.arange(0, 1200e-9 * c, 25e-9 * c) # Realistic filling scheme for 1 bunch train
z_new = np.array([bunch_train+(b*1425e-9*c) for b in range(n_bunch_train)]).flatten() # Repeating bunch trains with an interval of 225 ns

n_bunches = len(z_new)

# Interpolation
tck = interpolate.splrep(z, dip_X, s=0)
dip_Xnew = interpolate.splev(z_new, tck, der=0)

tck = interpolate.splrep(z, dip_Y, s=0)
dip_Ynew = interpolate.splev(z_new, tck, der=0)

tck = interpolate.splrep(z, quad_X, s=0)
quad_Xnew = interpolate.splev(z_new, tck, der=0)

tck = interpolate.splrep(z, quad_Y, s=0)
quad_Ynew = interpolate.splev(z_new, tck, der=0)

# Parameters for transfer matrix
dia_ele1 = np.cos(mu) + (alpha * np.sin(mu)) - damper_gain
dia_ele2 = np.cos(mu) - (alpha * np.sin(mu))
ele_1 = beta * np.sin(mu)
ele_2 = (-gamma * np.sin(mu))
eles = [dia_ele1, dia_ele2, ele_1, ele_2]
row = np.array([0, 1, 0, 1])
col = np.array([0, 1, 1, 0])

# Part for obtaining the general transfer matrix for all particles
mat_row = np.array([r for i in range(n_bunches) for r in row+(2*i)])
mat_col = np.array([c for i in range(n_bunches) for c in col+(2*i)])
transfer_data = np.array(eles*n_bunches)

T = coo_matrix((transfer_data, (mat_row, mat_col)), shape=(2*n_bunches, 2*n_bunches)).toarray() # Defines transfer matrix
#print(T)

# Part for obtaining the wake matrix
mat_row = np.array([r for i in range(1, 2*n_bunches, 2) for c in range(0, 2*n_bunches, 2) for r in [i] if i > c], dtype=np.int32)
mat_col = np.array([c for i in range(1, 2*n_bunches, 2) for c in range(0, 2*n_bunches, 2) for r in [i] if i > c], dtype=np.int32)

# Part for only driving wakes
wake_data = np.zeros((np.sum([i+1 for i in range(n_bunches)])), dtype=np.float64)
index = 0;
for (r,c) in zip(mat_row, mat_col):
	if r - c == 1:
		wake_data[index] = 0.
	else:
		wake_data[index] = dip_Xnew[np.searchsorted(z_new, z_new[int((r-1)/2)]-z_new[int(c/2)])] # Generates the dipolar wake term
	index = index + 1

W = coo_matrix((wake_data, (mat_row, mat_col)), shape=(2*n_bunches, 2*n_bunches)).toarray() # Defines wake matrix
#print(W)

I = np.identity(2*n_bunches)
full_mat = (I + W)@T # Addition of the two matrices
[eig_val, eig_vec] = np.linalg.eig(full_mat) # Function finds Eigenvalues and Eigenvectors
	
mu_ = np.array([np.arccos(0.5*np.real((eig_val[2*i] + eig_val[1+(2*i)])/np.sqrt((eig_val[2*i] * eig_val[1+(2*i)])))) for i in range(n_bunches)]) # Finding tunes
l_im = np.array([(0.5*np.imag((eig_val[2*i] - eig_val[1+(2*i)]))) for i in range(n_bunches)]) # Imaginary part of the eigenvalues

ind_sorted = np.argsort(-l_im) # Sort imaginary part
im_sorted = np.take(l_im, ind_sorted)

modes = [i for i in range(n_bunches)]
Q_ = mu_/(2*np.pi)
re_sorted = np.take(Q_, ind_sorted) # Arrange according to sorted imaginary part

plt.figure(2)
plt.scatter(modes, re_sorted,color='b',label='Only driving')
plt.ylabel('${\mu}/{2\pi}$')
plt.xlabel('modes')

# Part for only detuning
wake_data = np.zeros((np.sum([i+1 for i in range(n_bunches)])), dtype=np.float64)
index = 0;
for (r,c) in zip(mat_row, mat_col):
	if r - c == 1:
		W_quad = [quad_Xnew[np.searchsorted(z_new, z_new[int((r-1)/2)]-z_new[w_q])] for w_q in range(int(((r-1)/2)))] # Generates list with the different terms to be summed for quadrupolar wake
		wake_data[index] = np.sum(W_quad)
		#wake_data[index] = 0. #Uncomment this line for no detuning
	index = index + 1

W = coo_matrix((wake_data, (mat_row, mat_col)), shape=(2*n_bunches, 2*n_bunches)).toarray() # Defines wake matrix
#print(W)

I = np.identity(2*n_bunches)
full_mat = (I + W)@T # Addition of the two matrices
[eig_val, eig_vec] = np.linalg.eig(full_mat) # Function finds Eigenvalues and Eigenvectors
	
mu_ = np.array([np.arccos(0.5*np.real((eig_val[2*i] + eig_val[1+(2*i)])/np.sqrt((eig_val[2*i] * eig_val[1+(2*i)])))) for i in range(n_bunches)]) # Finding tunes
l_im = np.array([(0.5*np.imag((eig_val[2*i] - eig_val[1+(2*i)]))) for i in range(n_bunches)]) # Imaginary part of the eigenvalues

ind_sorted = np.argsort(l_im) # Sort imaginary part
im_sorted = np.take(l_im, ind_sorted)

modes = [i for i in range(n_bunches)]
Q_ = mu_/(2*np.pi)
re_sorted = np.take(Q_, ind_sorted) # Arrange according to sorted imaginary part

plt.figure(2)
plt.scatter(modes, re_sorted,color='g',label='Only detuning')
plt.ylabel('${\mu}/{2\pi}$')
plt.xlabel('modes')

# Part for driving + detuning
wake_data = np.zeros((np.sum([i+1 for i in range(n_bunches)])), dtype=np.float64)
index = 0;
for (r,c) in zip(mat_row, mat_col):
	if r - c == 1:
		W_quad = [quad_Xnew[np.searchsorted(z_new, z_new[int((r-1)/2)]-z_new[w_q])] for w_q in range(int(((r-1)/2)))] # Generates list with the different terms to be summed for quadrupolar wake
		wake_data[index] = np.sum(W_quad)
		#wake_data[index] = 0. #Uncomment this line for no detuning case
	else:
		wake_data[index] = dip_Xnew[np.searchsorted(z_new, z_new[int((r-1)/2)]-z_new[int(c/2)])] # Generates the dipolar wake term
	index = index + 1

W = coo_matrix((wake_data, (mat_row, mat_col)), shape=(2*n_bunches, 2*n_bunches)).toarray() # Defines wake matrix
#print(W)

I = np.identity(2*n_bunches)
full_mat = (I + W)@T # Addition of the two matrices
[eig_val, eig_vec] = np.linalg.eig(full_mat) # Function finds Eigenvalues and Eigenvectors

mu_ = np.array([np.arccos(0.5*np.real((eig_val[2*i] + eig_val[1+(2*i)])/np.sqrt((eig_val[2*i] * eig_val[1+(2*i)])))) for i in range(n_bunches)]) # Finding tunes
#l_im = np.array([(0.5*np.imag((eig_val[2*i] - eig_val[1+(2*i)]))) for i in range(n_bunches)]) # Imaginary part of the eigenvalues
l_im = np.array([(np.imag((eig_val[2*i]))) for i in range(n_bunches)]) # Imaginary part of the eigenvalues

ind_sorted = np.argsort(l_im) # Sort imaginary part
im_sorted = np.take(l_im, ind_sorted)

modes = [i for i in range(n_bunches)]
Q_ = mu_/(2*np.pi)
re_sorted = np.take(Q_, ind_sorted) # Arrange according to sorted imaginary part

plt.figure(2)
plt.scatter(modes, re_sorted,color='r',label='Driving + detuning')
plt.ylabel('${\mu}/{2\pi}$')
plt.xlabel('modes')

plt.legend()
plt.show()


