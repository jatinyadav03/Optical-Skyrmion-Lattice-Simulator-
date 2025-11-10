# optical_skyrmion_simulator.py
# Requires: numpy, matplotlib, scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ---- Parameters (you can change these) ----
wavelength = 0.636  # micrometers (636 nm, as in the paper)
k0 = 2*np.pi / wavelength

# choose transverse wavevector slightly above k0 (plasmonic example in paper)
k_parallel = 1.038 * k0

# magnitude of evanescent axial decay (|kz|)
kz_mag = np.sqrt(k_parallel**2 - k0**2)  # positive real

E0 = 1.0  # amplitude normalization
z = 0.0   # evaluate at surface z=0 (unit cell is in x-y plane)

# angles q = { -pi/3, 0, +pi/3 } used in Eq. 2/3
qs = np.array([-np.pi/3.0, 0.0, np.pi/3.0])

# Spatial grid over a single unit cell
# The spatial periodicity of cos(k_parallel * ... ) is 2*pi/k_parallel.
# Use domain [-L/2, L/2] with L = 2*pi / k_parallel (a single period).
L = 2*np.pi / k_parallel
Nx = Ny = 400  # grid resolution (increase for better accuracy)
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

# ---- Construct fields using Eqs. (2) and (3) from the paper ----
# Eq. 2: E_z = E0 * exp(-|kz| z) * sum_q cos( k_par * (cos q * x + sin q * y) )
phi_q = [k_parallel * (np.cos(q)*X + np.sin(q)*Y) for q in qs]

E_z = np.zeros_like(X)
for phi in phi_q:
    E_z += np.cos(phi)
E_z *= E0 * np.exp(-kz_mag * z)

# Eq. 3 (real part for evanescent TM waves)
# [E_x, E_y]^T = -E0 * ( j*|kz| / k_parallel ) * e^{-j|kz| z} * sum_q [cos q; sin q] * sin(phi_q)
# Taking real part (derivation in note) yields:
E_x = np.zeros_like(X)
E_y = np.zeros_like(X)
pref = -E0 * (kz_mag / k_parallel) * np.exp(-kz_mag * z)  # real prefactor from Re(...)
for q, phi in zip(qs, phi_q):
    E_x += np.cos(q) * np.sin(phi)
    E_y += np.sin(q) * np.sin(phi)
E_x *= pref
E_y *= pref

# Compose 3-component field
E_vec = np.stack([E_x, E_y, E_z], axis=-1)  # shape (Ny, Nx, 3)

# ---- Unit vector e = E / |E| (careful near zeros) ----
amp = np.linalg.norm(E_vec, axis=-1)
eps = 1e-12
amp_safe = amp + eps
e_vec = E_vec / amp_safe[..., None]  # shape (Ny, Nx, 3)

# ---- Compute spatial derivatives of e with central differences ----
# Use numpy.gradient: returns derivatives along axis order (y,x) because meshgrid used 'xy'
de_dy = np.gradient(e_vec, dy, axis=0)  # derivative w.r.t y
de_dx = np.gradient(e_vec, dx, axis=1)  # derivative w.r.t x

# ---- Compute skyrmion density s = e · (∂_x e × ∂_y e) ----
cross = np.cross(de_dx, de_dy)  # shape (Ny,Nx,3)
s_map = np.einsum('ijk,ijk->ij', e_vec, cross)  # dot product across vector components

# ---- Integrate s over unit cell to obtain skyrmion number S = (1/(4*pi)) * ∫ s dA ----
# Do a simple midpoint/rectangular integral
S_numeric = (1.0 / (4.0*np.pi)) * np.sum(s_map) * dx * dy

# ---- Print results ----
print("Parameters:")
print(f" wavelength = {wavelength} µm, k0 = {k0:.4f} µm^-1")
print(f" k_parallel = {k_parallel:.4f} µm^-1  (ratio k_par/k0 = {k_parallel/k0:.6f})")
print(f" kz_mag = {kz_mag:.4f} µm^-1")
print()
print(f"Computed skyrmion number S (numerical over one unit cell) = {S_numeric:.6f}")

# ---- Visualizations ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.flat

# 1) E_z (colormap)
im0 = ax[0].imshow(E_z, extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
ax[0].set_title("Axial (out-of-plane) E_z")
ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")
fig.colorbar(im0, ax=ax[0], orientation='vertical')

# 2) In-plane amplitude sqrt(E_x^2 + E_y^2)
inplane_amp = np.sqrt(E_x**2 + E_y**2)
im1 = ax[1].imshow(inplane_amp, extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
ax[1].set_title("Transverse (in-plane) amplitude |E_xy|")
ax[1].set_xlabel("x (µm)"); ax[1].set_ylabel("y (µm)")
fig.colorbar(im1, ax=ax[1], orientation='vertical')

# 3) Vector representation (quiver) of in-plane unit vectors (subsample for clarity)
skip = max(1, Nx//30)
ux = E_x / (inplane_amp + eps)
uy = E_y / (inplane_amp + eps)
Xq = X[::skip, ::skip]; Yq = Y[::skip, ::skip]
Uq = ux[::skip, ::skip]; Vq = uy[::skip, ::skip]
ax[2].quiver(Xq, Yq, Uq, Vq, scale=25)
ax[2].set_title("In-plane field direction (subsampled)")
ax[2].set_xlabel("x (µm)"); ax[2].set_ylabel("y (µm)")
ax[2].set_xlim([x[0], x[-1]]); ax[2].set_ylim([y[0], y[-1]])

# 4) Skyrmion number density map s(x,y)
im3 = ax[3].imshow(s_map, extent=[x[0], x[-1], y[0], y[-1]], origin='lower')
ax[3].set_title("Skyrmion number density s(x,y)")
ax[3].set_xlabel("x (µm)"); ax[3].set_ylabel("y (µm)")
fig.colorbar(im3, ax=ax[3], orientation='vertical')

plt.tight_layout()
plt.show()
