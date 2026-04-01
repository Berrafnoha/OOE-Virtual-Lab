import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS
# -----------------------------
A = 0.05       # wave amplitude (small for smooth caustics)
kx = 4*np.pi   # wave number x
ky = 4*np.pi   # wave number y
n1 = 1.0       # air
n2 = 1.33      # water
N = 600        # grid resolution
floor_z = 1.0  # distance from water surface to pool floor

# -----------------------------
# 2. CREATE WAVY WATER SURFACE
# -----------------------------
x = np.linspace(0, 2, N)
y = np.linspace(0, 2, N)
X, Y = np.meshgrid(x, y)

# Water surface: sum of sine waves
h = A*np.sin(kx*X) + A*np.sin(ky*Y)

# Surface gradients
hx, hy = np.gradient(h, x[1]-x[0], y[1]-y[0])
normals = np.dstack((-hx, -hy, np.ones_like(h)))
normals /= np.linalg.norm(normals, axis=2)[:,:,None]

# -----------------------------
# 3. MATH TEST (Snell's law)
# -----------------------------
theta_inc_deg = 30  # degrees of incidence
theta_inc = np.radians(theta_inc_deg)
theta_refr = np.arcsin(n1/n2 * np.sin(theta_inc))
theta_refr_deg = np.degrees(theta_refr)
print(f"Math Test: θ₁ = {theta_inc_deg}° → θ₂ ≈ {theta_refr_deg:.2f}°")

# -----------------------------
# 4. CAST RAYS TO FLOOR
# -----------------------------
dz = floor_z / normals[:,:,2]
X_floor = X + normals[:,:,0]*dz
Y_floor = Y + normals[:,:,1]*dz

# -----------------------------
# 5. COMPUTE INTENSITY (CAUSTICS)
# -----------------------------
H, xedges, yedges = np.histogram2d(X_floor.ravel(), Y_floor.ravel(),
                                   bins=N, range=[[0,2],[0,2]])
H = H / np.max(H)  # normalize

# -----------------------------
# 6. PLOT TOP-DOWN VIEW
# -----------------------------
plt.figure(figsize=(8,8))
plt.imshow(H.T, extent=[0,2,0,2], origin='lower', cmap='viridis')
plt.title("Top-Down View: Pool Floor Caustics 🌊", fontsize=16)
plt.axis('off')
plt.show()

    