import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from io_utils import save_hdf5

use_gpu = True # Set to True for GPU acceleration

# ======== TOGGLE BETWEEN GPU AND CPU ========
if use_gpu:
    try:
        import cupy as cp
        print("Using GPU acceleration with CuPy")
        xp = cp
    except (ModuleNotFoundError, ImportError):
        print("CuPy not found, falling back to CPU")
        use_gpu = False
        xp = np
        print("Using CPU mode with NumPy")
else:
    xp = np
    print("Using CPU mode with NumPy")
    
# ================== PARAMETERS ==================
Lx = 2*xp.pi # Physical domain size in x
Ly = Lx # Physical domain size in y
dx = (2/512)*xp.pi # Spatial resolution in x
dy = dx # Spatial reolution in Y

u_noise = 0
v_noise = u_noise

xlength = int(Lx/dx) # Number of grid points in x
ylength = int(Ly/dy) # Add 1 to account for the 0 index


nt = 101 # Time steps, includes t=0
dt = 0.001 # Time step size
CFL_limit = 0.24 #nu = 0.4
nit = 100  # Pressure solver iterations

# Physical properties
rho = 0.01 # Density
nu = 0.4 # Kinematic viscosity

parameters = {
    "nu": nu,
    "rho": rho,
    "nt": nt,
    "dt": dt,
    "dx": dx,
    "dy": dy,
    "Lx": Lx,
    "Ly": Ly
}

q_vis = 4 # Visualisation downsampling
vmax = None # Colourmap range
vmin = None
levels = 32 # Colourmap levels

x = xp.linspace(0, Lx-dx, num=xlength) # Create grid
y = xp.linspace(0, Ly-dy, num=ylength)
X, Y = xp.meshgrid(x, y)

#awkward fix
vis_X, vis_Y = np.meshgrid(xp.linspace(0, Lx, num=xlength+1), xp.linspace(0, Ly, num=ylength+1))

# ================== HELPER FUNCTIONS ==================

def get_array(a):
    """Convert to numpy array if using GPU"""
    return cp.asnumpy(a) if use_gpu else a

def pad_array(a):
    """Applies periodic BC ghost cells to a 2D array"""
    padded_a = xp.zeros((xlength+2, ylength+2))
    
    padded_a[1:-1, 1:-1] = a
    padded_a[0, 1:-1] = a[-1] # top a -> bottom padded_a
    padded_a[-1, 1:-1] = a[0] # bottom a -> top padded_a
    padded_a[1:-1, 0] = a[:, -1] # right a -> left padded_a
    padded_a[1:-1, -1] = a[:, 0] # left a -> right padded_a
    
    padded_a[-1, -1] = a[0, 0]
    padded_a[0, 0] = a[-1, -1]
    padded_a[-1, 0] = a[0, -1]
    padded_a[0, -1] = a[-1, 0]
    
    return padded_a

def build_up_b(b, u, v, dt, dx, dy, rho):
    """Build RHS of Poisson equation"""
    
    dudx = (u[1:-1, 2:] - u[1:-1, :-2])/(2*dx)
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dx)
    dvdy = (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy)
    
    b[1:-1, 1:-1] = rho*(1/dt*(dudx + dvdy) - dudx**2 - 2*(dudy*dvdx) - dvdy**2)
    b = pad_array(b[1:-1, 1:-1])
    
    return b

def pressure_poisson(p, b, dx, dy, nit):
    """Solve pressure Poisson equation in nit iterations"""
    
    for _ in range(nit):
        pn = p.copy()
        
        p[1:-1, 1:-1] = (pn[1:-1, 2:] - pn[1:-1, :-2])*(dy**2)+(pn[2:, 1:-1] - pn[:-2, 1:-1])*(dx**2)
        p[1:-1, 1:-1] -= b[1:-1, 1:-1]*(dx**2)*(dy**2)
        p[1:-1, 1:-1] /= 2*(dx**2 + dy**2)
        p = pad_array(p[1:-1, 1:-1])
    
    return p

def a_fix(a):
    vis_a = xp.zeros((xlength+1, ylength+1))
    vis_a[:-1, :-1] = a
    vis_a[-1, :-1] = a[0] # bottom a -> top padded_a
    vis_a[:-1, -1] = a[:, 0] # left a -> right padded_a
    vis_a[-1, -1] = a[0, 0] #corner
    
    return vis_a


if __name__ == "__main__":
    # ================== MAIN INIT. ==================
    u = xp.zeros((xlength, ylength))
    v = xp.zeros((xlength, ylength))
    u = xp.sin(X)*xp.cos(Y) 
    v = -xp.cos(X)*xp.sin(Y) 
    
    u = pad_array(u)
    v = pad_array(v)
    
    un = xp.empty_like(u)
    vn = xp.empty_like(v)
    
    b = xp.zeros((xlength, ylength))
    b = pad_array(b)
    
    p = xp.zeros((xlength, ylength))
    p = pad_array(p)
    
    u_data = []
    v_data = []
    p_data = []
            
    xp.savez('grid.npz', X, Y)
    '''
    plt.contourf(get_array(vis_X), get_array(vis_Y), get_array(a_fix(p[1:-1, 1:-1])), alpha=0.6, cmap='jet',
                 vmax=vmax, vmin=vmin, levels=levels)
    plt.colorbar()
    plt.title(str(xlength)+'x'+str(ylength)+' mesh, t=0, dt='+str(dt))
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.quiver(get_array(vis_X)[::q_vis, ::q_vis], get_array(vis_Y)[::q_vis, ::q_vis],
               get_array(a_fix(u[1:-1, 1:-1]))[::q_vis, ::q_vis], get_array(a_fix(v[1:-1, 1:-1]))[::q_vis, ::q_vis])
    plt.show()'''
    
    # ================== MAIN SIMULATION LOOP ==================   
    for i in range(nt):  
        
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, u, v, dt, dx, dy, rho)
        p = pressure_poisson(p, b, dx, dy, nit)

        # ================== CALCULATION ================== 
        #u_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(un[1:-1, 2:] - un[1:-1, :-2]) + 
                  #vn[1:-1, 1:-1]*(dt/(2*dy))*(un[2:, 1:-1] - un[:-2, 1:-1]))
        
        u_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(un[1:-1, 2:] - un[1:-1, 1:-1]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(un[2:, 1:-1] - un[1:-1, 1:-1]))
        
        u_diff = (nu*((dt/dx**2)*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) + 
                      (dt/dy**2)*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])))
        
        u_pressure = dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, :-2])
        
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - u_conv - u_pressure + u_diff
    
        #v_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(vn[1:-1, 2:] - vn[1:-1, :-2]) + 
                  #vn[1:-1, 1:-1]*(dt/(2*dy))*(vn[2:, 1:-1] - vn[:-2, 1:-1]))
        
        v_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(vn[1:-1, 2:] - vn[1:-1, 1:-1]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(vn[2:, 1:-1] - vn[1:-1, 1:-1]))
        
        v_diff = (nu*((dt/dx**2)*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) + 
                      (dt/dy**2)*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
        
        v_pressure = dt/(2*rho*dy)*(p[2:, 1:-1] - p[:-2, 1:-1])
        
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - v_conv - v_pressure + v_diff
        
        u = pad_array(u[1:-1, 1:-1])
        v = pad_array(v[1:-1, 1:-1]) 
        
        # ================== DATA MANAGEMENT AND VISUALISATION ==================
        
        u_data.append(u[1:-1, 1:-1])
        v_data.append(v[1:-1, 1:-1])
        p_data.append(p[1:-1, 1:-1])
        
        if (i)%999==0:
            print(str(i)+"/"+str(nt))
            plt.contourf(get_array(vis_X), get_array(vis_Y), get_array(a_fix(p[1:-1, 1:-1])), alpha=0.6, cmap='jet',
                         vmax=vmax, vmin=vmin, levels=levels)
            plt.colorbar()
            plt.title(str(xlength)+'x'+str(ylength)+' mesh, t='+str(xp.round((i+1)*dt, 3))+', dt='+str(dt))
            plt.xlim(0, Lx)
            plt.ylim(0, Ly)
            plt.quiver(get_array(vis_X)[::q_vis, ::q_vis], get_array(vis_Y)[::q_vis, ::q_vis],
                       get_array(a_fix(u[1:-1, 1:-1]))[::q_vis, ::q_vis], get_array(a_fix(v[1:-1, 1:-1]))[::q_vis, ::q_vis])
            plt.show()

    print("Saving as HDF5...")

    #converts list of 2D np arrays to 3D np array
    #could be simplified by initializing arrays as np and appending?
    u_data_xp = xp.stack(u_data, axis=0)
    v_data_xp = xp.stack(v_data, axis=0)
    p_data_xp = xp.stack(p_data, axis=0)

    h5_time = timeit("save_hdf5(parameters, u_data_xp, v_data_xp, p_data_xp)", number=1, globals=globals())
    print(f'Executed in {h5_time} seconds.')

    plt.close()

# Investigating HDF5 for saving parameters and data
# Speed during saving must be considered
# Other compressions must be considered
# NP only hands out references to arrays so modularity is useful