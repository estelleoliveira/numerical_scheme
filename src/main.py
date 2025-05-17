import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

#PAramètres globaux
L = 2.0
T = 0.1
nx = 315
x = np.linspace(0, L, nx)
dx = x[1] - x[0]

# Define nt values for each method
nt_explicit = int(T / (0.500 * dx**2))     #CFL (r <= 0.5)
nt_implicit = int(T / (0.500 * dx**2))
nt_cn = int(T / (0.500* dx**2))

#compute dt et r 
dt_explicit = T / nt_explicit
r_explicit = dt_explicit / dx**2

dt_implicit = T / nt_implicit
r_implicit = dt_implicit / dx**2

dt_cn = T / nt_cn
r_cn = dt_cn / dx**2


# On défini la condition intiale f(x) = sin(2*pi*x)
def initial_condition(x, n_mode=2):
    return np.sin(n_mode * np.pi * x)

def exact_solution(x, t, n_mode=2):
    return np.exp(- (n_mode * np.pi)**2 * t) * np.sin(n_mode * np.pi * x)

def erreur_L2(u_num, u_exact, dx):
    return np.sqrt(np.sum((u_num - u_exact)**2) * dx)

#méthode explicite nous permet une résolution direct via le schéma numérique déterminé
def euler_explicit(u0, r, nt):
    """
    schéma explicite : u^{n+1} = u^n + r (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)
    condition CFL nécessaire pour la stabilité : r <= 0.5
    r = dt/(dx**2)
    ordre 2 en espace, ordre 1 en temps
    """
    u = u0.copy()
    start_explicit = time.time()
    for _ in range(nt):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u[0] = 0
        u[-1] = 0

    end_explicit = time.time()
    print("Temps pour modèle explicite :", end_explicit - start_explicit)
    return u

#méthode impicite nous oblige à résoudre avec implémentation d'une matrice A
def euler_implicit(u0, r, nt):
    """
    schéma implicite : u_i^{n+1} - r * (u_{i+1}^{n+1} - 2 u_i^{n+1} + u_{i-1}^{n+1}) = u_i^n
    non conditionné par CFL (stable quel que soit r)
    r = dt/(dx**2)
    ordre 2 en espace, ordre 1 en temps
    """
    u = u0.copy()
    main_diag = (1 + 2*r) * np.ones(nx-2)
    off_diag = -r * np.ones(nx-3)
    A = diags([main_diag, off_diag, off_diag], [0, -1, 1])
    A_csc = A.tocsc()
    start_implicit = time.time()
    for _ in range(nt):
        b = u[1:-1]
        u[1:-1] = spsolve(A_csc, b)
        u[0] = 0
        u[-1] = 0
    end_implicit = time.time()
    print("Temps pour modèle implicite :", end_implicit - start_implicit)
    return u

def crank_nicholson(u0, r, nt):
    """
    schéma crank-nicholson : u_i^{n+1} - (r/2) * (u_{i+1}^{n+1} - 2 u_i^{n+1} + u_{i-1}^{n+1}) = u_i^n + (r/2) * (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)
    moyenne du schéma implicite et explicite, il nous permet d'avoir une solution stable et très proche de la solution exacte
    non conditionné par CFL (stable quel que soit r)
    r = dt/(dx**2)
    ordre 2 en espace, ordre 2 en temps
    """
    u = u0.copy()
    main_diag_A = (1 + r) * np.ones(nx-2) #diag principale 1 + delta(t)/(delta(x))**2
    off_diag_A = (-r/2) * np.ones(nx-3) #diag sup et sous -delta(t)/2(delta(x))**2
    A = diags([main_diag_A, off_diag_A, off_diag_A], [0, -1, 1])
    
    main_diag_B = (1 - r) * np.ones(nx-2)   #diag principale 1 - delta(t)/(delta(x))**2
    off_diag_B = (r/2) * np.ones(nx-3)  #diag sup et sous +delta(t)/2(delta(x))**2
    B = diags([main_diag_B, off_diag_B, off_diag_B], [0, -1, 1])
    
    A_csc = A.tocsc()
    B_csc = B.tocsc()
    start_crank_nicholson = time.time()
    for _ in range(nt):
        b = B_csc.dot(u[1:-1])  #calcul de B*un avec les conditions au bord fixées plus tard
        u[1:-1] = spsolve(A_csc, b) #resolution de A un+1 = B un 
        u[0] = 0    #conditions au bord
        u[-1] = 0   #conditions au bord
    end_crank_nicholson = time.time()
    print("Temps pour modèle de Crank-Nicholson :", end_crank_nicholson - start_crank_nicholson)
    return u

def plot_results(x, u_exact, u_exp, u_imp, u_cn, err_exp, err_imp, err_cn):
    plt.figure(figsize=(10, 7))

    plt.subplot(2,1,1)
    plt.plot(x, u_exact, 'k-', label='Exacte')
    plt.plot(x, u_exp, 'r--', label='Euler explicite')
    plt.plot(x, u_imp, 'b-.', label='Euler implicite')
    plt.plot(x, u_cn, 'g:', label='Crank-Nicholson')
    plt.title('Solutions à t = {:.3f}'.format(T))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(x, err_exp, 'r--', label='Erreur explicite')
    plt.plot(x, err_imp, 'b-.', label='Erreur implicite')
    plt.plot(x, err_cn, 'g:', label='Erreur Crank-Nicholson')
    plt.title('Erreur absolue')
    plt.xlabel('x')
    plt.ylabel('Erreur')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

def main():
    u0 = initial_condition(x, 2)
    u_exact = exact_solution(x, T, 2)

    u_exp = euler_explicit(u0, r_explicit, nt_explicit)
    u_imp = euler_implicit(u0, r_implicit, nt_implicit)
    u_cn = crank_nicholson(u0, r_cn, nt_cn)

    err_exp = np.abs(u_exp - u_exact)
    err_imp = np.abs(u_imp - u_exact)
    err_cn = np.abs(u_cn - u_exact)
    
    err_explicit = erreur_L2(u_exp, u_exact, dx)
    err_implicit = erreur_L2(u_imp, u_exact, dx)
    err_crank = erreur_L2(u_cn, u_exact, dx)
    
    print(f"r_explicit = {r_explicit:.3f} (doit être ≤ 0.5)")
    print(f"r_implicit = {r_implicit:.3f}")
    print(f"r_cn = {r_cn:.3f}")

    print(f"Erreurs L2 calculées au temps T={T}:")
    print(f" Euler explicite : {err_explicit:.6e} ")
    print(f" Euler implicite : {err_implicit:.6e} ")
    print(f" Crank-Nicholson : {err_crank:.6e} ")

    plot_results(x, u_exact, u_exp, u_imp, u_cn, err_exp, err_imp, err_cn)

    # Conservation de l'intégrale
    print("\nIntégrales (doivent décroître exponentiellement) :")
    print(f"Exacte : {np.trapz(np.abs(u_exact), x):.2e}")
    print(f"Explicite : {np.trapz(np.abs(u_exp), x):.2e}")

if __name__ == "__main__":
    main()