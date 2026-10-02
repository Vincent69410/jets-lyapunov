"""Table 1 of the article: for the polytope co{A1, A2(a)}, compare
  - the order N required by the hypotheses of Theorem 7 (exact constants c=a, lambda=1),
  - the smallest N for which the universal form V_{T,N} works for some T (grid search),
  - the smallest level of the vertex LMIs with slack variables (free P).
Requires numpy, scipy, mpmath, cvxpy (Clarabel). Runtime: a few minutes.
"""
import numpy as np, mpmath as mp, warnings
warnings.filterwarnings('ignore')
from scipy.linalg import expm
import polytope_levels as pl          # jet_finsler(a, N)

mp.mp.dps = 30
A1 = np.array([[-1., -1.], [1., -1.]])
def A2(a): return np.array([[-1., -a], [1 / a, -1.]])
def fam(a, K=21): return [al * A1 + (1 - al) * A2(a) for al in np.linspace(0, 1, K)]

def M_max(A, T, N):
    """largest eigenvalue of sym(A^T P_{T,N}(A) + P_{T,N}(A) A), high precision"""
    h = lambda i, j: mp.mpf(T) ** (i + j + 1) / ((i + j + 1) * mp.factorial(i) * mp.factorial(j))
    Am = mp.matrix(A.tolist()); pw = [mp.eye(2)]
    for k in range(N): pw.append(pw[-1] * Am)
    P = mp.zeros(2)
    for i in range(N + 1):
        for j in range(N + 1): P += h(i, j) * (pw[i].T * pw[j])
    M = Am.T * P + P * Am; M = (M + M.T) / 2
    return float(max(mp.eig(M)[0]).real)

def N_univ(a, Ts=np.linspace(0.3, 3.5, 17), Nmax=40):
    best = None
    for T in Ts:
        for N in range(0, Nmax + 1):
            if best is not None and N >= best: break
            if all(M_max(A, T, N) < 0 for A in fam(a, 11)):
                best = N; break
    return best

def N_thm7(a):
    F = fam(a, 41); ts = np.linspace(0, 12, 1201)
    c = max(np.linalg.norm(expm(A * t), 2) * np.exp(t) for A in F for t in ts); lam = 1.0
    an = max(np.linalg.norm(A, 2) for A in F); T = np.log(2 * c ** 2) / (2 * lam); N = 0
    while mp.e ** (2 * T * an) * mp.mpf(T * an) ** (N + 1) / mp.factorial(N + 1) > mp.mpf(1) / 16: N += 1
    return N, c, T

def N_lmi(a):
    for N in range(0, 4):
        if pl.jet_finsler(a, N): return N
    return None

if __name__ == "__main__":
    print("%6s %8s %8s %10s %14s %6s" % ("a", "c", "T", "N_Thm7", "N_univ(bestT)", "N_LMI"))
    for a in [4, 6, 8, 12, 20, 50]:
        n7, c, T = N_thm7(a); nu = N_univ(a); nl = N_lmi(a)
        print("%6g %8.2f %8.2f %10d %14s %6s" % (a, c, T, n7, str(nu), str(nl)), flush=True)
