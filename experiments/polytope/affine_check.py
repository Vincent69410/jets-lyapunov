import numpy as np, cvxpy as cp, warnings, sys; warnings.filterwarnings('ignore')
sys.argv=['x','24','3','3']; exec(open('margins.py').read().split("for N in (0,1,2):")[0])
V=list(np.load('example_N2_final.npy')); n=3; grid=simplex_grid(3,10)
def best_margin_poly(deg,K=10,bound=1e3):
    # P(alpha) homogeneous of degree deg in alpha (all monomials), exact conditions on the grid
    import itertools
    monos=[k for k in itertools.product(range(deg+1),repeat=3) if sum(k)==deg]
    Ps={k:cp.Variable((n,n),symmetric=True) for k in monos}; gam=cp.Variable()
    cons=[]
    for k in monos: cons+=[Ps[k]<<bound*np.eye(n),Ps[k]>>-bound*np.eye(n)]
    for al in simplex_grid(3,K):
        A=sum(a*Vv for a,Vv in zip(al,V)); Pa=sum(np.prod([al[i]**k[i] for i in range(3)])*Ps[k] for k in monos)
        cons.append(Pa>>np.eye(n)); cons.append(Pa@A+A.T@Pa<<-gam*np.eye(n))
    pr=cp.Problem(cp.Maximize(gam),cons); pr.solve(solver=cp.CLARABEL); return gam.value,pr.status
for deg in (1,2):
    print("affine/HPD degree %d, exact grid conditions: gamma* = %.4f [%s]"%((deg,)+best_margin_poly(deg)))
print("level 1 grid with bound 1e6:",best_margin_grid(1,10) if False else "")
# level-1 grid margin with a much larger bound on P
def best_margin_grid_b(N,K,bound):
    mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); gam=cp.Variable(); cons=[P<<bound*np.eye(mm),P>>-bound*np.eye(mm)]
    for al in simplex_grid(3,K):
        A=sum(a*Vv for a,Vv in zip(al,V)); G1=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+1)]); G2=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+2)])
        cons.append(G1.T@P@G1>>np.eye(n)); cons.append(G2.T@Dop(P,N,n)@G2<<-gam*np.eye(n))
    pr=cp.Problem(cp.Maximize(gam),cons); pr.solve(solver=cp.CLARABEL); return gam.value,pr.status
for b in (1e3,1e6):
    print("level 1 grid, ||P||<=%g: gamma* = %s"%((b,)+tuple(str(x) for x in best_margin_grid_b(1,10,b))))
