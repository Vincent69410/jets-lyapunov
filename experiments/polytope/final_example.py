import numpy as np, cvxpy as cp, warnings, sys; warnings.filterwarnings('ignore')
sys.argv=['x','24','3','3']; exec(open('search4.py').read().split("rng=np.random.default_rng(seed)")[0])
V=[10*np.round(A,3) for A in np.load('exN2_3_3_24_85.npy')]
for i,A in enumerate(V): print("A%d =\n"%(i+1),A)
g=simplex_grid(3,200); print("max Re eig over simplex (200-grid): %.4f"%robust_margin(V,g))
print("eigenvalues of vertices:",[np.round(np.linalg.eigvals(A),3) for A in V])
def grid_level_scs(verts,N,grid,eps=1e-3):
    n=3; mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for al in grid:
        A=sum(a*Vv for a,Vv in zip(al,verts)); G1=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+1)]); G2=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+2)])
        cons.append(G1.T@P@G1>>eps*np.eye(n)); cons.append(G2.T@Dop(P,N,n)@G2<<-eps*np.eye(n))
    pr=cp.Problem(cp.Minimize(0),cons); pr.solve(solver=cp.SCS,max_iters=50000,eps=1e-9); return pr.status
print("level 0 vertex:",status(V,0),"| grid:",grid_level(V,0,simplex_grid(3,8)))
print("level 1 vertex:",status(V,1),"| grid (Clarabel):",grid_level(V,1,simplex_grid(3,12)),"| grid (SCS):",grid_level_scs(V,1,simplex_grid(3,12)))
print("level 2 vertex:",status(V,2),"| grid:",grid_level(V,2,simplex_grid(3,8)))
np.save('example_N2_final.npy',np.array(V))
