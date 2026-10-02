import numpy as np, cvxpy as cp, warnings, sys; warnings.filterwarnings('ignore')
sys.argv=['x','0','3','3']; exec(open('search3.py').read().split("t=time.time(); found=0")[0])
def grid_level(verts,N,grid,eps=1e-3):
    # necessary condition: exact level-N conditions on a grid of alpha (no multipliers)
    n=verts[0].shape[0]; mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for al in grid:
        A=sum(a*V for a,V in zip(al,verts)); G1=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+1)]); G2=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+2)])
        cons.append(G1.T@P@G1>>eps*np.eye(n)); cons.append(G2.T@Dop(P,N,n)@G2<<-eps*np.eye(n))
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL); return pr.status
    except Exception: return 'fail'
for f in ['found2_3_3_4_39.npy','found2_3_3_4_26.npy']:
    V=list(np.load(f)); s=max(np.abs(A).max() for A in V); Vs=[A/s for A in V]   # time scaling
    print(f,"scaled by 1/%.2f"%s)
    print("  vertex-LMI statuses N=0,1,2 (scaled):",[status(Vs,N) for N in (0,1,2)])
    g=simplex_grid(3,12)
    print("  gridded exact conditions N=0,1,2 (scaled, %d points):"%len(g),[grid_level(Vs,N,g) for N in (0,1,2)],flush=True)
