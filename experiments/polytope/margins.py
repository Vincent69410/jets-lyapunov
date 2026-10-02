import numpy as np, cvxpy as cp, warnings, sys; warnings.filterwarnings('ignore')
sys.argv=['x','24','3','3']; exec(open('search4.py').read().split("rng=np.random.default_rng(seed)")[0])
V=list(np.load('example_N2_final.npy')); n=3
def best_margin_grid(N,K):
    mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); gam=cp.Variable(); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for al in simplex_grid(3,K):
        A=sum(a*Vv for a,Vv in zip(al,V)); G1=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+1)]); G2=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+2)])
        cons.append(G1.T@P@G1>>np.eye(n)); cons.append(G2.T@Dop(P,N,n)@G2<<-gam*np.eye(n))
    pr=cp.Problem(cp.Maximize(gam),cons)
    for sv in (cp.CLARABEL,cp.SCS):
        try:
            pr.solve(solver=sv,**({'max_iters':50000,'eps':1e-8} if sv==cp.SCS else {}))
            if gam.value is not None: return gam.value,pr.status,str(sv)
        except Exception: pass
    return None,'fail',''
def best_margin_vertex(N):
    mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); gam=cp.Variable(); G=cp.Variable((n*(N+2),mm)); G0=cp.Variable((mm,n*N)) if N>0 else None
    E0=np.zeros((n,mm)); E0[:,:n]=np.eye(n); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm),G<<1e3,G>>-1e3] if False else [P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for A in V:
        Nm=Nmat(A,N,n); cons.append(Dop(P,N,n)+G@Nm+(G@Nm).T<<-gam*np.eye(n*(N+2)))
        if N>0: N0=Nmat(A,N-1,n); cons.append(P+G0@N0+(G0@N0).T>>E0.T@E0)
        else: cons.append(P>>E0.T@E0)
    pr=cp.Problem(cp.Maximize(gam),cons)
    for sv in (cp.CLARABEL,cp.SCS):
        try:
            pr.solve(solver=sv,**({'max_iters':50000,'eps':1e-8} if sv==cp.SCS else {}))
            if gam.value is not None: return gam.value,pr.status,str(sv)
        except Exception: pass
    return None,'fail',''
for N in (0,1,2):
    g,st,sv=best_margin_grid(N,10); print("level %d, grid (66 pts): best decrease margin gamma* = %s  [%s %s]"%(N,None if g is None else "%.4f"%g,st,sv),flush=True)
for N in (0,1,2):
    g,st,sv=best_margin_vertex(N); print("level %d, vertex LMIs: best decrease margin gamma* = %s  [%s %s]"%(N,None if g is None else "%.4f"%g,st,sv),flush=True)
