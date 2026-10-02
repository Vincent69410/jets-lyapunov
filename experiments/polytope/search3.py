import numpy as np, cvxpy as cp, sys, time, warnings
warnings.filterwarnings('ignore')
src=open('search.py').read()
exec(src.split("def level(")[0])
exec("def simplex_grid"+src.split("def simplex_grid")[1].split("n=int(sys.argv[2])")[0])
seed=int(sys.argv[1]); n=int(sys.argv[2]); m=int(sys.argv[3]); rng=np.random.default_rng(seed)
grid=simplex_grid(m,400 if m==2 else 60)
def status(verts,N,eps=1e-3):
    n=verts[0].shape[0]; mm=n*(N+1)
    P=cp.Variable((mm,mm),symmetric=True); G=cp.Variable((n*(N+2),mm)); G0=cp.Variable((mm,n*N)) if N>0 else None
    E0=np.zeros((n,mm)); E0[:,:n]=np.eye(n); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for A in verts:
        Nm=Nmat(A,N,n); cons.append(Dop(P,N,n)+G@Nm+(G@Nm).T<<-eps*np.eye(n*(N+2)))
        if N>0: N0=Nmat(A,N-1,n); cons.append(P+G0@N0+(G0@N0).T>>E0.T@E0)
        else: cons.append(P>>E0.T@E0)
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL); return pr.status
    except Exception: return 'fail'
def family(b,S):
    J=-np.eye(n)
    for i in range(n-1): J[i,i+1]=b
    return [Si@J@np.linalg.inv(Si) for Si in S]
t=time.time(); found=0
for trial in range(400):
    S=[]
    while len(S)<m:
        Si=rng.standard_normal((n,n))
        if np.linalg.cond(Si)<4: S.append(Si)
    if robust_margin(family(0.05,S),grid)>-0.05: continue
    lo,hi=0.05,8.
    for _ in range(25):
        mid=(lo+hi)/2
        if robust_margin(family(mid,S),grid)<=-0.05: lo=mid
        else: hi=mid
    if lo>7.9: continue
    verts=family(lo,S)
    if max(np.abs(V).max() for V in verts)>30: continue
    st=[status(verts,N) for N in (0,1,2)]
    if st[0]=='infeasible' and st[1]=='infeasible' and st[2]=='optimal':
        found+=1; print("FOUND n=%d m=%d seed=%d trial=%d b=%.3f maxabs=%.1f"%(n,m,seed,trial,lo,max(np.abs(V).max() for V in verts)),flush=True)
        np.save('ex_%d_%d_%d_%d.npy'%(n,m,seed,trial),np.array(verts))
        if found>=4: break
    if trial%40==0: print("trial",trial,"b=%.2f"%lo,st,"%.0fs"%(time.time()-t),flush=True)
print("done found",found)
