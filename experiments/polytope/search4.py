import numpy as np, cvxpy as cp, warnings, sys, time; warnings.filterwarnings('ignore')
seed=int(sys.argv[1]); n=int(sys.argv[2]); m=int(sys.argv[3])
sys.argv=['x',str(seed),str(n),str(m)]; exec(open('search3.py').read().split("t=time.time(); found=0")[0])
exec(open('validate.py').read().split("for f in [")[0].split("exec(open('search3.py')")[0])  # nothing
def grid_level(verts,N,grid,eps=1e-3):
    n=verts[0].shape[0]; mm=n*(N+1); P=cp.Variable((mm,mm),symmetric=True); cons=[P<<1e3*np.eye(mm),P>>-1e3*np.eye(mm)]
    for al in grid:
        A=sum(a*V for a,V in zip(al,verts)); G1=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+1)]); G2=np.vstack([np.linalg.matrix_power(A,k) for k in range(N+2)])
        cons.append(G1.T@P@G1>>eps*np.eye(n)); cons.append(G2.T@Dop(P,N,n)@G2<<-eps*np.eye(n))
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL); return pr.status
    except Exception: return 'fail'
rng=np.random.default_rng(seed); gridc=simplex_grid(m,400 if m==2 else 60); gridl=simplex_grid(m,15 if m==2 else 8)
t=time.time(); found=0
for trial in range(600):
    S=[]
    while len(S)<m:
        Si=rng.standard_normal((n,n))
        if np.linalg.cond(Si)<6: S.append(Si)
    if robust_margin(family(0.05,S),gridc)>-0.02: continue
    lo,hi=0.05,20.
    for _ in range(25):
        mid=(lo+hi)/2
        if robust_margin(family(mid,S),gridc)<=-0.02: lo=mid
        else: hi=mid
    if lo>19.9: continue
    verts=family(lo,S); s=max(np.abs(A).max() for A in verts); verts=[A/s for A in verts]
    g1=grid_level(verts,1,gridl)
    if g1!='infeasible': 
        if trial%40==0: print("trial",trial,"b=%.2f"%lo,"grid L1:",g1,"%.0fs"%(time.time()-t),flush=True)
        continue
    v2=status(verts,2); g2=grid_level(verts,2,gridl)
    print("candidate trial %d b=%.2f: grid L1=%s, vertex L2=%s, grid L2=%s"%(trial,lo,g1,v2,g2),flush=True)
    if v2=='optimal':
        found+=1; np.save('exN2_%d_%d_%d_%d.npy'%(n,m,seed,trial),np.array(verts))
        if found>=3: break
print("done found",found)
