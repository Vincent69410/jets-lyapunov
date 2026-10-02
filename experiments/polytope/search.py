import numpy as np, cvxpy as cp, warnings, itertools, time, sys
warnings.filterwarnings('ignore')
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def solve(cons):
    pr=cp.Problem(cp.Minimize(0),cons)
    try:
        pr.solve(solver=cp.CLARABEL)
        if pr.status=='optimal': return True
        if pr.status=='infeasible': return False
    except Exception: pass
    try:
        pr.solve(solver=cp.SCS,max_iters=30000,eps=1e-8); return pr.status=='optimal'
    except Exception: return False
def Dop(P,N,n):
    m=n*(N+1); Pi0=np.hstack([np.eye(m),np.zeros((m,n))]); Pi1=np.hstack([np.zeros((m,n)),np.eye(m)])
    M=Pi0.T@P@Pi1; return M+M.T
def Nmat(A,N,n):
    rows=[]
    for k in range(N+1):
        r=np.zeros((n,n*(N+2))); r[:,n*(k+1):n*(k+2)]=np.eye(n); r[:,n*k:n*(k+1)]=-A; rows.append(r)
    return np.vstack(rows)
def level(verts,N,eps=1e-3):
    n=verts[0].shape[0]; m=n*(N+1)
    P=cp.Variable((m,m),symmetric=True); G=cp.Variable((n*(N+2),m)); G0=cp.Variable((m,n*N)) if N>0 else None
    E0=np.zeros((n,m)); E0[:,:n]=np.eye(n)
    cons=[P<<1e4*np.eye(m),P>>-1e4*np.eye(m)]
    for A in verts:
        Nm=Nmat(A,N,n); cons.append(Dop(P,N,n)+G@Nm+(G@Nm).T<<-eps*np.eye(n*(N+2)))
        if N>0:
            N0=Nmat(A,N-1,n); cons.append(P+G0@N0+(G0@N0).T>>E0.T@E0)
        else: cons.append(P>>E0.T@E0)
    return solve(cons)
def simplex_grid(m,K):
    if m==2: return [np.array([a,1-a]) for a in np.linspace(0,1,K)]
    pts=[]
    for i in range(K+1):
        for j in range(K+1-i):
            pts.append(np.array([i,j,K-i-j])/K)
    return pts
def robust_margin(verts,grid):
    # max real part of eigenvalues over the grid (negative = robustly Hurwitz on the grid)
    return max(np.linalg.eigvals(sum(a*V for a,V in zip(al,verts))).real.max() for al in grid)
n=int(sys.argv[2]) if len(sys.argv)>2 else 2; m=int(sys.argv[3]) if len(sys.argv)>3 else 2
grid=simplex_grid(m,200 if m==2 else 40)
t=time.time(); found=0
for trial in range(400):
    M=[rng.standard_normal((n,n)) for _ in range(m)]
    # shift sigma: smallest sigma with robust margin <= -0.02
    lo,hi=-5.,10.
    for _ in range(30):
        mid=(lo+hi)/2
        if robust_margin([Mi-mid*np.eye(n) for Mi in M],grid)<=-0.02: hi=mid
        else: lo=mid
    verts=[Mi-hi*np.eye(n) for Mi in M]
    r=[level(verts,N) for N in (0,1,2)]
    if (not r[1]) and r[2]:
        found+=1
        print("FOUND (n=%d,m=%d) trial %d: levels 0,1,2 = %s ; margin=%.3f"%(n,m,trial,r,robust_margin(verts,grid)))
        np.save('found_%d_%d_%d_%d.npy'%(n,m,int(sys.argv[1]) if len(sys.argv)>1 else 0,trial),np.array(verts))
        if found>=3: break
    if trial%50==0: print("trial",trial,"levels",r,"%.0fs"%(time.time()-t),flush=True)
print("done, found",found)
