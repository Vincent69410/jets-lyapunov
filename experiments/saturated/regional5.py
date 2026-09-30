import numpy as np, cvxpy as cp, itertools, time
exec(open('regional2.py').read().split('def feasible')[0])
R0=np.load('regional3_res.npz')['PQ'][:2,:2]   # classical ellipsoid shape as reference
S2=np.zeros((3,3)); S2[:2,:2]=np.outer(k,k); S2[:2,2]=-u0*k; S2[2,:2]=-u0*k; S2[2,2]=u0**2   # (Kx-u0)^2
def lin(hvec,h0):
    # z^T S z = (Kx-u0)*(h0 + h^T x)
    S=np.zeros((3,3),dtype=object)
    Sk=np.zeros((3,3)); Sk[:2,2]=k/2; Sk[2,:2]=k/2; Sk[2,2]=-u0  # (Kx-u0)
    # product (Kx-u0)(h0+h x) : symmetric matrix  0.5*(a b^T + b a^T) with a=[k;-u0], b=[h;h0]
    a=np.array([k[0],k[1],-u0])
    b=cp.hstack([hvec,cp.reshape(h0,(1,),order='C')])
    return 0.5*(cp.reshape(a,(3,1),order='C')@cp.reshape(b,(1,3),order='C')+cp.reshape(b,(3,1),order='C')@cp.reshape(a,(1,3),order='C'))

def feasible(beta,le,mu,jet=True,ref='ball',eps=1e-4):
    P=cp.Variable((3,3),symmetric=True)
    cons=[] if jet else [P[0:2,2]==0,P[2,2]==0]
    Pxx=P[:2,:2]
    la=cp.Variable(nonneg=True); lb=cp.Variable(nonneg=True); lc=cp.Variable(nonneg=True); ld=cp.Variable(nonneg=True)
    nu=cp.Variable(nonneg=True); nu0=cp.Variable(nonneg=True); l0=cp.Variable(nonneg=True)
    H=cp.Variable(2)          # H = le*G
    Gv=H/le
    cons+=[P<<1e3*np.eye(3), Pxx>>eps*np.eye(2), Pxx@Acl+Acl.T@Pxx<<-eps*np.eye(2)]
    VP=Lp.T@P@Lp
    cons+=[VP - eps*E - la*S1 >> 0]
    dV=Lp.T@P@Lpd; dV=dV+dV.T
    # product multiplier: (Kx-u0)*(u0-(K-G)x) = -le*(Kx-u0)^2 + (Kx-u0)*(H x)
    prod=-le*S2 + lin(H,0*cp.Variable())  # h0=0
    cons+=[-dV - eps*E - lc*S1 - mu*(E22-VP) - prod >> 0]
    colp=cp.vstack([cp.reshape(k-Gv,(2,1),order='C'), np.zeros((1,1))])
    cons+=[cp.vstack([cp.hstack([VP - lb*S1, colp]), cp.hstack([colp.T, np.array([[u0**2]])])]) >> 0]
    col0=cp.reshape(k-Gv,(2,1),order='C')
    cons+=[cp.vstack([cp.hstack([Pxx, col0]), cp.hstack([col0.T, np.array([[u0**2]])])]) >> 0]
    Rm=np.eye(2) if ref=='ball' else R0
    Rb=np.block([[-Rm,np.zeros((2,1))],[np.zeros((1,2)),np.array([[beta**2]])]])
    cons+=[cp.bmat([[-Pxx,np.zeros((2,1))],[np.zeros((1,2)),np.ones((1,1))]]) - nu0*Rb - l0*Q0 >> 0]
    cons+=[E22 - VP - nu*Rb - ld*S1 >> 0]
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL)
    except Exception: return None
    return P.value if pr.status in ('optimal','optimal_inaccurate') else None

def best(jet,ref,les=np.logspace(-2,2,9),mus=np.r_[0,np.logspace(-2,1.5,8)]):
    lo,hi=(0.2,4.0); Pbest=None
    for _ in range(14):
        mid=(lo+hi)/2; ok=None
        for le,mu in itertools.product(les,mus):
            Pv=feasible(mid,le,mu,jet,ref)
            if Pv is not None: ok=Pv;break
        if ok is not None: lo=mid;Pbest=ok
        else: hi=mid
    return lo,Pbest
t=time.time()
for ref in ['ball','ell']:
    bJ,PJ=best(True,ref); bQ,PQ=best(False,ref)
    print("ref=%s  jet: beta=%.3f  quadratic: beta=%.3f  (%.0fs)"%(ref,bJ,bQ,time.time()-t))
    np.savez('regional5_%s.npz'%ref,PJ=PJ,PQ=PQ,K=K)
