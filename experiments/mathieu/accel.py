import numpy as np, cvxpy as cp, itertools, warnings
warnings.filterwarnings('ignore')
from math import comb
from scipy.integrate import solve_ivp
zeta,epsv=0.05,0.5
A0=np.array([[0.,1.],[-1.,-2*zeta]]); A1=np.array([[0.,0.],[-epsv,0.]]); n=2
def solve(cons):
    pr=cp.Problem(cp.Minimize(0),cons)
    try:
        pr.solve(solver=cp.CLARABEL)
        if pr.status=='optimal': return True
        if pr.status=='infeasible': return False
    except Exception: pass
    return None
def Nmat(theta,K):
    # constraints x_{k+1} = sum_i C(k,i) A^{(i)} x_{k-i}, k=0..K-1, on jets (x_0..x_K); theta=(w,w',w'',...)
    rows=[]
    for k in range(K):
        r=np.zeros((n,n*(K+1))); r[:,n*(k+1):n*(k+2)]=np.eye(n)
        for i in range(k+1):
            Ai=(A0+theta[0]*A1) if i==0 else theta[i]*A1
            r[:,n*(k-i):n*(k-i+1)]-=comb(k,i)*Ai
        rows.append(r)
    return np.vstack(rows)
def jet_cert(N,bounds,eps=1e-3):
    # bounds = (1, d1, d2, ..., dN) box for (w, w', ..., w^{(N)}); V on N-jets, decrease on (N+1)-jets
    m=n*(N+1); P=cp.Variable((m,m),symmetric=True); G=cp.Variable((n*(N+2),n*(N+1))); G0=cp.Variable((m,n*N))
    E0=np.zeros((n,m)); E0[:,:n]=np.eye(n)
    cons=[P<<1e4*np.eye(m),P>>-1e4*np.eye(m)]
    for sg in itertools.product([-1,1],repeat=N+1):
        th=[s*b for s,b in zip(sg,bounds)]
        Nm=Nmat(th,N+1); M=Dop(P,N)+G@Nm+(G@Nm).T; cons.append(M<<-eps*np.eye(n*(N+2)))
    for sg in itertools.product([-1,1],repeat=N):
        th=[s*b for s,b in zip(sg,bounds[:N])]
        N0=Nmat(th,N); cons.append(P+G0@N0+(G0@N0).T>>E0.T@E0)
    return solve(cons)
def Dop(P,N):
    m=n*(N+1); Pi0=np.hstack([np.eye(m),np.zeros((m,n))]); Pi1=np.hstack([np.zeros((m,n)),np.eye(m)])
    M=Pi0.T@P@Pi1; return M+M.T
# sanity: resonance w=cos(2t) is destabilizing?
sol=solve_ivp(lambda t,x:(A0+np.cos(2*t)*A1)@x,[0,60],[1,0],max_step=0.01)
print("growth under w=cos 2t: |x(60)|/|x(0)| = %.2e"%np.linalg.norm(sol.y[:,-1]))
# classical: N=1 (rate bound only) for delta1=2 ; then N=2 with delta2
for d1 in [0.5,1.0,2.0]:
    print("delta1=%.1f  N=1 (rate-only certificate):"%d1, jet_cert(1,(1.0,d1)))
def maxd2(N,d1,bounds_extra=()):
    lo,hi=0.0,50.0
    if not jet_cert(N,(1.0,d1,1e-3)+bounds_extra): return None
    for _ in range(14):
        mid=(lo+hi)/2
        if jet_cert(N,(1.0,d1,mid)+bounds_extra): lo=mid
        else: hi=mid
    return lo
for d1 in [0.5,1.0,2.0,4.0]:
    print("delta1=%.1f  N=2: max certified delta2 = %s"%(d1,maxd2(2,d1)))
