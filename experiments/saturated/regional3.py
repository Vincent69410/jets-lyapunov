import numpy as np, cvxpy as cp
from scipy.linalg import solve_continuous_are
exec(open('regional2.py').read().split('def feasible')[0])   # reuse setup (A,B,K,Lp,S1,Q0,...)

def Theta(TG,T):
    Z=np.zeros((2,2))
    top=cp.hstack([Z, cp.reshape(TG,(2,1),order='C')])
    bot=cp.hstack([cp.reshape(TG,(1,2),order='C'), cp.reshape(-2*T,(1,1),order='C')])
    return cp.vstack([top,bot])

def feasible(beta,T,jet=True,eps=1e-4):
    P=cp.Variable((3,3),symmetric=True)
    cons=[] if jet else [P[0:2,2]==0,P[2,2]==0]
    Pxx=P[:2,:2]
    M=cp.Variable(2); T2=cp.Variable(); G1=cp.Variable((6,2))
    la=cp.Variable(nonneg=True); lb=cp.Variable(nonneg=True); ld=cp.Variable(nonneg=True)
    nu=cp.Variable(nonneg=True); nu0=cp.Variable(nonneg=True); l0=cp.Variable(nonneg=True)
    Gv=M/T
    cons+=[P<<1e3*np.eye(3), Pxx>>eps*np.eye(2)]
    # derivative LMI on zeta=(x,psi,xdot,psidot): sector(T,G) on S(G,u0), slope equality, Finsler
    Pi0=np.hstack([np.eye(3),np.zeros((3,3))]); Pi1=np.hstack([np.zeros((3,3)),np.eye(3)])
    D=Pi0.T@P@Pi1; D=D+D.T
    Nm=np.hstack([-Acl, B, np.eye(2), np.zeros((2,1))]); Fin=G1@Nm; Fin=Fin+Fin.T
    Sec=cp.vstack([cp.hstack([Theta(M,T), np.zeros((3,3))]), np.zeros((3,6))])
    Sl=np.zeros((6,6)); Sl[3:5,5]=k; Sl[5,3:5]=k; Sl[5,5]=-2
    E6=np.zeros((6,6)); E6[:2,:2]=np.eye(2)
    cons+=[D+Fin+Sec+T2*Sl << -eps*E6]
    # region-wise positivity in R+
    VP=Lp.T@P@Lp
    cons+=[VP - eps*E - la*S1 >> 0]
    # region-wise inclusion {V<=1} ⊂ {(K-G)x <= u0} (R+) and {|(K-G)x|<=u0} (R0), Schur complements
    colp=cp.vstack([cp.reshape(k-Gv,(2,1),order='C'), np.zeros((1,1))])
    cons+=[cp.vstack([cp.hstack([VP - lb*S1, colp]), cp.hstack([colp.T, np.array([[u0**2]])])]) >> 0]
    col0=cp.reshape(k-Gv,(2,1),order='C')
    cons+=[cp.vstack([cp.hstack([Pxx, col0]), cp.hstack([col0.T, np.array([[u0**2]])])]) >> 0]
    # ball(beta) ⊂ {V<=1}, region-wise
    cons+=[cp.bmat([[-Pxx,np.zeros((2,1))],[np.zeros((1,2)),np.ones((1,1))]]) - nu0*np.diag([-1,-1,beta**2]) - l0*Q0 >> 0]
    cons+=[E22 - VP - nu*np.diag([-1,-1,beta**2]) - ld*S1 >> 0]
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL)
    except Exception: return None
    return P.value if pr.status in ('optimal','optimal_inaccurate') else None

def best_beta(jet,Ts=np.logspace(-2,2,17)):
    lo,hi=0.1,6.0; Pbest=None
    for _ in range(16):
        mid=(lo+hi)/2; ok=None
        for T in Ts:
            Pv=feasible(mid,T,jet)
            if Pv is not None: ok=Pv;break
        if ok is not None: lo=mid;Pbest=ok
        else: hi=mid
    return lo,Pbest
bJ,PJ=best_beta(True); bQ,PQ=best_beta(False)
print("jet: beta=%.3f  quadratic: beta=%.3f"%(bJ,bQ))
np.savez('regional3_res.npz',PJ=PJ,PQ=PQ,K=K)
