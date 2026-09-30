import numpy as np, cvxpy as cp, warnings
warnings.filterwarnings('ignore')
from math import comb
n=2
A1=np.array([[-1.,-1.],[1.,-1.]])
def A2(a): return np.array([[-1.,-a],[1/a,-1.]])
def solve(cons):
    pr=cp.Problem(cp.Minimize(0),cons)
    try:
        pr.solve(solver=cp.CLARABEL)
        if pr.status=='optimal': return True
        if pr.status in ('infeasible',): return False
    except Exception: pass
    try:
        pr.solve(solver=cp.SCS,max_iters=50000,eps=1e-8)
        if pr.status=='optimal': return True
        if pr.status=='infeasible': return False
    except Exception: pass
    return None
def Dop(P,N):
    m=n*(N+1); Pi0=np.hstack([np.eye(m),np.zeros((m,n))]); Pi1=np.hstack([np.zeros((m,n)),np.eye(m)])
    M=Pi0.T@P@Pi1; return M+M.T
def Nmat(A,N):
    rows=[]
    for k in range(N+1):
        r=np.zeros((n,n*(N+2))); r[:,n*(k+1):n*(k+2)]=np.eye(n); r[:,n*k:n*(k+1)]=-A; rows.append(r)
    return np.vstack(rows)
def jet_finsler(a,N,eps=1e-3):
    m=n*(N+1); P=cp.Variable((m,m),symmetric=True); G=cp.Variable((n*(N+2),m)); G0=cp.Variable((m,n*N)) if N>0 else None
    E0=np.zeros((n,m)); E0[:,:n]=np.eye(n)
    cons=[P<<1e4*np.eye(m), P>>-1e4*np.eye(m)]
    for A in [A1,A2(a)]:
        Nm=Nmat(A,N); M=Dop(P,N)+G@Nm+(G@Nm).T; cons.append(M<<-eps*np.eye(n*(N+2)))
        if N>0:
            N0=Nmat(A,N-1)   # constraints on the N-jet: x_{k+1}=A x_k, k<N   (size nN x n(N+1))
            cons.append(P+G0@N0+(G0@N0).T>>E0.T@E0)      # positivity on admissible jets
        else: cons.append(P>>E0.T@E0)
    return solve(cons)
def affine_slack(a,eps=1e-3):
    P1=cp.Variable((2,2),symmetric=True); P2=cp.Variable((2,2),symmetric=True); F=cp.Variable((4,2))
    cons=[P1>>np.eye(2),P2>>np.eye(2),P1<<1e4*np.eye(2),P2<<1e4*np.eye(2)]
    for P,Am in [(P1,A1),(P2,A2(a))]:
        M=cp.bmat([[np.zeros((2,2)),P],[P,np.zeros((2,2))]]); Nm=np.hstack([-Am,np.eye(2)])
        cons.append(M+F@Nm+(F@Nm).T<<-eps*np.eye(4))
    return solve(cons)
def hpd(a,g,d,eps=1e-3):
    Ps=[cp.Variable((2,2),symmetric=True) for _ in range(g+1)]; Amat={1:A1,0:A2(a)}; cons=[]
    for p in range(g+d+1):
        e=0
        for i in range(g+1):
            l=p-i
            if 0<=l<=d: e=e+comb(d,l)*Ps[i]
        cons.append(e>>eps*np.eye(2))
    for p in range(g+1+d+1):
        e=0
        for i in range(g+1):
            for jA in (0,1):
                l=p-i-jA
                if 0<=l<=d: e=e+comb(d,l)*(Ps[i]@Amat[jA]+Amat[jA].T@Ps[i])
        cons.append(e<<-eps*np.eye(2))
    cons+=[P<<1e4*np.eye(2) for P in Ps]
    return solve(cons)
if __name__ == "__main__":
    grid=[5,5.9,8,20,50,100,200,500,1000]
    def row(name,fn):
        print("%-38s"%name, " ".join({True:' ok ',False:' -- ',None:' ?? '}[fn(a)] for a in grid))
    print("%-38s"%"a =", " ".join("%4g"%a for a in grid))
    row("common quadratic (jet N=0)",lambda a:jet_finsler(a,0))
    row("affine PDLF, Polya d=0 (vertex)",lambda a:hpd(a,1,0))
    row("affine PDLF + constant slack",affine_slack)
    row("HPD-QLF deg 2, Polya d=0",lambda a:hpd(a,2,0))
    row("jet N=1, Finsler pos+dec, const G",lambda a:jet_finsler(a,1))
    row("jet N=2, Finsler pos+dec, const G",lambda a:jet_finsler(a,2))
