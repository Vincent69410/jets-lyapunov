import numpy as np, cvxpy as cp, itertools, warnings, time
warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
# damped pendulum: theta'' + 2 zeta theta' + sin theta = 0 ; x=(theta, theta'), psi = theta - sin theta
zeta=0.25
A=np.array([[0.,1.],[-1.,-2*zeta]]); B=np.array([[0.],[1.]]); C=np.array([[1.,0.]])
MU_G=1.2173  # global sector bound of phi(y)=y-sin y : max (1 - sin y / y) ~ 1.2172
S_G=2.0      # global slope bound  phi'(y)=1-cos y in [0,2]
def local_bounds(ybar,vbar):
    mu=1-np.sin(ybar)/ybar if ybar<np.pi else MU_G
    mup=1-np.cos(ybar) if ybar<np.pi else 2.0
    mu2=np.sin(ybar) if ybar<=np.pi/2 else 1.0
    return mu,mup,mu2*vbar     # rho = phi''(y) * ydot bounded by mu2*vbar
def solve(cons):
    pr=cp.Problem(cp.Minimize(0),cons)
    try:
        pr.solve(solver=cp.CLARABEL)
        if pr.status=='optimal': return True
        if pr.status=='infeasible': return False
    except Exception: pass
    try:
        pr.solve(solver=cp.SCS,max_iters=40000,eps=1e-8)
        return pr.status=='optimal'
    except Exception: return False

def build(L):
    # zeta layout
    if L==1:  # zeta=(x(2),psi,xd(2),psid)
        nz=6; iota={'x':[0,1],'psi':[2],'xd':[3,4],'psid':[5]}
        eta_idx=[0,1,2]; deta_idx=[3,4,5]
    else:     # zeta=(x(2),psi,psid,xd(2),psidd)
        nz=7; iota={'x':[0,1],'psi':[2],'psid':[3],'xd':[4,5],'psidd':[6]}
        eta_idx=[0,1,2,3]; deta_idx=[4,5,3,6]   # d/dt (x,psi,psid) = (xd, psid, psidd)
    m=len(eta_idx)
    Pe=np.zeros((m,nz)); Pd=np.zeros((m,nz))
    for k,i in enumerate(eta_idx): Pe[k,i]=1
    for k,i in enumerate(deta_idx): Pd[k,i]=1
    def N_dec(s,rho):
        rows=[]
        r=np.zeros((2,nz)); r[:,iota['xd']]=np.eye(2); r[:,iota['x']]-=A; r[:,iota['psi']]-=B; rows.append(r)  # xd=Ax+B psi
        r=np.zeros((1,nz)); r[:,iota['psid']]=1; r[:,iota['xd']]-=s*C; rows.append(r)                       # psid = s C xd
        if L==2:
            r=np.zeros((1,nz)); r[:,iota['psidd']]=1; r[:,iota['xd']]-=s*(C@A)+rho*C; r[:,iota['psid']]-=s*(C@B); rows.append(r)  # psidd = s C(A xd + B psid) + rho C xd
        return np.vstack(rows)
    def N_eta(s):
        # constraints on eta only (level 2: psid = s C (A x + B psi))
        if L==1: return None
        r=np.zeros((1,m)); r[0,3]=1; r[0,0:2]-=s*(C@A)[0]; r[0,2]-=s*(C@B)[0,0]; return r
    def Theta_sec(mu,dim,ix,ipsi):
        # psi (mu y - psi) >= 0 with y = C x = x1
        T=np.zeros((dim,dim)); T[ipsi,ix[0]]+=mu/2; T[ix[0],ipsi]+=mu/2; T[ipsi,ipsi]-=1; return T
    return nz,m,Pe,Pd,N_dec,N_eta,Theta_sec,iota,eta_idx

def certify(L,ybar,vbar,beta,eps=1e-4):
    nz,m,Pe,Pd,N_dec,N_eta,Theta_sec,iota,eta_idx=build(L)
    mu,mup,rhob=local_bounds(ybar,vbar)
    P=cp.Variable((m,m),symmetric=True)
    cons=[P<<1e3*np.eye(m),P>>-1e3*np.eye(m)]
    Ex=np.zeros((2,nz)); Ex[:,iota['x']]=np.eye(2)
    # --- decrease on admissible jets with base in Y (local bounds), vertices of (s,rho) box, constant G
    D=Pe.T@P@Pd; D=D+D.T
    G=cp.Variable((nz,N_dec(0,0).shape[0])); l1=cp.Variable(nonneg=True)
    verts=[(s,r) for s in (0,mup) for r in ((-rhob,rhob) if L==2 else (0,))]
    for s,r in verts:
        Nm=N_dec(s,r)
        cons.append(D+G@Nm+(G@Nm).T+l1*Theta_sec(mu,nz,iota['x'],iota['psi'][0])<<-eps*(Ex.T@Ex))
    # --- positivity on admissible eta with base in Y
    Exe=np.zeros((2,m)); Exe[:,0:2]=np.eye(2)
    l0=cp.Variable(nonneg=True)
    if L==2:
        G0=cp.Variable((m,1))
        for s in (0,mup):
            N0=N_eta(s); cons.append(P+G0@N0+(G0@N0).T-l0*Theta_sec(mu,m,[0,1],2)>>eps*(Exe.T@Exe))
    else:
        cons.append(P-l0*Theta_sec(mu,m,[0,1],2)>>eps*(Exe.T@Exe))
    # --- inclusion Omega ⊂ Y = {|x1|<=ybar, |x2|<=vbar}: global constraints (sector MU_G, slope in [0,S_G])
    for k,(bound) in enumerate([ybar,vbar]):
        Ek=np.zeros((m,m)); Ek[k,k]=1/bound**2
        lk=cp.Variable(nonneg=True)
        if L==2:
            Gk=cp.Variable((m,1))
            for s in (0,S_G):
                N0=N_eta(s); cons.append(P+Gk@N0+(Gk@N0).T-lk*Theta_sec(MU_G,m,[0,1],2)-Ek>>0)
        else:
            cons.append(P-lk*Theta_sec(MU_G,m,[0,1],2)-Ek>>0)
    # --- ball of radius beta inside Omega: homogeneous z=(eta,1)
    mz=m+1; E1=np.zeros((mz,mz)); E1[m,m]=1
    Pz=cp.bmat([[P,np.zeros((m,1))],[np.zeros((1,m)),np.zeros((1,1))]])
    Bb=np.zeros((mz,mz)); Bb[0,0]=-1; Bb[1,1]=-1; Bb[m,m]=beta**2
    nu=cp.Variable(nonneg=True); lb=cp.Variable(nonneg=True)
    Tz=np.zeros((mz,mz)); Tz[:m,:m]=Theta_sec(MU_G,m,[0,1],2)
    if L==2:
        Gb=cp.Variable((mz,1))
        for s in (0,S_G):
            N0z=np.hstack([N_eta(s),np.zeros((1,1))]); cons.append(E1-Pz+Gb@N0z+(Gb@N0z).T-nu*Bb-lb*Tz>>0)
    else:
        cons.append(E1-Pz-nu*Bb-lb*Tz>>0)
    ok=solve(cons)
    return (P.value if ok else None)

def best_beta(L,ybar,vbar):
    lo,hi=0.05,3.0; Pb=None
    if certify(L,ybar,vbar,lo) is None: return 0.0,None
    for _ in range(12):
        mid=(lo+hi)/2; Pv=certify(L,ybar,vbar,mid)
        if Pv is not None: lo=mid;Pb=Pv
        else: hi=mid
    return lo,Pb

t=time.time()
res={}
for ybar in [0.5,1.0,1.5,2.0,2.5,3.0]:
    for vbar in [1.0,1.5,2.0,3.0]:
        b1,_=best_beta(1,ybar,vbar); b2,_=best_beta(2,ybar,vbar)
        res[(ybar,vbar)]=(b1,b2)
        print("ybar=%.1f vbar=%.1f : level1 beta=%.3f   level2 beta=%.3f   (%.0fs)"%(ybar,vbar,b1,b2,time.time()-t),flush=True)
import pickle; pickle.dump(res,open('res.pkl','wb'))
