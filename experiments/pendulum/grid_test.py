import numpy as np, cvxpy as cp, warnings, time
warnings.filterwarnings('ignore')
zeta=0.25
phi=lambda y: y-np.sin(y); dphi=lambda y: 1-np.cos(y); d2phi=lambda y: np.sin(y)
def f(x): return np.array([x[1], -x[0]-2*zeta*x[1]+phi(x[0])])
def eta_deta(L,x):
    y=x[0]; yd=x[1]; xd=f(x); psi=phi(y); psid=dphi(y)*yd
    ydd=xd[1]; psidd=d2phi(y)*yd**2+dphi(y)*ydd
    if L==1: return np.r_[x,psi], np.r_[xd,psid]
    return np.r_[x,psi,psid], np.r_[xd,psid,psidd]
def feasible(L,ybar,vbar,beta,ng=31,eps=1e-3):
    m=2+L; P=cp.Variable((m,m),symmetric=True); cons=[P<<1e3*np.eye(m),P>>-1e3*np.eye(m)]
    xs=np.linspace(-ybar,ybar,ng); vs=np.linspace(-vbar,vbar,ng)
    for x1 in xs:
        for x2 in vs:
            x=np.array([x1,x2]); e,de=eta_deta(L,x); n2=x1**2+x2**2
            if n2<1e-6: continue
            cons.append(2*e@P@de<=-eps*n2)          # decrease on the box
            cons.append(e@P@e>=eps*n2)              # positivity on the box
            if n2<=beta**2: cons.append(e@P@e<=1)   # ball inside level set
    for x1 in xs:                                    # V>=1 on the boundary of the box
        for x2 in (-vbar,vbar):
            e,_=eta_deta(L,np.array([x1,x2])); cons.append(e@P@e>=1)
    for x2 in vs:
        for x1 in (-ybar,ybar):
            e,_=eta_deta(L,np.array([x1,x2])); cons.append(e@P@e>=1)
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL); return pr.status=='optimal'
    except Exception: return False
def maxbeta(L,ybar,vbar):
    if not feasible(L,ybar,vbar,0.05): return 0.0
    lo,hi=0.05,min(ybar,vbar)
    for _ in range(8):
        mid=(lo+hi)/2
        if feasible(L,ybar,vbar,mid): lo=mid
        else: hi=mid
    return lo
t=time.time()
for ybar in [2.0,2.5,2.8,3.0,3.1]:
    for vbar in [1.5,2.0,2.5]:
        b1=maxbeta(1,ybar,vbar); b2=maxbeta(2,ybar,vbar)
        print("box |theta|<=%.1f |thetadot|<=%.1f : level1 beta=%.3f  level2 beta=%.3f  (%.0fs)"%(ybar,vbar,b1,b2,time.time()-t),flush=True)
