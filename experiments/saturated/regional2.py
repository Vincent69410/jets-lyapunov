import numpy as np, cvxpy as cp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

A=np.array([[0.3,1.],[-1.,0.3]]); B=np.array([[0.],[1.]]); u0=1.0
Rw=float(__import__("os").environ.get("RW","1.0")); X=solve_continuous_are(A,B,np.eye(2),np.array([[Rw]])); K=-(B.T@X)/Rw; Acl=A+B@K
k=K[0]
def dz(u): return u-np.clip(u,-u0,u0)
def f(t,x): return Acl@x - B[:,0]*dz(K@x)[0]

# homogeneous coordinates z=[x;1]; region R+ : psi = Kx-u0
Lp=np.array([[1,0,0],[0,1,0],[k[0],k[1],-u0]])          # xi=(x,psi)=Lp z
Lpd=np.vstack([np.hstack([A,B*u0]), np.hstack([K@A, K@B*u0])])  # (xdot,psidot)=Lpd z in R+
S1=np.zeros((3,3)); S1[:2,2]=k/2; S1[2,:2]=k/2; S1[2,2]=-u0  # z^T S1 z = Kx-u0
Q0=np.zeros((3,3)); Q0[:2,:2]=-np.outer(k,k); Q0[2,2]=u0**2   # u0^2-(Kx)^2
E=np.diag([1.,1.,0.]); E22=np.diag([0.,0.,1.])

def feasible(beta,mu,jet=True,eps=1e-4):
    P=cp.Variable((3,3),symmetric=True)
    cons=[] if jet else [P[0:2,2]==0,P[2,2]==0]
    Pxx=P[:2,:2]
    la=cp.Variable(nonneg=True); lc=cp.Variable(nonneg=True); ld=cp.Variable(nonneg=True)
    nu=cp.Variable(nonneg=True); nu0=cp.Variable(nonneg=True); l0=cp.Variable(nonneg=True)
    cons+=[P<<1e3*np.eye(3)]
    # region R0 (psi=0)
    cons+=[Pxx>>eps*np.eye(2), Pxx@Acl+Acl.T@Pxx<<-eps*np.eye(2)]
    cons+=[cp.bmat([[-Pxx,np.zeros((2,1))],[np.zeros((1,2)),np.ones((1,1))]]) - nu0*np.diag([-1,-1,beta**2]) - l0*Q0 >> 0]
    # region R+
    VP=Lp.T@P@Lp
    cons+=[VP - eps*E - la*S1 >> 0]                                  # V >= eps|x|^2 on R+
    dV=Lp.T@P@Lpd; dV=dV+dV.T
    cons+=[-dV - eps*E - lc*S1 - mu*(E22-VP) >> 0]                   # dV<0 on R+ ∩ {V<=1}
    cons+=[E22 - VP - nu*np.diag([-1,-1,beta**2]) - ld*S1 >> 0]      # ball(beta) ∩ R+ ⊂ {V<=1}
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver=cp.CLARABEL)
    except Exception: return None
    return P.value if pr.status in ('optimal','optimal_inaccurate') else None

def best_beta(jet,mus=np.logspace(-2,1.5,15)):
    lo,hi=0.1,6.0; Pbest=None
    for _ in range(18):
        mid=(lo+hi)/2; ok=None
        for mu in mus:
            Pv=feasible(mid,mu,jet)
            if Pv is not None: ok=Pv;break
        if ok is not None: lo=mid;Pbest=ok
        else: hi=mid
    return lo,Pbest

bJ,PJ=best_beta(True); bQ,PQ=best_beta(False)
print("jet piecewise: beta=%.3f"%bJ); print("quadratic regionwise: beta=%.3f"%bQ)
np.savez('regional2_res.npz',PJ=PJ,PQ=PQ,K=K)

RoA=np.load('roa.npy'); L=4.0; ng=121
xs=np.linspace(-L,L,ng); XX,YY=np.meshgrid(xs,xs)
def Vjet(P,x):
    xi=np.array([x[0],x[1],dz(K@x)[0]]); return xi@P@xi
VJ=np.zeros_like(XX); VQ=np.zeros_like(XX)
for i in range(ng):
    for j in range(ng):
        x=np.array([XX[i,j],YY[i,j]]); VJ[i,j]=Vjet(PJ,x); VQ[i,j]=Vjet(PQ,x)
# classical generalized-sector ellipsoid from regional.py
PQ0=np.load('regional_res.npz')['PQ']
VQ0=np.array([[Vjet(PQ0,np.array([XX[i,j],YY[i,j]])) for j in range(ng)] for i in range(ng)])

fig,ax=plt.subplots(figsize=(6.4,6))
ax.contourf(XX,YY,RoA.astype(float),levels=[0.5,1.5],colors=['#e3ecf5'])
ax.contour(XX,YY,RoA.astype(float),levels=[0.5],colors='#5b7fa6',linewidths=1,linestyles='--')
ax.contour(XX,YY,VQ0,levels=[1],colors='#c0504d',linewidths=1.8)
ax.contour(XX,YY,VQ,levels=[1],colors='#e0a030',linewidths=1.5,linestyles='-.')
ax.contour(XX,YY,VJ,levels=[1],colors='#1f4e79',linewidths=2.4)
for sgn in (1,-1):
    xx=np.linspace(-L,L,2); ax.plot(xx,(sgn*u0-k[0]*xx)/k[1],color='gray',lw=0.8,ls=':')
th=np.linspace(0,2*np.pi,13)[:-1]
for t in th:
    d=np.array([np.cos(t),np.sin(t)]); lo,hi=0,L*1.5
    for _ in range(40):
        mid=(lo+hi)/2
        if Vjet(PJ,mid*d)<=1: lo=mid
        else: hi=mid
    x0=0.99*lo*d
    sol=solve_ivp(f,[0,25],x0,rtol=1e-8,atol=1e-10,max_step=0.05)
    ax.plot(sol.y[0],sol.y[1],color='#404040',lw=0.6); ax.plot(x0[0],x0[1],'o',color='#404040',ms=2.5)
ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.legend([Line2D([],[],color='#5b7fa6',ls='--'),Line2D([],[],color='#c0504d',lw=1.8),Line2D([],[],color='#e0a030',ls='-.'),Line2D([],[],color='#1f4e79',lw=2.4),Line2D([],[],color='gray',ls=':')],
 ['true region of attraction (simulation)','quadratic $x^\\top Px$, generalized sector condition','quadratic $x^\\top Px$, region-wise S-procedure','jet function $V(x,\\psi)$, $\\psi=\\mathrm{dz}(Kx)$, region-wise','saturation limits $|Kx|=u_0$'],loc='upper left',fontsize=7.5,framealpha=0.95)
ax.set_title('$\\dot x=Ax+B\\,\\mathrm{sat}(Kx)$: estimates of the region of attraction',fontsize=10)
fig.tight_layout(); fig.savefig('fig_regional.pdf'); fig.savefig('fig_regional.png',dpi=160)

# V along trajectories: |x|^2 vs V
fig2,ax2=plt.subplots(figsize=(6.4,3.2))
d=np.array([np.cos(0.7),np.sin(0.7)]); lo,hi=0,L*1.5
for _ in range(40):
    mid=(lo+hi)/2
    if Vjet(PJ,mid*d)<=1: lo=mid
    else: hi=mid
sol=solve_ivp(f,[0,12],0.99*lo*d,rtol=1e-8,atol=1e-10,max_step=0.01)
V=[Vjet(PJ,sol.y[:,i]) for i in range(sol.y.shape[1])]
psi=[dz(K@sol.y[:,i])[0] for i in range(sol.y.shape[1])]
ax2.plot(sol.t,V,color='#1f4e79',lw=2,label='$V(x(t),\\psi(t))$')
ax2.plot(sol.t,np.sum(sol.y**2,0)/np.sum(sol.y[:,0]**2),color='#c0504d',lw=1.2,label='$|x(t)|^2/|x(0)|^2$')
ax2.fill_between(sol.t,0,1,where=np.abs(psi)>1e-9,color='#f0e0c0',alpha=0.6,label='saturated')
ax2.set_xlabel('$t$'); ax2.legend(fontsize=8); ax2.set_ylim(0,1.05); ax2.set_title('Along a trajectory from the boundary of the jet estimate',fontsize=10)
fig2.tight_layout(); fig2.savefig('fig_regional_traj.pdf'); fig2.savefig('fig_regional_traj.png',dpi=160)
