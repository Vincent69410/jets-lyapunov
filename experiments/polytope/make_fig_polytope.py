import numpy as np, cvxpy as cp
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
n=2; a=20.0
A1=np.array([[-1.,-1.],[1.,-1.]]); A2=np.array([[-1.,-a],[1/a,-1.]])
def A(al): return al*A1+(1-al)*A2
# level-1 jet LMI with constant Finsler multiplier (vertex LMIs), maximize decay margin
N=1; m=n*(N+1)
P=cp.Variable((m,m),symmetric=True); G=cp.Variable((n*(N+2),m)); eps=cp.Variable()
Pi0=np.hstack([np.eye(m),np.zeros((m,n))]); Pi1=np.hstack([np.zeros((m,n)),np.eye(m)])
D=Pi0.T@P@Pi1; D=D+D.T
def Nmat(Am):
    rows=[]
    for k in range(N+1):
        r=np.zeros((n,n*(N+2))); r[:,n*(k+1):n*(k+2)]=np.eye(n); r[:,n*k:n*(k+1)]=-Am; rows.append(r)
    return np.vstack(rows)
E0=np.zeros((n,m)); E0[:,:n]=np.eye(n)
cons=[P>>E0.T@E0, P<<50*np.eye(m)]
for Am in [A1,A2]:
    Nm=Nmat(Am); cons.append(D+G@Nm+(G@Nm).T << -eps*np.eye(n*(N+2)))
cp.Problem(cp.Maximize(eps),cons).solve(solver=cp.CLARABEL); PP=P.value; print("eps=",eps.value)
def Pal(al):
    Gm=np.vstack([np.eye(n),A(al)]); return Gm.T@PP@Gm
def V(x,xd): j=np.r_[x,xd]; return j@PP@j
# --- Fig A: PDLF level sets + frozen trajectory
fig,ax=plt.subplots(1,2,figsize=(10,4.6))
th=np.linspace(0,2*np.pi,400); circ=np.vstack([np.cos(th),np.sin(th)])
cols=plt.cm.viridis(np.linspace(0.1,0.9,5))
for c,al in zip(cols,[0,0.25,0.5,0.75,1.0]):
    Lc=np.linalg.cholesky(Pal(al)); pts=np.linalg.solve(Lc.T,circ)
    ax[0].plot(pts[0],pts[1],color=c,lw=1.6,label=r'$\alpha=%.2f$'%al)
ax[0].set_aspect('equal'); ax[0].legend(fontsize=8,title='level set $x^\\top P(\\alpha)x=1$',title_fontsize=8)
ax[0].set_xlabel('$x_1$'); ax[0].set_ylabel('$x_2$'); ax[0].set_title('Parameter-dependent Lyapunov function\n$P(\\alpha)=\\Gamma_1(A(\\alpha))^\\top P_J\\,\\Gamma_1(A(\\alpha))$ from a single $P_J$ ($N=1$)',fontsize=9)
# frozen alpha=0 (A2, strongly non-normal): |x|^2 vs V
al=0.0; Am=A(al); x0=np.array([0.05,1.0]); x0=x0/np.sqrt(V(x0,Am@x0))
sol=solve_ivp(lambda t,x:Am@x,[0,6],x0,max_step=0.005,rtol=1e-9)
Vt=[V(sol.y[:,i],Am@sol.y[:,i]) for i in range(sol.y.shape[1])]
ax[1].plot(sol.t,np.sum(sol.y**2,0)/np.sum(x0**2),color='#c0504d',lw=1.4,label='$|x(t)|^2/|x(0)|^2$')
ax[1].plot(sol.t,Vt,color='#1f4e79',lw=2,label='$V(x(t),\\dot x(t))$')
ax[1].set_yscale('log'); ax[1].set_xlabel('$t$'); ax[1].legend(fontsize=8); ax[1].set_title('Frozen $\\alpha=0$ ($A_2$, $a=20$): the norm overshoots, $V$ decreases',fontsize=9)
fig.tight_layout(); fig.savefig('fig_polytope.pdf'); fig.savefig('fig_polytope.png',dpi=160)
# --- Fig B: time-varying alpha, slow vs fast
fig2,ax2=plt.subplots(1,2,figsize=(10,3.6))
for axx,om,lab in zip(ax2,[0.2,6.0],['slow','fast']):
    alf=lambda t:(1+np.sin(om*t))/2
    x0=np.array([1.0,0.3]); 
    sol=solve_ivp(lambda t,x:A(alf(t))@x,[0,12],x0,max_step=0.002,rtol=1e-9)
    Vt=np.array([V(sol.y[:,i],A(alf(sol.t[i]))@sol.y[:,i]) for i in range(sol.y.shape[1])])
    axx.plot(sol.t,Vt/Vt[0],color='#1f4e79',lw=2,label='$V(x,\\dot x)/V(0)$')
    axx.plot(sol.t,np.sum(sol.y**2,0)/np.sum(x0**2),color='#c0504d',lw=1.2,label='$|x|^2/|x(0)|^2$')
    axx.set_yscale('log'); axx.set_xlabel('$t$'); axx.legend(fontsize=8)
    axx.set_title('%s parameter: $\\alpha(t)=(1+\\sin %.1f t)/2$, $|\\dot\\alpha|\\leq %.1f$'%(lab,om,om/2),fontsize=9)
fig2.tight_layout(); fig2.savefig('fig_polytope_tv.pdf'); fig2.savefig('fig_polytope_tv.png',dpi=160)
