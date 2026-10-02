import os; RW=os.environ.setdefault('RW','0.01')
import numpy as np
exec(open('regional3.py').read().split('bJ,PJ=best_beta')[0])
bC,PC=best_beta(False)      # classical: quadratic + generalized sector (T line search)
print("classical generalized sector: beta=%.3f"%bC)
d=np.load('regional5_R%s_ball.npz'%RW); PJ=d['PJ']; PQ=d['PQ']
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
print("K=",K)
def converges(x0,Tf=30):
    sol=solve_ivp(f,[0,Tf],x0,rtol=1e-6,atol=1e-9,max_step=0.2)
    return np.linalg.norm(sol.y[:,-1])<1e-2 and np.all(np.abs(sol.y)<50)
L=3.0; ng=141; xs=np.linspace(-L,L,ng); XX,YY=np.meshgrid(xs,xs)
RoA=np.array([[converges(np.array([XX[i,j],YY[i,j]])) for j in range(ng)] for i in range(ng)])
np.save('roa_R0.01.npy',RoA)
def Vjet(P,x):
    xi=np.array([x[0],x[1],dz(K@x)[0]]); return xi@P@xi
VJ=np.array([[Vjet(PJ,np.array([XX[i,j],YY[i,j]])) for j in range(ng)] for i in range(ng)])
VC=np.array([[Vjet(PC,np.array([XX[i,j],YY[i,j]])) for j in range(ng)] for i in range(ng)])
fig,ax=plt.subplots(figsize=(6.4,6))
ax.contourf(XX,YY,RoA.astype(float),levels=[0.5,1.5],colors=['#e3ecf5'])
ax.contour(XX,YY,RoA.astype(float),levels=[0.5],colors='#5b7fa6',linewidths=1,linestyles='--')
ax.contour(XX,YY,VC,levels=[1],colors='#c0504d',linewidths=1.8)
ax.contour(XX,YY,VJ,levels=[1],colors='#1f4e79',linewidths=2.4)
for sgn in (1,-1):
    xx=np.linspace(-L,L,2); ax.plot(xx,(sgn*u0-k[0]*xx)/k[1],color='gray',lw=0.8,ls=':')
for t in np.linspace(0,2*np.pi,15)[:-1]:
    dd=np.array([np.cos(t),np.sin(t)]); lo,hi=0,L*1.5
    for _ in range(40):
        mid=(lo+hi)/2
        if Vjet(PJ,mid*dd)<=1: lo=mid
        else: hi=mid
    x0=0.99*lo*dd; sol=solve_ivp(f,[0,20],x0,rtol=1e-8,atol=1e-10,max_step=0.02)
    ax.plot(sol.y[0],sol.y[1],color='#404040',lw=0.6); ax.plot(x0[0],x0[1],'o',color='#404040',ms=2.5)
ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.legend([Line2D([],[],color='#5b7fa6',ls='--'),Line2D([],[],color='#c0504d',lw=1.8),Line2D([],[],color='#1f4e79',lw=2.4),Line2D([],[],color='gray',ls=':')],
 ['true region of attraction (simulation)','quadratic $x^\\top Px$ + generalized sector condition','jet function $V(x,\\psi)$, $\\psi=\\mathrm{dz}(Kx)$ (order-1 jet)','saturation limits $|Kx|=u_0$'],loc='upper left',fontsize=8,framealpha=0.95)
ax.set_title('Saturated LQR feedback $\\dot x=Ax+B\\,\\mathrm{sat}(Kx)$, unstable focus, $u_0=1$',fontsize=10)
fig.tight_layout(); fig.savefig('fig_regional.pdf'); fig.savefig('fig_regional.png',dpi=160)
# V and |x| along a trajectory
fig2,ax2=plt.subplots(figsize=(6.4,3.0))
dd=np.array([np.cos(2.2),np.sin(2.2)]); lo,hi=0,L*1.5
for _ in range(40):
    mid=(lo+hi)/2
    if Vjet(PJ,mid*dd)<=1: lo=mid
    else: hi=mid
sol=solve_ivp(f,[0,10],0.99*lo*dd,rtol=1e-8,atol=1e-10,max_step=0.005)
V=np.array([Vjet(PJ,sol.y[:,i]) for i in range(sol.y.shape[1])]); psi=np.array([dz(K@sol.y[:,i])[0] for i in range(sol.y.shape[1])])
ax2.fill_between(sol.t,0,1.05,where=np.abs(psi)>1e-9,color='#f3e6c8',alpha=0.8,label='input saturated')
ax2.plot(sol.t,V,color='#1f4e79',lw=2,label='$V(x(t),\\psi(t))$')
ax2.plot(sol.t,np.sum(sol.y**2,0)/np.sum(sol.y[:,0]**2),color='#c0504d',lw=1.2,label='$|x(t)|^2/|x(0)|^2$')
ax2.set_xlabel('$t$'); ax2.set_ylim(0,1.05); ax2.legend(fontsize=8,loc='upper right'); ax2.set_title('Jet Lyapunov function along a trajectory starting on the boundary of its estimate',fontsize=9)
fig2.tight_layout(); fig2.savefig('fig_regional_traj.pdf'); fig2.savefig('fig_regional_traj.png',dpi=160)
np.savez('final_regional.npz',PJ=PJ,PC=PC,K=K,bC=bC)
