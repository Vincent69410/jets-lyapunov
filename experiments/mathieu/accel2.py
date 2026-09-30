import numpy as np, sys
exec(open('accel.py').read().split('# sanity')[0].replace('zeta,epsv=0.05,0.5','zeta,epsv=float(sys.argv[1]),float(sys.argv[2])'))
from scipy.integrate import solve_ivp
sol=solve_ivp(lambda t,x:(A0+np.cos(2*t)*A1)@x,[0,60],[1,0],max_step=0.01)
print("zeta=%.2f eps=%.2f  growth under w=cos 2t: %.2e"%(zeta,epsv,np.linalg.norm(sol.y[:,-1])))
print("frozen (delta1=0), N=1:",jet_cert(1,(1.0,0.0)))
def maxd1(N,extra=()):
    lo,hi=0.0,10.0
    for _ in range(12):
        mid=(lo+hi)/2
        if jet_cert(N,(1.0,mid)+extra): lo=mid
        else: hi=mid
    return lo
d1max=maxd1(1); print("N=1 (rate only): max delta1 = %.3f"%d1max)
for d1 in [d1max*1.5, d1max*2, d1max*3, 2.0]:
    lo,hi=0.0,20.0; ok0=jet_cert(2,(1.0,d1,0.0))
    if not ok0: print("delta1=%.3f N=2: infeasible even with delta2=0"%d1); continue
    for _ in range(12):
        mid=(lo+hi)/2
        if jet_cert(2,(1.0,d1,mid)): lo=mid
        else: hi=mid
    print("delta1=%.3f N=2: max delta2 = %.3f"%(d1,lo))
