import os; os.environ['RW']='0.01'
import numpy as np, itertools, time
src=open('regional5.py').read().split("t=time.time()")[0]
exec(src)
les=np.logspace(-2,2,17); mus=np.r_[0,np.logspace(-2,1.5,12)]
def best2(jet):
    lo,hi=0.2,4.0; Pbest=None
    for _ in range(14):
        mid=(lo+hi)/2; ok=None
        for le,mu in itertools.product(les,mus):
            Pv=feasible(mid,le,mu,jet,'ball')
            if Pv is not None: ok=Pv;break
        if ok is not None: lo=mid;Pbest=ok
        else: hi=mid
    return lo,Pbest
t=time.time()
bQ,PQ=best2(False); print("quadratic (Prop. 26 with P=diag(P,0)): beta=%.3f (%.0fs)"%(bQ,time.time()-t))
bJ,PJ=best2(True); print("jet: beta=%.3f (%.0fs)"%(bJ,time.time()-t))
np.savez('recheck_R0.01.npz',PJ=PJ,PQ=PQ,bJ=bJ,bQ=bQ)
