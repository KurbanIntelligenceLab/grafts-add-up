"""Verify the CORRECTED statements hold, from first principles."""
import numpy as np
from scipy.stats import beta as Beta
rng = np.random.default_rng(3)
L, d = 6, 4
def blk(x, W1, W2, b): return np.tanh(x @ W1 + b) @ W2
def traj(x0, th):
    xs=[x0]
    for l in range(L): xs.append(xs[-1] + blk(xs[-1], *th[l]))
    return xs
def jac(f, x, eps=1e-6):
    x=x.ravel(); n=x.size; J=np.zeros((n,n))
    for i in range(n):
        e=np.zeros(n); e[i]=eps
        J[:,i]=((f((x+e).reshape(1,-1))-f((x-e).reshape(1,-1)))/(2*eps)).ravel()
    return J

x0 = rng.normal(size=(1,d))
base=[(rng.normal(0,.6,(d,d)), rng.normal(0,.6,(d,d)), rng.normal(0,.6,d)) for _ in range(L)]
pert=lambda s:[(W1+s*rng.normal(size=(d,d)),W2+s*rng.normal(size=(d,d)),b+s*rng.normal(size=d)) for W1,W2,b in base]
host, donor = pert(.03), pert(.03)
xh = traj(x0, host)
v=[blk(xh[l],*donor[l])-blk(xh[l],*host[l]) for l in range(L)]
Jh=[jac(lambda z,l=l: blk(z,*host[l]), xh[l]) for l in range(L)]
def prop(ls):
    P=np.eye(d)
    for k in range(L-1,ls-1,-1): P=P@(np.eye(d)+Jh[k])
    return P
P=[prop(l+1) for l in range(L)]
Gamma=max(np.linalg.norm(P[l],2) for l in range(L))

# ---- CORRECTED Corollary 4 bound uses ||grad phi||, not L_phi ---------------
g = rng.normal(size=d)
gradnorm = np.linalg.norm(g)                    # linear readout: grad phi = g
phi = lambda x: float(x.ravel()@g)
S={1,3,4}
# kappa_theta on the segment, estimated by finite differences of d^2/dalpha^2
def v_alpha(l,a):
    th=(host[l][0]+a*(donor[l][0]-host[l][0]), host[l][1]+a*(donor[l][1]-host[l][1]),
        host[l][2]+a*(donor[l][2]-host[l][2]))
    return blk(xh[l],*th)-blk(xh[l],*host[l])
def second_deriv(l,a,h=1e-4):
    return (v_alpha(l,a+h)-2*v_alpha(l,a)+v_alpha(l,a-h))/h**2
kap_dtheta2 = max(np.linalg.norm(second_deriv(l,a)) for l in S for a in (0.1,0.3,0.5,0.7,0.9))

print("CORRECTED Corollary 4:  |rho| <= ||grad phi|| * Gamma * sum_l 0.5*a(1-a)*kappa_theta*||dtheta||^2")
ok=True
for a in (0.1,0.25,0.5,0.75,0.9):
    th_a=[(host[l][0]+a*(donor[l][0]-host[l][0]), host[l][1]+a*(donor[l][1]-host[l][1]),
           host[l][2]+a*(donor[l][2]-host[l][2])) if l in S else host[l] for l in range(L)]
    true=phi(traj(x0,th_a)[L])-phi(xh[L])
    first=a*sum(float((P[l].T@g)@v[l].ravel()) for l in S)
    rho=abs(true-first)
    bound=gradnorm*Gamma*sum(0.5*a*(1-a)*kap_dtheta2 for _ in S)
    holds = rho <= bound
    ok &= holds
    print(f"   alpha={a:<5} |rho|={rho:.3e}  bound={bound:.3e}  {'holds' if holds else 'VIOLATED'}")
print(f"   -> corrected constant {'VERIFIED' if ok else 'FAILS'}")

# ---- Clopper-Pearson upper limit coverage, computed from scratch ------------
def cp_upper(k,n,delta):
    return 1.0 if k==n else Beta.ppf(1-delta, k+1, n-k)
print("\nClopper-Pearson upper limit, independent coverage check (n=200, delta=0.05):")
for pi in (0.01,0.05,0.10,0.30):
    K=rng.binomial(200,pi,20000)
    cov=np.mean([pi<=cp_upper(k,200,0.05) for k in K])
    print(f"   pi={pi:<5} coverage {cov:.4f}  {'OK' if cov>=0.95 else 'FAIL'}")

# ---- subset DP degenerate grid --------------------------------------------
ar=np.zeros(8)
lo,hi=ar[ar<0].sum(), ar[ar>0].sum()
print(f"\nSubset DP: all risk scores zero -> lo={lo}, hi={hi}, h=(hi-lo)/N = 0 "
      f"-> division by zero in the rounding rule (degenerate case unhandled)")

# ---- margin condition soundness, from scratch ------------------------------
bad=0
for t in range(5000):
    r_=np.random.default_rng(t)
    A=r_.normal(size=40); rr=r_.normal(scale=0.3,size=40); T=A+rr
    kh=int(np.argmax(A))
    for i in range(40):
        if i==kh: continue
        if A[kh]-A[i] > abs(rr[i])+abs(rr[kh]) and T[i] > T[kh]:
            bad+=1
print(f"\nMargin condition: false certifications in 5000 random instances = {bad}  "
      f"{'OK' if bad==0 else 'FAIL'}")
