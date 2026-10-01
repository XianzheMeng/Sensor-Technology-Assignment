"""Matched-budget pilot of classical adaptive measurement with unknown nuisances.
This is not a claimed new Fisher-design algorithm or a neural policy.
"""
from run_study import *
from scipy.stats import chi2

def observation(theta,f,u,photons,rng):
    z=np.sqrt((28*(theta[0]+u))**2+theta[1]**2)
    p=np.ones(len(f))
    for sign in [-1,1]:p-=theta[4]/(1+((f-theta[2]-sign*z)/theta[3])**2)
    return p

def estimate(f,u,ratio,lam,initial,rng,starts=1):
    def fun(th):
        pred=observation(th,f,u,lam,rng);obs=ratio.clip(1e-9)
        dev=2*lam*(pred-obs+obs*np.log(obs/pred))
        return np.sign(pred-obs)*np.sqrt(np.maximum(dev,0))
    candidates=[initial]+list(sample_params(rng,starts-1))
    best=None;cost=np.inf
    for init in candidates:
        r=least_squares(fun,np.clip(init,LOW+1e-6,HIGH-1e-6),bounds=(LOW,HIGH),x_scale=SCALE,max_nfev=90)
        if r.cost<cost:best=r.x;cost=r.cost
    if 2*cost>chi2.ppf(.99,len(f)-5):
        for init in sample_params(rng,12):
            r=least_squares(fun,np.clip(init,LOW+1e-6,HIGH-1e-6),bounds=(LOW,HIGH),x_scale=SCALE,max_nfev=90)
            if r.cost<cost:best=r.x;cost=r.cost
    return best

def gradient(th,f,u):
    grad=[]
    for k in range(5):
        up=th.copy();dn=th.copy();h=SCALE[k]*1e-4
        up[k]+=h;dn[k]-=h
        grad.append((observation(up,f,u,0,None)-observation(dn,f,u,0,None))/(2e-4))
    return np.stack(grad,1)

def acquire(theta,budget,method,seed):
    start=time.perf_counter()
    rng=np.random.default_rng(seed)
    coarse=np.linspace(2854,2886,16)
    f=np.tile(coarse,2);u=np.repeat(U,16);lam=np.full(32,budget*.25/32)
    p=observation(theta,f,u,lam,rng);counts=rng.poisson(lam*p)
    ratio=counts/lam
    th=estimate(f,u,ratio,lam,CENTER.copy(),rng,4)
    candf=np.tile(np.linspace(2854,2886,129),2);candu=np.repeat(U,129)
    uniformf=np.tile(np.linspace(2854,2886,32),2);uniformu=np.repeat(U,32)
    late=budget*.75/64
    traj=[]
    for b in range(8):
        if method=='uniform':
            nf=uniformf[b*8:(b+1)*8];nu=uniformu[b*8:(b+1)*8]
        else:
            grad=gradient(th,f,u);p=observation(th,f,u,lam,rng)
            # Prior regularization is only a numerical design aid, not a CRLB.
            J=3*np.eye(5)+(grad.T*(lam/np.maximum(p,.01)))@grad
            cg=gradient(th,candf,candu);cp=observation(th,candf,candu,late,rng)
            nf=[];nu=[]
            for _ in range(8):
                if method=='B_only':score=late*cg[:,0]**2/cp
                else:
                    cov=np.linalg.inv(J);v=cg@cov[:,0]
                    denominator=cp/late+np.einsum('ij,jk,ik->i',cg,cov,cg)
                    score=v*v/denominator
                idx=int(np.argmax(score));nf.append(candf[idx]);nu.append(candu[idx])
                g=cg[idx];J+=late/cp[idx]*np.outer(g,g)
            nf=np.array(nf);nu=np.array(nu)
        nl=np.full(8,late);npred=observation(theta,nf,nu,nl,rng)
        nc=rng.poisson(nl*npred)
        f=np.r_[f,nf];u=np.r_[u,nu];lam=np.r_[lam,nl];ratio=np.r_[ratio,nc/nl]
        th=estimate(f,u,ratio,lam,th,rng,1)
        traj.append(float(th[0]))
    return th,time.perf_counter()-start,{'frequency_MHz':f.tolist(),'bias_mT':u.tolist(),'lambda':lam.tolist(),'trajectory_B_mT':traj}

def main():
    rows=[];traces=[]
    for bi,budget in enumerate([1e5,1e6,1e7]):
        th=sample_params(np.random.default_rng(1223),96)
        for method in ['uniform','B_only','nuisance_aware']:
            estimates=[];seconds=[]
            for i,truth in enumerate(th):
                pred,sec,trace=acquire(truth,budget,method,2000+bi*100+i)
                estimates.append(pred);seconds.append(sec)
                if i==0:traces.append({'budget':int(budget),'method':method,'truth':truth.tolist(),**trace})
            e=np.stack(estimates)-th
            rows.append({'budget':int(budget),'method':method,'ncase':len(th),'B_RMSE_uT':float(np.sqrt(np.mean(e[:,0]**2))*1000),'E_RMSE_MHz':float(np.sqrt(np.mean(e[:,1]**2))),'catastrophic_rate_30uT':float(np.mean(abs(e[:,0])>.03)),'mean_computation_seconds':float(np.mean(seconds))})
            print('adaptive',rows[-1],flush=True)
            (OUT/'adaptive_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'adaptive_traces.json').write_text(json.dumps(traces),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),constrained_layout=True)
    for method in ['uniform','B_only','nuisance_aware']:
        a=[r for r in rows if r['method']==method]
        axs[0].plot([r['budget'] for r in a],[r['B_RMSE_uT'] for r in a],'o-',label=method)
        axs[1].plot([r['budget'] for r in a],[r['mean_computation_seconds'] for r in a],'o-',label=method)
    axs[0].set(xscale='log',yscale='log',xlabel='Nominal photon budget',ylabel='Field RMSE (uT)',title='Adaptive measurement pilot')
    axs[1].set(xscale='log',xlabel='Nominal photon budget',ylabel='Computation time per trial (s)',title='Measured software overhead')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'adaptive.png',dpi=220);plt.close(fig)
if __name__=='__main__':main()
