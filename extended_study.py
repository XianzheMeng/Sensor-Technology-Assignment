"""New independent tests: spin-1 x 14N, bias design and gain calibration.
All quantitative results are simulated. No field labels are inferred from public data.
"""
from run_study import *
from scipy.stats import chi2
import concurrent.futures

EXT=OUT/'extended';EXT.mkdir(exist_ok=True)
SX=np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
SY=np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])/np.sqrt(2)
SZ=np.diag([1,0,-1]);I3=np.eye(3)
DZ=np.kron(SZ@SZ,I3);EX=np.kron(SX@SX-SY@SY,I3)
ZE=np.kron(SZ,I3);ZN=np.kron(I3,SZ)
HF=-2.16*np.kron(SZ,SZ)-2.70*(np.kron(SX,SX)+np.kron(SY,SY))-4.95*np.kron(I3,SZ@SZ)
MX=np.kron(SX,I3);MY=np.kron(SY,I3)

def spin_spectrum(theta,f=F,u=U,hyperfine=True):
    """Unpolarized nuclear states, equally averaged transverse MW polarizations.
    Equal Lorentzian linewidth; total oscillator strength is normalized per bias.
    This is a ground-state spectral model, not an optical master-equation simulator.
    """
    theta=np.asarray(theta);B,E,D,g,c=np.moveaxis(theta,-1,0)
    field=B[...,None]+u
    h=D[...,None,None,None]*DZ+E[...,None,None,None]*EX
    h=h+28*field[...,None,None]*ZE
    if hyperfine:h=h+HF-.003077*field[...,None,None]*ZN
    eig,v=np.linalg.eigh(h)
    freqs=eig[...,3:,None]-eig[...,None,:3]
    xx=v.conj().swapaxes(-1,-2)@MX@v
    yy=v.conj().swapaxes(-1,-2)@MY@v
    weight=(abs(xx[...,3:,:3])**2+abs(yy[...,3:,:3])**2)/6
    weight=weight/weight.sum(axis=(-2,-1),keepdims=True)
    delta=(f-freqs[...,None])/g[...,None,None,None,None]
    return 1-2*c[...,None,None]*(weight[...,None]/(1+delta**2)).sum(axis=(-3,-2))

def fit_general(y,u,model,initials,gain_free=False):
    lo=np.r_[LOW,.7] if gain_free else LOW
    hi=np.r_[HIGH,1.3] if gain_free else HIGH
    scale=np.r_[SCALE,.1] if gain_free else SCALE
    def fun(th):
        p=model(th[:5],u=u*th[5] if gain_free else u)
        obs=np.maximum(y,1e-9)
        dev=2*(p-obs+obs*np.log(obs/p))
        return (np.sign(p-obs)*np.sqrt(np.maximum(dev,0))).ravel()
    best=None;cost=np.inf
    for th in initials:
        init=np.r_[th,1.] if gain_free else th
        r=least_squares(fun,np.clip(init,lo+1e-6,hi-1e-6),bounds=(lo,hi),x_scale=scale,max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        if r.cost<cost:best=r.x;cost=r.cost
    return best,2*cost

def dump(name,value):
    (EXT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def spin_tests(n):
    rng=np.random.default_rng(90123);truth=sample_params(rng,n)
    check=float(np.max(abs(spin_spectrum(truth[:30],hyperfine=False)-spectrum(truth[:30]))))
    assert check<1e-7,check
    dump('spin_validation.json',{'zero_hyperfine_reduces_to_two_line_max_error':check,'S':1,'I':1,'A_parallel_MHz':-2.16,'A_perpendicular_MHz':-2.70,'quadrupole_MHz':-4.95,'nuclear_gamma_MHz_per_mT':.003077,'scope':'axial 9-level Hamiltonian, unpolarized nuclear population, transverse polarization average, normalized oscillator weights; not full optical dynamics'})
    models=[]
    import run_study as rs
    def generate_spin(rng,n,scenario='ID',budget=None):
        theta=sample_params(rng,n)
        budgets=10**rng.uniform(5,7,n) if budget is None else np.full(n,budget)
        ratios=[]
        for k in range(0,n,500):
            response=spin_spectrum(theta[k:k+500])
            lam=budgets[k:k+500]/322
            ratios.append(rng.poisson(lam[:,None,None]*response)/lam[:,None,None])
        return theta,np.concatenate(ratios).astype(np.float32),budgets.astype(np.float32)
    original_generate=rs.generate;rs.generate=generate_spin
    for seed in [1,2,3]:
        ckpt=OUT/f'spin9_cnn_{seed}.pt'
        if ckpt.exists():
            m=CNN().to(DEVICE);m.load_state_dict(torch.load(ckpt,weights_only=True,map_location=DEVICE));m.eval()
        else:m=train('spin9_cnn',seed,20000,22)
        models.append(('spin9_cnn',seed,m))
    rs.generate=original_generate
    for seed in [1,2,3]:
        for kind in ['cnn','physics_cnn']:
            m=CNN().to(DEVICE);m.load_state_dict(torch.load(OUT/f'{kind}_{seed}.pt',weights_only=True,map_location=DEVICE));m.eval();models.append((kind,seed,m))
    details=[]
    for budget in [100000,1000000,10000000]:
        ratio=rng.poisson(budget/322*spin_spectrum(truth))/(budget/322)
        for kind,seed,m in models:
            pred,_=predict(m,ratio.astype(np.float32),np.full(n,budget,dtype=np.float32))
            details.append(dict(scenario='spin9_hyperfine',budget=budget,method=kind,seed=seed,truth=truth.tolist(),prediction=pred.tolist()))
        starts=sample_params(np.random.default_rng(6123+budget),n*4).reshape(n,4,5)
        for name,model in [('two_line_MLE4',spectrum),('spin9_MLE4',spin_spectrum)]:
            predictions=[];deviances=[]
            for i,y in enumerate(ratio):
                pred,dev=fit_general(y,U,model,starts[i]);predictions.append(pred);deviances.append(dev*budget/322)
                if i%48==0:print('spin',budget,name,i,flush=True)
            details.append(dict(scenario='spin9_hyperfine',budget=budget,method=name,seed=0,truth=truth.tolist(),prediction=np.asarray(predictions).tolist(),deviance=deviances,misfit_threshold=float(chi2.ppf(.99,317))))
        dump('spin_predictions.json',details)
    fig,ax=plt.subplots(figsize=(6.4,2.4),constrained_layout=True)
    th=truth[0];ax.plot(F,spectrum(th)[0],label='Two-line approximation')
    ax.plot(F,spin_spectrum(th)[0],label='9-level $^{14}$N model');ax.set(xlabel='Microwave frequency (MHz)',ylabel='Normalized fluorescence');ax.legend(fontsize=9)
    fig.savefig(EXT/'spin_model_comparison.png',dpi=250);plt.close(fig)

def design_tests(n):
    truth=sample_params(np.random.default_rng(78234),n);details=[]
    for budget in [100000,1000000,10000000]:
        for sep in [.025,.05,.10,.25]:
            u=np.array([0,sep]);rng=np.random.default_rng(8510+budget+int(sep*1000))
            ratio=rng.poisson(budget/322*spectrum(truth,u=u))/(budget/322)
            starts=sample_params(np.random.default_rng(8610),n*4).reshape(n,4,5)
            pred=np.stack([fit_general(y,u,spectrum,starts[i])[0] for i,y in enumerate(ratio)])
            details.append(dict(scenario='bias_separation',budget=budget,separation_mT=sep,method='MLE4',truth=truth.tolist(),prediction=pred.tolist()))
            print('design',budget,sep,flush=True);dump('design_predictions.json',details)

def gain_tests(n):
    truth=sample_params(np.random.default_rng(91831),n)
    gain=np.random.default_rng(91732).uniform(.95,1.05,n);details=[]
    u3=np.array([0,.125,.25]);u2=np.array([0,.25])
    for budget in [100000,1000000,10000000]:
        rng=np.random.default_rng(91842+budget);starts=sample_params(np.random.default_rng(52345),n*4).reshape(n,4,5)
        observations={}
        for u in [u2,u3]:
            npoint=len(u)*len(F);ratio=np.stack([rng.poisson(budget/npoint*spectrum(t,u=u*g))/(budget/npoint) for t,g in zip(truth,gain)])
            observations[len(u)]=ratio
        for method,u,gain_free in [('two_bias_fixed_gain',u2,False),('three_bias_fixed_gain',u3,False),('three_bias_gain_aware',u3,True),('two_bias_oracle_gain',u2,False),('three_bias_oracle_gain',u3,False)]:
            ratio=observations[len(u)]
            pred=np.stack([fit_general(y,u*gain[i] if 'oracle' in method else u,spectrum,starts[i],gain_free)[0] for i,y in enumerate(ratio)])
            details.append(dict(scenario='bias_gain',budget=budget,method=method,nominal_biases_mT=u.tolist(),truth=truth.tolist(),gain_truth=gain.tolist(),prediction=pred.tolist()))
            print('gain',budget,method,flush=True);dump('gain_predictions.json',details)

def adaptive_tests(n):
    from adaptive_study import acquire
    def batch(k):
        true=sample_params(np.random.default_rng(73100+k),n);records=[]
        for budget in [100000,1000000,10000000]:
            for method in ['uniform','B_only','nuisance_aware']:
                pred=[];traces=[]
                for i,t in enumerate(true):
                    estimate,seconds,trace=acquire(t,budget,method,900000+k*10000+budget+i)
                    pred.append(estimate.tolist());traces.append(trace)
                records.append(dict(scenario='adaptive_replication',batch=k,budget=budget,method=method,truth=true.tolist(),prediction=pred,traces=traces))
                print('active',k,budget,method,flush=True)
                dump(f'adaptive_batch_{k}.json',records)
        return records
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records=sum(list(pool.map(batch,[1,2,3])),[])
    dump('adaptive_predictions.json',records)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['spin','design','gain','adaptive']);p.add_argument('--cases',type=int,default=192);a=p.parse_args()
    torch.set_num_threads(1)
    {'spin':spin_tests,'design':design_tests,'gain':gain_tests,'adaptive':adaptive_tests}[a.mode](a.cases)
if __name__=='__main__':main()
