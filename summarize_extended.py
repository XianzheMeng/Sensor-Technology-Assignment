"""Summarize saved extended predictions and paired, stratified test bootstrap."""
from extended_study import *

def main():
    rng=np.random.default_rng(83021);rows=[]
    for kind,filename in [('spin','spin_predictions.json'),('design','design_predictions.json'),('gain','gain_predictions.json'),('adaptive','adaptive_predictions.json')]:
        source=json.loads((EXT/filename).read_text());keys=[]
        for d in source:
            key=(d['budget'],d['method'],d.get('separation_mT'))
            if key not in keys:keys.append(key)
        for budget,method,sep in keys:
            ds=[d for d in source if (d['budget'],d['method'],d.get('separation_mT'))==(budget,method,sep)]
            errs=[(np.asarray(d['prediction'])[:,0]-np.asarray(d['truth'])[:,0])*1000 for d in ds]
            if kind=='adaptive':
                flat=np.concatenate(errs);point=float(np.sqrt(np.mean(flat**2)));boots=[]
                for _ in range(2000):boots.append(np.sqrt(np.mean(np.concatenate([e[rng.integers(len(e),size=len(e))] for e in errs])**2)))
                ncase=len(flat)
            else:
                e=np.stack(errs);point=float(np.mean(np.sqrt(np.mean(e**2,axis=1))))
                idx=rng.integers(e.shape[1],size=(2000,e.shape[1]))
                boots=np.sqrt((e[:,idx]**2).mean(axis=-1)).mean(axis=0);flat=e.flatten();ncase=e.shape[1]
            row={'experiment':kind,'budget':budget,'method':method,'separation_mT':sep,'ncase':ncase,'B_RMSE_uT':point,'test_case_bootstrap95_uT':np.quantile(boots,[.025,.975]).tolist(),'bias_uT':float(np.mean(flat)),'failure30uT':float(np.mean(abs(flat)>30))}
            if 'deviance' in ds[0]:row['misfit_flag_rate']=float(np.mean(np.asarray(ds[0]['deviance'])>ds[0]['misfit_threshold']))
            rows.append(row)
    source=json.loads((EXT/'adaptive_predictions.json').read_text());differences=[]
    for budget in [100000,1000000,10000000]:
        a=[d for d in source if d['budget']==budget and d['method']=='uniform'];b=[d for d in source if d['budget']==budget and d['method']=='nuisance_aware']
        a.sort(key=lambda d:d['batch']);b.sort(key=lambda d:d['batch']);ae=[];be=[]
        for x,y in zip(a,b):
            assert x['batch']==y['batch'] and np.allclose(x['truth'],y['truth'])
            ae.append((np.array(x['prediction'])[:,0]-np.array(x['truth'])[:,0])*1000);be.append((np.array(y['prediction'])[:,0]-np.array(y['truth'])[:,0])*1000)
        boot=[]
        for _ in range(2000):
            ids=[rng.integers(len(e),size=len(e)) for e in ae]
            x=np.concatenate([e[i] for e,i in zip(ae,ids)]);y=np.concatenate([e[i] for e,i in zip(be,ids)])
            boot.append(np.sqrt(np.mean(x*x))-np.sqrt(np.mean(y*y)))
        point=np.sqrt(np.mean(np.concatenate(ae)**2))-np.sqrt(np.mean(np.concatenate(be)**2))
        differences.append({'budget':budget,'uniform_minus_aware_RMSE_uT':float(point),'paired_stratified_bootstrap95_uT':np.quantile(boot,[.025,.975]).tolist(),'batches':3,'cases_per_batch':len(ae[0])})
    dump('summary.json',rows);dump('adaptive_paired.json',differences)
    with (EXT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    proof_rng=np.random.default_rng(77001);th=sample_params(proof_rng,10000);g=proof_rng.uniform(.95,1.05,10000)
    u=np.array([0.,.125,.25]);r=(28*(th[:,0,None]+g[:,None]*u))**2+th[:,1,None]**2
    coeff=np.linalg.solve(np.stack([u*u,u,np.ones(3)],axis=1),r.T).T
    ghat=np.sqrt(coeff[:,0])/28;Bhat=coeff[:,1]/(2*28**2*ghat)
    assert np.max(abs(ghat-g))<1e-10;assert np.max(abs(Bhat-th[:,0]))<1e-10
    dump('gain_algebra_validation.json',{'independent_cases':10000,'max_gain_error':float(np.max(abs(ghat-g))),'max_B_error_mT':float(np.max(abs(Bhat-th[:,0]))),'status':'passed'})
    print('Extended summary',len(rows),'groups; paired differences',differences,flush=True)

if __name__=='__main__':main()
