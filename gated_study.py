"""Residual-gated hybrid: compare warm start and a validation-independent GOF rule."""
from run_study import *
from scipy.stats import chi2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',type=int,default=192);args=parser.parse_args()
    rows=[];details=[];cutoff=float(chi2.ppf(.99,322-5))
    models=[]
    for seed in [1,2,3]:
        m=CNN().to(DEVICE);m.load_state_dict(torch.load(OUT/f'physics_cnn_{seed}.pt',weights_only=True,map_location=DEVICE));m.eval();models.append((seed,m))
    for si,scenario in enumerate(['ID','ZFS_shift','strain_shift','linewidth_shift','hyperfine','baseline_drift']):
        for bi,budget in enumerate([1e5,1e6,1e7]):
            theta,y,N=generate(np.random.default_rng(1000+si*20+bi),args.cases,scenario,budget)
            for seed,m in models:
                nn_pred,nn_time=predict(m,y,N);pred=[];fallback=[];misfit=[]
                start=time.perf_counter()
                rng=np.random.default_rng(400000+si*20+bi)
                for obs,init in zip(y,nn_pred):
                    warm=fit(obs,rng,1,initial=init)
                    score=budget/322*np.sum(residual(warm,obs)**2)
                    retry=score>cutoff
                    if retry:
                        candidate=fit(obs,rng,4,initial=warm)
                        warm=candidate
                    final_score=budget/322*np.sum(residual(warm,obs)**2)
                    pred.append(warm);fallback.append(retry);misfit.append(final_score>cutoff)
                pred=np.stack(pred);err=pred-theta
                rows.append({'scenario':scenario,'budget':int(budget),'method':'gated_hybrid','seed':seed,'ncase':args.cases,'B_RMSE_uT':float(np.sqrt(np.mean(err[:,0]**2))*1000),'E_RMSE_MHz':float(np.sqrt(np.mean(err[:,1]**2))),'D_RMSE_MHz':float(np.sqrt(np.mean(err[:,2]**2))),'fallback_rate':float(np.mean(fallback)),'model_misfit_flag_rate':float(np.mean(misfit)),'catastrophic_rate_30uT':float(np.mean(np.abs(err[:,0])>.03)),'latency_ms_per_spectrum':(time.perf_counter()-start)*1000/len(y)+nn_time})
                details.append({'scenario':scenario,'budget':int(budget),'seed':seed,'truth':theta.tolist(),'prediction':pred.tolist()})
            print('gated',scenario,int(budget),flush=True)
    (OUT/'gated_metrics.json').write_text(json.dumps({'chi_square_threshold':cutoff,'degrees_of_freedom':317,'rows':rows},indent=2),encoding='utf-8')
    (OUT/'gated_predictions.json').write_text(json.dumps(details),encoding='utf-8')
if __name__=='__main__':main()
