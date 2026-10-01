from run_study import *
import hashlib

def main():
    rows=json.loads((OUT/'metrics.json').read_text())
    gated=json.loads((OUT/'gated_metrics.json').read_text())
    rows+=gated['rows']
    detail=json.loads((OUT/'predictions.json').read_text())
    detail += [dict(d,method='gated_hybrid') for d in json.loads((OUT/'gated_predictions.json').read_text())]
    summary=[];rng=np.random.default_rng(501)
    for scenario in ['ID','ZFS_shift','strain_shift','linewidth_shift','hyperfine','baseline_drift']:
        for budget in [100000,1000000,10000000]:
            for method in ['single_bias','cnn','physics_cnn','hybrid','gated_hybrid','Poisson_MLE_1','Poisson_MLE_4']:
                group=[r for r in rows if r['scenario']==scenario and r['budget']==budget and r['method']==method]
                point=float(np.mean([r['B_RMSE_uT'] for r in group]))
                obj={'scenario':scenario,'budget':budget,'method':method,'B_RMSE_uT_mean':point,'B_RMSE_uT_seed_sd':float(np.std([r['B_RMSE_uT'] for r in group],ddof=1)) if len(group)>1 else None,'ncase':group[0]['ncase']}
                d=[q for q in detail if q['scenario']==scenario and q['budget']==budget and q['method']==method]
                if d:
                    square=np.array([(np.asarray(q['prediction'])[:,0]-np.asarray(q['truth'])[:,0])**2 for q in d])
                    idx=rng.integers(0,square.shape[1],size=(1000,square.shape[1]))
                    boot=np.sqrt(square[:,idx].mean(2)).mean(0)*1000
                    obj['test_case_bootstrap95_uT']=[float(x) for x in np.quantile(boot,[.025,.975])]
                summary.append(obj)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['scenario','budget','method','B_RMSE_uT_mean','B_RMSE_uT_seed_sd','ncase','test_case_bootstrap95_uT'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(summary)
    plot_metrics(rows)
    fig,ax=plt.subplots(figsize=(7,2.8),constrained_layout=True)
    scenarios=['ID','ZFS_shift','strain_shift','hyperfine']
    for j,method in enumerate(['cnn','physics_cnn','single_bias']):
        vals=[np.mean([r['coverage95_IDcal'] for r in rows if r['method']==method and r['scenario']==s and r['budget']==1000000]) for s in scenarios]
        ax.bar(np.arange(4)+(j-1)*.23,vals,width=.22,label=method)
    ax.axhline(.95,c='black',ls='--',lw=1);ax.set(xticks=np.arange(4),xticklabels=scenarios,ylim=(0,1.03),ylabel='Coverage',title='95% interval calibrated on independent ID data at budget 1e6');ax.legend(fontsize=8)
    fig.savefig(OUT/'coverage.png',dpi=220);plt.close(fig)
    public=np.loadtxt(ROOT/'odmr_public.dat',comments='#')
    baseline=public[:,:20].mean(1)
    lag=float(np.corrcoef(baseline[:-1],baseline[1:])[0,1])
    centered=public-public[:3754].mean(0)
    train=centered[:3754]
    cov=train.T@train/(len(train)-1)
    eigen,vec=np.linalg.eigh(cov);eigen=eigen[::-1];vec=vec[:,::-1]
    projection=centered@vec[:,:3]
    stats={'scans':len(public),'frequency_points':public.shape[1],'baseline_lag1_correlation':lag,'pca_train_scans':len(train),'top3_variance_fractions':(eigen[:3]/eigen.sum()).tolist(),'pca_test_scans':len(public)-len(train),'sha256':hashlib.sha256((ROOT/'odmr_public.dat').read_bytes()).hexdigest(),'interpretation':'Descriptive time correlation and PCA only; no inference of field truth or shot-noise model.'}
    (OUT/'public_noise_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),constrained_layout=True)
    axs[0].plot(baseline,lw=.5);axs[0].set(xlabel='Sweep index',ylabel='Off-resonance mean (file units)',title=f'Lag-1 correlation = {lag:.3f}')
    axs[1].plot(projection[:,0],lw=.5);axs[1].axvline(3754,c='red',ls='--',lw=1);axs[1].set(xlabel='Sweep index',ylabel='First PCA score',title='PCA basis fitted on first 80% of sweeps')
    fig.savefig(OUT/'public_noise.png',dpi=220);plt.close(fig)
    print('public audit',stats,flush=True)
    for s in scenarios:
        print(s,[(r['method'],round(r['B_RMSE_uT_mean'],2)) for r in summary if r['scenario']==s and r['budget']==1000000],flush=True)
if __name__=='__main__':main()
