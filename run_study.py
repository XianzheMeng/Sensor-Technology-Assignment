"""Reproducible NV ODMR simulation study. No hardware data are simulated as experiments.
Units: frequency MHz; field mT. Study assumptions are stated in README/report.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json,time,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
GAMMA=28.0
F=np.linspace(2854,2886,161,dtype=np.float32)
U=np.array([0,.25],dtype=np.float32)
CENTER=np.array([0,2.25,2870,2,.08],dtype=np.float32)
SCALE=np.array([.12,1.75,2,1,.04],dtype=np.float32)
LOW=np.array([-.3,.02,2863,.35,.005])
HIGH=np.array([.3,8,2877,7,.25])
torch.set_num_threads(4)
DEVICE=os.environ.get('NV_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')

def spectrum(theta,f=F,u=U,hyperfine=False):
    theta=np.asarray(theta)
    B,E,D,g,c=np.moveaxis(theta,-1,0)
    z=np.sqrt((GAMMA*(B[...,None]+u))**2+E[...,None]**2)
    delta=f.reshape((1,)*(theta.ndim-1)+(1,-1))-D[...,None,None]
    y=np.ones(delta.shape[:-2]+(len(u),len(f)))
    shifts=[-2.16,0,2.16] if hyperfine else [0]
    for sign in [-1,1]:
        for shift in shifts:
            y-=c[...,None,None]/len(shifts)/(1+((delta-sign*z[...,None]-shift)/g[...,None,None])**2)
    return y

def sample_params(rng,n,scenario='ID'):
    v=rng.uniform(-1,1,(n,5))*SCALE+CENTER
    if scenario=='ZFS_shift':v[:,2]+=3.5
    if scenario=='strain_shift':v[:,1]+=2.5
    if scenario=='linewidth_shift':v[:,3]+=2
    return v.astype(np.float32)

def generate(rng,n,scenario='ID',budget=None):
    theta=sample_params(rng,n,scenario)
    budgets=10**rng.uniform(5,7,n) if budget is None else np.full(n,budget)
    lam=budgets/322
    y=spectrum(theta,hyperfine=scenario=='hyperfine')
    if scenario=='baseline_drift':
        slope=rng.uniform(-.008,.008,(n,1,1))
        y+=slope*(F[None,None,:]-2870)/16
    counts=rng.poisson(lam[:,None,None]*np.maximum(y,1e-6))
    ratios=counts/lam[:,None,None]
    return theta,ratios.astype(np.float32),budgets.astype(np.float32)

class CNN(nn.Module):
    def __init__(self,single=False):
        super().__init__();self.single=single
        self.body=nn.Sequential(nn.Conv1d(1 if single else 2,16,7,padding=3),nn.GELU(),nn.MaxPool1d(2),
          nn.Conv1d(16,32,5,padding=2),nn.GELU(),nn.MaxPool1d(2),
          nn.Conv1d(32,32,5,padding=2),nn.GELU())
        self.head=nn.Sequential(nn.Linear(32*40+1,128),nn.GELU(),nn.Linear(128,5))
    def forward(self,y,budget):
        x=(1-y)/.08
        if self.single:x=x[:,:1]
        z=self.body(x).flatten(1)
        return self.head(torch.cat([z,(torch.log10(budget)-6)[:,None]],1))

def torch_spectrum(normalized):
    center=torch.as_tensor(CENTER,device=DEVICE)
    scale=torch.as_tensor(SCALE,device=DEVICE)
    th=center+scale*normalized
    B,E,D,g,c=th.unbind(1)
    g=g.clamp(.2,8);c=c.clamp(.002,.3)
    f=torch.as_tensor(F,device=DEVICE);u=torch.as_tensor(U,device=DEVICE)
    z=torch.sqrt((GAMMA*(B[:,None]+u))**2+E[:,None]**2+1e-12)
    delta=f[None,None,:]-D[:,None,None]
    y=torch.ones((len(th),2,len(F)),device=DEVICE)
    for sign in [-1,1]:y=y-c[:,None,None]/(1+((delta-sign*z[:,:,None])/g[:,None,None])**2)
    return y.clamp(.01,2)

def train(kind,seed,ntrain,epochs):
    torch.manual_seed(seed);np.random.seed(seed)
    rng=np.random.default_rng(seed)
    theta,y,N=generate(rng,ntrain)
    vt,vy,vN=generate(np.random.default_rng(50000+seed),2000)
    tx=torch.as_tensor(y,device=DEVICE);tn=torch.as_tensor(N,device=DEVICE)
    tt=torch.as_tensor((theta-CENTER)/SCALE,device=DEVICE)
    vx=torch.as_tensor(vy,device=DEVICE);vn=torch.as_tensor(vN,device=DEVICE)
    target=torch.as_tensor((vt-CENTER)/SCALE,device=DEVICE)
    model=CNN(kind=='single_bias').to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=np.inf;history=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();perm=torch.randperm(ntrain,device=DEVICE)
        total=0
        for ids in perm.split(256):
            pred=model(tx[ids],tn[ids]);loss=((pred-tt[ids])**2).mean()
            if kind=='physics_cnn':
                reconstructed=torch_spectrum(pred);obs=tx[ids].clamp_min(1e-6)
                dev=2*(reconstructed-obs+obs*torch.log(obs/reconstructed))/.08**2
                loss=loss+.03*dev.mean()
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():val=float(((model(vx,vn)-target)**2).mean())
        history.append({'epoch':epoch+1,'train_loss':total/ntrain,'validation_loss':val})
        if val<best:
            best=val;torch.save(model.state_dict(),OUT/f'{kind}_{seed}.pt')
        if epoch%5==0 or epoch==epochs-1:print(f'{kind} seed={seed} epoch={epoch+1}/{epochs} val={val:.5f}',flush=True)
    seconds=time.perf_counter()-start
    (OUT/f'{kind}_{seed}_train.json').write_text(json.dumps({'seed':seed,'kind':kind,'ntrain':ntrain,'epochs':epochs,'seconds':seconds,'history':history}),encoding='utf-8')
    model.load_state_dict(torch.load(OUT/f'{kind}_{seed}.pt',weights_only=True,map_location=DEVICE));model.eval()
    return model

def predict(model,y,N):
    start=time.perf_counter()
    with torch.no_grad():out=model(torch.as_tensor(y,device=DEVICE),torch.as_tensor(N,device=DEVICE)).cpu().numpy()
    return out*SCALE+CENTER,(time.perf_counter()-start)*1000/len(y)

def residual(th,y,f=F,u=U):
    p=spectrum(th,f,u)
    obs=np.maximum(y,1e-9)
    dev=2*(p-obs+obs*np.log(obs/p))
    return (np.sign(p-obs)*np.sqrt(np.maximum(dev,0))).ravel()

def fit(y,rng,starts=1,initial=None,f=F,u=U):
    initial=[] if initial is None else [np.asarray(initial)]
    initial+=list(sample_params(rng,max(0,starts-len(initial))))
    best=None;bestcost=np.inf
    for x0 in initial:
        r=least_squares(residual,np.clip(x0,LOW+1e-6,HIGH-1e-6),args=(y,f,u),bounds=(LOW,HIGH),x_scale=SCALE,ftol=1e-7,xtol=1e-7,gtol=1e-7,max_nfev=110)
        if r.cost<bestcost:best=r.x;bestcost=r.cost
    return best

def identifiability():
    a=np.array([.08,2.0,2870,2,.08])
    z=np.sqrt((28*a[0])**2+a[1]**2)
    b=a.copy();b[0]=.04;b[1]=np.sqrt(z*z-(28*b[0])**2)
    errors={str(u):float(np.max(np.abs(spectrum(a,u=np.array([u]))-spectrum(b,u=np.array([u]))))) for u in [0,.25]}
    recovered=[]
    rng=np.random.default_rng(2)
    th=sample_params(rng,10000)
    r=(GAMMA*(th[:,0,None]+U))**2+th[:,1,None]**2
    brec=(r[:,1]-r[:,0]-GAMMA**2*(U[1]**2-U[0]**2))/(2*GAMMA**2*(U[1]-U[0]))
    result={'single_bias_identical_max_error':errors['0'],'second_bias_difference':errors['0.25'],'noise_free_recovery_max_error_mT':float(np.max(np.abs(th[:,0]-brec))),'pairs':[a.tolist(),b.tolist()],'scope':'axial spin-1, known gamma and calibrated biases; E,D fixed across probes'}
    (OUT/'identifiability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),constrained_layout=True)
    for ax,u in zip(axs,[0,.25]):
        ax.plot(F,spectrum(a,u=np.array([u]))[0],label='B=80 uT, E=2.00 MHz')
        ax.plot(F,spectrum(b,u=np.array([u]))[0],'--',label=f'B=40 uT, E={b[1]:.2f} MHz')
        ax.set(title=f'Known bias = {1000*u:.0f} uT',xlabel='Microwave frequency (MHz)',ylabel='Normalized fluorescence');ax.legend(fontsize=8)
    fig.savefig(OUT/'identifiability.png',dpi=220);plt.close(fig)
    print('identifiability',result,flush=True)

def public_audit():
    p=ROOT/'odmr_public.dat'
    if not p.exists():return
    a=np.loadtxt(p,comments='#')
    info={'shape':list(a.shape),'doi':'10.6084/m9.figshare.28788437','source':'https://figshare.com/articles/dataset/28788437','ground_truth_field_labels':'not present in the downloaded data header','noise_model':'header says counts/s but channel /Dev1/AI0 and arbitrary-looking units; do not assume integer Poisson counts'}
    (OUT/'public_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    freq=np.arange(2740,3051)
    if a.shape[0]==len(freq):scans=a
    elif a.shape[1]==len(freq):scans=a.T
    else:print('public shape needs interpretation',a.shape,flush=True);return
    mean=scans.mean(1);idx=np.argmax(np.var(scans,axis=1))
    fig,ax=plt.subplots(figsize=(7,2.7),constrained_layout=True)
    ax.plot(freq,scans[:,0],alpha=.5,lw=.6,label='First sweep');ax.plot(freq,mean,lw=1.2,label=f'Mean of {scans.shape[1]} sweeps')
    ax.set(xlabel='Microwave frequency (MHz)',ylabel='Recorded signal (file units)',title='Public NV-ensemble ODMR data: raw-data audit');ax.legend(fontsize=9)
    fig.savefig(OUT/'public_data.png',dpi=220);plt.close(fig)

def evaluate(models,ncase):
    scenarios=['ID','ZFS_shift','strain_shift','linewidth_shift','hyperfine','baseline_drift']
    budgets=[1e5,1e6,1e7];rows=[];detail=[]
    ct,cy,cN=generate(np.random.default_rng(9031),1500,budget=1e6)
    calibration={}
    for kind,seed,m in models:
        cp,_=predict(m,cy,cN);calibration[(kind,seed)]=float(np.quantile(np.abs(cp[:,0]-ct[:,0]),.95,method='higher'))
    for si,scenario in enumerate(scenarios):
        for bi,budget in enumerate(budgets):
            th,y,N=generate(np.random.default_rng(1000+si*20+bi),ncase,scenario,budget)
            for kind,seed,m in models:
                pred,ms=predict(m,y,N)
                predictions=[(kind,pred,ms)]
                if kind=='physics_cnn':
                    rng=np.random.default_rng(400000+si*20+bi)
                    start=time.perf_counter()
                    hp=np.stack([fit(obs,rng,1,initial=ip) for obs,ip in zip(y,pred)])
                    predictions.append(('hybrid',hp,(time.perf_counter()-start)*1000/ncase+ms))
                for label,prediction,latency in predictions:
                    err=prediction-th
                    coverage=float(np.mean(np.abs(err[:,0])<=calibration[(kind,seed)])) if label!='hybrid' else None
                    rows.append({'scenario':scenario,'budget':int(budget),'method':label,'seed':seed,'ncase':ncase,'B_RMSE_uT':float(np.sqrt(np.mean(err[:,0]**2))*1000),'E_RMSE_MHz':float(np.sqrt(np.mean(err[:,1]**2))),'D_RMSE_MHz':float(np.sqrt(np.mean(err[:,2]**2))),'B_bias_uT':float(np.mean(err[:,0])*1000),'catastrophic_rate_30uT':float(np.mean(np.abs(err[:,0])>.03)),'coverage95_IDcal':coverage,'latency_ms_per_spectrum':latency})
                    detail.append({'scenario':scenario,'budget':int(budget),'method':label,'seed':seed,'truth':th.tolist(),'prediction':prediction.tolist()})
            for starts in [1,4]:
                rng=np.random.default_rng(7000+si*20+bi)
                start=time.perf_counter();pred=np.stack([fit(obs,rng,starts) for obs in y]);ms=(time.perf_counter()-start)*1000/ncase
                err=pred-th
                rows.append({'scenario':scenario,'budget':int(budget),'method':f'Poisson_MLE_{starts}','seed':0,'ncase':ncase,'B_RMSE_uT':float(np.sqrt(np.mean(err[:,0]**2))*1000),'E_RMSE_MHz':float(np.sqrt(np.mean(err[:,1]**2))),'D_RMSE_MHz':float(np.sqrt(np.mean(err[:,2]**2))),'B_bias_uT':float(np.mean(err[:,0])*1000),'catastrophic_rate_30uT':float(np.mean(np.abs(err[:,0])>.03)),'coverage95_IDcal':None,'latency_ms_per_spectrum':ms})
            print(f'evaluated {scenario} budget={budget:.0e}',flush=True)
            (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'predictions.json').write_text(json.dumps(detail),encoding='utf-8')
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    plot_metrics(rows)

def plot_metrics(rows):
    methods=['single_bias','cnn','physics_cnn','hybrid','Poisson_MLE_1','Poisson_MLE_4']
    fig,axs=plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
    for ax,scenario in zip(axs,['ID','strain_shift','hyperfine']):
        for method in methods:
            ys=[]
            for budget in [1e5,1e6,1e7]:
                a=[r['B_RMSE_uT'] for r in rows if r['scenario']==scenario and r['method']==method and r['budget']==budget]
                ys.append(np.mean(a))
            ax.plot([1e5,1e6,1e7],ys,'o-',label=method,ms=3,lw=1)
        ax.set(xscale='log',yscale='log',xlabel='Nominal photon budget',ylabel='Field RMSE (uT)',title=scenario);ax.grid(alpha=.2)
    axs[-1].legend(fontsize=7)
    fig.savefig(OUT/'benchmark.png',dpi=240);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=int,default=20000);parser.add_argument('--epochs',type=int,default=22);parser.add_argument('--cases',type=int,default=192);parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--retrain',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    identifiability();public_audit();models=[]
    for kind in ['cnn','physics_cnn','single_bias']:
        for seed in range(1,args.seeds+1):
            ckpt=OUT/f'{kind}_{seed}.pt'
            if ckpt.exists() and not args.retrain:
                record=OUT/f'{kind}_{seed}_train.json'
                if not record.exists():raise RuntimeError(f'Missing checkpoint provenance: {record}')
                metadata=json.loads(record.read_text())
                if metadata['ntrain']!=args.train or metadata['epochs']!=args.epochs:
                    raise RuntimeError('Cached training protocol differs. Use --retrain to replace checkpoints.')
                model=CNN(kind=='single_bias').to(DEVICE);model.load_state_dict(torch.load(ckpt,weights_only=True,map_location=DEVICE));model.eval()
            else:model=train(kind,seed,args.train,args.epochs)
            models.append((kind,seed,model))
    evaluate(models,args.cases)
    info={'args':vars(args),'device':DEVICE,'torch':torch.__version__,'total_seconds':time.perf_counter()-start,'model_parameters':sum(x.numel() for x in models[0][2].parameters()),'training_parameter_ranges':{'B_mT':[-.12,.12],'E_MHz':[.5,4],'D_MHz':[2868,2872],'linewidth_HWHM_MHz':[1,3],'per_dip_contrast':[.04,.12]},'biases_mT':U.tolist(),'frequency_MHz':[2854,2886,161],'nominal_budget_definition':'expected off-resonance photons summed over 2x161 points; fixed optical flux, total dwell allocation'}
    (OUT/'run_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('DONE',info,flush=True)
if __name__=='__main__':main()
