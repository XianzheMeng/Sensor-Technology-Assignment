from run_study import *

def main():
    rng=np.random.default_rng(72)
    sx=np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
    sy=np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])/np.sqrt(2)
    sz=np.diag([1,0,-1])
    errors=[]
    for th in sample_params(rng,100):
        B,E,D,_,_=th
        for u in U:
            h=D*(sz@sz)+E*(sx@sx-sy@sy)+28*(B+u)*sz
            eig=np.linalg.eigvalsh(h)
            z=np.sqrt((28*(B+u))**2+E**2)
            errors.append(np.max(np.abs(eig-np.array([0,D-z,D+z]))))
    assert max(errors)<1e-8,max(errors)
    th=sample_params(rng,25)
    with torch.no_grad():ys=torch_spectrum(torch.tensor((th-CENTER)/SCALE,device=DEVICE)).cpu().numpy()
    deviation=np.max(np.abs(ys-spectrum(th)))
    assert deviation<2e-5,deviation
    # The symmetry and the extra bias are independent of training code.
    true=np.array([.08,2,2870,2,.08])
    signflip=true.copy();signflip[0]*=-1
    assert np.allclose(spectrum(true,u=np.array([0.])),spectrum(signflip,u=np.array([0.])))
    assert not np.allclose(spectrum(true),spectrum(signflip))
    result={'Hamiltonian_eigenvalue_max_error_MHz':float(max(errors)),'numpy_torch_forward_max_difference':float(deviation),'zero_bias_sign_symmetry':True,'second_bias_breaks_sign_symmetry':True,'independent_cases':100,'status':'passed'}
    (OUT/'physics_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
if __name__=='__main__':main()
