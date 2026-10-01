"""CPU checks for physics, saved metrics, and every supplied checkpoint.

Does not retrain models, rerun the benchmark, or rewrite recorded evidence.
"""
import os
os.environ['NV_DEVICE'] = 'cpu'
import json
from run_study import CENTER, SCALE, U, CNN, OUT, generate, sample_params, spectrum, torch_spectrum
from extended_study import SX, SY, SZ, spin_spectrum
import numpy as np
import torch


def main():
    rng = np.random.default_rng(72)
    parameters = sample_params(rng, 100)
    maximum = 0.0
    for field, strain, splitting, _, _ in parameters:
        for bias in U:
            hamiltonian = splitting * (SZ @ SZ) + strain * (SX @ SX - SY @ SY)
            hamiltonian = hamiltonian + 28 * (field + bias) * SZ
            separation = np.sqrt((28 * (field + bias)) ** 2 + strain ** 2)
            expected = np.array([0, splitting - separation, splitting + separation])
            maximum = max(maximum, float(np.max(np.abs(np.linalg.eigvalsh(hamiltonian) - expected))))
    assert maximum < 1e-8, maximum
    with torch.no_grad():
        response = torch_spectrum(torch.tensor((parameters - CENTER) / SCALE)).numpy()
    assert np.max(np.abs(response - spectrum(parameters))) < 2e-5
    spin_error = float(np.max(np.abs(spin_spectrum(parameters[:30], hyperfine=False) - spectrum(parameters[:30]))))
    assert spin_error < 1e-7, spin_error
    state = np.array([.08, 2., 2870., 2., .08])
    flipped = state.copy()
    flipped[0] *= -1
    assert np.allclose(spectrum(state, u=np.array([0.])), spectrum(flipped, u=np.array([0.])))
    assert not np.allclose(spectrum(state), spectrum(flipped))

    _, ratios, budgets = generate(np.random.default_rng(987), 4, budget=1e6)
    checkpoints = sorted(OUT.glob('*.pt'))
    assert len(checkpoints) == 12, len(checkpoints)
    for checkpoint in checkpoints:
        metadata = json.loads(checkpoint.with_name(checkpoint.stem + '_train.json').read_text())
        assert metadata['ntrain'] == 20000 and metadata['epochs'] == 22
        model = CNN(single=checkpoint.stem.startswith('single_bias')).eval()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
        with torch.no_grad():
            prediction = model(torch.tensor(ratios), torch.tensor(budgets)).numpy()
        assert prediction.shape == (4, 5) and np.isfinite(prediction).all(), checkpoint

    metrics = json.loads((OUT / 'metrics.json').read_text())
    metrics += json.loads((OUT / 'gated_metrics.json').read_text())['rows']
    original = json.loads((OUT / 'summary.json').read_text())
    assert len(original) == 126, len(original)
    for row in original:
        selected = [r for r in metrics if all(r[k] == row[k] for k in ('scenario', 'budget', 'method'))]
        assert selected
        assert np.isclose(np.mean([r['B_RMSE_uT'] for r in selected]), row['B_RMSE_uT_mean'], rtol=1e-10)
    extended = json.loads((OUT / 'extended' / 'summary.json').read_text())
    assert len(extended) == 51, len(extended)
    sources = {kind: json.loads((OUT / 'extended' / filename).read_text()) for kind, filename in [
        ('spin', 'spin_predictions.json'), ('design', 'design_predictions.json'),
        ('gain', 'gain_predictions.json'), ('adaptive', 'adaptive_predictions.json')]}
    for row in extended:
        selected = [r for r in sources[row['experiment']] if r['budget'] == row['budget']
                    and r['method'] == row['method'] and r.get('separation_mT') == row['separation_mT']]
        errors = [(np.asarray(r['prediction'])[:, 0] - np.asarray(r['truth'])[:, 0]) * 1000 for r in selected]
        assert errors
        value = (np.sqrt(np.mean(np.concatenate(errors) ** 2)) if row['experiment'] == 'adaptive'
                 else np.mean([np.sqrt(np.mean(e ** 2)) for e in errors]))
        assert np.isclose(value, row['B_RMSE_uT'], rtol=1e-10), row
    print(f'PASS: axial and spin-9 physics; {len(checkpoints)} CPU checkpoint loads/inferences; '
          f'{len(original)} original and {len(extended)} extended summary groups.')


if __name__ == '__main__':
    main()
