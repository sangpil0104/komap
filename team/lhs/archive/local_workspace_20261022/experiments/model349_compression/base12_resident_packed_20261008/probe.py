"""Real GPU reset audit; no synthetic label scoring or speed claims."""
import importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path('/workspace/model349_compress/base12_resident_packed_20261008')
sys.path.insert(0, str(ROOT / 'bundles/control12/src'))
torch.set_num_threads(2)
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = module('old_tent', ROOT / 'bundles/control12/src/tent_style_search.py')
new = module('resident_tent', ROOT / 'bundles/resident12/src/tent_style_search.py')
names = ['restart59_aspp768_seed456', 'restart61_pairwise_margin', 'restart59_aspp768_seed456']
rows = []; model_id = None
for name in names:
    p = ROOT / 'bundles/resident12/weights' / name / 'best.pt'
    expected = old.load_aspp(p, 'cuda:0'); actual = new.load_aspp(p, 'cuda:0')
    if model_id is None: model_id = id(actual)
    assert id(actual) == model_id
    a, b = expected.state_dict(), actual.state_dict()
    exact = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert exact and all(not m.training for m in actual.modules())
    assert all(p.grad is None and p.requires_grad for p in actual.parameters())
    rows.append({'source': name, 'all_parameters_and_buffers_exact': exact, 'state_entries': len(a), 'same_model_object': True, 'grad_reset': True, 'eval_reset': True})
    with torch.no_grad():
        for parameter in actual.parameters(): parameter.add_(0.125); parameter.grad = torch.ones_like(parameter)
        for buffer in actual.buffers(): buffer.add_(1)
    del expected, a, b
assert len(new._pristine_states) == 2 and new._restores == 3
(ROOT / 'probe_result.json').write_text(json.dumps({'restoration_checks': rows, 'cpu_source_cache_count': 2, 'model_constructions': 1, 'scope': 'state restoration after deliberate parameter/BN/counter mutation and source switching; full valid will verify TENT outputs'}, indent=2)+'\n')
print('RESIDENT_RESET_PROBE_PASSED')
