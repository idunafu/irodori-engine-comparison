"""Validate A/B coverage, all outputs, preservation, and actual installed runtimes."""
import json
import subprocess

import runtime_ab as ab
from runtime_ab_setup import ENV, snapshot
from runtime_ab_provenance import digest


def main():
    data = ab.records()
    decision = json.loads((ab.OUT / 'decision.json').read_text())
    rows = ab.report(data, decision, final=True)
    assert len([r for r in rows if r['phase'] == 'screen']) == 64
    for r in data:
        if r['kind'] not in ('wall', 'profile', 'cold', 'preflight'):
            continue
        results = r['outputs'] if r['kind'] == 'preflight' else [r['result']]
        for out in results:
            assert out['finite'] and out['rms'] > 1e-8 and out['sample_rate'] == 48000
            assert out['samples'] > 0 and out['frames'] == (out['samples'] + 1919) // 1920
            if r['kind'] in ('wall', 'profile'):
                ab.check(out, r['seconds'], r['kind'] == 'profile', r['case'])
    current = snapshot()
    assert current == json.loads((ab.OUT / 'preservation.json').read_text())
    provenance = json.loads((ab.OUT / 'provenance.json').read_text())
    for name, expected in provenance['hashes'].items():
        assert digest(ab.ROOT / name) == expected, name
    for engine in ('Irodori-TTS', 'Irodori-TTS-Optimiz', 'onnx'):
        for prefix, filename in (('.venvs', f'{engine}-original-packages.txt'),
                                 ('.venvs/runtime-ab-20261015', f'{engine}-packages.txt')):
            actual = subprocess.check_output(['uv', 'pip', 'freeze', '--python',
                        str(ab.ROOT / prefix / engine / 'bin/python')], env=ENV, text=True)
            assert actual == (ab.OUT / filename).read_text(), (engine, prefix)
    # The Rust engine exercises ORT directly; also smoke-test the newly installed
    # Python ONNX/export packages against the unchanged graphs.
    probe = '''import json,sys
from pathlib import Path
import onnx,onnxscript,onnxruntime
models = sorted(Path(sys.argv[1]).glob('*.onnx'))
assert models
for path in models:
    graph = onnx.load_model(str(path), load_external_data=False)
    assert graph.graph.node
providers = onnxruntime.get_available_providers()
assert 'CUDAExecutionProvider' in providers
print(json.dumps(dict(onnx=onnx.__version__,onnxscript=onnxscript.__version__,
                     onnxruntime=onnxruntime.__version__,providers=providers,
                     models_read=len(models))))
'''
    tooling = json.loads(subprocess.check_output([
        str(ab.ROOT / '.venvs/runtime-ab-20261015/onnx/bin/python'), '-c', probe,
        str(ab.ROOT / 'artifacts/onnx/export/ref')], env=ENV, text=True))
    expected = json.loads((ab.ROOT / 'environments/runtime-ab-20261015/versions.json').read_text())
    for field, package in (('onnx', 'onnx'), ('onnxscript', 'onnxscript'), ('onnxruntime', 'onnxruntime-gpu')):
        assert tooling[field] == expected[package]['version'], tooling
    telemetry = [json.loads(s) for s in (ab.OUT / 'telemetry.jsonl').read_text().splitlines()]
    # The only compute PIDs should belong to the single worker of each request.
    concurrent = [p for p in telemetry if sum('compute' in x['kinds'] for x in p['processes']) > 1]
    assert not concurrent, 'Concurrent GPU compute processes observed'
    validation = dict(records=len(data), screen_conditions=64,
                      expanded_conditions=len(decision['affected'])*130,
                      original_environments_preserved=True, package_freezes_preserved=True,
                      concurrent_compute_observations=len(concurrent),
                      onnx_tooling=tooling,
                      temperature_range=[min(p['temperature'] for p in telemetry), max(p['temperature'] for p in telemetry)],
                      clock_range=[min(p['sm_clock'] for p in telemetry), max(p['sm_clock'] for p in telemetry)])
    (ab.DEST / 'independent-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
