"""FP32 reference intervention, using the existing pinned official diagnostic."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/quality/runs/20261018-jvs/analysis/one-reference/official.py'
# Keep the original experiment and its hashes immutable. This small overlay adds
# automatic length only for the oracle pass; controlled requests supply seconds.
source = SOURCE.read_text()
anchor = "ROOT=OUT.parents[5]"
assert source.count(anchor) == 1
source = source.replace(anchor, 'ROOT=Path(' + repr(str(ROOT)) + ')')
anchor = "seconds=r['seconds']"
assert source.count(anchor) == 1
source = source.replace(anchor, "seconds=r.get('seconds')")
anchor = "if not used and len(shape)==3 and shape[-1]==32:"
assert source.count(anchor) == 1
source = source.replace(anchor, "if not used and not r.get('native_noise') and len(shape)==3 and shape[-1]==32:")
anchor = 'assert used\n'
assert source.count(anchor) == 1
source = source.replace(anchor, "assert used or r.get('native_noise')\n")
source = source.replace('noise_used=True', 'noise_used=used')
exec(compile(source, str(SOURCE), 'exec'), {'__name__': '__main__', '__file__': str(SOURCE)})
