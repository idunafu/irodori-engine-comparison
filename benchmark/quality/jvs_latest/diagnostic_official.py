"""Run the pinned official worker with reference caches local to this diagnostic."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
source_path=HERE.parent/'python_worker.py'
source=source_path.read_text()
anchor="runtime._reference_latent_cache_dir = ROOT / 'artifacts/quality/reference-cache'"
assert source.count(anchor)==1
source=source.replace(anchor,"runtime._reference_latent_cache_dir = Path(os.environ['QUALITY_REFERENCE_CACHE'])")
exec(compile(source,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
