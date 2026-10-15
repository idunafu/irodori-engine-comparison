"""Export static old/new runtime scaling figures from validated aggregates."""
import collections
import csv
import os
from pathlib import Path
import statistics as stats

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from runtime_ab import DEST


def main():
    rows = list(csv.DictReader((DEST / 'per-input.csv').open()))
    grouped = collections.defaultdict(list)
    for r in rows:
        if r['phase']=='expanded' and r['complete']=='True' and r['cache']=='native_cache':
            grouped[(r['case'],r['mode'])].append(r)
    cases = sorted({c for c,m in grouped})
    if not cases:
        return
    for mode, xlabel in (('audio_scaling','Requested audio (s)'), ('text_scaling','Text tokens')):
        fig, axes = plt.subplots((len(cases)+1)//2, 2, figsize=(12,3.5*((len(cases)+1)//2)), squeeze=False)
        for ax, case in zip(axes.flat,cases):
            xs = grouped[(case,mode)]
            key = 'seconds' if mode=='audio_scaling' else 'tokens'
            bins = collections.defaultdict(list)
            for r in xs:
                bins[float(r[key])].append(r)
            x = sorted(bins)
            for version, label in (('old','Original runtime'),('new','Latest / CUDA 13.0')):
                y = [stats.median(float(r[version+'_ms']) for r in bins[v]) for v in x]
                ax.plot(x,y,'o-',label=label,linewidth=1.5,markersize=4)
            ax.set_title(case,fontsize=9)
            ax.set(xlabel=xlabel,ylabel='Warm inference (ms)')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(DEST/f'{mode}.png',dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
