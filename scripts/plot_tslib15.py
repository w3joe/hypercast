"""Render completed held-out comparisons without selecting test-set winners."""
import argparse
import json
from pathlib import Path
import numpy as np

def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    report=json.loads((run/'analysis.json').read_text());rows=report['bootstrap_intervals'][str(report['primary_block_length'])]
    models=list(dict.fromkeys(r['backbone'] for r in rows));arms=['complex','quaternion','octonion']
    lookup={(r['backbone'],r['arm']):r for r in rows}
    values=np.array([[lookup[(b,a)]['improvement_pct'] for a in arms] for b in models])
    bound=max(2,float(np.max(np.abs(values))))
    fig,ax=plt.subplots(figsize=(8,9));fig.patch.set_facecolor('#fafafa')
    heat=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    ax.set_xticks(range(3),['2D complex','4D quaternion','8D octonion']);ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)),models);ax.tick_params(length=0,pad=8)
    for i,b in enumerate(models):
        for j,a in enumerate(arms):
            r=lookup[(b,a)];star=' *' if r['lower_pct']>0 else ''
            ax.text(j,i,f"{r['improvement_pct']:+.2f}%{star}",ha='center',va='center',fontsize=10,
                color='white' if abs(r['improvement_pct'])>bound*.6 else '#17212b')
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('HyperDense accuracy versus controlled dense',x=.05,ha='left',y=.99,fontsize=17,fontweight='bold')
    fig.text(.05,.944,'ETTh1 · 15 backbones · 3 paired seeds · 32-hour context / 5-hour forecast',fontsize=10,color='#444')
    fig.colorbar(heat,ax=ax,fraction=.045,pad=.04,label='MAE reduction (%) · positive is better')
    fig.text(.05,.038,'* Simultaneous 95% interval excludes zero in favour of HyperDense.\nSelected internal sites; FiLM is a spectral intervention. See report for native and low-rank controls.',fontsize=9,color='#444')
    fig.subplots_adjust(left=.22,right=.9,top=.87,bottom=.09)
    fig.savefig(run/'accuracy-comparison.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);plot(p.parse_args().run)
