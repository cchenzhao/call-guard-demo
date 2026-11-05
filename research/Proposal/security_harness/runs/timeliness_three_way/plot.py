import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent
r=json.loads((p/'report.json').read_text()); modes=[('rules_only','Rules only','#ce7224','o'),('minilm_only','MiniLM only','#506ac1','s'),('cascade','Rules + MiniLM','#0a806a','^')]
fig,ax=plt.subplots(figsize=(11,6.7)); summary=r['summary']
for key,label,color,marker in modes:
 group=[a for a in r['results'] if a['scheme']==key and a['target']=='INTERVENE']; denom=len(group)
 vals=sorted(x['delta_ms'] for x in group if x['delta_ms'] is not None)
 y=np.arange(1,len(vals)+1)*100/denom
 ax.scatter(vals,y,c=color,marker=marker,s=35,alpha=.85,edgecolor='white',linewidth=.35,label=f'{label}: {len(vals)}/{denom} warned')
 ax.step(vals,y,where='post',color=color,lw=1.1,alpha=.42)
ax.axvline(0,color='#5d6670',lw=1,ls='--');ax.text(30,3,'Password onset',fontsize=9,color='#515860')
ax.set_xlabel('Decision time relative to estimated password onset: T3 − T1 (ms)',labelpad=12)
ax.set_ylabel('Risk scenarios with a warning decision by this time (%)',labelpad=12)
ax.set_title('When do warning decisions occur relative to password onset?',fontsize=16,weight='bold',pad=15)
ax.text(.5,1.005,'EXPLORATORY · synthetic voice / cached ASR · estimated onset and replayed timing',transform=ax.transAxes,ha='center',fontsize=10,color='#59666d')
ax.grid(axis='both',alpha=.16);ax.set_ylim(0,102);ax.set_yticks([0,20,40,60,80,100]);ax.legend(loc='upper left',frameon=True)
fig.text(.10,.025,'Earlier than zero = before password begins, not necessarily a justified warning. Misses remain in the denominator (75).',fontsize=9,color='#555')
fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(p/'timeliness_scatter.png',dpi=200);fig.savefig(p/'timeliness_scatter.pdf');plt.close(fig)
print('Wrote scatter plot in PNG and PDF')
