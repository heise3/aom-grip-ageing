# Academic Figure Skill Asset Confirmation
# Existing manuscript production assets reused as native vector renderers.
# Data: permitted aggregate CSVs only; no individual records and no model fitting.
"""Figure4 and five-cohort supplement, aggregate-only. No data/model imports."""
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 8, "figure.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False,
})
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
GREY,BLACK = "#999999","#222222"
mpl.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none","savefig.bbox":"tight","savefig.dpi":300,"savefig.pad_inches":0})
mpl.use('Agg')
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

B=Path(__file__).resolve().parent
fig_width_mm=183
DEFS=['EWGSOP2','SDOC','FNIH'];PROG=['CHARLS','MHAS','SHARE'];FIVE=['CHARLS','ELSA','MHAS','SHARE','KLoSA']
MARK={'EWGSOP2':'o','SDOC':'s','FNIH':'D'}
LOW,NORMAL='#C8432B','#2F6DB5'

def aggregate_csv(path):
    d=pd.read_csv(path)
    if {'id','pid','mergeid','person','hhidpn','idauniq','source_row'}&set(d.columns):raise RuntimeError('PARTICIPANT_INPUT_FORBIDDEN')
    if len(d)>15 or not {'cohort','definition'}.issubset(d):raise RuntimeError('NOT_EXPECTED_AGGREGATE_SCHEMA')
    return d

def dot(ax,x,lo,hi,y,marker='o',filled=True,color=BLACK):
    ax.plot([lo,hi],[y,y],lw=1,color=color)
    ax.plot([lo,lo],[y-.045,y+.045],lw=.7,color=color);ax.plot([hi,hi],[y-.045,y+.045],lw=.7,color=color)
    ax.plot(x,y,marker=marker,ms=3.5,mec=color,mfc=color if filled else 'white',mew=.8,ls='none')

def clean_axis(ax,ylim,xlim,xticks,xlabel):
    ax.set_ylim(*ylim);ax.set_xlim(*xlim);ax.set_yticks([]);ax.set_xticks(xticks)
    ax.spines[['left','top','right']].set_visible(False);ax.tick_params(length=2)
    ax.set_xlabel(xlabel,fontsize=7,labelpad=3)

def title(fig,letter,text,y,note):
    fig.text(.035,y,letter,fontsize=10,fontweight='bold',va='top')
    fig.text(.075,y,text,fontsize=8.5,fontweight='bold',va='top')
    fig.text(.075,y-.022,note,fontsize=7,va='top',color='#555555')

def interval(r,col,dec=4,scale=1):
    return f'{r[col]*scale:.{dec}f} ({r[col+"_ci_low"]*scale:.{dec}f}–{r[col+"_ci_high"]*scale:.{dec}f})'

def save_cns_figure(fig,filename):
    fig.add_artist(Rectangle((0,0),1,1,transform=fig.transFigure,facecolor='none',edgecolor='none',in_layout=True))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        q=t.get_window_extent(renderer)
        if q.x0< -1 or q.y0< -1 or q.x1>fig.bbox.width+1 or q.y1>fig.bbox.height+1:raise RuntimeError('TEXT_OUTSIDE_FIGURE')
    fig.savefig(str(filename)+'.pdf',bbox_inches='tight',dpi=300)
    fig.savefig(str(filename)+'.svg',bbox_inches='tight')
    fig.savefig(str(filename)+'.png',bbox_inches='tight',dpi=300)
    plt.close(fig)

def flag_panel(fig,d,cohorts,bottom,height,source='matched'):
    n=len(cohorts);ax=fig.add_axes([.31,bottom,.39,height]);clean_axis(ax,(-.6,n-.4),(0,100),[0,25,50,75,100],'Flagged per 100 candidates')
    ta=fig.add_axes([.035,bottom,.93,height],frameon=False);ta.set_xlim(0,1);ta.set_ylim(ax.get_ylim());ta.axis('off')
    ta.text(.995,n-.4,'EWGSOP2 / SDOC / FNIH',fontsize=7,ha='right',va='bottom',color='#555555')
    for i,c in enumerate(cohorts):
        y=n-1-i
        if source=='matched':
            q=d[d.cohort.eq(c)].iloc[0]
            label=c+('¹' if c=='MHAS' else '')
            ta.text(0,y+.16,label,fontsize=7.2,va='center')
            ta.text(0,y-.18,f'proportion n={int(q.positive_weight_n):,}; prognosis n={int(q.prognostic_n):,}',fontsize=6.5,va='center',color='#555555')
        else:
            ta.text(0,y,c+('¹' if c=='MHAS' else ''),fontsize=7.2,va='center')
        values=[]
        for j,de in enumerate(DEFS):
            r=d[d.cohort.eq(c)&d.definition.eq(de)].iloc[0]
            lo=r.flagged_ci_low_per100 if source=='matched' else r.ci_low_per100
            hi=r.flagged_ci_high_per100 if source=='matched' else r.ci_high_per100
            dot(ax,r.flagged_per100,lo,hi,y+.22-j*.22,MARK[de],de=='EWGSOP2')
            values.append(f'{r.flagged_per100:.1f}')
        ta.text(.995,y,' / '.join(values),fontsize=7,va='center',ha='right')

def marker_key(fig,y):
    ax=fig.add_axes([.075,y-.012,.36,.024],frameon=False);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    # Direct labels outside data panels avoid legend occlusion in grayscale.
    for x,de in zip([.025,.43,.735],DEFS):
        ax.plot(x,.5,marker=MARK[de],ms=3.5,mec=BLACK,mfc=BLACK if de=='EWGSOP2' else 'white',mew=.8,ls='none')
        ax.text(x+.045,.5,de,fontsize=7,va='center')


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
d=aggregate_csv(OUT/'figdata/Figure4_data.csv')
assert len(d)==9 and d.prevalence_status.eq('OK').all()
assert set(d.loc[d.population_match_status.ne('EXACT_PRIMARY_PEOPLE'),'cohort'])=={'MHAS'}
assert d[d.cohort.eq('MHAS')].positive_weight_n.eq(262).all() and d[d.cohort.eq('MHAS')].prognostic_n.eq(270).all()
fig=plt.figure(figsize=(183/25.4,244/25.4),facecolor='white')
def head(letter,text,y):
    fig.text(.035,y,letter,fontsize=10,fontweight='bold',va='top')
    fig.text(.075,y,text,fontsize=8.2,fontweight='bold',va='top')
head('a','SDOC identifies more candidates than EWGSOP2',.977)
ax=fig.add_axes([.31,.785,.39,.145]);clean_axis(ax,(-.6,2.6),(0,100),[0,25,50,75,100],'Identified per 100 candidates')
ta=fig.add_axes([.035,.785,.93,.145],frameon=False);ta.set_xlim(0,1);ta.set_ylim(ax.get_ylim());ta.axis('off')
ta.text(.995,2.58,'EWGSOP2 / SDOC / FNIH',fontsize=7,ha='right',va='bottom',color=GREY)
for i,c in enumerate(PROG):
    y=2-i;ta.text(0,y+.1,c,fontsize=7.2,va='center',fontweight='bold')
    if c=='MHAS':ta.text(0,y-.23,'Proportion n=262; prognosis n=270',fontsize=6.4,va='center',color='#555555')
    vals=[]
    for j,de in enumerate(DEFS):
        r=d[d.cohort.eq(c)&d.definition.eq(de)].iloc[0]
        dot(ax,r.flagged_per100,r.flagged_ci_low_per100,r.flagged_ci_high_per100,y+.22-j*.22,MARK[de],de=='EWGSOP2')
        vals.append(f'{r.flagged_per100:.1f}')
    ta.text(.995,y,' / '.join(vals),fontsize=7,ha='right',va='center')
marker_key(fig,.738)
head('b','SHARE: EWGSOP2 identifies a smaller group with higher risk',.698)
ax=fig.add_axes([.32,.405,.38,.238]);clean_axis(ax,(-.55,10.7),(0,100),[0,25,50,75,100],'Model-standardised 9-year risk (%)')
ta=fig.add_axes([.035,.405,.93,.238],frameon=False);ta.set_xlim(0,1);ta.set_ylim(ax.get_ylim());ta.axis('off')
ta.text(.995,10.6,'Risk difference, pp (95% CI)',fontsize=7,ha='right',va='bottom',color=GREY)
ys=[10,9,8,6,5,4,2,1,0]
for start in [8,4,0]:
    ax.axhspan(start-.45,start+2.45,facecolor='#F3F3F3',lw=0,zorder=0)
for (c,de),y in zip(((c,de) for c in PROG for de in DEFS),ys):
    r=d[d.cohort.eq(c)&d.definition.eq(de)].iloc[0]
    ax.plot([r.risk_unflagged*100,r.risk_flagged*100],[y,y],color='#BBBBBB',lw=1)
    for pre,off,mk,col,fill in [('risk_unflagged',-.11,'o',NORMAL,False),('risk_flagged',.11,'s',LOW,True)]:dot(ax,r[pre]*100,r[pre+'_ci_low']*100,r[pre+'_ci_high']*100,y+off,mk,fill,col)
    ta.text(0,y,f'{c} · {de}',fontsize=7,va='center');ta.text(.995,y,interval(r,'RD',1,100),fontsize=7,ha='right',va='center')
fig.text(.33,.658,'○ Not identified     ■ Identified',fontsize=7,color='#555555')
head('c','Each definition adds little to discrimination in SHARE',.342)
ax=fig.add_axes([.31,.071,.39,.218]);clean_axis(ax,(-.55,10.7),(-.02,.14),[0,.05,.10],'AUC increment (definition added to base model)')
ax.axvline(0,color=GREY,lw=.7,ls=(0,(3,3)))
ta=fig.add_axes([.035,.071,.93,.218],frameon=False);ta.set_xlim(0,1);ta.set_ylim(ax.get_ylim());ta.axis('off')
ta.text(.995,10.6,'ΔAUC (95% CI)',fontsize=7,ha='right',va='bottom',color=GREY)
for start in [8,4,0]:
    ax.axhspan(start-.45,start+2.45,facecolor='#F3F3F3',lw=0,zorder=0)
for (c,de),y in zip(((c,de) for c in PROG for de in DEFS),ys):
    r=d[d.cohort.eq(c)&d.definition.eq(de)].iloc[0]
    dot(ax,r.AUC_increment,r.AUC_increment_ci_low,r.AUC_increment_ci_high,y,MARK[de],de=='EWGSOP2')
    ta.text(0,y,f'{c} · {de}',fontsize=7,va='center');ta.text(.995,y,interval(r,'AUC_increment'),fontsize=6.8,ha='right',va='center')
fig.text(.965,.025,'CHARLS n = 206; MHAS n = 270',ha='right',fontsize=7)
save_cns_figure(fig,OUT/'figures/Figure4_definition_comparison')
# Five-cohort descriptive supplement: original estimates and original design.
five=aggregate_csv(ROOT/'outputs/A_AGGREGATE_REVIEW_v0_8_1_20261031/code/resources/v07_T3_flagged_per100.csv')
fig=plt.figure(figsize=(183/25.4,130/25.4),facecolor='white')
head('','Three definitions among all eligible candidates',.96)
flag_panel(fig,five,FIVE,.24,.58,'all');marker_key(fig,.15)
fig.text(.075,.09,'¹ MHAS: four-state physical-measurement subsample. SHARE pools 16 countries.',fontsize=7)
fig.text(.075,.055,'Age ≥65; survey-weighted estimates and 95% CIs; all candidates, not prognosis samples.',fontsize=7)
save_cns_figure(fig,OUT/'supplement/FigureS3_five_cohort_definitions')
