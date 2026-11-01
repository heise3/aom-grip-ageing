# Academic Figure Skill Asset Confirmation
# Existing manuscript production assets reused as native vector renderers.
# Data: permitted aggregate CSVs only; no individual records and no model fitting.
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from common import *

d, t = load()
out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/A4/out'

m = t['QC_A_models']
prim = m[(m.outcome == 'composite') & (m.analysis == 'primary') & (m.status == 'OK')].set_index('cohort')
meta = t['QC_A_meta'].set_index('analysis').loc['composite_primary']
het = t['REPORT_heterogeneity'].set_index('definition').loc['EWGSOP2']
crude = pd.DataFrame(d['crude'], columns=['cohort', 'dfn', 'hr', 'lo', 'hi', 'adj']).query("dfn=='EWGSOP2'").set_index('cohort')
ab = pd.DataFrame(d['abs'], columns=['cohort', 'm', 'g', 'v', 'lo', 'hi'])
cif = t['QC_CIF_intervals']
ms = t['QC_A_multistate']
tab = t['REPORT_table1_long']
fu = pd.DataFrame(d['fu'], columns=['cohort', 'group', 'n', 'med', 'q1', 'q3', 'max', 'py']).set_index(['cohort', 'group'])

cohorts = ['CHARLS', 'MHAS', 'SHARE']
iv = {c: 1 / prim.loc[c, 'se'] ** 2 for c in cohorts}
wsum = sum(iv.values())
wt = {c: 100 * iv[c] / wsum for c in cohorts}      # display: inverse-variance share (tau^2 = 0)


fig = plt.figure(figsize=(8.8, 10.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.5], wspace=0.34, hspace=0.55,
                      left=0.01, right=0.99, top=0.93, bottom=0.07)
def ptitle(axref, letter, text, dx=0.0, dy=0.035):
    bb = axref.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=10, fontweight='bold', va='bottom')
    fig.text(bb.x0 + dx + 0.022, bb.y1 + dy + 0.002, text, fontsize=7.5, fontweight='bold', va='bottom')

def tabaxes(spec, ratios):
    sg = spec.subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.0)
    axs = [fig.add_subplot(sg[0, i]) for i in range(len(ratios))]
    for a_ in axs:
        a_.set_yticks([])
    for a_ in (axs[0], axs[-1]):
        a_.axis('off'); a_.set_xlim(0, 1)
    return axs

# ---------- a: forest ----------
la, ax, va = tabaxes(gs[0, 0], [0.43, 0.29, 0.28])
YL = (-0.8, 3.8)
for a_ in (la, ax, va):
    a_.set_ylim(*YL)
ax.set_xscale('log'); ax.set_xlim(0.55, 4.6)
rows = [('CHARLS', 3.0), ('MHAS', 2.0), ('SHARE', 1.0)]
info = {'CHARLS': ('China', '206 (22)', '128 (16)'),
        'MHAS': ('Mexico†', f"{int(float(fu.loc[('MHAS','All'),'n']))} ({int(float(fu.loc[('MHAS','Low'),'n']))})",
                 f"{int(float(prim.loc['MHAS','events_n']))} ({int(float(prim.loc['MHAS','events_group1_n']))})"),
        'SHARE': ('16 countries', f"{int(float(fu.loc[('SHARE','All'),'n'])):,} ({int(float(fu.loc[('SHARE','Low'),'n']))})",
                  f"{int(float(prim.loc['SHARE','events_n'])):,} ({int(float(prim.loc['SHARE','events_group1_n']))})")}
for c, y in rows:
    r = prim.loc[c]
    sz = 10.0 * np.sqrt(wt[c] / 100)
    ax.plot([r.ci_low, r.ci_high], [y + 0.14] * 2, color=LOW, lw=1.5, solid_capstyle='round')
    ax.plot(r.effect, y + 0.14, marker='s', ms=sz, color=LOW, mec='white', mew=0.5, zorder=3)
    cr = crude.loc[c]
    ax.plot([cr.lo, cr.hi], [y - 0.22] * 2, color=GREY, lw=0.8)
    ax.plot(cr.hr, y - 0.22, marker='o', ms=3.4, mfc='white', mec=GREY, mew=0.8, zorder=3)
    la.text(0.04, y + 0.12, c, fontweight='bold', va='center')
    la.text(0.04, y - 0.25, info[c][0], va='center', fontsize=6, color=GREY)
    la.text(0.56, y, info[c][1], va='center', ha='center', fontsize=6)
    la.text(0.87, y, info[c][2], va='center', ha='center', fontsize=6)
    va.text(0.06, y + 0.14, fmt_ci(r.effect, r.ci_low, r.ci_high), va='center', fontsize=6.3)
    va.text(0.06, y - 0.24, fmt_ci(cr.hr, cr.lo, cr.hi), va='center', fontsize=5.6, color=GREY)
    va.text(1.0, y + 0.14, f'{wt[c]:.0f}%', va='center', ha='right', fontsize=6.2)
pe, plo, phi = meta.effect, meta.ci_low, meta.ci_high
yp = 0.0
ax.add_patch(Polygon([[plo, yp + 0.06], [pe, yp + 0.30], [phi, yp + 0.06], [pe, yp - 0.18]], closed=True, color=LOW, zorder=3))
cp = crude.loc['POOLED']
ax.plot([cp.lo, cp.hi], [yp - 0.45] * 2, color=GREY, lw=0.8)
ax.plot(cp.hr, yp - 0.45, marker='o', ms=3.4, mfc='white', mec=GREY, mew=0.8)
la.text(0.04, yp + 0.06, 'Pooled (random effects)', fontweight='bold', va='center')
la.text(0.04, yp - 0.40, f"I² {meta.I2:.0f}% (95% CI 0–{het.I2_ci_high:.0f}%); Q P = {meta.Q_p:.2f}", va='center', fontsize=5.6, color=GREY)
va.text(0.06, yp + 0.06, fmt_ci(pe, plo, phi), va='center', fontsize=6.3, fontweight='bold')
va.text(0.06, yp - 0.45, fmt_ci(cp.hr, cp.lo, cp.hi), va='center', fontsize=5.6, color=GREY)
va.text(1.0, yp + 0.06, '100%', va='center', ha='right', fontsize=6.2)
ax.axvline(1, color=INK, lw=0.6)
ax.axvline(pe, color=LOW, lw=0.5, ls=(0, (2, 2)))
ax.spines['left'].set_visible(False)
ax.set_xticks([0.6, 1, 2, 4]); ax.set_xticklabels(['0.6', '1', '2', '4']); ax.minorticks_off()
ax.set_xlabel('HR, low vs normal grip (log)')
hy = 3.55
la.text(0.56, hy, 'n (low grip)', ha='center', va='bottom', fontsize=5.8, color=GREY)
la.text(0.87, hy, 'Events\n(low grip)', ha='center', va='bottom', fontsize=5.8, color=GREY)
va.text(0.06, hy, 'Adjusted HR (95% CI)\n\u25cb crude, in grey', va='bottom', fontsize=5.8, color=GREY)
va.text(1.0, hy, 'Weight', ha='right', va='bottom', fontsize=5.8, color=GREY)
leg_h = [Line2D([], [], marker='s', color=LOW, ls='-', lw=1.2, ms=4, label='Adjusted'),
         Line2D([], [], marker='o', mfc='white', mec=GREY, color=GREY, ls='-', lw=0.8, ms=3.4, label='Crude')]


# ---------- b: absolute 9-year risk ----------
axb, vb = tabaxes(gs[0, 1], [0.78, 0.22])[0:2] if False else (None, None)
sg = gs[0, 1].subgridspec(1, 2, width_ratios=[0.77, 0.23], wspace=0.0)
axb = fig.add_subplot(sg[0, 0]); vb = fig.add_subplot(sg[0, 1]); vb.axis('off'); vb.set_xlim(0, 1)
yb = {'CHARLS': 2, 'MHAS': 1, 'SHARE': 0}
for c in cohorts:
    a = ab[ab.cohort == c]
    n0 = a[a.g == '0.0'].iloc[0]; n1 = a[a.g == '1.0'].iloc[0]; rd = a[a.g == ''].iloc[0]
    y = yb[c]
    axb.annotate('', xy=(100 * n1.v, y), xytext=(100 * n0.v, y),
                 arrowprops=dict(arrowstyle='-|>', color=LGREY, lw=2.2, shrinkA=4, shrinkB=4, mutation_scale=7), zorder=1)
    for r_, col, mk, dy in [(n0, NORM, 'o', 0.0), (n1, LOW, '^', 0.0)]:
        axb.plot([100 * r_.lo, 100 * r_.hi], [y + (0.16 if col == NORM else -0.16)] * 2, color=col, lw=0.9, zorder=2)
        axb.plot(100 * r_.v, y, marker=mk, ms=5.4, color=col, mec='white', mew=0.5, zorder=3)
    axb.text(100 * n0.v, y + 0.30, f'{100*n0.v:.0f}', ha='center', va='bottom', fontsize=6, color=NORM)
    axb.text(100 * n1.v, y - 0.30, f'{100*n1.v:.0f}', ha='center', va='top', fontsize=6.3, color=LOW, fontweight='bold')
    lo_s = f'{100*rd.lo:.1f}'.replace('-', '−')
    vb.text(0.08, y + 0.08, f'+{100*rd.v:.1f}', va='center', fontsize=6.5, fontweight='bold')
    vb.text(0.08, y - 0.22, f'({lo_s} to {100*rd.hi:.1f})', va='center', fontsize=5.6, color=GREY)
vb.set_ylim(-0.7, 2.75); axb.set_ylim(-0.7, 2.75)
vb.text(0.08, 2.5, 'RD, pp', fontsize=5.8, color=GREY, va='bottom')
axb.set_yticks([])
for c_, lab_ in [('CHARLS', 'CHARLS'), ('MHAS', 'MHAS'), ('SHARE', 'SHARE§')]:
    axb.text(40.8, yb[c_] + 0.30, lab_, fontweight='bold', va='bottom', fontsize=6.6)
axb.tick_params(axis='y', length=0)
axb.spines['left'].set_visible(False)
axb.set_xlim(40, 95)
axb.set_xticks([40, 50, 60, 70, 80, 90])
axb.grid(axis='x', color=LGREY, lw=0.4)
axb.set_xlabel('Model-standardised 9-year risk of first ADL\ndifficulty or death, %')
axb.legend(handles=[Line2D([], [], marker='o', color=NORM, ls='', ms=4.5, label='Normal grip'),
                    Line2D([], [], marker='^', color=LOW, ls='', ms=4.5, label='Low grip')],
           loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.06), handletextpad=0.2, columnspacing=0.8)


# ---------- c: archived alternatives; no new fits ----------
from pathlib import Path
alts=pd.read_csv(Path(__file__).resolve().parents[1]/'figdata/Figure3c_alternative_explanations.csv')
sc=gs[1,:].subgridspec(1,3,width_ratios=[.51,.25,.24],wspace=.02)
lc=fig.add_subplot(sc[0]);ac=fig.add_subplot(sc[1]);vc=fig.add_subplot(sc[2])
for a in [lc,vc]:a.axis('off');a.set_xlim(0,1)
y=12.2;positions=[];last=None
for _,r in alts.iterrows():
    if r.group!=last:
        y-=.6
        lc.text(.005,y,r.group.upper(),fontsize=6.3,fontweight='bold',color=GREY,va='center')
        y-=.65;last=r.group
    else:y-=.85
    positions.append(y)
    lc.text(.035,y,r.label,fontsize=7,va='center')
    lo,hi=r.ci_low,r.ci_high
    ac.plot([max(.8,lo),min(2.5,hi)],[y,y],lw=1,color=INK)
    if lo<.8:ac.annotate('',xy=(.801,y),xytext=(.87,y),arrowprops=dict(arrowstyle='->',lw=1,color=INK))
    if hi>2.5:ac.annotate('',xy=(2.49,y),xytext=(2.30,y),arrowprops=dict(arrowstyle='->',lw=1,color=INK))
    ac.plot(r.estimate,y,marker='D' if r.post_hoc else 's',mfc='white' if r.post_hoc else LOW,mec=INK,ms=4.5,mew=.8)
    vc.text(.04,y,fmt_ci(r.estimate,lo,hi),fontsize=7,va='center')
for a in [lc,ac,vc]:a.set_ylim(min(positions)-.6,12.6)
ac.set_xscale('log');ac.set_xlim(.8,2.5);ac.set_xticks([.8,1,1.5,2,2.5]);ac.set_xticklabels(['0.8','1','1.5','2','2.5']);ac.minorticks_off()
ac.axvline(1,color=GREY,lw=.7);ac.set_yticks([]);ac.spines['left'].set_visible(False)
ac.set_xlabel('Effect estimate (log scale)')
vc.text(.04,12.1,'HR / ratio of HRs (95% CI)',fontsize=6.5,color=GREY)
fig.text(.02,.031,r'$\diamond$ Post hoc. Arrow: confidence interval extends below the axis; full interval is printed.',fontsize=6.6,color=GREY)
fig.canvas.draw()
ptitle(la,'a','Low grip strength and ADL difficulty or death')
ptitle(axb,'b','Model-standardised 9-year ADL-or-death risk',dx=-.01)
ptitle(lc,'c','Tests of alternative explanations',dy=.018)
save(fig,'Figure3_prognosis',out)
