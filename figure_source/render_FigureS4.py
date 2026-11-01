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


fig=plt.figure(figsize=(8.8,4.4))
gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=.40,left=.01,right=.99,top=.86,bottom=.18)
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

# ---------- c: cumulative incidence ----------
gsc = gs[0, 0].subgridspec(1, 2, wspace=0.28)
cax = []
for j, co in enumerate(['SHARE', 'MHAS']):
    axc = fig.add_subplot(gsc[0, j]); cax.append(axc)
    cc = cif[cif.cohort == co]
    for g, col, mk, bandc in [(0, NORM, 'o', NORM_L), (1, LOW, '^', LOW_L)]:
        for outc, ls in [('incident_ADL', '-'), ('death_before_observed_ADL', (0, (3, 1.5)))]:
            s_ = cc[(cc.low_grip == g) & (cc.outcome == outc)].sort_values('years_from_baseline')
            x = np.r_[0, s_.years_from_baseline.values]
            y = 100 * np.r_[0, s_.cif.values]
            lo = 100 * np.r_[0, s_.ci_low.values]; hi = 100 * np.r_[0, s_.ci_high.values]
            axc.fill_between(x, lo, hi, color=bandc, alpha=0.45, lw=0)
            axc.plot(x, y, color=col, ls=ls, lw=1.3, marker=mk, ms=3, mec='white', mew=0.4)
            yy = y[-1] + (1.6 if (co == 'MHAS' and g == 1 and outc == 'incident_ADL') else (-1.6 if (co == 'MHAS' and g == 0 and outc == 'incident_ADL') else 0))
            axc.text(x[-1] + 0.3, yy, f'{y[-1]:.0f}', fontsize=5.8, color=col, va='center',
                     fontweight='bold' if g == 1 else 'normal')
    axc.text(0.04, 0.97, co, transform=axc.transAxes, fontweight='bold', va='top')
    xmax = cc.years_from_baseline.max()
    axc.set_xlim(0, xmax + 1.7); axc.set_ylim(0, 60)
    axc.set_xticks(sorted(cc.years_from_baseline.unique().tolist() + [0]))
    axc.grid(axis='y', color=LGREY, lw=0.4)
    axc.set_xlabel('Years from baseline')
    if j == 0:
        axc.set_ylabel('Cumulative incidence, %')
        axc.legend(handles=[Line2D([], [], color=INK, lw=1.2, label='First ADL difficulty'),
                            Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 1.5)), label='Death before observed\nADL difficulty')],
                   loc='upper left', bbox_to_anchor=(0.0, 0.90), frameon=False, fontsize=5.6, handlelength=2.2)
    else:
        axc.set_yticklabels([])
pc = cax[0].get_position()
for a_ in cax:
    p = a_.get_position(); a_.set_position([p.x0 + 0.035, p.y0, p.width * 0.93, p.height])

# ---------- d: route-specific ORs ----------
ld, axd, vd = tabaxes(gs[0, 1], [0.40, 0.36, 0.24])
for a_ in (ld, axd, vd):
    a_.set_ylim(-0.3, 4.3)
axd.set_xscale('log')
items = [('SHARE', 'incident_ADL', 3.2), ('SHARE', 'death_before_observed_ADL', 2.55),
         ('MHAS', 'incident_ADL', 1.35), ('MHAS', 'death_before_observed_ADL', 0.7)]
for co, tr_, y in items:
    r = ms[(ms.cohort == co) & (ms.transition == tr_)].iloc[0]
    ls = '-' if tr_ == 'incident_ADL' else (0, (3, 1.5))
    axd.plot([r.ci_low, r.ci_high], [y, y], color=LOW, lw=1.4, ls=ls)
    axd.plot(r.effect, y, marker='D', ms=4.5, color=LOW, mec='white', mew=0.5, zorder=3)
    vd.text(0.08, y, fmt_ci(r.effect, r.ci_low, r.ci_high), va='center', fontsize=6.2)
    ld.text(0.98, y, 'First ADL difficulty' if tr_ == 'incident_ADL' else 'Death, no prior ADL\ndifficulty',
            va='center', ha='right', fontsize=6.1, linespacing=1.0)
for co, y in [('SHARE', 3.8), ('MHAS', 1.95)]:
    ld.text(0.98, y, co, va='center', ha='right', fontweight='bold')
ld.text(0.98, 0.0, 'CHARLS: not estimable\n(small transition cells)', va='center', ha='right', fontsize=5.6, color=GREY)
axd.axvline(1, color=INK, lw=0.6)
axd.set_xlim(0.6, 4.6)
axd.set_xticks([0.6, 1, 2, 4]); axd.set_xticklabels(['0.6', '1', '2', '4']); axd.minorticks_off()
axd.spines['left'].set_visible(False)
axd.set_xlabel('Transition-specific OR (log)')
vd.text(0.08, 4.05, 'OR (95% CI)', fontsize=5.8, color=GREY, va='center')


ptitle(cax[0],'a','Outcome-specific cumulative incidence',dx=-.035)
ptitle(ld,'b','Transition-specific odds ratios')
fig.text(.02,.03,'Pointwise 95% confidence bands. Transition effects are ORs. CHARLS transitions were not estimable.',fontsize=6.6)
save(fig,'FigureS4_competing_outcomes',out)
