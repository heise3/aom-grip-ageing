# Academic Figure Skill Asset Confirmation
# Existing manuscript production assets reused as native vector renderers.
# Data: permitted aggregate CSVs only; no individual records and no model fitting.
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from common import *

d, t = load()
out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/A4/out'

m = t['QC_A_models'].copy()
from pathlib import Path
lm=pd.read_csv(Path(__file__).resolve().parents[3]/'outputs/A_AGGREGATE_REVIEW_v0_7_20261031/figdata/fig3c_landmark.csv')
meta = t['QC_A_meta'].set_index('analysis')
phm = t['QC_posthoc_meta'].set_index('analysis')
phc = t['QC_posthoc_models']
em = t['QC_effect_modification']
crude = pd.DataFrame(d['crude'], columns=['cohort', 'dfn', 'hr', 'lo', 'hi', 'adj'])
crude = crude[crude.dfn == 'EWGSOP2'].set_index('cohort')
on = t['QC_old_new_effect_comparison']
qc = pd.DataFrame(d['qc'], columns=['a', 'v', 'oe', 'olo', 'ohi', 'ne', 'nlo', 'nhi']).set_index('a')
C4 = ['CHARLS', 'MHAS', 'SHARE', 'KLoSA']


def mod(co, outc, an):
    r = m[(m.cohort == co) & (m.outcome == outc) & (m.analysis == an)]
    if len(r) == 0 or r.iloc[0].status != 'OK':
        return None
    r = r.iloc[0]
    return (r.effect, r.ci_low, r.ci_high)


def inv(x):
    return None if x is None else (1 / x[0], 1 / x[2], 1 / x[1])


def mrow(name):
    r = meta.loc[name]
    return (r.effect, r.ci_low, r.ci_high, int(r.k), r.I2)


def prow(name):
    r = phm.loc[name]
    return (r.effect, r.ci_low, r.ci_high, int(r.k), r.I2)


def ph(co, an):
    r = phc[(phc.cohort == co) & (phc.analysis == an)].iloc[0]
    return (r.effect, r.ci_low, r.ci_high)


def old(co):
    r = on[(on.table == 'A_models.csv') & (on.cohort == co) & (on.outcome == 'composite') & (on.analysis == 'primary')].iloc[0]
    return (r.old_effect, r.old_ci_low, r.old_ci_high)


cp = crude.loc['POOLED']
mc = meta.loc['composite_continuous_5kg']; dc = meta.loc['death_continuous_5kg']
# (group, label, pooled(e,lo,hi,k,I2), cohort cells dict, prespecified?, reason for NA)
specs = [
    ('Reference', 'Primary: composite, EWGSOP2, adjusted', mrow('composite_primary'),
     {c: mod(c, 'composite', 'primary') for c in C4}, 'pre', {'KLoSA': 'ADL≠'}),
    ('Model and data', 'Unadjusted', (cp.hr, cp.lo, cp.hi, 3, None),
     {c: (crude.loc[c].hr, crude.loc[c].lo, crude.loc[c].hi) for c in ['CHARLS', 'MHAS', 'SHARE']}, 'post', {'KLoSA': 'ADL≠'}),
    ('Model and data', 'Censoring-weighted (IPCW)', prow('IPCW'),
     {c: ph(c, 'IPCW') for c in ['CHARLS', 'MHAS', 'SHARE']}, 'post', {'KLoSA': 'ADL≠'}),
    ('Model and data','First-revisit landmark', tuple(lm[lm.cohort.eq('POOLED')].iloc[0][['effect','ci_low','ci_high']])+ (3,0.0),
     {r.cohort:(r.effect,r.ci_low,r.ci_high) for _,r in lm[lm.cohort.ne('POOLED')].iterrows()},'post',{'KLoSA':'ADL≠'}),
    ('Model and data', 'Measured BMI cohorts only', mrow('composite_measured_BMI'),
     {c: mod(c, 'composite', 'primary') for c in ['CHARLS', 'MHAS']}, 'pre', {'SHARE': 'self-rep', 'KLoSA': 'ADL≠'}),
    ('Model and data', 'Before measurement QC (v1)', (qc.loc['composite_primary'].oe, qc.loc['composite_primary'].olo, qc.loc['composite_primary'].ohi, 3, 0.0),
     {c: old(c) for c in ['CHARLS', 'MHAS', 'SHARE']}, 'ver', {'KLoSA': 'ADL≠'}),
    ('Definition of weakness', 'SDOC cut-offs', mrow('composite_SDOC'),
     {c: mod(c, 'composite', 'SDOC') for c in C4}, 'pre', {'KLoSA': 'ADL≠'}),
    ('Definition of weakness', 'FNIH grip/BMI', prow('FNIH'),
     {c: ph(c, 'FNIH') for c in ['CHARLS', 'MHAS', 'SHARE']}, 'post', {'KLoSA': 'ADL≠'}),
    ('Definition of weakness', 'Continuous, per 5 kg lower', (1 / mc.effect, 1 / mc.ci_high, 1 / mc.ci_low, int(mc.k), mc.I2),
     {c: inv(mod(c, 'composite', 'continuous_5kg')) for c in C4}, 'pre', {'KLoSA': 'ADL≠'}),
    ('Death alone (all candidates)', 'EWGSOP2', mrow('death_primary'),
     {c: mod(c, 'death', 'primary') for c in C4}, 'pre', {'KLoSA': '<10'}),
    ('Death alone (all candidates)', 'SDOC cut-offs, adding KLoSA', mrow('death_SDOC'),
     {c: mod(c, 'death', 'SDOC') for c in C4}, 'pre', {}),
    ('Death alone (all candidates)', 'Historic date-based exclusion (superseded)', mrow('death_exclude_first_year_death'),
     {c: mod(c, 'death', 'exclude_first_year_death') for c in C4}, 'pre', {'KLoSA': '<10'}),
    ('Death alone (all candidates)', 'Continuous, per 5 kg lower', (1 / dc.effect, 1 / dc.ci_high, 1 / dc.ci_low, int(dc.k), dc.I2),
     {c: inv(mod(c, 'death', 'continuous_5kg')) for c in C4}, 'pre', {'KLoSA': '<10'}),
]

fig = plt.figure(figsize=(8.6, 8.8))
outer = fig.add_gridspec(2, 1, height_ratios=[0.30, 1], hspace=0.44, left=0.01, right=0.99, top=0.95, bottom=0.085)


def ptitle(axref, letter, text, dx=0.0, dy=0.022):
    bb = axref.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=10, fontweight='bold', va='bottom')
    fig.text(bb.x0 + dx + 0.022, bb.y1 + dy + 0.002, text, fontsize=7.5, fontweight='bold', va='bottom')


# ---------- a: effect modification ----------
sga = outer[0].subgridspec(1, 3, width_ratios=[0.26, 0.40, 0.34], wspace=0.06)
la = fig.add_subplot(sga[0, 0]); axa = fig.add_subplot(sga[0, 1]); va = fig.add_subplot(sga[0, 2])
for a_ in (la, va):
    a_.axis('off'); a_.set_xlim(0, 1)
for a_ in (la, axa, va):
    a_.set_ylim(-0.6, 3.75)
axa.set_xscale('log'); axa.set_xlim(0.43, 2.8)
rows = [('CHARLS', 3), ('MHAS', 2), ('SHARE', 1)]
for co, y in rows:
    e = em[em.cohort == co].set_index('analysis')
    for an, col, mk, dy in [('eligible_HR', LOW, 's', 0.16), ('ineligible_HR', GREY, 'o', -0.16)]:
        r = e.loc[an]
        axa.plot([r.ci_low, r.ci_high], [y + dy] * 2, color=col, lw=1.3)
        axa.plot(r.effect, y + dy, marker=mk, ms=4.2, color=col, mec='white', mew=0.5, zorder=3)
        va.text(0.02 if an == 'eligible_HR' else 0.36, y, fmt_ci(r.effect, r.ci_low, r.ci_high), va='center', fontsize=6,
                color=INK if an == 'eligible_HR' else GREY)
    rr = e.loc['interaction_HR_ratio']
    va.text(0.70, y, fmt_ci(rr.effect, rr.ci_low, rr.ci_high), va='center', fontsize=6)
    la.text(0.06, y, co, fontweight='bold', va='center')
    n_ = e.loc['eligible_HR'].persons_n
    la.text(0.52, y, 'Not released' if not str(n_).replace('.', '').isdigit() else f'{int(float(n_)):,}', va='center', fontsize=6, ha='center')
yp = 0
for an, col, dy in [('eligible_HR', LOW, 0.16), ('ineligible_HR', GREY, -0.16)]:
    r = phm.loc[an]
    axa.add_patch(Polygon([[r.ci_low, yp + dy], [r.effect, yp + dy + 0.13], [r.ci_high, yp + dy], [r.effect, yp + dy - 0.13]],
                          closed=True, color=col, zorder=3))
    va.text(0.02 if an == 'eligible_HR' else 0.36, yp, fmt_ci(r.effect, r.ci_low, r.ci_high), va='center', fontsize=6,
            fontweight='bold', color=INK if an == 'eligible_HR' else GREY)
ri = phm.loc['interaction_HR_ratio']
va.text(0.70, yp, fmt_ci(ri.effect, ri.ci_low, ri.ci_high), va='center', fontsize=6.2, fontweight='bold')
va.text(0.70, yp - 0.48, f'P = {ri.p:.2f}', va='center', fontsize=5.8, color=GREY)
la.text(0.06, yp, 'Pooled', fontweight='bold', va='center')
la.text(0.52, 3.55, 'n analysed', ha='center', va='bottom', fontsize=5.8, color=GREY)
for x, s_ in [(0.02, 'Candidates'), (0.36, 'Non-candidates'), (0.70, 'Ratio of HRs')]:
    va.text(x, 3.55, s_, va='bottom', fontsize=5.8, color=GREY)
axa.axvline(1, color=INK, lw=0.6)
axa.set_yticks([]); axa.spines['left'].set_visible(False)
axa.set_xticks([0.5, 1, 1.5, 2]); axa.set_xticklabels(['0.5', '1', '1.5', '2']); axa.minorticks_off()
axa.set_xlabel('HR for first ADL difficulty or death, low vs normal grip (log)')
axa.legend(handles=[Line2D([], [], marker='s', color=LOW, ms=4, lw=1.2, label='Label-defined candidates'),
                    Line2D([], [], marker='o', color=GREY, ms=4, lw=1.2, label='Non-candidates (≥65 y, ADL-free)')],
           loc='upper left', bbox_to_anchor=(0.0, 1.12), ncol=2, frameon=False, fontsize=5.8, columnspacing=1.0, handlelength=1.6)

# ---------- b: specification panel ----------
sgb = outer[1].subgridspec(1, 5, width_ratios=[0.30, 0.22, 0.17, 0.07, 0.24], wspace=0.07)
lb = fig.add_subplot(sgb[0, 0]); axb = fig.add_subplot(sgb[0, 1]); vb = fig.add_subplot(sgb[0, 2])
kb = fig.add_subplot(sgb[0, 3]); hb = fig.add_subplot(sgb[0, 4])
for a_ in (lb, vb, kb):
    a_.axis('off'); a_.set_xlim(0, 1)
# y layout with group headers
ypos, heads = [], []
y = 0
last = None
for g, *_ in specs:
    if g != last:
        y -= 0.55; heads.append((g, y)); y -= 0.75; last = g
    else:
        y -= 1.0
    ypos.append(y)
ymin, ymax = min(ypos) - 0.8, 0.6
for a_ in (lb, axb, vb, kb, hb):
    a_.set_ylim(ymin, ymax)
prim = specs[0][2]
axb.axvspan(prim[1], prim[2], color=LOW_L, alpha=0.55, lw=0, zorder=0)
axb.axvline(1, color=INK, lw=0.6)
axb.set_xscale('log'); axb.set_xlim(0.5, 4.6)
norm = mpl.colors.TwoSlopeNorm(vmin=np.log(0.6), vcenter=0.0, vmax=np.log(2.6))
cmap = mpl.colormaps['RdBu_r']
for (g, lab, pr, cells, kind, na), y in zip(specs, ypos):
    e, lo, hi, k, i2 = pr
    hi_c = min(hi, 4.5)
    col = LOW if kind == 'pre' and lab.startswith('Primary') else INK
    axb.plot([max(lo, 0.5), hi_c], [y, y], color=col, lw=1.1)
    if lo < 0.5:
        axb.annotate('', xy=(0.505, y), xytext=(0.60, y), arrowprops=dict(arrowstyle='->', color=col, lw=1.0))
    if hi > 4.5:
        axb.annotate('', xy=(4.55, y), xytext=(4.0, y), arrowprops=dict(arrowstyle='->', color=col, lw=1.0))
    if lab.startswith('Primary'):
        axb.add_patch(Polygon([[lo, y], [e, y + 0.3], [hi, y], [e, y - 0.3]], closed=True, color=LOW, zorder=3))
    else:
        mk = {'pre': 'o', 'post': 'o', 'ver': 's'}[kind]
        fc = INK if kind == 'pre' else 'white'
        axb.plot(e, y, marker=mk, ms=4.2, mfc=fc, mec=INK, mew=0.9, zorder=3)
    lb.text(0.06, y, lab, va='center', fontsize=6.3, fontweight='bold' if lab.startswith('Primary') else 'normal')
    vb.text(0.04, y, fmt_ci(e, lo, hi), va='center', fontsize=6.1, fontweight='bold' if lab.startswith('Primary') else 'normal')
    kb.text(0.30, y, f'{k}', va='center', ha='center', fontsize=6)
    kb.text(0.95, y, '–' if i2 is None else f'{i2:.0f}%', va='center', ha='right', fontsize=6)
    for j, co in enumerate(C4):
        c = cells.get(co)
        if c is None:
            hb.add_patch(Rectangle((j + 0.04, y - 0.45), 0.92, 0.9, color='#F2F2F2', lw=0))
            hb.text(j + 0.5, y, na.get(co, 'n.e.'), ha='center', va='center', fontsize=5.0, color=GREY)
            continue
        ce, clo, chi = c
        rgba = cmap(norm(np.log(ce)))
        hb.add_patch(Rectangle((j + 0.04, y - 0.45), 0.92, 0.9, color=rgba, lw=0))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        sig = (clo > 1) or (chi < 1)
        hb.text(j + 0.5, y, f'{ce:.2f}', ha='center', va='center', fontsize=6,
                color='white' if lum < 0.5 else INK, fontweight='bold' if sig else 'normal')
for g, y in heads:
    lb.text(0.02, y, g.upper(), va='center', fontsize=5.9, color=GREY, fontweight='bold')
hb.set_xlim(0, 4); hb.axis('off')
for j, co in enumerate(C4):
    hb.text(j + 0.5, ymax - 0.15, co, ha='center', va='top', fontsize=6.2, fontweight='bold')
vb.text(0.04, ymax - 0.15, 'Pooled HR (95% CI)', va='top', fontsize=5.8, color=GREY)
kb.text(0.30, ymax - 0.15, 'k', va='top', ha='center', fontsize=5.8, color=GREY)
kb.text(0.95, ymax - 0.15, 'I²', va='top', ha='right', fontsize=5.8, color=GREY)
axb.set_yticks([]); axb.spines['left'].set_visible(False)
axb.set_xticks([0.5, 1, 2, 4]); axb.set_xticklabels(['0.5', '1', '2', '4']); axb.minorticks_off()
axb.set_xlabel('Pooled HR (log)')
# colourbar
cax = fig.add_axes([0.80, 0.030, 0.15, 0.010])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
cb.solids.set_rasterized(False)
cb.set_ticks(np.log([0.6, 1, 1.5, 2.5])); cb.set_ticklabels(['0.6', '1', '1.5', '2.5'])
cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=5.6, length=1.5)
cax.set_title('Cohort HR (bold: 95% CI excludes 1)', fontsize=5.6, pad=2, loc='center', fontweight='normal')
fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc=INK, mec=INK, ms=4, label='Pre-specified'),
                    Line2D([], [], marker='o', ls='', mfc='white', mec=INK, ms=4, label='Post hoc / reporting'),
                    Line2D([], [], marker='s', ls='', mfc='white', mec=INK, ms=4, label='Version comparison'),
                    Rectangle((0, 0), 1, 1, color=LOW_L, label='Primary 95% CI')],
           loc='lower left', bbox_to_anchor=(0.03, -0.012), ncol=4, frameon=False, fontsize=5.8, columnspacing=1.2, handletextpad=0.3)

fig.canvas.draw()
ptitle(la, 'a', 'No clear evidence of effect modification by label eligibility')
ptitle(lb, 'b', 'Pooled estimates point the same way across models, definitions and outcomes', dy=0.012)
save(fig, 'FigureS6_sensitivity', out)
