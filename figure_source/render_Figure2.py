# Academic Figure Skill Asset Confirmation
# Existing manuscript production assets reused as native vector renderers.
# Data: permitted aggregate CSVs only; no individual records and no model fitting.
import matplotlib.transforms as mtrans
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *

d, t = load()
out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/A4/out'

# ---------- data ----------
std = pd.DataFrame(d['std'], columns=['cohort', 'country', 'standard', 'n', 'p', 'lo', 'hi', 'status'])
std['country'] = std.country.map(cname)
pooled = std[std.standard == 'poo'].set_index('country')
who = std[std.standard == 'WHO'].set_index('country')
wd = pd.DataFrame(d['wd'], columns=['cohort', 'country', 'def', 'n', 'pos', 'p', 'lo', 'hi'])
wd['country'] = wd.country.map(cname)
wds = wd.pivot(index='country', columns='def', values='p')
w = t['QC_A_weighted_descriptions'].copy()
w['country'] = w.country.map(cname)

rep = pooled[pooled.p.notna()].sort_values('p', ascending=True)
order = list(rep.index) + ['Denmark']          # Denmark (suppressed) at the bottom
median = float(np.median(rep.p))

fig = plt.figure(figsize=(8.6, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[0.86, 1], height_ratios=[1, 1.02],
                      wspace=1.02, hspace=0.76, left=0.115, right=0.985, top=0.93, bottom=0.08)

# ---------- a: standardised burden ----------
ax = fig.add_subplot(gs[:, 0])
ys = {c: i for i, c in enumerate(reversed(order))}  # top = highest
ys = {c: len(order) - 1 - order.index(c) for c in order}
ys = {c: i for i, c in enumerate(order[::-1])}
# rows from top: highest first
rows = order[:-1][::-1] + ['Denmark']
ypos = {c: len(rows) - 1 - i for i, c in enumerate(rows)}
for c in rows:
    y = ypos[c]
    if c in SINGLE:
        ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
    if c == 'Denmark':
        ax.text(1.22, y, 'suppressed (<10 cases)', va='center', ha='center', fontsize=5.8, color=GREY, style='italic', transform=mtrans.blended_transform_factory(ax.transAxes, ax.transData))
        continue
    r = rep.loc[c]
    col = REG[REGION_OF[c]]
    ax.plot([100 * r.lo, 100 * r.hi], [y, y], color=col, lw=1.6, solid_capstyle='round', zorder=2)
    ax.plot(100 * r.p, y, marker='D' if c in SINGLE else 'o', ms=5.2 if c in SINGLE else 4.6,
            mfc=col, mec='white', mew=0.6, zorder=3)
    ax.plot(100 * who.loc[c].p, y, marker='|', ms=6, mew=1.0, color=INK, zorder=4)
ax.axvline(100 * median, color=GREY, ls=(0, (2, 2)), lw=0.7, zorder=1)
ax.text(100 * median + 0.5, -0.1, f'median {100*median:.1f}%', fontsize=5.8, color=GREY, va='bottom')
labels = []
for c in rows:
    lab = c + ('†' if c == 'Mexico' else '')
    labels.append(lab)
ax.set_yticks([ypos[c] for c in rows])
ax.set_yticklabels(labels)
for tl, c in zip(ax.get_yticklabels(), rows):
    if c in SINGLE:
        tl.set_fontweight('bold')
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlim(0, 36)
ax.set_xticks([0, 10, 20, 30])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Candidates with low grip strength, %\n(age–sex standardised, 65–84 years)')
# value + '1 in N' columns
import matplotlib.transforms as mtrans
tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
xv, xn = 1.22, 1.47
ax.text(xv, len(rows) - 0.45, '% (95% CI)', ha='center', fontsize=6, color=GREY, va='bottom', transform=tr)
ax.text(xn, len(rows) - 0.45, '1 in N', ha='right', fontsize=6, color=GREY, va='bottom', transform=tr)
for c in rows:
    if c == 'Denmark':
        continue
    r = rep.loc[c]
    ax.text(xv, ypos[c], f'{100*r.p:.1f} ({100*r.lo:.1f}–{100*r.hi:.1f})', ha='center', va='center', fontsize=5.8, transform=tr)
    ax.text(xn, ypos[c], f'{1/r.p:.0f}', ha='right', va='center', fontsize=6.2, transform=tr,
            fontweight='bold' if c in SINGLE else 'normal')
for c in rows:
    if c in SINGLE:
        ax.add_patch(plt.Rectangle((1.0, ypos[c]-0.5), 0.5, 1.0, transform=tr, color=BAND, lw=0, clip_on=False, zorder=0))
panel(ax, 'a', None, x=-0.30)
fig.text(.015,.989,'About one in ten, rising steeply with age',fontsize=10,fontweight='bold',va='bottom')
# legend for a
h = [Line2D([], [], marker='o', ls='', mfc=REG[k], mec='white', ms=5, label=k) for k in REG]
h += [Line2D([], [], marker='|', ls='', color=INK, ms=6, mew=1, label='WHO 2000–2025 standard'),
      Line2D([], [], marker='D', ls='', mfc='white', mec=INK, ms=4, label='Single-country cohort')]
ax.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 0.07), frameon=False, fontsize=5.5,
          handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2)

# ---------- b: age gradient ----------
axb = fig.add_subplot(gs[:, 1])
ages = ['65-74', '75-84', '85+']
xb = np.arange(3)
ew = w[(w.measure == 'EWGSOP2') & (w.sex == 'All') & (w.agegroup.isin(ages))]
share_c = sorted(ew[ew.cohort == 'SHARE'].country.unique())
for c in share_c:
    s = ew[(ew.country == c)].set_index('agegroup').reindex(ages)
    v = 100 * s.proportion.values.astype(float)
    axb.plot(xb, v, color='#B9C6D6', lw=0.7, marker='o', ms=1.8, zorder=1)
single_cols = {'ELSA': REG['England'], 'MHAS': REG['Latin America'], 'KLoSA': REG['East Asia'], 'CHARLS': '#6B3E8F'}
lab_name = {'ELSA': 'England', 'MHAS': 'Mexico†', 'KLoSA': 'South Korea', 'CHARLS': 'China'}
for co, col in single_cols.items():
    s = ew[ew.cohort == co].set_index('agegroup').reindex(ages)
    v = 100 * s.proportion.values.astype(float)
    lo = 100 * s.ci_low.values.astype(float); hi = 100 * s.ci_high.values.astype(float)
    off = {'ELSA': -0.06, 'MHAS': 0.06, 'KLoSA': 0.0, 'CHARLS': -0.03}[co]
    axb.errorbar(xb + off, v, yerr=[v - lo, hi - v], color=col, lw=1.4, marker='D', ms=3.6,
                 mec='white', mew=0.5, elinewidth=0.6, capsize=0, zorder=3)
    last = np.where(~np.isnan(v))[0][-1]
    dy = {'ELSA': 0, 'MHAS': 0, 'KLoSA': 4.5, 'CHARLS': -3.0}[co]
    axb.text(xb[last] + off + 0.09, v[last] + dy, f'{lab_name[co]} {v[last]:.0f}%', color=col, fontsize=6,
             va='center', fontweight='bold')
axb.text(2.09, 31, 'SHARE countries\n(grey lines)', color='#7F93AB', fontsize=5.6, va='center')
axb.set_xticks(xb); axb.set_xticklabels(['65–74', '75–84', '≥85'])
axb.set_xlim(-0.25, 2.95)
axb.set_ylim(0, 85)
axb.set_ylabel('Candidates with low grip, %\n(survey-weighted, EWGSOP2)')
axb.set_xlabel('Age band, years')
axb.grid(axis='y', color=LGREY, lw=0.4)
panel(axb, 'b', 'Higher proportions in older age groups\nin most cohorts', x=-0.16)
axb.text(0.0, 79, 'Unreported cells omitted', fontsize=5.4, color=GREY)


save(fig, 'Figure2_burden', out)
