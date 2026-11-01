# Academic Figure Skill Asset Confirmation
# Existing manuscript production assets reused as native vector renderers.
# Data: permitted aggregate CSVs only; no individual records and no model fitting.
import sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from common import *

d, t = load()
out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/A4/out'
fl = pd.read_csv(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'figdata/Figure1_QCv2_flow.csv')).set_index(['cohort', 'stage']).n
s1 = t['REPORT_tableS1']


def num(c, st):
    v = str(fl.loc[(c, st)])
    return int(float(v)) if v.replace('.', '').isdigit() else None


def s1v(c, var):
    r = s1[(s1.cohort == c) & (s1.variable == var)].iloc[0]
    return float(r['mean'])


C = [('CHARLS', 'China', '2011 (W1)', 'Measured', [2011, 2013, 2015, 2018, 2020], 'comp'),
     ('ELSA', 'England', '2012 (W6)', 'Measured', [2012], 'none'),
     ('MHAS', 'Mexico (4 states),\nbiomarker subsample', '2012 (W3)', 'Measured', [2012, 2015, 2018, 2021], 'comp'),
     ('SHARE', '16 European\ncountries', '2011 (W4)', 'Self-report', [2011, 2013, 2015, 2017, 2019, 2022], 'comp'),
     ('KLoSA', 'South Korea', '2012 (W4)', 'Self-report', [2012, 2014, 2016, 2018, 2020], 'death')]

fig = plt.figure(figsize=(8.8, 6.6))
ax = fig.add_axes([0.0, 0.49, 1.0, 0.46]); ax.axis('off')
ax.set_xlim(0, 100); ax.set_ylim(-0.2, 6.2)
cols = {'cohort': 0.8, 'base': 23.0, 'bmi': 31.0, 'n65': 39.0, 'valid': 46.5, 'cand': 52.0, 'mean': 72.5, 'tl0': 79.5, 'tl1': 98.5}
hy = 5.85
heads = [('cohort', 'Cohort and setting', 'left'), ('base', 'Baseline', 'center'), ('bmi', 'BMI', 'center'),
         ('n65', 'Aged ≥65\ninterviewed', 'center'), ('valid', 'Valid grip\nand BMI', 'center'),
         ('cand', 'Label-defined candidates\n(% of valid)', 'left'), ('mean', 'Candidates:\nBMI | grip', 'center')]
for k, s_, ha in heads:
    ax.text(cols[k], hy, s_, ha=ha, va='center', fontsize=6, color=GREY, linespacing=1.05)
# timeline axis
t0, t1 = 2011, 2022
X = lambda yr: cols['tl0'] + (yr - t0) / (t1 - t0) * (cols['tl1'] - cols['tl0'])
ax.text((cols['tl0'] + cols['tl1']) / 2, hy + 0.12, 'Follow-up waves used', ha='center', va='center', fontsize=6, color=GREY)
for yr in (2011, 2014, 2017, 2020):
    ax.text(X(yr), hy - 0.28, str(yr), ha='center', va='center', fontsize=5.3, color=GREY)
ax.plot([0.5, 99.5], [5.4, 5.4], color=INK, lw=0.6)
for i, (co, setting, base, bmi, waves, role) in enumerate(C):
    y = 4.75 - i * 1.0
    if i % 2 == 0:
        ax.add_patch(Rectangle((0.3, y - 0.48), 99.4, 0.96, color=BAND, lw=0, zorder=0))
    ax.text(cols['cohort'], y + 0.14, co, fontweight='bold', fontsize=7, va='center')
    ax.text(cols['cohort'] + 7.6, y, setting, fontsize=5.9, va='center', linespacing=1.0)
    ax.text(cols['base'], y, base, ha='center', va='center', fontsize=6)
    ax.text(cols['bmi'], y, bmi, ha='center', va='center', fontsize=6,
            color=INK if bmi == 'Measured' else '#7A5C00', style='normal' if bmi == 'Measured' else 'italic')
    n65 = num(co, 'baseline_age65'); nv = num(co, 'sex_BMI_grip_available_after_QC'); nc = num(co, 'confirmed_eligible_with_grip')
    ax.text(cols['n65'], y, f'{n65:,}', ha='center', va='center', fontsize=6.2)
    ax.text(cols['valid'], y, f'{nv:,}' if nv else 'n/s‡', ha='center', va='center', fontsize=6.2,
            color=INK if nv else GREY)
    bx, bw = cols['cand'], 7.0
    ax.add_patch(Rectangle((bx, y - 0.13), bw, 0.26, color=LGREY, lw=0))
    if nc and nv:
        p = nc / nv
        ax.add_patch(Rectangle((bx, y - 0.13), bw * p, 0.26, color=LOW, lw=0))
        ax.text(bx + bw + 0.8, y, f'{nc:,} ({100*p:.1f}%)', va='center', fontsize=6.2)
    else:
        ax.add_patch(Rectangle((bx, y - 0.13), bw, 0.26, fill=False, hatch='/////', ec=GREY, lw=0))
        ax.text(bx + bw + 0.8, y, 'n/s‡', va='center', fontsize=6.2, color=GREY)
    ax.text(cols['mean'], y, f'{s1v(co, "bmi"):.1f} | {s1v(co, "grip"):.1f}', ha='center', va='center', fontsize=6.2)
    # timeline
    ax.plot([X(t0), X(t1)], [y, y], color=LGREY, lw=0.6, zorder=1)
    if role == 'none':
        ax.plot(X(waves[0]), y, marker='o', ms=4.2, mfc='white', mec=INK, mew=0.8, zorder=3)
        ax.text(X(waves[0]) + 0.9, y, 'burden only (no complete\nmortality chain)', fontsize=5.4, color=GREY, va='center', linespacing=1.0)
    else:
        col = LOW if role == 'comp' else GREY
        ax.plot([X(waves[0]), X(waves[-1])], [y, y], color=col, lw=1.4, zorder=2)
        for k_, yr in enumerate(waves):
            ax.plot(X(yr), y, marker='o', ms=4.2, mfc='white' if k_ == 0 else col, mec=col, mew=0.8, zorder=3)
        if role == 'death':
            ax.text(X(waves[-1]) + 0.8, y, 'death\nonly§', fontsize=5.3, color=GREY, va='center', linespacing=1.0)
ax.plot([0.5, 99.5], [0.2, 0.2], color=INK, lw=0.6)
ax.text(0.8, -0.05, 'Open circle, baseline. Red, ADL-or-death follow-up; grey, death-only follow-up. '
        'Candidates: weighted mean BMI (kg/m²) and grip (kg).\n'
        '§KLoSA ADL items record need for help, not difficulty.', fontsize=5.3, color=GREY, va='top')
fig.text(0.005, 0.955, 'a', fontsize=10, fontweight='bold', va='bottom')
fig.text(0.03, 0.957, 'Five harmonised ageing cohorts, 20 countries and regions, baselines 2011–2012', fontsize=7.5,
         fontweight='bold', va='bottom')

# ---------- b: one analytical flow; no decorative question cards ----------
from matplotlib.patches import FancyArrowPatch
fig.text(0.005, 0.392, 'b', fontsize=10, fontweight='bold')
fig.text(0.03, 0.394, 'Eligibility and analysis populations', fontsize=7.5, fontweight='bold')

def flowbox(x,y,w,h,title,body,edge=INK):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor='white',edgecolor=edge,lw=0.8))
    fig.text(x+0.012,y+h-0.019,title,fontsize=7,fontweight='bold',va='top',color=INK)
    fig.text(x+0.012,y+h-0.048,body,fontsize=6.3,va='top',linespacing=1.35,color=INK)

def arrow(x1,y1,x2,y2):
    fig.add_artist(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,
                   arrowstyle='-|>',mutation_scale=9,lw=0.9,color=INK))

flowbox(0.01,0.185,0.19,0.155,'Baseline measurement sample',
        'Age ≥65 years\nKnown sex and BMI\nAt least one valid grip trial')
flowbox(0.25,0.185,0.29,0.155,'Observable label eligibility',
        'BMI ≥30 kg/m²\nor BMI 27 to <30 kg/m² with\nat least one recorded qualifying condition¹')
arrow(0.205,0.262,0.244,0.262)
# Single candidate stream separates into two genuinely different analysis populations.
fig.lines.append(plt.Line2D([0.54,0.57],[0.262,0.262],transform=fig.transFigure,color=INK,lw=0.9))
fig.lines.append(plt.Line2D([0.57,0.57],[0.155,0.285],transform=fig.transFigure,color=INK,lw=0.9))
arrow(0.57,0.285,0.604,0.285);arrow(0.57,0.155,0.604,0.155)
flowbox(0.61,0.24,0.38,0.10,'Descriptive analysis — five cohorts',
        'Eligible participants with positive survey weights\nCountry proportions; standardisation at ages 65–84',NORM)
flowbox(0.61,0.065,0.38,0.14,'Primary prognosis — CHARLS, MHAS, SHARE',
        'Eligible participants free of ADL difficulty at baseline,\ncomplete model covariates and usable follow-up\nOutcome: first ADL difficulty or death',LOW)
fig.text(0.012,0.16,'Outside the candidate stream:',fontsize=6.2,fontweight='bold',va='top')
fig.text(0.012,0.139,'BMI <27: not eligible; BMI 27 to <30 without a recorded\nqualifying condition: undetermined. Invalid BMI is not classified.\nDetailed inclusion and exclusion counts: Figure S1.',fontsize=6.0,va='top',linespacing=1.3)
fig.text(0.012,0.035,'Low grip strength (EWGSOP2): men <27 kg; women <16 kg. Maximum of valid trials from both hands.\n¹Recorded hypertension, diabetes, heart disease/stroke or dyslipidaemia, as available. Observable eligibility is a proxy.',
         fontsize=6.1,va='top',linespacing=1.3,color=INK)

save(fig, 'Figure1_design', out)
