"""Shared style + frozen-data loader for A manuscript v0.4 figures.
All plotted values are read from frozen aggregate tables (figdata_full_precision.json and
source_tables/*.csv from A_SUBMISSION_PREP_v0_3). No model is fitted here. The only
display derivations are: reciprocal of per-5-kg-higher HRs, inverse-variance weight shares
(tau^2 = 0), '1 in N' = 1/proportion, and percentage formatting.
"""
import json, os, math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

DATA = os.environ.get('A_DATA', os.path.join(os.path.dirname(__file__), '..'))
ST = os.path.join(DATA, 'source_tables')

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.dpi': 300, 'figure.dpi': 150,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})

# palette (colour-blind safe; grip groups also separated by marker / line style)
LOW = '#C8432B'      # low grip
LOW_L = '#F2C4B8'
NORM = '#2F6DB5'     # normal grip
NORM_L = '#C9DAF0'
INK = '#222222'
GREY = '#8A8A8A'
LGREY = '#D9D9D9'
BAND = '#F4F1EC'
REG = {  # region colours for burden figure
    'Northern & Western Europe': '#7FA7CF',
    'Central & Eastern Europe': '#2E5A8A',
    'Southern Europe': '#14324F',
    'England': '#2A9D7F',
    'East Asia': '#A0559E',
    'Latin America': '#E08A1E',
}
REGION_OF = {
    'Germany': 'Northern & Western Europe', 'Netherlands': 'Northern & Western Europe',
    'Belgium': 'Northern & Western Europe', 'France': 'Northern & Western Europe',
    'Switzerland': 'Northern & Western Europe', 'Austria': 'Northern & Western Europe',
    'Sweden': 'Northern & Western Europe', 'Denmark': 'Northern & Western Europe',
    'Czechia': 'Central & Eastern Europe', 'Poland': 'Central & Eastern Europe',
    'Hungary': 'Central & Eastern Europe', 'Slovenia': 'Central & Eastern Europe',
    'Estonia': 'Central & Eastern Europe',
    'Italy': 'Southern Europe', 'Spain': 'Southern Europe', 'Portugal': 'Southern Europe',
    'England': 'England', 'China': 'East Asia', 'South Korea': 'East Asia',
    'Mexico': 'Latin America',
}
ISO = {'Germany': 'DE', 'Netherlands': 'NL', 'Belgium': 'BE', 'France': 'FR', 'Switzerland': 'CH',
       'Austria': 'AT', 'Sweden': 'SE', 'Denmark': 'DK', 'Czechia': 'CZ', 'Poland': 'PL',
       'Hungary': 'HU', 'Slovenia': 'SI', 'Estonia': 'EE', 'Italy': 'IT', 'Spain': 'ES',
       'Portugal': 'PT', 'England': 'ENG', 'China': 'CN', 'South Korea': 'KR', 'Mexico': 'MX'}
SINGLE = {'China': 'CHARLS', 'England': 'ELSA', 'South Korea': 'KLoSA', 'Mexico': 'MHAS'}


def cname(c):
    c = c.replace('Czech Republic', 'Czechia')
    return 'Mexico' if c.startswith('Mexico') else c


def load():
    d = json.load(open(os.path.join(DATA, 'figdata_full_precision.json')))
    t = {}
    for f in os.listdir(ST):
        if f.endswith('.csv'):
            t[f[:-4]] = pd.read_csv(os.path.join(ST, f))
    return d, t


def fmt_ci(e, lo, hi, nd=2):
    f = f'{{:.{nd}f}}'
    lo_s = f.format(lo) if lo >= 0 else '−' + f.format(-lo)
    return f'{f.format(e)} ({lo_s}–{f.format(hi)})'


def pct(x, nd=1):
    return f'{100 * x:.{nd}f}'


def panel(ax, letter, title=None, x=-0.02, y=1.0, fig=None):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='right')
    if title:
        ax.set_title(title, pad=4)


def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.03)
