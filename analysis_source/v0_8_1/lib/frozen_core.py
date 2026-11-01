"""Copied frozen scientific functions; no import-time data I/O and no analysis main."""
from pathlib import Path
import numpy as np
import pandas as pd
SCHEMA={};INV=[];MANIFEST=[];BRIDGE_AUDIT=[]
def suppress(n):return int(n)
def read(*a,**k):raise RuntimeError('DATA_READER_NOT_BOUND')
def extend_followup(*a,**k):raise RuntimeError('WAVE_BRIDGE_NOT_BOUND')
SPEC = {'CHARLS': ('H_CHARLS_D_Data.dta', 'ID', [1, 2, 3], 'mbmi'), 'ELSA': ('h_elsa_g3.dta', 'idauniq', [2, 4, 6], 'mbmi'), 'MHAS': ('H_MHAS_c2.dta', 'unhhidnp', [3], 'mbmi'), 'SHARE': ('H_SHARE_f2.dta', 'mergeid', [1, 2, 4, 5, 6, 7, 8], 'bmi'), 'KLoSA': ('H_KLoSA_e2.dta', 'pid', list(range(1, 9)), 'bmi')}

def num(h, col):
    return pd.to_numeric(h[col], errors='coerce') if col in h else pd.Series(np.nan, index=h.index)

def bn(h, col):
    return num(h, col).where(lambda x: x.isin([0, 1]))

def derive(h, cohort, idcol, w, bmi):
    d = pd.DataFrame(index=h.index)
    d['id'] = h[idcol]
    d['wave'] = w
    d['country'] = num(h, 'country') if cohort == 'SHARE' else cohort
    d['age'] = num(h, f'r{w}agey').combine_first(num(h, f'r{w}agey_e'))
    d['sex'] = num(h, 'ragender')
    d['bmi'] = num(h, f'r{w}{bmi}').where(lambda x: x > 0)
    grips = pd.DataFrame({f'{hand}{t}': num(h, f'r{w}{hand}grip{t}').where(lambda x: x.between(0, 100)) for hand in ['l', 'r'] for t in range(1, 4)})
    d['grip'] = grips.max(axis=1)
    q = lambda s: bn(h, f'r{w}{s}')
    d['nondiabetes_comorb'] = pd.concat([q('hibpe'), q('hrtatte' if cohort == 'MHAS' else 'hearte'), q('stroke'), q('hchole'), q('dyslipe')], axis=1).eq(1).any(axis=1)
    d['any_comorb'] = d.nondiabetes_comorb | q('diabe').eq(1)
    d['low_EWGSOP2'] = d.grip < np.where(d.sex.eq(1), 27, 16)
    d['low_SDOC'] = d.grip < np.where(d.sex.eq(1), 35.5, 20)
    ok = d.age.ge(65) & d.sex.isin([1, 2]) & d.bmi.notna() & d.grip.notna() & num(h, f'r{w}iwstat').eq(1)
    return d.loc[ok].copy()

def eligibility(d, lower=27, upper=30):
    x = d.copy()
    x['eligible'] = np.where(x.bmi >= upper, 'yes', np.where(x.bmi < lower, 'no', np.where(x.any_comorb, 'yes', 'unknown')))
    x['strict_eligible'] = (x.bmi >= upper) | (x.bmi >= lower) & x.nondiabetes_comorb
    return x
ADLMAP = {'bath': 'batha', 'dress': 'dressa', 'eat': 'eata', 'bed': 'beda', 'toilet': 'toilta', 'room': 'walkra'}

HELPMAP = {'bath': 'bathehlp', 'dress': 'dresshlp', 'eat': 'eathlp', 'bed': 'bedhlp', 'toilet': 'toilethlp', 'room': 'walkhlp'}

CHRONMAP = {'htn': 'hibpe', 'cancer': 'cancre', 'lung': 'lunge', 'heart': 'hearte', 'stroke': 'stroke', 'arthritis': 'arthre'}
BASE = {'CHARLS': 1, 'ELSA': 6, 'MHAS': 3, 'SHARE': 4, 'KLoSA': 4}

MAXW = {'CHARLS': 4, 'ELSA': 9, 'MHAS': 5, 'SHARE': 8, 'KLoSA': 8}

WEIGHTS = {'CHARLS': 'r1wtrespbiob', 'ELSA': 'r6nwtresp', 'MHAS': 'factora_12', 'SHARE': 'r4wtresp', 'KLoSA': 'r4wtresp'}

YEARS = {'CHARLS': {1: 2011, 2: 2013, 3: 2015, 4: 2018}, 'ELSA': {6: 2012, 7: 2014, 8: 2016, 9: 2018}, 'MHAS': {3: 2012, 4: 2015, 5: 2018}, 'SHARE': {4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2019}, 'KLoSA': {4: 2012, 5: 2014, 6: 2016, 7: 2018, 8: 2020}}

def cleanstr(s):
    s = s.astype('string')
    return s.mask(s.str.strip().isin(['', 'nan', '<NA>', '.']))

def load(c):
    name, idcol, _, bmi = SPEC[c]
    schema = SCHEMA[name]
    w = BASE[c]
    suffix = list(ADLMAP.values()) + list(CHRONMAP.values()) + list(HELPMAP.values()) + ['bathb', 'dressb', 'eatb', 'bedb_k', 'toiltb', 'respe', 'hrtatte', 'diabe', 'smoken', 'agey', 'agey_e', 'iwstat', 'iwy', 'iwm', 'iwindy', 'iwindm', 'bmi', 'mbmi', 'hibpe', 'hchole', 'dyslipe'] + [f'{hand}grip{i}' for hand in ['l', 'r'] for i in range(1, 4)]
    cols = [idcol, 'ragender', 'raeducl', 'country', 'communityID', 'householdID', 'unhhid', 'np', 'hhid' + str(w), 'rapsu1', 'raestrat1', 'raeclust', 'raestrat', 'radyear', 'radmonth', WEIGHTS[c]] + [f'r{k}{s}' for k in range(w, MAXW[c] + 1) for s in suffix]
    cols = list(dict.fromkeys((k for k in cols if k in schema['labels'])))
    h = read(schema['path'], cols)
    assert h[idcol].is_unique
    h['id'] = h[idcol].astype(str)
    if c == 'MHAS':
        p = next((x['path'] for x in INV if x['path'].endswith('/sect_l_biomarkers_2012.dta')))
        a = read(p, ['cunicah', 'np', 'factora_12']).rename(columns={'cunicah': 'unhhid'})
        h = h.merge(a, on=['unhhid', 'np'], how='left', validate='one_to_one')
    if c == 'ELSA':
        p = next((x['path'] for x in INV if x['path'].endswith('/Raw_data/wave6/wave_6_elsa_data_v2.dta')))
        with pd.io.stata.StataReader(p) as rr:
            lookup = {k.lower(): k for k in rr.variable_labels()}
        a = read(p, [lookup[k] for k in ['idauniq', 'gor', 'idahhw6']])
        a.columns = ['idauniq', 'raw_stratum', 'raw_psu']
        h = h.merge(a, on='idauniq', how='left', validate='one_to_one')
    if c == 'KLoSA':
        p = Path(schema['path']).parent.parent / 'wave4/w04_e.dta'
        a = read(p, ['pid', 'ed']).rename(columns={'ed': 'raw_enumeration_district'})
        h = h.merge(a, on='pid', how='left', validate='one_to_one')
    h['weight'] = num(h, WEIGHTS[c]).where(lambda x: x > 0)
    country = num(h, 'country').astype(str) if c == 'SHARE' else pd.Series(c, index=h.index)
    if c == 'CHARLS':
        ps = h.communityID
        st = pd.Series('ALL', index=h.index)
        hh = h.householdID
        note = 'community_cluster_approximation'
    elif c == 'ELSA':
        ps = h.raw_psu
        st = h.raw_stratum
        hh = ps
        note = 'region_household_cluster'
    elif c == 'SHARE':
        ps = h.rapsu1
        st = h.raestrat1
        hh = h['hhid4']
        note = 'PSU_stratum_or_household_fallback'
    elif c == 'KLoSA':
        ps = h.raw_enumeration_district
        st = h.raestrat
        hh = ps
        note = 'raw_enumeration_district_harmonized_stratum_binding_correction'
    else:
        ps = h.unhhid
        st = pd.Series('ALL', index=h.index)
        hh = ps
        note = 'household_cluster_approximation_four_states'
    ps = cleanstr(ps)
    st = cleanstr(st)
    fallback = ps.isna()
    ps = ps.fillna(cleanstr(hh)).fillna(h.id)
    st = st.fillna('UNKNOWN')
    h['psu'] = country + '_' + ps
    h['stratum'] = country + '_' + st
    h['design_note'] = note
    h = extend_followup(h, c)
    d = eligibility(derive(h, c, idcol, w, bmi))
    d['id'] = d.id.astype(str)
    d['weight'] = h.loc[d.index, 'weight']
    d['agegroup'] = pd.cut(d.age, [65, 75, 85, np.inf], right=False, labels=['65-74', '75-84', '85+']).astype(str)
    d['educ'] = num(h, 'raeducl').where(lambda x: x.isin([1, 2, 3]))
    d['smoke'] = bn(h, f'r{w}smoken')
    d['diabetes'] = bn(h, f'r{w}diabe')
    terms = [('respe' if k == 'lung' else 'hrtatte' if k == 'heart' else s) if c == 'MHAS' else s for k, s in CHRONMAP.items()]
    d['chronic'] = pd.concat([bn(h, f'r{w}{s}') for s in terms], axis=1).sum(axis=1, min_count=6)
    if c == 'SHARE':
        with pd.io.stata.StataReader(schema['path']) as rr:
            cmap = rr.value_labels()['country']
        d['country_name'] = d.country.map(cmap)
    else:
        d['country_name'] = {'CHARLS': 'China', 'ELSA': 'England', 'MHAS': 'Mexico (4-state subsample)', 'KLoSA': 'South Korea'}[c]
    d['grip5'] = d.grip / 5
    return (h, d[d.eligible.eq('yes')].copy(), int(fallback.sum()))

def adl(h, c, w, helped=False):
    if c == 'KLoSA':
        terms = ['bathb', 'dressb', 'eatb', 'bedb_k', 'toiltb']
    else:
        terms = [(HELPMAP if helped else ADLMAP)[k] for k in ['bath', 'dress', 'eat', 'bed', 'toilet']]
    a = pd.concat([bn(h, f'r{w}{s}') for s in terms], axis=1)
    return pd.Series(np.where(a.eq(1).any(axis=1), 1, np.where(a.eq(0).all(axis=1), 0, np.nan)), index=h.index)

def intervals(h, d, c, kind='composite', helped=False):
    w = BASE[c]
    baseadl = adl(h, c, w, helped).reindex(d.index)
    active = pd.Series(True, index=d.index) if kind == 'death' else baseadl.eq(0)
    rows = []
    audit = []
    for v in range(w + 1, MAXW[c] + 1):
        status = num(h, f'r{v}iwstat').reindex(d.index)
        future = pd.concat([num(h, f'r{k}iwstat').isin([1, 4]) for k in range(v + 1, MAXW[c] + 1)], axis=1).any(axis=1).reindex(d.index) if v < MAXW[c] else pd.Series(False, index=d.index)
        bad = status.eq(5) & future
        dead = status.eq(5) & ~bad
        alive = status.isin([1, 4])
        a = adl(h, c, v, helped).reindex(d.index)
        valid = dead | alive if kind == 'death' else dead | status.eq(1) & a.notna()
        keep = active & valid & ~bad
        g = d.loc[keep].copy()
        g['interval'] = v - w
        g['event'] = dead.loc[keep].astype(int)
        if kind != 'death':
            g['event'] = (dead.loc[keep] | a.loc[keep].eq(1)).astype(int)
        g['state'] = np.where(dead.loc[keep], 2, np.where(a.loc[keep].eq(1), 1, 0))
        rows.append(g)
        audit.append({'cohort': c, 'outcome': kind, 'followup_wave': v, 'available_n': suppress(int(keep.sum())), 'censored_unknown_n': suppress(int((active & ~valid).sum())), 'death_alive_conflict_n': suppress(int((active & bad).sum()))})
        active = keep & ~dead
        if kind != 'death':
            active = active & a.eq(0)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), audit)

COV = ['age', 'sex', 'educ', 'bmi', 'smoke', 'diabetes', 'chronic', 'country', 'interval']
