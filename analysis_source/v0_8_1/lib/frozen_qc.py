"""Exact frozen QC functions; no old output paths or analysis main."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import frozen_core as f
MEASURED = {'CHARLS': (1, 150, False), 'ELSA': (6, 130, True), 'MHAS': (3, 150, True)}

TOL = 1e-06

@lru_cache(maxsize=8)
def measurement(c, lower=1.0):
    w, upper, inclusive = MEASURED[c]
    name, idcol, _, _ = f.SPEC[c]
    sc = f.SCHEMA[name]
    cols = [idcol, 'ID_w1', f'r{w}agey', f'r{w}agey_e', f'r{w}iwstat', f'r{w}mheight', f'r{w}mweight', f'r{w}mbmi', f'r{w}htcomp', f'r{w}wtcomp']
    q = f.read(sc['path'], [x for x in cols if x in sc['labels']])
    q['id'] = q[idcol].astype(str)
    age = f.num(q, f'r{w}agey').combine_first(f.num(q, f'r{w}agey_e'))
    base = age.ge(65) & f.num(q, f'r{w}iwstat').eq(1)
    h = f.num(q, f'r{w}mheight')
    wt = f.num(q, f'r{w}mweight')
    b = f.num(q, f'r{w}mbmi')
    bv = b.gt(0) & np.isfinite(b)
    flags = {'height_below_min': h.notna() & h.lt(lower - TOL), 'height_above210': h.gt(2.1 + TOL), 'weight_nonpositive': wt.notna() & wt.le(0), 'weight_over_limit': wt.gt(upper + TOL) if inclusive else wt.ge(upper - TOL)}
    flags['BMI_present_measure_missing'] = bv & (~np.isfinite(h) | ~np.isfinite(wt))
    calc = wt / h ** 2
    flags['BMI_formula_conflict'] = bv & h.gt(0) & wt.gt(0) & np.isfinite(calc) & ((b - calc).abs() > 1e-06 + 1e-05 * calc.abs())
    flags['completion_conflict'] = bv & (f.num(q, f'r{w}htcomp').eq(0) | f.num(q, f'r{w}wtcomp').eq(0))
    flags['source_link_or_unit_conflict'] = pd.Series(False, index=q.index)
    flags['explicit_not_completed'] = pd.Series(False, index=q.index)
    if c == 'CHARLS':
        path = next((x['path'] for x in f.INV if x['path'].endswith('/2011charls/biomarkers.dta')))
        with pd.io.stata.StataReader(path) as reader:
            labels = reader.variable_labels()
        raw = f.read(path, [x for x in ['ID', 'qi002', 'ql002', 'pi001', 'pl001', 'pl002'] if x in labels])
        assert raw.ID.is_unique
        raw = raw.set_index('ID').reindex(q.ID_w1.astype(str))
        raw.index = q.index
        rh = raw.qi002.where(raw.qi002.between(1, 2), (raw.qi002 / 100).where(raw.qi002.between(3, 200)))
        flags['source_link_or_unit_conflict'] = bv & (~np.isfinite(rh) | ~np.isfinite(raw.ql002) | ((rh - h).abs() > 1e-06 + 1e-05 * h.abs()) | ((raw.ql002 - wt).abs() > 1e-06 + 1e-05 * wt.abs()))
        flags['explicit_not_completed'] = raw.pi001.eq(5) | raw.pl001.eq(5) | raw.pl002.eq(5)
        if 'pi001' in raw:
            flags['completion_conflict'] |= bv & raw.pi001.eq(5)
        if 'pl001' in raw:
            flags['completion_conflict'] |= bv & raw.pl001.eq(5) & ~flags['weight_over_limit']
        if 'pl002' in raw:
            flags['completion_conflict'] |= bv & raw.pl002.eq(5)
    bad = pd.concat([flags[k] for k in ['height_below_min', 'height_above210', 'weight_nonpositive', 'weight_over_limit', 'explicit_not_completed']], axis=1).any(axis=1)
    conflict = pd.concat([flags[k] for k in ['BMI_present_measure_missing', 'BMI_formula_conflict', 'completion_conflict', 'source_link_or_unit_conflict']], axis=1).any(axis=1)
    conflict &= ~bad
    return (q, base, bv, flags, bad, conflict)

def qc_load(c, lower=1.0):
    """Apply measurement validity before every original eligibility derive, without changing original code."""
    if c not in MEASURED:
        return f.load(c)
    q, base, bv, flags, bad, conflict = measurement(c, lower)
    assert not (base & conflict).any(), 'Unresolved measurement-source conflict'
    invalid_ids = set(q.loc[bad, 'id'])
    original = f.derive

    def derive_qc(h, cohort, idcol, w, bmi):
        if cohort == c:
            h = h.copy()
            mask = h[idcol].astype(str).isin(invalid_ids)
            h.loc[mask, f'r{w}{bmi}'] = np.nan
        return original(h, cohort, idcol, w, bmi)
    f.derive = derive_qc
    try:
        h, d, a = f.load(c)
    finally:
        f.derive = original
    h.loc[h.id.isin(invalid_ids), f'r{f.BASE[c]}mbmi'] = np.nan
    return (h, d, a)
