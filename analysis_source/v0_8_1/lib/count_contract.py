"""Validate frozen aggregate count metadata before any participant reader."""
import csv,math

COHORTS={'CHARLS','MHAS','SHARE'}
FIELDS=['n','events_n','Low_n','Low_events_n']

def validate_expected_counts(path):
    with open(path,newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=3 or {r.get('cohort') for r in rows}!=COHORTS:
        raise RuntimeError('FROZEN_COUNT_COHORT_SET_INVALID')
    out={}
    for row in rows:
        c=row['cohort'];parsed={}
        for key in FIELDS:
            try:value=float(row.get(key,''))
            except (TypeError,ValueError):raise RuntimeError('MISSING_OR_INVALID_FROZEN_COUNT:'+c+':'+key) from None
            if not math.isfinite(value) or value<0 or value!=int(value):
                raise RuntimeError('MISSING_OR_INVALID_FROZEN_COUNT:'+c+':'+key)
            parsed[key]=int(value)
        if not (0<parsed['Low_n']<parsed['n'] and 0<parsed['events_n']<=parsed['n']
                and 0<=parsed['Low_events_n']<=min(parsed['Low_n'],parsed['events_n'])
                and parsed['events_n']-parsed['Low_events_n']<=parsed['n']-parsed['Low_n']):
            raise RuntimeError('FROZEN_COUNT_PARTITION_INVALID:'+c)
        out[c]=parsed
    return out
