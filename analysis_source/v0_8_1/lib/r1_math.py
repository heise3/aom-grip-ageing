"""Paired-person relative log-HR bootstrap. Pure in-memory, no participant readers."""
from pathlib import Path
import json,os
import numpy as np
from extension_math import NoValidResult,effect
from prognosis import frequency_fit

def relative_change(beta0,beta1,tolerance=1e-8):
    if not np.isfinite([beta0,beta1]).all() or abs(beta0)<=tolerance:
        raise NoValidResult('RELATIVE_LOGHR_UNDEFINED_NEAR_ZERO_REFERENCE')
    return float(100*(beta0-beta1)/beta0)

def paired_bootstrap(m0,m1,checkpoint=None,reps=500,min_valid=450,seed=20261031,tolerance=1e-8,progress=None):
    z0=m0['data'];z1=m1['data']
    if not z0[['id','interval','event']].equals(z1[['id','interval','event']]):
        raise NoValidResult('R1_PEOPLE_INTERVALS_NOT_IDENTICAL')
    people=z0.id.drop_duplicates();n=len(people);lookup={x:i for i,x in enumerate(people)}
    group=z0.id.map(lookup).to_numpy(int);y=z0.event.to_numpy(float)
    X0=m0['X'].to_numpy(float);X1=m1['X'].to_numpy(float)
    b0=np.asarray(m0['result'].params);b1=np.asarray(m1['result'].params)
    i0=m0['X'].columns.get_loc('x');i1=m1['X'].columns.get_loc('x')
    point=relative_change(b0[i0],b1[i1],tolerance)
    rng=np.random.default_rng(seed);records=[]
    path=Path(checkpoint) if checkpoint else None
    binding={'n':n,'rows':len(z0),'seed':seed,'reps':reps,'point_beta0':float(b0[i0]),'point_beta1':float(b1[i1])}
    if path and path.exists():
        old=json.loads(path.read_text())
        if old['binding']!=binding:raise NoValidResult('R1_CHECKPOINT_BINDING_CHANGED')
        records=old['records'];rng.bit_generator.state=old['rng_state']
    for rep in range(len(records),reps):
        sampled=rng.choice(n,n,replace=True);freq=np.bincount(sampled,minlength=n);w=freq[group]
        try:
            c0=frequency_fit(X0,y,w,b0);c1=frequency_fit(X1,y,w,b1)
            value=relative_change(c0[i0],c1[i1],tolerance)
            records.append({'replicate':rep+1,'beta0':float(c0[i0]),'beta1':float(c1[i1]),'relative_change_percent':value,'status':'OK'})
        except (NoValidResult,np.linalg.LinAlgError,ValueError,FloatingPointError) as e:
            records.append({'replicate':rep+1,'relative_change_percent':None,'status':str(e) if isinstance(e,NoValidResult) else type(e).__name__})
        if path and ((rep+1)%10==0 or rep+1==reps):
            value={'binding':binding,'records':records,'rng_state':rng.bit_generator.state,'individual_rows_written':False}
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False));os.replace(tmp,path)
        if progress and (rep+1)%50==0:progress(rep+1,reps)
    good=[r['relative_change_percent'] for r in records if r['status']=='OK'];fail={}
    for r in records:
        if r['status']!='OK':fail[r['status']]=fail.get(r['status'],0)+1
    row={'relative_logHR_change_percent':point,'bootstrap_requested':reps,'bootstrap_successful':len(good),
        'bootstrap_failure_types':json.dumps(fail,sort_keys=True),'seed':seed,
        'relative_change_ci_low':None,'relative_change_ci_high':None,
        'relative_change_status':'OK' if len(good)>=min_valid else 'POINT_ONLY_INSUFFICIENT_VALID_BOOTSTRAP'}
    if len(good)>=min_valid:
        lo,hi=np.quantile(good,[.025,.975]);row.update(relative_change_ci_low=float(lo),relative_change_ci_high=float(hi))
    return row
