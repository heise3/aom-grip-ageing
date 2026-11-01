"""Same estimands/models; paired participant bootstrap with aggregate-only checkpoints."""
from pathlib import Path
import json,os
import numpy as np,pandas as pd,patsy,statsmodels.api as sm
from analysis import DEFS,TAG,estimate,unavailable
from runtime import f,require,B
import runtime as rt
from extension_math import fit_cloglog,risk_prediction,standardized_risk,first_followup_landmark,NoValidResult

def weighted_auc(risk,time,event,freq,horizon=9.):
    risk=np.asarray(risk,float);time=np.asarray(time,float);event=np.asarray(event,int);freq=np.asarray(freq,float)
    G=1.;left={};Gt=1.
    for tt in sorted(np.unique(time[freq>0])):
        left[tt]=G;at=freq[time>=tt].sum();cen=freq[(time==tt)&(event==0)].sum();G*=1-cen/at
        if tt<horizon:Gt=G
    cases=np.flatnonzero((freq>0)&(event==1)&(time<=horizon));controls=np.flatnonzero((freq>0)&((time>horizon)|((time==horizon)&(event==0))))
    if not len(cases) or not len(controls) or Gt<=0:raise NoValidResult('AUC_CASE_CONTROL_OR_CENSOR_SUPPORT')
    wc=np.array([freq[i]/left[time[i]] if left[time[i]]>0 else np.inf for i in cases])
    if not np.isfinite(wc).all():raise NoValidResult('AUC_CENSOR_POSITIVITY')
    order=np.argsort(risk[controls]);rc=risk[controls][order];ww=freq[controls][order];cs=np.r_[0,np.cumsum(ww)]
    lo=np.searchsorted(rc,risk[cases],side='left');hi=np.searchsorted(rc,risk[cases],side='right')
    concordance=(cs[lo]+.5*(cs[hi]-cs[lo]))/ww.sum()
    return float(wc@concordance/wc.sum())

def projection(m,people,bounds):
    mats=[]
    for it in sorted(m['data'].interval.unique()):
        start,end=bounds[it];fraction=np.clip((9-start)/(end-start),0,1)
        if not fraction:continue
        q=people.copy();q['interval']=it
        x=pd.DataFrame(patsy.build_design_matrices([m['X'].design_info],q)[0],columns=m['X'].columns)
        for k,(mu,sd) in m['scale'].items():x[k]=(x[k]-mu)/sd
        mats.append((x.to_numpy(float),float(fraction)))
    return mats

def predicted_risk(mats,beta):
    L=sum(np.exp(X@beta)*fraction for X,fraction in mats)
    return 1-np.exp(-L)

def frequency_fit(X,y,weight,start):
    active=weight>0
    if np.unique(y[active]).size<2:raise NoValidResult('BOOTSTRAP_NO_OUTCOME_CONTRAST')
    if np.linalg.matrix_rank(X[active])<X.shape[1]:raise NoValidResult('BOOTSTRAP_RANK_DEFICIENT')
    fit=sm.GLM(y,X,family=sm.families.Binomial(link=sm.families.links.CLogLog()),freq_weights=weight).fit(start_params=start,maxiter=150,tol=1e-8)
    if not fit.converged or not np.isfinite(fit.params).all() or np.max(np.abs(fit.params))>30:raise NoValidResult('BOOTSTRAP_NONCONVERGENCE')
    return np.asarray(fit.params)

def bootstrap_auc(z,c,definition,bounds,checkpoint,seed,reps=500):
    expo='low_'+definition
    row=dict(cohort=c,definition=definition,n=z.id.nunique(),method='9-year nominal-wave cumulative_dynamic_IPCW_AUC',performance='apparent_not_external_validation',bootstrap_requested=reps,seed=seed,post_hoc=TAG)
    try:
        base=fit_cloglog(z,expo,c,prediction_base=True,without_exposure=True)
        added=fit_cloglog(z,expo,c,prediction_base=True)
    except NoValidResult as e:return dict(row,status='NOT_ESTIMABLE',reason=str(e))
    people=z.drop_duplicates('id').copy();people['x']=people[expo].astype(float)
    terminal=z.sort_values('interval').groupby('id',sort=False).tail(1).set_index('id').reindex(people.id)
    times=np.array([bounds[int(i)][1] for i in terminal.interval]);ev=terminal.event.to_numpy(int);n=len(people)
    lookup={i:k for k,i in enumerate(people.id)};gid=z.id.map(lookup).to_numpy(int)
    P0=projection(base,people,bounds);P1=projection(added,people,bounds)
    r0=predicted_risk(P0,np.asarray(base['result'].params));r1=predicted_risk(P1,np.asarray(added['result'].params))
    try:b=weighted_auc(r0,times,ev,np.ones(n));a=weighted_auc(r1,times,ev,np.ones(n))
    except NoValidResult as e:return dict(row,status='NOT_ESTIMABLE',reason=str(e))
    row.update(base_AUC=b,added_AUC=a,increment=a-b)
    X0=base['X'].to_numpy(float);X1=added['X'].to_numpy(float);y=z.event.to_numpy(float)
    checkpoint=Path(checkpoint);rng=np.random.default_rng(seed);records=[]
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text());require(old['definition']==definition and old['cohort']==c and old['seed']==seed,'BOOTSTRAP_CHECKPOINT_MISMATCH')
        require(abs(old['point_increment']-(a-b))<1e-10,'BOOTSTRAP_POINT_CHANGED')
        records=old['records'];rng.bit_generator.state=old['rng_state']
    for rep in range(len(records),reps):
        sampled=rng.choice(n,n,replace=True);freq=np.bincount(sampled,minlength=n);weight=freq[gid]
        try:
            beta0=frequency_fit(X0,y,weight,np.asarray(base['result'].params));beta1=frequency_fit(X1,y,weight,np.asarray(added['result'].params))
            delta=weighted_auc(predicted_risk(P1,beta1),times,ev,freq)-weighted_auc(predicted_risk(P0,beta0),times,ev,freq)
            records.append({'replicate':rep+1,'increment':float(delta),'status':'OK'})
        except (NoValidResult,np.linalg.LinAlgError,ValueError,FloatingPointError) as e:
            records.append({'replicate':rep+1,'increment':None,'status':str(e) if isinstance(e,NoValidResult) else type(e).__name__})
        if (rep+1)%10==0 or rep+1==reps:
            payload={'cohort':c,'definition':definition,'seed':seed,'point_increment':a-b,'rng_state':rng.bit_generator.state,'records':records}
            temp=checkpoint.with_suffix('.tmp');temp.write_text(json.dumps(payload,allow_nan=False));os.replace(temp,checkpoint)
        if (rep+1)%50==0:print(c,definition,'bootstrap',rep+1,'/',reps,flush=True)
    good=[r['increment'] for r in records if r['status']=='OK'];fail={}
    for r in records:
        if r['status']!='OK':fail[r['status']]=fail.get(r['status'],0)+1
    row.update(bootstrap_successful=len(good),failure_types=json.dumps(fail))
    if len(good)>=450:
        ci=np.quantile(good,[.025,.975]);row.update(ci_low=float(ci[0]),ci_high=float(ci[1]),status='OK')
    else:row.update(status='POINT_ESTIMATE_ONLY_NO_VALID_BOOTSTRAP_CI')
    return row

def run(c,state,config):
    require(c in ['MHAS','SHARE'],'PROGNOSTIC_COHORT_FORBIDDEN')
    h,d=rt.load(c);panel,_=f.intervals(h,d,c,'composite');z=panel.dropna(subset=f.COV+['low_EWGSOP2','event']).copy()
    w=f.BASE[c];year=f.YEARS[c][w]
    bounds={int(i):(f.YEARS[c][w+int(i)-1]-year,f.YEARS[c][w+int(i)]-year) for i in z.interval.unique()}
    old=pd.read_csv(B/'resources/frozen_primary_models.csv');old=old[(old.cohort==c)&(old.outcome=='composite')&(old.analysis=='primary')].iloc[0]
    oldrisk=pd.read_csv(B/'resources/frozen_risks.csv');oldrisk=oldrisk[oldrisk.cohort==c]
    models=[];risks=[];checks=[];counts=[]
    for definition in DEFS:
        r,m=estimate(z,c,'low_'+definition,'primary_definition');r['definition']=definition;models.append(r)
        if m is None:continue
        if definition=='EWGSOP2':
            require(abs(r['log_effect']-float(old.log_effect))<1e-6 and abs(r['se']-float(old.se))<1e-6,'FROZEN_PRIMARY_REPRODUCTION_FAILED:'+c)
            checks.append(dict(cohort=c,check='primary_logHR_clusterSE',status='PASS'))
        for rr in standardized_risk(m,bounds):
            risks.append(dict(cohort=c,definition=definition,nominal_years=9,**rr,status='OK',post_hoc=TAG))
            if definition=='EWGSOP2':
                q=oldrisk[oldrisk.measure.eq('risk')&oldrisk.low_grip.eq(rr['flagged'])] if rr['measure']=='risk' else oldrisk[oldrisk.measure.eq('risk_difference_low_minus_normal')]
                require(len(q)==1,'FROZEN_RISK_BINDING_FAILED:'+c)
                for newkey,oldkey in [('estimate','risk'),('ci_low','ci_low'),('ci_high','ci_high')]:require(abs(rr[newkey]-float(q.iloc[0][oldkey]))<1e-6,'FROZEN_RISK_REPRODUCTION_FAILED:'+c)
    require(any(r.get('definition')=='EWGSOP2' and r['status']=='OK' for r in models),'FROZEN_PRIMARY_NOT_REPRODUCIBLE:'+c)
    checks.append(dict(cohort=c,check='nine_year_risks_RD_and_CIs',status='PASS'))
    lag=first_followup_landmark(z);r,_=estimate(lag,c,'low_EWGSOP2','first_followup_landmark');r['definition']='EWGSOP2';models.append(r)
    counts=[dict(cohort=c,stage='original_primary',n=z.id.nunique(),events_n=int(z.event.sum())),dict(cohort=c,stage='first_followup_observed_eventfree',n=z.loc[z.interval.eq(1)&z.event.eq(0),'id'].nunique()),dict(cohort=c,stage='landmark_with_later_observable_interval',n=lag.id.nunique(),events_n=int(lag.event.sum()))]
    aucs=[]
    for definition in DEFS:
        aucs.append(bootstrap_auc(z,c,definition,bounds,state/f'{c}_{definition}_bootstrap.json',config['seeds'][definition],config['bootstrap_replicates']))
    return {'T3_T4_models':pd.DataFrame(models),'definition_risks':pd.DataFrame(risks),'definition_AUC':pd.DataFrame(aucs),'landmark_flow':pd.DataFrame(counts),'frozen_reproduction':pd.DataFrame(checks)}
