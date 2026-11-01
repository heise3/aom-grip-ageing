"""Pure in-memory post hoc mathematics. No file readers or writers, no HRS code.

Not an execution authorization. Callers must bind/lock methods, sources and terms.
Synthetic verification does not establish real-data validity or equivalence.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.stats import norm

FROZEN_RHS='age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'

class NoValidResult(Exception):
    pass

def fi_score(deficits, min_observed=0, max_missing_fraction=.20):
    a=np.asarray(deficits,float)
    if a.ndim!=2 or a.shape[1]<30:raise ValueError('At least 30 mapped deficits required')
    if np.any(np.isfinite(a)&((a<0)|(a>1))):raise ValueError('FI scores outside 0-1')
    n=np.isfinite(a).sum(1)
    threshold=max(int(min_observed),int(np.ceil((1-max_missing_fraction)*a.shape[1])))
    value=np.divide(np.nansum(a,axis=1),n,out=np.full(a.shape[0],np.nan),where=n>0)
    value[n<threshold]=np.nan
    return value,n

def first_followup_landmark(z):
    """Only use after the author chooses this estimand. No calendar death dates."""
    first=z.loc[z.interval.eq(1)&z.event.eq(0),'id']
    q=z[z.id.isin(first)&z.interval.gt(1)].copy()
    return q  # Preserve relative interval labels; formula uses a categorical term.

def fit_cloglog(z, exposure, cohort, add_fi=False, prediction_base=False,
                without_exposure=False, event_gate=1, continuous_exposure=False):
    z=z.copy();z['x']=z[exposure].astype(float)
    rhs='age + C(sex) + bmi + chronic + C(interval)' if prediction_base else FROZEN_RHS
    if not without_exposure:rhs='x + '+rhs
    if add_fi:rhs+=' + FI'
    if cohort=='SHARE':rhs+=' + C(country)'
    X=patsy.dmatrix(rhs,z,return_type='dataframe',NA_action='raise')
    if z.id.nunique()<2 or z.event.nunique()!=2:raise NoValidResult('NO_OUTCOME_CONTRAST')
    if not without_exposure:
        if continuous_exposure:
            if z.x.nunique()<2:raise NoValidResult('NO_EXPOSURE_VARIATION')
        elif z.x.nunique()!=2:raise NoValidResult('NO_EXPOSURE_CONTRAST')
        # The historic <10 fit-screen was a project disclosure gate, removed by T1.
        # A genuinely zero-event arm still has no finite regular binary PH contrast.
        if not continuous_exposure and any(z.loc[z.x.eq(v),'event'].sum()<event_gate for v in [0,1]):
            raise NoValidResult('ZERO_EVENT_EXPOSURE_GROUP')
    scale={}
    for k in ['age','bmi','chronic']:
        if k in X and X[k].std()>0:
            scale[k]=(float(X[k].mean()),float(X[k].std()))
            X[k]=(X[k]-scale[k][0])/scale[k][1]
    if np.linalg.matrix_rank(X.to_numpy())<X.shape[1]:raise NoValidResult('DESIGN_RANK_DEFICIENT')
    rr=sm.GLM(z.event.to_numpy(float),X.to_numpy(float),
              family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
                  cov_type='cluster',cov_kwds={'groups':z.id},maxiter=150,tol=1e-8)
    if not rr.converged or not np.isfinite(rr.params).all() or np.max(np.abs(rr.params))>30:
        raise NoValidResult('NONCONVERGENCE_OR_EXTREME_COEFFICIENT')
    if not np.isfinite(rr.cov_params()).all():raise NoValidResult('INVALID_CLUSTER_COVARIANCE')
    return {'result':rr,'X':X,'scale':scale,'data':z,'rhs':rhs}

def effect(m):
    i=m['X'].columns.get_loc('x');r=m['result'];b=float(r.params[i]);s=float(r.bse[i])
    if not s>0 or not np.isfinite(s):raise NoValidResult('INVALID_EXPOSURE_SE')
    return {'log_effect':b,'se':s,'effect':float(np.exp(b)),
            'ci_low':float(np.exp(b-1.96*s)),'ci_high':float(np.exp(b+1.96*s)),
            'p':float(2*norm.sf(abs(b/s)))}

def risk_prediction(m, people, interval_bounds, horizon=9.):
    """Original fixed-baseline standardization and joint delta gradient.

    interval_bounds is a locked dict {original interval: (start_year,end_year)}.
    Missing support is an error; no new interval or covariate extrapolation is added.
    """
    B=np.asarray(m['result'].params);L=np.zeros(len(people));G=np.zeros((len(people),len(B)))
    for it in sorted(m['data'].interval.unique()):
        start,end=interval_bounds[it]
        if end<=start:raise ValueError('Invalid locked interval bounds')
        frac=np.clip((horizon-start)/(end-start),0,1)
        if frac==0:continue
        q=people.copy();q['interval']=it
        xx=pd.DataFrame(patsy.build_design_matrices([m['X'].design_info],q)[0],columns=m['X'].columns)
        for k,(mu,sd) in m['scale'].items():xx[k]=(xx[k]-mu)/sd
        x=xx.to_numpy();H=np.exp(x@B)*frac;L+=H;G+=H[:,None]*x
    S=np.exp(-L)
    return 1-S,(S[:,None]*G).mean(axis=0)

def standardized_risk(m, interval_bounds, horizon=9.):
    people=m['data'].drop_duplicates('id').copy();V=np.asarray(m['result'].cov_params());values=[];gradients=[];out=[]
    for val in [0,1]:
        q=people.copy();q['x']=val
        indiv,g=risk_prediction(m,q,interval_bounds,horizon);p=float(indiv.mean());se=float(np.sqrt(g@V@g))
        if not 0<p<1:raise NoValidResult('BOUNDARY_STANDARDIZED_RISK')
        ci=expit(logit(p)+np.array([-1,1])*1.96*se/(p*(1-p)))
        out.append({'measure':'risk','flagged':val,'estimate':p,'ci_low':float(ci[0]),'ci_high':float(ci[1])})
        values.append(p);gradients.append(g)
    g=gradients[1]-gradients[0];rd=values[1]-values[0];se=float(np.sqrt(g@V@g))
    out.append({'measure':'risk_difference','estimate':rd,'ci_low':rd-1.96*se,'ci_high':rd+1.96*se})
    return out

def auc_ipcw(risk,time,event,horizon=9.):
    """Nominal-endpoint cumulative/dynamic IPCW AUC; known event-free through horizon.

    Marginal censoring KM, left limit at case times. Independent censoring assumed.
    Not Harrell's C, not external validation, not exact-date outcome discrimination.
    """
    risk=np.asarray(risk,float);time=np.asarray(time,float);event=np.asarray(event,int)
    if not(len(risk)==len(time)==len(event)) or not np.isfinite(risk).all():raise ValueError('AUC input mismatch')
    survival=1.;left={};Gt=1.
    for tt in sorted(np.unique(time)):
        left[tt]=survival
        at=np.sum(time>=tt);censored=np.sum((time==tt)&(event==0))
        survival*=1-censored/at
        if tt<horizon:Gt=survival
    cases=np.flatnonzero((event==1)&(time<=horizon))
    controls=np.flatnonzero((time>horizon)|((time==horizon)&(event==0)))
    if not len(cases) or not len(controls) or Gt<=0:raise NoValidResult('AUC_NO_CASE_OR_CONTROL_OR_CENSOR_SUPPORT')
    wc=np.array([1/left[time[i]] if left[time[i]]>0 else np.inf for i in cases])
    if not np.isfinite(wc).all():raise NoValidResult('AUC_CENSOR_POSITIVITY')
    # All controls have the same marginal IPCW; it cancels in the weighted rank ratio.
    rc=np.sort(risk[controls]);lo=np.searchsorted(rc,risk[cases],side='left');hi=np.searchsorted(rc,risk[cases],side='right')
    concordant=(lo+.5*(hi-lo))/len(controls)
    return float(np.dot(wc,concordant)/wc.sum())
