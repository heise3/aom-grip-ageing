"""Locked calculations only. All results returned by these functions are aggregate tables."""
from pathlib import Path
import json,hashlib,warnings
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.special import expit,logit
from scipy.stats import t,chi2
from scipy.optimize import minimize_scalar
import runtime as rt
from runtime import f,qc,require,SafeStop,B
from extension_math import fit_cloglog,effect,standardized_risk,risk_prediction,first_followup_landmark,NoValidResult

DEFS=['EWGSOP2','SDOC','FNIH']
TAG='post hoc; v0.7 author decisions2026-10-31'

def unavailable(c,analysis,reason,**extra):
    return dict(cohort=c,analysis=analysis,status='NOT_ESTIMABLE',reason=reason,post_hoc=TAG,**extra)

def estimate(z,c,exposure,analysis,**kw):
    if z.empty:return unavailable(c,analysis,'EMPTY_RISK_SET'),None
    try:
        m=fit_cloglog(z,exposure,c,**kw)
        return dict(cohort=c,analysis=analysis,n=int(z.id.nunique()),events_n=int(z.event.sum()),**effect(m),status='OK',post_hoc=TAG),m
    except NoValidResult as e:return unavailable(c,analysis,str(e)),None
    except (np.linalg.LinAlgError,FloatingPointError):return unavailable(c,analysis,'NUMERICAL_FIT_FAILURE'),None

def screen_fi(h,d,c):
    p=B/'resources/FI_candidate_pool_v0_7.json'
    pool=json.loads(p.read_text());lock=json.loads((B/'resources/FI_candidate_pool_lock.json').read_text())
    require(hashlib.sha256(p.read_bytes()).hexdigest()==lock['sha256'],'FI_POOL_LOCK_MISMATCH')
    items=[i for i in pool['items'] if i['cohort']==c]
    require(all(not i['requires_help'] for i in items),'HELP_ITEM_IN_FI_POOL')
    sc=f.SCHEMA[f.SPEC[c][0]];idcol=f.SPEC[c][1]
    raw=rt.read(sc['path'],[idcol]+[i['field'] for i in items]);raw['id']=raw[idcol].astype(str)
    require(raw.id.is_unique,'NONUNIQUE_FI_LINKAGE')
    raw=raw.set_index('id').reindex(d.id);rows=[];S=[];age=d.age.to_numpy(float)
    require(len(d)>0,'EMPTY_CANDIDATE_UNIVERSE')
    for it in items:
        x=pd.to_numeric(raw[it['field']],errors='coerce').map({float(k):v for k,v in it['codes'].items()}).to_numpy(float)
        ok=np.isfinite(x);n=int(ok.sum());missing=1-n/len(d);prev=float(np.mean(x[ok]>0)) if n else np.nan
        slope=float(np.cov(age[ok],x[ok],ddof=0)[0,1]/np.var(age[ok])) if n>1 and np.var(age[ok])>0 else np.nan
        keep=bool(.01<=prev<=.8 and missing<=.1 and slope>0)
        rows.append(dict(cohort=c,concept=it['concept'],field=it['field'],domain=it['domain'],n=len(d),nonmissing_n=n,missing_n=len(d)-n,missing_fraction=missing,prevalence_score_positive=prev,mean_score=float(np.mean(x[ok])) if n else np.nan,age_slope=slope,selected_main=keep,selected_noMob=keep and not it['mobility_strength'],common_equiv=it['common_equiv'],status='SCREENED',post_hoc=TAG))
        S.append(x)
    return pd.DataFrame(rows),np.column_stack(S),items

def describe(d,c,groups,weighted=False):
    rows=[];variables=['age','sex','bmi','grip'] if weighted else ['age','sex','educ','bmi','smoke','diabetes','chronic','country','grip','eligible']
    for group,sub in groups:
        for var in variables:
            a=sub[var];valid=a.notna();x=a[valid];w=sub.loc[valid,'weight'].to_numpy(float) if weighted else np.ones(len(x))
            common=dict(cohort=c,group=group,variable=var,n=len(x),missing_n=int((~valid).sum()),weighting=f.WEIGHTS[c] if weighted else 'unweighted',status='OK')
            if var in ['age','bmi','grip','chronic']:
                if not len(x):rows.append(dict(common,type='continuous',status='EMPTY_GROUP'));continue
                if weighted:
                    v=x.to_numpy(float);mu=float(np.average(v,weights=w));sd=float(np.sqrt(np.average((v-mu)**2,weights=w)));order=np.argsort(v);cs=np.cumsum(w[order]);qs=[v[order][min(np.searchsorted(cs,p*cs[-1],side='left'),len(v)-1)] for p in [.25,.5,.75]]
                else:mu=float(x.mean());sd=float(x.std());qs=x.quantile([.25,.5,.75]).to_numpy()
                rows.append(dict(common,type='continuous',level='',mean=mu,sd=sd,q1=qs[0],median=qs[1],q3=qs[2]))
            else:
                for lev in sorted(d[var].dropna().unique(),key=str):
                    yes=x.eq(lev);n=int(yes.sum());pc=100*float(w[yes].sum()/w.sum()) if len(w) and w.sum()>0 else np.nan
                    label=str(lev)
                    if var=='country':label=str(d.loc[d.country.eq(lev),'country_name'].iloc[0])
                    rows.append(dict(common,type='categorical',level=label,category_n=n,percent=pc))
    return pd.DataFrame(rows)

def prevalences(h,d,c):
    """Inherited PSU/stratum linearization. No project <10 numerical screen."""
    out=[]
    for definition in DEFS:
        g=d[d.weight.notna()];y=g['low_'+definition].to_numpy(float);w=g.weight.to_numpy();n=len(g)
        r=dict(cohort=c,definition=definition,n=n,positive_n=int(y.sum()),weight=f.WEIGHTS[c],scope='eligible positive-weight age>=65; not age-standardised',post_hoc=TAG)
        if not n:r['status']='EMPTY_DOMAIN';out.append(r);continue
        p=float(w@y/w.sum());r.update(flagged_per100=100*p,n_eff=float(w.sum()**2/(w@w)))
        if p<=0 or p>=1:r.update(status='BOUNDARY_PROPORTION_CI_UNAVAILABLE');out.append(r);continue
        u=pd.Series(0.,index=h.index);u.loc[g.index]=w*(y-p)/w.sum()
        frame=h.loc[h.weight.notna(),['psu','stratum']].copy();frame['u']=u.loc[frame.index]
        scores=frame.groupby(['stratum','psu'],dropna=False).u.sum();var=0.;df=0;single=[]
        for _,q in scores.groupby(level=0):
            m=len(q)
            if m>1:var+=m/(m-1)*float(((q-q.mean())**2).sum());df+=m-1
            else:single.extend(q.tolist())
        if single:var+=float(np.square(np.array(single)-scores.mean()).sum())
        se=np.sqrt(var);ci=expit(logit(p)+np.array([-1,1])*t.ppf(.975,max(df,1))*se/(p*(1-p)))
        r.update(ci_low_per100=100*float(ci[0]),ci_high_per100=100*float(ci[1]),design_df=df,singleton_strata=len(single),status='OK');out.append(r)
    return pd.DataFrame(out)

def baseline(c):
    h,d=rt.load(c);w=f.BASE[c];age=f.num(h,f'r{w}agey').combine_first(f.num(h,f'r{w}agey_e'));inter=f.num(h,f'r{w}iwstat').eq(1)
    stages=[('source_harmonized_records',len(h)),('baseline_interviewed',int(inter.sum())),('baseline_age65',int((inter&age.ge(65)).sum())),('sex_BMI_grip_available_after_QC',len(f.derive(h,c,f.SPEC[c][1],w,f.SPEC[c][3]))),('confirmed_eligible_with_grip',len(d)),('description_positive_weight',int(d.weight.notna().sum())),('baseline_ADL0',int(f.adl(h,c,w).reindex(d.index).eq(0).sum()))]
    table=pd.DataFrame();counts=[]
    if c in ['CHARLS','MHAS','SHARE']:
        intervals,_=f.intervals(h,d,c,'composite');z=intervals.dropna(subset=f.COV+['low_EWGSOP2','event']);g=d[d.id.isin(set(z.id))]
        stages +=[('any_observable_interval',intervals.id.nunique()),('primary_complete_case',len(g))]
        table=describe(g,c,[('Normal',g[~g.low_EWGSOP2]),('Low',g[g.low_EWGSOP2]),('All',g)])
        for group,q in [('Normal',z[~z.low_EWGSOP2]),('Low',z[z.low_EWGSOP2]),('All',z)]:counts.append(dict(cohort=c,group=group,n=q.id.nunique(),events_n=int(q.event.sum())))
    screen,_,_=screen_fi(h,d,c)
    qcrules=[]
    if c in qc.MEASURED:
        q,base,bv,flags,bad,conflict=qc.measurement(c)
        for rule,flag in flags.items():qcrules.append(dict(cohort=c,rule=rule,age65_n=int((base&flag).sum()),previously_positive_BMI_n=int((base&bv&flag).sum()),overlap='nonexclusive',version='QCv2_only'))
    else:qcrules.append(dict(cohort=c,rule='measured_anthropometry_QC',status='NOT_APPLIED_SELF_REPORTED_BMI'))
    return {'FI_item_screen':screen,'flow':pd.DataFrame([dict(cohort=c,stage=s,n=int(n),version='QCv2_only') for s,n in stages]),'table1':table,'tableS1':describe(d,c,[('All',d[d.weight.notna()])],True),'fig3_counts':pd.DataFrame(counts),'definition_prevalences':prevalences(h,d,c),'QC_rules':pd.DataFrame(qcrules)}

def fi_versions(S,screen,common):
    result={}
    for version,mask in [('main',screen.selected_main.to_numpy(bool)),('noMob',screen.selected_noMob.to_numpy(bool)),('common',screen.selected_main.to_numpy(bool)&screen.common_equiv.to_numpy(bool)&screen.concept.isin(common).to_numpy())]:
        q=S[:,mask];k=q.shape[1]
        if not k or (version!='common' and k<30):result[version]=(None,k,'INSUFFICIENT_ITEMS_LT30' if version!='common' else 'NO_COMMON_ITEMS');continue
        observed=np.isfinite(q).sum(1);v=np.divide(np.nansum(q,axis=1),observed,out=np.full(len(q),np.nan),where=observed>0);v[observed<np.ceil(.8*k)]=np.nan
        result[version]=(v,k,'OK' if k>=30 else 'FI_COMMON_EXPLICIT_LT30_EXCEPTION')
    return result

def fi(c,common,expected_screen):
    h,d=rt.load(c);screen,S,_=screen_fi(h,d,c)
    cols=['concept','selected_main','selected_noMob','common_equiv']
    require(screen[cols].reset_index(drop=True).equals(expected_screen[cols].reset_index(drop=True)),'FI_SCREEN_CHANGED_ON_RESUME')
    versions=fi_versions(S,screen,common);status=[];validity=[];models=[];sexrows=[]
    for version,(value,k,state) in versions.items():
        if value is None:
            status.append(dict(cohort=c,FI_version=version,item_n=k,status=state))
            validity.append(dict(cohort=c,FI_version=version,item_n=k,status=state,reason='FI_NOT_COMPUTED'))
            sexrows.append(dict(cohort=c,FI_version=version,item_n=k,status=state,reason='FI_NOT_COMPUTED'))
            for kind in (['not_applicable'] if c=='ELSA' else ['death'] if c=='KLoSA' else ['composite','death']):
                models.append(unavailable(c,'grip_adjusted_FI',state,FI_version=version,outcome=kind))
            continue
        valid=np.isfinite(value);cuts=np.quantile(value[valid],[1/3,2/3]) if valid.any() else [np.nan,np.nan]
        status.append(dict(cohort=c,FI_version=version,item_n=k,n=len(d),valid_FI_n=int(valid.sum()),missing_FI_n=int((~valid).sum()),tertile_cut1=cuts[0],tertile_cut2=cuts[1],status=state))
        groups=[('All',np.ones(len(d),bool)),('Men',d.sex.eq(1).to_numpy()),('Women',d.sex.eq(2).to_numpy())]
        groups +=[(f'{int(a)}-{int(a+4)}',d.age.ge(a).to_numpy()&d.age.lt(a+5).to_numpy()) for a in sorted((np.floor(d.age/5)*5).unique())]
        for group,mask in groups:
            x=value[mask];xx=x[np.isfinite(x)]
            validity.append(dict(cohort=c,FI_version=version,group=group,n=len(x),valid_n=len(xx),missing_n=len(x)-len(xx),mean=float(np.mean(xx)) if len(xx) else np.nan,median=float(np.median(xx)) if len(xx) else np.nan,P99=float(np.quantile(xx,.99)) if len(xx) else np.nan,max='SUPPRESSED_PROVIDER_UNIT_RECORD_RULE' if c=='ELSA' else float(np.max(xx)) if len(xx) else np.nan,status=state))
        women=value[d.sex.eq(2).to_numpy()&valid];men=value[d.sex.eq(1).to_numpy()&valid]
        sexrows.append(dict(cohort=c,FI_version=version,women_n=len(women),men_n=len(men),women_mean=float(women.mean()) if len(women) else np.nan,men_mean=float(men.mean()) if len(men) else np.nan,difference_women_minus_men=float(women.mean()-men.mean()) if len(women) and len(men) else np.nan,status='DESCRIPTIVE'))
        if c=='ELSA':models.append(unavailable(c,'FI_prognosis','POST_W6_MORTALITY_UNAVAILABLE',FI_version=version));continue
        dd=d.copy();dd['FI']=value/.1;dd['FI01']=value/.1
        for kind in (['death'] if c=='KLoSA' else ['composite','death']):
            panel,_=f.intervals(h,dd,c,kind);z=panel.dropna(subset=f.COV+['event','FI','low_EWGSOP2'])
            r,_=estimate(z,c,'low_EWGSOP2','grip_adjusted_FI',add_fi=True);models.append(dict(r,FI_version=version,outcome=kind))
            if np.isfinite(cuts).all() and cuts[0]<cuts[1]:
                which=pd.cut(z.FI*.1,[-np.inf,cuts[0],cuts[1],np.inf],labels=[1,2,3],right=True)
                for tertile in [1,2,3]:
                    r,_=estimate(z.loc[which.eq(tertile)],c,'low_EWGSOP2',f'grip_FI_tertile{tertile}');models.append(dict(r,FI_version=version,outcome=kind))
            else:models.append(unavailable(c,'FI_tertiles','TIED_OR_MISSING_CUTPOINTS',FI_version=version,outcome=kind))
            if kind=='death':
                r,_=estimate(z,c,'FI01','FI_per0.1_mortality',continuous_exposure=True);models.append(dict(r,FI_version=version,outcome='death'))
    return {'FI_versions_status':pd.DataFrame(status),'FI_validity':pd.DataFrame(validity),'FI_sex_difference':pd.DataFrame(sexrows),'FI_models':pd.DataFrame(models)}

def pool(rows,label):
    z=rows[rows.status.eq('OK')].copy() if len(rows) and 'status' in rows else pd.DataFrame();k=len(z)
    r=dict(cohort='POOLED',analysis=label,k=k,cohorts=';'.join(z.cohort) if k else '',status='OK' if k>=2 else 'NOT_ESTIMABLE_TOO_FEW_COHORTS',prediction_interval='NOT_REPORTED',post_hoc=TAG)
    if k<2:return r
    require(z.cohort.is_unique,'DUPLICATE_COHORT_EFFECT_IN_POOL')
    y=z.log_effect.to_numpy(float);v=z.se.to_numpy(float)**2
    def objective(tau):
        w=1/(v+tau);mu=np.sum(w*y)/w.sum();return .5*(np.log(v+tau).sum()+np.log(w.sum())+np.sum(w*(y-mu)**2))
    opt=minimize_scalar(objective,bounds=(0,max(1.,float(np.var(y)*10))),method='bounded');tau=float(opt.x) if objective(opt.x)<objective(0) else 0.
    w=1/(v+tau);mu=float(np.sum(w*y)/w.sum());scale=max(1.,float(np.sum(w*(y-mu)**2)/(k-1)));se=np.sqrt(scale/w.sum());crit=t.ppf(.975,k-1)
    wf=1/v;mf=np.sum(wf*y)/wf.sum();Q=float(np.sum(wf*(y-mf)**2));I2=max(0,(Q-k+1)/Q)*100 if Q>0 else 0.
    return dict(r,log_effect=mu,se=float(se),effect=float(np.exp(mu)),ci_low=float(np.exp(mu-crit*se)),ci_high=float(np.exp(mu+crit*se)),tau2=tau,I2=I2,Q=Q,Q_p=float(chi2.sf(Q,k-1)),method='REML_modified_Hartung_Knapp',cohorts=';'.join(z.cohort))
