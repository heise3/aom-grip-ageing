"""Frozen A v1, non-HRS allowlist; in-memory participant processing, aggregate outputs only."""
from pathlib import Path
import sys,json,re,hashlib,datetime,warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit,logit
from scipy.stats import t,chi2
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/A_formal_v1_20261027'
sys.path.insert(0,str(ROOT/'work/a_validation'))
from count_a import SPEC,derive,eligibility,num,bn,suppress
from harmonized_adapter import ADLMAP,CHRONMAP,HELPMAP
BASE={'CHARLS':1,'ELSA':6,'MHAS':3,'SHARE':4,'KLoSA':4}
MAXW={'CHARLS':4,'ELSA':9,'MHAS':5,'SHARE':8,'KLoSA':8}
WEIGHTS={'CHARLS':'r1wtrespbiob','ELSA':'r6nwtresp','MHAS':'factora_12','SHARE':'r4wtresp','KLoSA':'r4wtresp'}
YEARS={'CHARLS':{1:2011,2:2013,3:2015,4:2018},'ELSA':{6:2012,7:2014,8:2016,9:2018},'MHAS':{3:2012,4:2015,5:2018},'SHARE':{4:2011,5:2013,6:2015,7:2017,8:2019},'KLoSA':{4:2012,5:2014,6:2016,7:2018,8:2020}}
SCHEMA=json.loads((ROOT/'work/nonHRS_selected_schema.json').read_text())
INV=json.loads(Path('${LOCAL_HOME}/Documents/Codex/2026-10-26/k0-k3-hrs-rand-gateway-hrs/outputs/K1_completion_v2/nonHRS_file_inventory_metadata_only.json').read_text())
MANIFEST=[]
BRIDGE_AUDIT=[]
def read(p,cols):
 p=Path(p);assert not re.search('hrs|rand',str(p),re.I)
 MANIFEST.append({'path':str(p),'columns':cols,'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
 return pd.read_stata(p,columns=cols,convert_categoricals=False)
def cleanstr(s):
 s=s.astype('string');return s.mask(s.str.strip().isin(['','nan','<NA>','.']))
def extend_followup(h,c):
 """Raw-wave mappings verified against producer labels before extension fitting."""
 schema=json.loads((ROOT/'work/a_formal/new_wave_schema.json').read_text())
 if c=='CHARLS':
  health=schema['/2020charls/Health_Status_and_Functioning.dta']['path']
  status=schema['/2020charls/Sample_Infor.dta']['path']
  a=read(health,['ID','db001','db003','db005','db007','db009']).set_index('ID')
  z=read(status,['ID','died','iyear','imonth']).set_index('ID')
  assert a.index.is_unique and z.index.is_unique
  aa=a.reindex(h.ID);zz=z.reindex(h.ID);aa.index=h.index;zz.index=h.index
  h['r5iwstat']=np.where(zz.died.eq(1),5,np.where(zz.died.eq(0)&aa.notna().any(axis=1),1,np.where(zz.died.eq(0),4,np.nan)))
  for raw,suff in [('db001','dressa'),('db003','batha'),('db005','eata'),('db007','beda'),('db009','toilta')]:h['r5'+suff]=aa[raw].map({1:0,2:1,3:1,4:1})
  h['r5iwy']=zz.iyear;h['r5iwm']=zz.imonth;MAXW[c]=5;YEARS[c][5]=2020
  BRIDGE_AUDIT.append({'cohort':c,'wave':5,'status':'MAPPED','join':'ID one-to-one','ADL_codes':'1=0;2/3/4=1;others unknown','matched_health_n':suppress(int(aa.notna().any(axis=1).sum())),'matched_status_n':suppress(int(zz.died.notna().sum()))})
 elif c=='MHAS':
  p=schema['/Core survey Data/sect_a_c_d_e_pc_f_h_i_2021.dta']['path'];q=schema['/Master Follow-up File/master_follow_up_file_2021.dta']['path']
  names=['h13_21','h16a_21','h17a_21','h18a_21','h19a_21']
  a=read(p,['unhhid','np']+names).set_index(['unhhid','np']);z=read(q,['unhhid','np','fallecido_21','tipent_21']).set_index(['unhhid','np'])
  assert a.index.is_unique and z.index.is_unique
  ix=pd.MultiIndex.from_frame(h[['unhhid','np']]);aa=a.reindex(ix);zz=z.reindex(ix);aa.index=h.index;zz.index=h.index
  h['r6iwstat']=np.where(zz.fallecido_21.isin([1,2])|zz.tipent_21.eq(5),5,np.where(zz.tipent_21.isin([1,2,3,4]),1,np.where(zz.fallecido_21.eq(0),4,np.nan)))
  for raw,suff in zip(names,['dressa','batha','eata','beda','toilta']):h['r6'+suff]=aa[raw].map({1:1,2:0,6:1})
  MAXW[c]=6;YEARS[c][6]=2021
  BRIDGE_AUDIT.append({'cohort':c,'wave':6,'status':'MAPPED_DIRECT_ADL_PROXY_UNKNOWN','join':'unhhid+np one-to-one','ADL_codes':'1/6=1;2=0;7/8/9 unknown','matched_health_n':suppress(int(aa.notna().any(axis=1).sum())),'matched_status_n':suppress(int(zz.fallecido_21.notna().sum()))})
 elif c=='ELSA':BRIDGE_AUDIT.append({'cohort':c,'wave':10,'status':'NOT_ADMISSIBLE_COMPOSITE_MORTALITY_UNAVAILABLE','reason':'G3 W6-W9 mortality unavailable; local W10 core is not a complete mortality tracker'})
 elif c=='SHARE':
  p=next(x['path'] for x in INV if x['path'].endswith('/sharew9_rel9-0-0_ph.dta'));q=next(x['path'] for x in INV if x['path'].endswith('/sharew9_rel9-0-0_cv_r.dta'))
  names=['ph049d1','ph049d3','ph049d4','ph049d5','ph049d6']
  a=read(p,['mergeid']+names).set_index('mergeid');z=read(q,['mergeid','deceased','interview','int_year','int_month']).set_index('mergeid')
  assert a.index.is_unique and z.index.is_unique
  aa=a.reindex(h.mergeid);zz=z.reindex(h.mergeid);aa.index=h.index;zz.index=h.index
  h['r9iwstat']=np.where(zz.deceased.eq(1)|zz.interview.eq(2),5,np.where(zz.interview.eq(1),1,np.where(zz.deceased.eq(0),4,np.nan)))
  for raw,suff in zip(names,['dressa','batha','eata','beda','toilta']):h['r9'+suff]=aa[raw].where(aa[raw].isin([0,1]))
  h['r9iwy']=zz.int_year;h['r9iwm']=zz.int_month;MAXW[c]=9;YEARS[c][9]=2022
  BRIDGE_AUDIT.append({'cohort':c,'wave':9,'status':'MAPPED','join':'mergeid one-to-one','ADL_codes':'0=no;1=yes;negative unknown','matched_health_n':suppress(int(aa.notna().any(axis=1).sum())),'matched_status_n':suppress(int(zz.deceased.notna().sum()))})
 elif c=='KLoSA':BRIDGE_AUDIT.append({'cohort':c,'wave':9,'status':'NOT_ADMISSIBLE_MORTALITY_TRACKER_MISSING','reason':'local W9 core/Lt files contain participants; respondent death/exit status chain not established'})
 return h.copy()
def load(c):
 name,idcol,_,bmi=SPEC[c];schema=SCHEMA[name];w=BASE[c]
 suffix=list(ADLMAP.values())+list(CHRONMAP.values())+list(HELPMAP.values())+['bathb','dressb','eatb','bedb_k','toiltb','respe','hrtatte','diabe','smoken','agey','agey_e','iwstat','iwy','iwm','iwindy','iwindm','bmi','mbmi','hibpe','hchole','dyslipe']+[f'{hand}grip{i}' for hand in ['l','r'] for i in range(1,4)]
 cols=[idcol,'ragender','raeducl','country','communityID','householdID','unhhid','np','hhid'+str(w),'rapsu1','raestrat1','raeclust','raestrat','radyear','radmonth',WEIGHTS[c]]+[f'r{k}{s}' for k in range(w,MAXW[c]+1) for s in suffix]
 cols=list(dict.fromkeys(k for k in cols if k in schema['labels']))
 h=read(schema['path'],cols);assert h[idcol].is_unique
 # Design universe is the complete source sample, not just the eligible domain.
 h['id']=h[idcol].astype(str)
 if c=='MHAS':
  p=next(x['path'] for x in INV if x['path'].endswith('/sect_l_biomarkers_2012.dta'))
  a=read(p,['cunicah','np','factora_12']).rename(columns={'cunicah':'unhhid'})
  h=h.merge(a,on=['unhhid','np'],how='left',validate='one_to_one')
 if c=='ELSA':
  p=next(x['path'] for x in INV if x['path'].endswith('/Raw_data/wave6/wave_6_elsa_data_v2.dta'))
  with pd.io.stata.StataReader(p) as rr:lookup={k.lower():k for k in rr.variable_labels()}
  a=read(p,[lookup[k] for k in ['idauniq','gor','idahhw6']]);a.columns=['idauniq','raw_stratum','raw_psu']
  h=h.merge(a,on='idauniq',how='left',validate='one_to_one')
 if c=='KLoSA':
  # The harmonized RAECLUST has only residential-type codes; bind actual enumeration district.
  p=Path(schema['path']).parent.parent/'wave4/w04_e.dta'
  a=read(p,['pid','ed']).rename(columns={'ed':'raw_enumeration_district'})
  h=h.merge(a,on='pid',how='left',validate='one_to_one')
 h['weight']=num(h,WEIGHTS[c]).where(lambda x:x>0)
 country=num(h,'country').astype(str) if c=='SHARE' else pd.Series(c,index=h.index)
 if c=='CHARLS':ps=h.communityID;st=pd.Series('ALL',index=h.index);hh=h.householdID;note='community_cluster_approximation'
 elif c=='ELSA':ps=h.raw_psu;st=h.raw_stratum;hh=ps;note='region_household_cluster'
 elif c=='SHARE':ps=h.rapsu1;st=h.raestrat1;hh=h['hhid4'];note='PSU_stratum_or_household_fallback'
 elif c=='KLoSA':ps=h.raw_enumeration_district;st=h.raestrat;hh=ps;note='raw_enumeration_district_harmonized_stratum_binding_correction'
 else:ps=h.unhhid;st=pd.Series('ALL',index=h.index);hh=ps;note='household_cluster_approximation_four_states'
 ps=cleanstr(ps);st=cleanstr(st);fallback=ps.isna();ps=ps.fillna(cleanstr(hh)).fillna(h.id);st=st.fillna('UNKNOWN')
 h['psu']=country+'_'+ps;h['stratum']=country+'_'+st
 h['design_note']=note
 h=extend_followup(h,c)
 d=eligibility(derive(h,c,idcol,w,bmi));d['id']=d.id.astype(str)
 d['weight']=h.loc[d.index,'weight'];d['agegroup']=pd.cut(d.age,[65,75,85,np.inf],right=False,labels=['65-74','75-84','85+']).astype(str)
 d['educ']=num(h,'raeducl').where(lambda x:x.isin([1,2,3]));d['smoke']=bn(h,f'r{w}smoken');d['diabetes']=bn(h,f'r{w}diabe')
 terms=[('respe' if k=='lung' else 'hrtatte' if k=='heart' else s) if c=='MHAS' else s for k,s in CHRONMAP.items()]
 d['chronic']=pd.concat([bn(h,f'r{w}{s}') for s in terms],axis=1).sum(axis=1,min_count=6)
 if c=='SHARE':
  with pd.io.stata.StataReader(schema['path']) as rr:cmap=rr.value_labels()['country']
  d['country_name']=d.country.map(cmap)
 else:d['country_name']={'CHARLS':'China','ELSA':'England','MHAS':'Mexico (4-state subsample)','KLoSA':'South Korea'}[c]
 d['grip5']=d.grip/5
 return h,d[d.eligible.eq('yes')].copy(),int(fallback.sum())
def adl(h,c,w,helped=False):
 if c=='KLoSA':terms=['bathb','dressb','eatb','bedb_k','toiltb']
 else:terms=[(HELPMAP if helped else ADLMAP)[k] for k in ['bath','dress','eat','bed','toilet']]
 a=pd.concat([bn(h,f'r{w}{s}') for s in terms],axis=1)
 return pd.Series(np.where(a.eq(1).any(axis=1),1,np.where(a.eq(0).all(axis=1),0,np.nan)),index=h.index)
def domain_est(h,d,mask,y):
 valid=mask&d.weight.notna();g=d.loc[valid];yy=y.loc[g.index].astype(float)
 n=len(g);pos=int(yy.eq(1).sum());neg=n-pos
 result={'n':suppress(n),'positive_n':suppress(pos),'status':'SMALL_CELL' if min(pos,neg)<10 else 'OK','weight_variable':WEIGHTS[g.iloc[0].cohort] if 'cohort' in g and n else ''}
 if not n or min(pos,neg)<10:return result
 weights=g.weight.to_numpy();p=float(np.dot(weights,yy)/weights.sum());result.update(proportion=p,n_eff=float(weights.sum()**2/np.square(weights).sum()))
 u=pd.Series(0.,index=h.index);u.loc[g.index]=weights*(yy-p)/weights.sum()
 frame=h.loc[h.weight.notna(),['psu','stratum']].copy();frame['u']=u.loc[frame.index]
 scores=frame.groupby(['stratum','psu'],dropna=False).u.sum();var=0.;df=0;single=[]
 for st,z in scores.groupby(level=0):
  m=len(z)
  if m>1:var+=m/(m-1)*float(((z-z.mean())**2).sum());df+=m-1
  else:single.extend(z.tolist())
 if single:var+=float(np.square(np.array(single)-scores.mean()).sum())
 se=np.sqrt(var);crit=t.ppf(.975,max(df,1));ci=expit(logit(p)+np.array([-1,1])*crit*se/(p*(1-p)))
 result.update(ci_low=float(ci[0]),ci_high=float(ci[1]),se=float(se),design_df=int(df),singleton_strata=len(single),design_note=h.design_note.iloc[0])
 return result
def describe(h,d,c):
 rows=[];ad=adl(h,c,BASE[c]).reindex(d.index);vul=d.low_EWGSOP2|ad.eq(1)
 d=d.copy();d['cohort']=c
 for country in sorted(d.country_name.dropna().unique()):
  masks=[('All','All',d.country_name.eq(country))]
  masks +=[(a,'All',d.country_name.eq(country)&d.agegroup.eq(a)) for a in ['65-74','75-84','85+']]
  masks +=[('All',s,d.country_name.eq(country)&d.sex.eq(v)) for s,v in [('Men',1),('Women',2)]]
  masks +=[(a,s,d.country_name.eq(country)&d.agegroup.eq(a)&d.sex.eq(v)) for a in ['65-74','75-84','85+'] for s,v in [('Men',1),('Women',2)]]
  for age,sex,mask in masks:
   for cut in ['EWGSOP2','SDOC']:
    r={'cohort':c,'country':country,'baseline_wave':BASE[c],'agegroup':age,'sex':sex,'measure':cut};r.update(domain_est(h,d,mask,d['low_'+cut]));rows.append(r)
   # Frailty branch unknown: confirmed positives from low grip or ADL; all others remain possible.
   r={'cohort':c,'country':country,'baseline_wave':BASE[c],'agegroup':age,'sex':sex,'measure':'any_vulnerability_bounds','comparison_scope':'SEPARATE_HELP_ADL' if c=='KLoSA' else 'DIFFICULTY_ADL'}
   est=domain_est(h,d,mask,vul);r.update(est)
   r['lower_bound']=r.pop('proportion',None);r['upper_bound']=1. if 'lower_bound' in r and r['lower_bound'] is not None else None
   r['identification_status']='FRAILTY_FIVE_COMPONENTS_NOT_FULLY_BOUND';rows.append(r)
 return rows
def intervals(h,d,c,kind='composite',helped=False):
 w=BASE[c];baseadl=adl(h,c,w,helped).reindex(d.index)
 active=pd.Series(True,index=d.index) if kind=='death' else baseadl.eq(0)
 rows=[];audit=[]
 for v in range(w+1,MAXW[c]+1):
  status=num(h,f'r{v}iwstat').reindex(d.index)
  future=pd.concat([num(h,f'r{k}iwstat').isin([1,4]) for k in range(v+1,MAXW[c]+1)],axis=1).any(axis=1).reindex(d.index) if v<MAXW[c] else pd.Series(False,index=d.index)
  bad=status.eq(5)&future
  dead=status.eq(5)&~bad;alive=status.isin([1,4]);a=adl(h,c,v,helped).reindex(d.index)
  valid=(dead|alive) if kind=='death' else (dead|(status.eq(1)&a.notna()))
  keep=active&valid&~bad
  g=d.loc[keep].copy();g['interval']=v-w;g['event']=dead.loc[keep].astype(int)
  if kind!='death':g['event']=(dead.loc[keep]|a.loc[keep].eq(1)).astype(int)
  g['state']=np.where(dead.loc[keep],2,np.where(a.loc[keep].eq(1),1,0));rows.append(g)
  audit.append({'cohort':c,'outcome':kind,'followup_wave':v,'available_n':suppress(int(keep.sum())),'censored_unknown_n':suppress(int((active&~valid).sum())),'death_alive_conflict_n':suppress(int((active&bad).sum()))})
  active=keep&~dead
  if kind!='death':active=active&a.eq(0)
 return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(),audit
COV=['age','sex','educ','bmi','smoke','diabetes','chronic','country','interval']
def fitmodel(dd,c,outcome,analysis='primary',exposure='low_EWGSOP2',multistate=False):
 r={'cohort':c,'outcome':outcome,'analysis':analysis,'exposure':exposure,'baseline_wave':BASE[c],'last_wave':MAXW[c]}
 if dd.empty:r['status']='NO_OBSERVABLE_INTERVALS';return r
 z=dd.dropna(subset=COV+[exposure,'event']).copy();z['x']=z[exposure].astype(float)
 r.update(persons_n=suppress(z.id.nunique()),intervals_n=suppress(len(z)),events_n=suppress(int(z.event.sum())))
 # Cell suppression gate uses EWGSOP2/SDOC groups even for continuous grip.
 cut=exposure if exposure.startswith('low_') else 'low_EWGSOP2'
 for val in [0,1]:r[f'events_group{val}_n']=suppress(int(z.loc[z[cut].eq(val),'event'].sum()))
 if z.empty or z.x.nunique()<2 or any(int(z.loc[z[cut].eq(v),'event'].sum())<10 for v in [0,1]):r['status']='NOT_ESTIMABLE_SMALL_EVENT_CELL';return r
 formula='event ~ x + age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'+(' + C(country)' if c=='SHARE' else '')
 try:
  with warnings.catch_warnings(record=True) as ws:
   fit=smf.glm(formula,data=z,family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster',cov_kwds={'groups':z.id},maxiter=150)
  beta=float(fit.params['x']);se=float(fit.bse['x'])
  r.update(status='OK' if fit.converged and np.isfinite(se) and se>0 else 'NOT_CONVERGED',log_effect=beta,se=se,effect=float(np.exp(beta)),ci_low=float(np.exp(beta-1.96*se)),ci_high=float(np.exp(beta+1.96*se)),p=float(fit.pvalues['x']),effect_type='grouped_time_PH',warnings=';'.join(str(w.message) for w in ws))
 except Exception as e:r.update(status='MODEL_ERROR',error=str(e))
 return r
def multistate(dd,c):
 r={'cohort':c,'outcome':'ADL_competing_death','analysis':'multistate','status':'NOT_RUN'}
 z=dd.dropna(subset=COV+['state']).copy();z['x']=z.low_EWGSOP2.astype(int)
 if any((z.state.eq(s)&z.x.eq(v)).sum()<10 for s in [1,2] for v in [0,1]):r['status']='NOT_ESTIMABLE_SMALL_TRANSITION_CELL';return [r]
 try:
  import patsy
  form='x + age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'+(' + C(country)' if c=='SHARE' else '')
  X=patsy.dmatrix(form,z,return_type='dataframe')
  # Center/scale nuisance continuous covariates without changing their model span.
  for k in ['age','bmi','chronic']:
   if k in X and X[k].std()>0:X[k]=(X[k]-X[k].mean())/X[k].std()
  # Preserve transform for standardized prediction below.
  transform={k:(float(z[k].mean()),float(z[k].std())) for k in ['age','bmi','chronic'] if z[k].std()>0}
  with warnings.catch_warnings(record=True) as ws:fit=sm.MNLogit(z.state,X).fit(method='bfgs',disp=False,maxiter=600,cov_type='cluster',cov_kwds={'groups':z.id})
  if not fit.mle_retvals['converged']:r['status']='NOT_CONVERGED';return [r]
  out=[]
  for j,label in [(0,'incident_ADL'),(1,'death_before_observed_ADL')]:
   b=float(fit.params.loc['x',j]);se=float(fit.bse.loc['x',j]);out.append(dict(r,status='OK',transition=label,effect_type='multinomial_transition_OR',effect=float(np.exp(b)),ci_low=float(np.exp(b-1.96*se)),ci_high=float(np.exp(b+1.96*se))))
  # Marginal standardized CIF, predict each baseline person at all available intervals.
  persons=z.drop_duplicates('id').copy()
  for val in [0,1]:
   survival=np.ones(len(persons));ci_adl=np.zeros(len(persons));ci_death=np.zeros(len(persons))
   for interval in sorted(z.interval.unique()):
    temp=persons.copy();temp['x']=val;temp['interval']=interval
    mat=pd.DataFrame(patsy.build_design_matrices([X.design_info],temp)[0],columns=X.columns)
    for k,(mean,sd) in transform.items():mat[k]=(mat[k]-mean)/sd
    prob=np.asarray(fit.predict(mat))
    ci_adl+=survival*prob[:,1];ci_death+=survival*prob[:,2];survival*=prob[:,0]
   out.append(dict(r,status='OK',transition='standardized_CIF_last_wave',low_grip=val,ADL_CIF=float(ci_adl.mean()),death_CIF=float(ci_death.mean()),estimand='model_based_complete_case_baseline_distribution',uncertainty='point_estimate_only'))
  return out
 except Exception as e:r.update(status='MODEL_ERROR',error=str(e));return [r]
def early_exclusion(h,d,c):
 w=BASE[c];by=num(h,f'r{w}'+('iwindy' if c=='ELSA' else 'iwy'));bm=num(h,f'r{w}'+('iwindm' if c=='ELSA' else 'iwm'))
 dy=num(h,'radyear');dm=num(h,'radmonth')
 bmin=by+(bm.fillna(1)-1)/12;bmax=by+(bm.fillna(12))/12
 dmin=dy+(dm.fillna(1)-1)/12;dmax=dy+dm.fillna(12)/12
 definitely_after=dmin-bmax>=1
 for k in range(w+1,MAXW[c]+1):
  ky=num(h,f'r{k}'+('iwindy' if c=='ELSA' else 'iwy'));km=num(h,f'r{k}'+('iwindm' if c=='ELSA' else 'iwm'))
  definitely_after=definitely_after|(num(h,f'r{k}iwstat').isin([1,4])&(ky+(km.fillna(1)-1)/12-bmax>=1))
 # First observed death with unclassified date cannot establish survival beyond 1y.
 died=pd.concat([num(h,f'r{k}iwstat').eq(5) for k in range(w+1,MAXW[c]+1)],axis=1).any(axis=1)
 excluded=died&~definitely_after
 return set(h.loc[excluded,'id'])
def meta(rows,label):
 z=pd.DataFrame(rows);z=z[z.status.eq('OK')].copy() if 'status' in z else pd.DataFrame();k=len(z)
 r={'analysis':label,'k':k,'status':'OK' if k>=2 else 'NOT_ESTIMABLE_TOO_FEW_COHORTS'}
 if k<2:return r
 y=z.log_effect.to_numpy();v=z.se.to_numpy()**2
 def objective(tau):
  w=1/(v+tau);mu=np.sum(w*y)/w.sum();return .5*(np.log(v+tau).sum()+np.log(w.sum())+np.sum(w*(y-mu)**2))
 opt=minimize_scalar(objective,bounds=(0,max(1.,float(np.var(y)*10))),method='bounded');tau=float(opt.x) if objective(opt.x)<objective(0) else 0.
 w=1/(v+tau);mu=np.sum(w*y)/w.sum();scale=max(1.,float(np.sum(w*(y-mu)**2)/(k-1)));var=scale/w.sum();se=np.sqrt(var);crit=t.ppf(.975,k-1)
 wf=1/v;mf=np.sum(wf*y)/wf.sum();Q=float(np.sum(wf*(y-mf)**2));I2=max(0,(Q-k+1)/Q)*100 if Q>0 else 0.
 r.update(log_effect=float(mu),se=float(se),effect=float(np.exp(mu)),ci_low=float(np.exp(mu-crit*se)),ci_high=float(np.exp(mu+crit*se)),tau2=tau,I2=I2,Q=Q,Q_p=float(chi2.sf(Q,k-1)),method='REML_modified_Hartung_Knapp',cohorts=';'.join(z.cohort))
 if k>=3:
  width=t.ppf(.975,k-2)*np.sqrt(tau+var);r.update(pi_low=float(np.exp(mu-width)),pi_high=float(np.exp(mu+width)))
 return r
def main():
 lock=json.loads((OUT/'freeze_receipt.json').read_text());assert hashlib.sha256((OUT/'A_analysis_plan_v1.md').read_bytes()).hexdigest()==lock['sha256']
 props=[];models=[];trans=[];audit=[];flows=[]
 for c in BASE:
  h,d,fallback=load(c);props+=describe(h,d,c)
  flows.append({'cohort':c,'baseline_wave':BASE[c],'eligible_grip_n':suppress(len(d)),'positive_weight_n':suppress(int(d.weight.notna().sum())),'baseline_ADL_zero_n':suppress(int(adl(h,c,BASE[c]).reindex(d.index).eq(0).sum())),'source_psu_fallback_n':suppress(fallback)})
  excluded=early_exclusion(h,d,c)
  for kind in ['composite','death']:
   if c=='ELSA':
    for analysis in ['primary','SDOC','continuous_5kg','exclude_first_year_death']:models.append({'cohort':c,'outcome':kind,'analysis':analysis,'status':'NOT_ESTIMABLE_MORTALITY_UNAVAILABLE'})
    if kind=='composite':trans.append({'cohort':c,'outcome':'ADL_competing_death','analysis':'multistate','status':'NOT_ESTIMABLE_MORTALITY_UNAVAILABLE'})
    continue
   if c=='KLoSA' and kind=='composite':models.append({'cohort':c,'outcome':kind,'analysis':'primary','status':'EXCLUDED_ADL_CONSTRUCT'});continue
   dd,aa=intervals(h,d,c,kind);audit+=aa
   for analysis,exposure,subset in [('primary','low_EWGSOP2',dd),('SDOC','low_SDOC',dd),('continuous_5kg','grip5',dd),('exclude_first_year_death','low_EWGSOP2',dd[~dd.id.isin(excluded)])]:models.append(fitmodel(subset,c,kind,analysis,exposure))
   if kind=='composite':trans+=multistate(dd,c)
  print(c,'formal aggregates complete',flush=True)
 for name,data in [('A_weighted_descriptions.csv',props),('A_models.csv',models),('A_multistate.csv',trans),('A_followup_audit.csv',audit),('A_flow.csv',flows)]:pd.DataFrame(data).to_csv(OUT/name,index=False)
 metas=[]
 for outcome in ['composite','death']:
  for analysis in ['primary','SDOC','continuous_5kg','exclude_first_year_death']:
   metas.append(meta([r for r in models if r['outcome']==outcome and r['analysis']==analysis],outcome+'_'+analysis))
  metas.append(meta([r for r in models if r['outcome']==outcome and r['analysis']=='primary' and r['cohort'] in ['CHARLS','ELSA','MHAS']],outcome+'_measured_BMI'))
 pd.DataFrame(metas).to_csv(OUT/'A_meta.csv',index=False)
 pd.DataFrame(BRIDGE_AUDIT).to_csv(OUT/'raw_followup_bridge_audit.csv',index=False)
 (OUT/'run_receipt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':lock['sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'HRS_read':False,'participant_output':False,'manifest':MANIFEST,'all_local_followup_status':'CHARLS2020_MHAS2021_SHARE_W9_INCLUDED_ELSA_KLOSA_W9_MORTALITY_NOT_AVAILABLE','help_sensitivity':'NOT_ESTIMABLE_FIVE_COHORT_HARMONIZED_FIELDS_MISSING','frailty':'PARTIAL_IDENTIFICATION_ONLY'},ensure_ascii=False,indent=2))
 print('DONE aggregate outputs only')
if __name__=='__main__':main()
