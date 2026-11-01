"""A v1.1 post hoc analyses; explicit non-HRS allowlist; aggregate output only."""
from pathlib import Path
import sys,json,hashlib,datetime,warnings
import numpy as np,pandas as pd,patsy,statsmodels.api as sm
from scipy.special import expit,logit
from scipy.stats import norm,t
O=Path(__file__).resolve().parents[1];R=O.parents[1];sys.path.insert(0,str(R/'work/a_formal'));import run_formal as f
assert hashlib.sha256((O/'A_amendment_v1.1.md').read_bytes()).hexdigest()==json.loads((O/'amendment_lock.json').read_text())['sha256']
assert hashlib.sha256((R/'outputs/A_formal_v1_20261027/A_analysis_plan_v1.md').read_bytes()).hexdigest()=='321cb13e11058d74962e40545b75bc46f98cea3e3903920f2357b25cac92b99b'
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
TAG='post hoc, added after reviewing v1 results'
MODELS=[];WEIGHTS=[];BOOT=[];RISK=[];INTER=[];FLOW=[];CHECK=[];REPS=[];BASE=[]
FORM='x + age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'
def all_baseline(h,c):
 w=f.BASE[c];d=f.eligibility(f.derive(h,c,f.SPEC[c][1],w,f.SPEC[c][3]));d.id=d.id.astype(str)
 d['educ']=f.num(h,'raeducl').where(lambda x:x.isin([1,2,3]));d['smoke']=f.bn(h,f'r{w}smoken');d['diabetes']=f.bn(h,f'r{w}diabe')
 terms=[('respe' if k=='lung' else 'hrtatte' if k=='heart' else s) if c=='MHAS' else s for k,s in f.CHRONMAP.items()]
 d['chronic']=pd.concat([f.bn(h,f'r{w}{s}') for s in terms],axis=1).sum(axis=1,min_count=6)
 d['grip5']=d.grip/5;d['low_FNIH']=d.grip/d.bmi<np.where(d.sex.eq(1),1.00,.56)
 return d
def matrix(form,z):
 X=patsy.dmatrix(form,z,return_type='dataframe');scale={}
 for k in ['age','bmi','chronic']:
  if k in X and X[k].std()>0:
   scale[k]=(X[k].mean(),X[k].std());X[k]=(X[k]-scale[k][0])/scale[k][1]
 return X,scale
def fit(X,y,ids=None,weight=None,link='cloglog',start=None):
 family=sm.families.Binomial(link=sm.families.links.CLogLog() if link=='cloglog' else sm.families.links.Logit())
 with warnings.catch_warnings(record=True) as caught:
  model=sm.GLM(np.asarray(y,float),np.asarray(X,float),family=family,freq_weights=weight)
  kw={'cov_type':'cluster','cov_kwds':{'groups':np.asarray(ids)}} if ids is not None else {}
  rr=model.fit(start_params=start,maxiter=100,tol=1e-8,**kw)
 if not rr.converged or not np.isfinite(rr.params).all() or np.max(np.abs(rr.params))>30:raise ValueError('nonconvergence_or_extreme_coefficient')
 return rr
def effect(b,se):return {'log_effect':float(b),'se':float(se),'effect':float(np.exp(b)),'ci_low':float(np.exp(b-1.96*se)),'ci_high':float(np.exp(b+1.96*se)),'p':float(2*norm.sf(abs(b/se))),'status':'OK'}
def censor_panel(h,d,c):
 d=d.dropna(subset=[x for x in f.COV if x!='interval']).copy();active=f.adl(h,c,f.BASE[c]).reindex(d.index).eq(0);out=[]
 for v in range(f.BASE[c]+1,f.MAXW[c]+1):
  s=f.num(h,f'r{v}iwstat').reindex(d.index);a=f.adl(h,c,v).reindex(d.index)
  fut=pd.concat([f.num(h,f'r{k}iwstat').isin([1,4]) for k in range(v+1,f.MAXW[c]+1)],axis=1).any(axis=1).reindex(d.index) if v<f.MAXW[c] else pd.Series(False,index=d.index)
  bad=s.eq(5)&fut;dead=s.eq(5)&~bad;obs=(dead|(s.eq(1)&a.notna()))&~bad
  g=d.loc[active].copy();g['interval']=v-f.BASE[c];g['observed']=obs.loc[active].astype(int);g['event']=(dead.loc[active]|a.loc[active].eq(1)).astype(int);out.append(g)
  active=active&obs&~dead&a.eq(0)
 return pd.concat(out,ignore_index=True).sort_values(['id','interval']).reset_index(drop=True)
def cumulative_ratio(pn,pd_,gid):
 logw=np.log(np.maximum(pn,1e-12))-np.log(np.maximum(pd_,1e-12));cs=np.cumsum(logw);start=np.r_[0,np.flatnonzero(gid[1:]!=gid[:-1])+1]
 offset=np.zeros(len(start));offset[1:]=cs[start[1:]-1]
 return np.exp(cs-offset[gid])
def ipcw(h,d,c):
 z=censor_panel(h,d,c);z['x']=z.low_EWGSOP2.astype(int);X,sc=matrix(FORM+(' + C(country)' if c=='SHARE' else ''),z);N,_=matrix('x + C(interval)',z)
 gid,people=pd.factorize(z.id,sort=False);obs=z.observed.to_numpy().astype(bool);y=z.event.to_numpy();Xn=np.asarray(X);Nn=np.asarray(N);ix=X.columns.get_loc('x')
 old=pd.read_csv(R/'outputs/A_formal_v1_20261027/A_models.csv');old=old[(old.cohort==c)&(old.outcome=='composite')&(old.analysis=='primary')].iloc[0]
 assert z.loc[obs,'id'].nunique()==int(old.persons_n) and y[obs].sum()==int(old.events_n)
 orig=fit(Xn[obs],y[obs],z.loc[obs,'id']);assert abs(orig.params[ix]-old.log_effect)<1e-6
 rn=fit(Nn,z.observed,link='logit');rd=fit(Xn,z.observed,link='logit');pn=rn.predict(Nn);pd_=rd.predict(Xn);raw=cumulative_ratio(pn,pd_,gid)[obs];cuts=np.quantile(raw,[.01,.99]);tr=np.clip(raw,*cuts)
 out=fit(Xn[obs],y[obs],weight=tr);b=float(out.params[ix]);row={'cohort':c,'analysis':'IPCW','post_hoc':TAG,'persons_n':f.suppress(z.loc[obs,'id'].nunique()),'censor_model_baseline_n':f.suppress(len(people)),'events_n':f.suppress(y[obs].sum()),'primary_HR':old.effect,'status':'PENDING_BOOTSTRAP'}
 for label,vv in [('untruncated',raw),('truncated_1_99',tr)]:
  q=np.quantile(vv,[0,.01,.05,.25,.5,.75,.95,.99,1]);WEIGHTS.append({'cohort':c,'distribution':label,'mean':float(vv.mean()),'sd':float(vv.std()),**{k:float(v) for k,v in zip(['min','p01','p05','p25','median','p75','p95','p99','max'],q)},'Kish_person_interval_n_eff':float(vv.sum()**2/(vv@vv)),'interval_rows_n':f.suppress(len(vv)),'truncation_low':float(cuts[0]),'truncation_high':float(cuts[1]),'fraction_clipped':float(np.mean(raw!=tr)),'minimum_observation_probability':float(pd_.min()),'minimum_numerator_probability':float(pn.min())})
 rng=np.random.default_rng(20261027+['CHARLS','MHAS','SHARE'].index(c));betas=[];fails={}
 for it in range(500):
  freq=rng.multinomial(len(people),np.full(len(people),1/len(people)));fw=freq[gid]
  try:
   nfit=fit(Nn,z.observed,weight=fw,link='logit',start=rn.params);dfit=fit(Xn,z.observed,weight=fw,link='logit',start=rd.params)
   ww=cumulative_ratio(nfit.predict(Nn),dfit.predict(Xn),gid)[obs];qq=np.quantile(np.repeat(ww,fw[obs]),[.01,.99]);wf=fw[obs]*np.clip(ww,*qq)
   ff=fit(Xn[obs],y[obs],weight=wf,start=out.params);betas.append(float(ff.params[ix]))
  except Exception as e:fails[type(e).__name__]=fails.get(type(e).__name__,0)+1
  if (it+1)%50==0:print(c,'IPCW bootstrap',it+1,'/',500,flush=True)
 if len(betas)>=450:
  bs=np.array(betas);ci=np.exp(np.quantile(bs,[.025,.975]));row.update(effect(b,bs.std(ddof=1)));row.update(ci_low=float(ci[0]),ci_high=float(ci[1]),p=np.nan,CI_method='individual_cluster_bootstrap_percentile_500',se_method='bootstrap_SD_logHR',HR_ratio_to_primary=float(np.exp(b)/old.effect))
 else:row.update(status='NOT_ESTIMABLE_BOOTSTRAP_FAILURE_RATE',effect=float(np.exp(b)))
 MODELS.append(row);BOOT.append({'cohort':c,'requested_replicates':500,'successful_replicates':len(betas),'failed_replicates':f.suppress(500-len(betas)),'failure_fraction':(500-len(betas))/500,'failure_types':json.dumps(fails),'seed':20261027+['CHARLS','MHAS','SHARE'].index(c),'CI_reliable_by_prespecified_rule':len(betas)>=450})
 CHECK.append({'cohort':c,'module':'IPCW','primary_beta_abs_difference':abs(orig.params[ix]-old.log_effect),'design_rank':int(np.linalg.matrix_rank(Xn)),'design_columns':Xn.shape[1],'numerator_converged':bool(rn.converged),'denominator_converged':bool(rd.converged),'weighted_outcome_converged':bool(out.converged),'weights_finite':bool(np.isfinite(tr).all())})
 return z.loc[obs].copy(),X.loc[obs].copy(),sc,orig
def absolute_risk(z,X,scale,rr,c):
 people=z.drop_duplicates('id');B=rr.params;V=rr.cov_params();est=[];grads=[];maxerr=0.
 for val in [0,1]:
  L=np.zeros(len(people));G=np.zeros((len(people),len(B)))
  for it in sorted(z.interval.unique()):
   w=f.BASE[c]+int(it);a=f.YEARS[c][w-1]-f.YEARS[c][f.BASE[c]];b=f.YEARS[c][w]-f.YEARS[c][f.BASE[c]];frac=np.clip((9-a)/(b-a),0,1)
   if frac==0:continue
   q=people.copy();q['x']=val;q['interval']=it;mat=pd.DataFrame(patsy.build_design_matrices([X.design_info],q)[0],columns=X.columns)
   for k,(mu,sd) in scale.items():mat[k]=(mat[k]-mu)/sd
   xx=mat.to_numpy();H=np.exp(xx@B)*frac;L+=H;G+=H[:,None]*xx
  s=np.exp(-L);p=float((1-s).mean());g=(s[:,None]*G).mean(axis=0);se=float(np.sqrt(g@V@g));ci=expit(logit(p)+np.array([-1,1])*1.96*se/(p*(1-p)));est.append(p);grads.append(g)
  RISK.append({'cohort':c,'measure':'risk','low_grip':val,'nominal_years':9,'risk':p,'ci_low':float(ci[0]),'ci_high':float(ci[1]),'standardization_n':f.suppress(len(people)),'status':'OK','method':'original_cloglog_fixed_empirical_standardization_cluster_delta','interpolation':'SHARE constant hazard within years8-11' if c=='SHARE' else 'observed_wave_endpoint','post_hoc':TAG})
 delta=est[1]-est[0];g=grads[1]-grads[0];se=np.sqrt(g@V@g)
 RISK.append({'cohort':c,'measure':'risk_difference_low_minus_normal','nominal_years':9,'risk':delta,'ci_low':float(delta-1.96*se),'ci_high':float(delta+1.96*se),'standardization_n':f.suppress(len(people)),'status':'OK','method':'paired_joint_delta_Wald','post_hoc':TAG})
def interaction(h,d,c):
 flow={'cohort':c,'age65_sex_BMI_grip_valid_n':f.suppress(len(d)),'eligible_yes_n':f.suppress(d.eligible.eq('yes').sum()),'eligible_no_n':f.suppress(d.eligible.eq('no').sum()),'eligible_unknown_excluded_n':f.suppress(d.eligible.eq('unknown').sum())}
 known=d[d.eligible.isin(['yes','no'])].copy();known['eligible_binary']=known.eligible.eq('yes').astype(int);dd,_=f.intervals(h,known,c);z=dd.dropna(subset=f.COV+['low_EWGSOP2','event']).copy();z['x']=z.low_EWGSOP2.astype(int);flow.update(model_n=f.suppress(z.id.nunique()),events_n=f.suppress(z.event.sum()));FLOW.append(flow)
 cells=z.groupby(['eligible_binary','x']).event.sum();base={'cohort':c,'post_hoc':TAG,'persons_n':f.suppress(z.id.nunique())}
 if len(cells)!=4 or cells.min()<10:
  for lab in ['interaction_HR_ratio','eligible_HR','ineligible_HR']:INTER.append({**base,'analysis':lab,'status':'NOT_ESTIMABLE_SMALL_EVENT_CELL'})
  return
 X,sc=matrix(FORM+' + eligible_binary + x:eligible_binary'+(' + C(country)' if c=='SHARE' else ''),z);rr=fit(X,z.event,z.id);ix=X.columns.get_loc('x');jj=X.columns.get_loc('x:eligible_binary');V=rr.cov_params()
 for lab,a in [('ineligible_HR',{ix:1}),('eligible_HR',{ix:1,jj:1}),('interaction_HR_ratio',{jj:1})]:
  v=np.zeros(X.shape[1]);
  for k,val in a.items():v[k]=val
  INTER.append({**base,'analysis':lab,**effect(float(v@rr.params),float(np.sqrt(v@V@v)))})
 CHECK.append({'cohort':c,'module':'interaction','converged':bool(rr.converged),'design_rank':int(np.linalg.matrix_rank(X)),'design_columns':X.shape[1]})
def baseline_table(h,d,c,ip):
 base=d[f.adl(h,c,f.BASE[c]).reindex(d.index).eq(0)].copy();pids=set(ip.id);base['included']=base.id.isin(pids)
 for group,g in base.groupby('included'):
  r={'cohort':c,'scope':'baseline_ADL0_eligible','included_model':bool(group),'n':f.suppress(len(g))}
  for key in ['age','bmi','grip','chronic']:
   r[key+'_mean']=float(g[key].mean());r[key+'_sd']=float(g[key].std());r[key+'_missing_n']=f.suppress(g[key].isna().sum())
  r['low_grip_n']=f.suppress(g.low_EWGSOP2.sum());r['low_grip_proportion']=float(g.low_EWGSOP2.mean()) if min(g.low_EWGSOP2.sum(),(~g.low_EWGSOP2).sum())>=10 else np.nan
  BASE.append(r)
def save():
 for name,rows in [('posthoc_models',MODELS),('IPCW_weight_distribution',WEIGHTS),('IPCW_bootstrap_diagnostics',BOOT),('absolute_risk_9year',RISK),('effect_modification',INTER),('effect_modification_flow',FLOW),('numerical_checks',CHECK),('baseline_inclusion_descriptions',BASE)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
 allm=[]
 for label in ['IPCW','FNIH']:
  if MODELS:allm.append(f.meta([r for r in MODELS if r['analysis']==label],label))
 for label in ['interaction_HR_ratio','eligible_HR','ineligible_HR']:
  if INTER:
   mm=f.meta([r for r in INTER if r['analysis']==label],label)
   if mm['status']=='OK':mm['p']=float(2*t.sf(abs(mm['log_effect']/mm['se']),mm['k']-1))
   allm.append(mm)
 for m in allm:m['post_hoc']=TAG;m['PI_df']=m['k']-2 if m['k']>=3 else np.nan
 pd.DataFrame(allm).to_csv(O/'posthoc_meta.csv',index=False)
if __name__=='__main__':
 for c in ['CHARLS','MHAS','SHARE']:
  print(c,'loading approved non-HRS data',flush=True);h,eligible,_=f.load(c);all_d=all_baseline(h,c)
  z,X,sc,original=ipcw(h,eligible,c);X.design_info=patsy.dmatrix(FORM+(' + C(country)' if c=='SHARE' else ''),z,return_type='dataframe').design_info
  absolute_risk(z,X,sc,original,c);baseline_table(h,eligible,c,z)
  dd,_=f.intervals(h,all_d[all_d.eligible.eq('yes')],c);r=f.fitmodel(dd,c,'composite','FNIH','low_FNIH');r['post_hoc']=TAG;MODELS.append(r)
  interaction(h,all_d,c);save();print(c,'new aggregate results saved',flush=True);del h,eligible,all_d,z,X,dd
 (O/'execution_receipt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'amendment_hash':hashlib.sha256((O/'A_amendment_v1.1.md').read_bytes()).hexdigest(),'code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_manifest':f.MANIFEST,'HRS_read':False,'participant_rows_written':False,'post_hoc':TAG,'packages':{'statsmodels':sm.__version__,'numpy':np.__version__,'pandas':pd.__version__}},ensure_ascii=False,indent=2))
 print('POSTHOC_PROGNOSIS_COMPLETE',flush=True)
