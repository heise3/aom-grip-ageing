"""A v1 closeout: explicit three-cohort allowlist; in-memory records, aggregate outputs only."""
from pathlib import Path
import sys,json,hashlib,datetime,warnings
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf,patsy
from scipy.special import softmax,expit,logit
from scipy.stats import chi2
O=Path(__file__).resolve().parents[1];R=O.parents[1];sys.path.insert(0,str(R/'work/a_formal'));import run_formal as f
S=R/'outputs/A_formal_v1_20261027';OLD=pd.read_csv(S/'A_models.csv');OLDMS=pd.read_csv(S/'A_multistate.csv')
LOCK=json.loads((O/'closeout_lock.json').read_text());assert hashlib.sha256((O/'A_closeout_inference_addendum.md').read_bytes()).hexdigest()==LOCK['sha256']
assert hashlib.sha256((S/'A_analysis_plan_v1.md').read_bytes()).hexdigest()=='321cb13e11058d74962e40545b75bc46f98cea3e3903920f2357b25cac92b99b'
DIAG=[];PH=[];CAL=[];CIF=[];FLOW=[];CENS=[];MISS=[];VALID=[];SUMMARY=[]
def sup(n):return '<10' if int(n)<10 else int(n)
def partition(vals):
 """Primary and complementary suppression in a known-total partition."""
 out={k:sup(v) for k,v in vals.items()};small=[k for k,v in vals.items() if v<10]
 if len(small)==1:
  others=[k for k in vals if k not in small]
  if others:out[min(others,key=lambda k:vals[k])]='SUPPRESSED_COMPLEMENT'
 return out

def design_condition(X):
 a=np.asarray(X,float);ss=a.std(axis=0);a=a.copy();mask=ss>0;a[:,mask]=(a[:,mask]-a[:,mask].mean(axis=0))/ss[mask]
 return int(np.linalg.matrix_rank(X)),float(np.linalg.cond(a))

def flow_audit(h,d,c,dd):
 w=f.BASE[c];age=f.num(h,f'r{w}agey').combine_first(f.num(h,f'r{w}agey_e'))
 eligibleage=f.num(h,f'r{w}iwstat').eq(1)&age.ge(65)
 baseadl=f.adl(h,c,w).reindex(d.index);ids=set(dd.id);z=dd.dropna(subset=f.COV+['low_EWGSOP2','event']).copy();zids=set(z.id)
 derv=f.derive(h,c,f.SPEC[c][1],w,f.SPEC[c][3]);el=f.eligibility(derv)
 stages={'interviewed_age65_n':int(eligibleage.sum()),'sex_BMI_grip_complete_n':len(derv),'observable_eligible_grip_n':len(d),'baseline_ADL_zero_n':int(baseadl.eq(0).sum()),'any_observable_interval_n':len(ids),'primary_complete_case_n':len(zids),'positive_weight_descriptive_n':int(d.weight.notna().sum())}
 row={'cohort':c,**{k:sup(v) for k,v in stages.items()}}
 # Broad exclusions are mutually exclusive on the ordered path; small cells suppressed.
 exclusions={'incomplete_sex_BMI_grip_n':stages['interviewed_age65_n']-len(derv),'not_confirmed_eligible_n':len(derv)-len(d),'baseline_ADL_nonzero_or_unknown_n':len(d)-stages['baseline_ADL_zero_n'],'no_observable_postbaseline_interval_n':stages['baseline_ADL_zero_n']-len(ids),'missing_fixed_covariates_n':len(ids)-len(zids)}
 row.update({k:sup(v) for k,v in exclusions.items()})
 row['no_interval_or_missing_covariates_n']=sup(stages['baseline_ADL_zero_n']-len(zids))
 if 0<exclusions['missing_fixed_covariates_n']<10:
  row['any_observable_interval_n']='SUPPRESSED_COMPLEMENT';row['no_observable_postbaseline_interval_n']='SUPPRESSED_COMPLEMENT'
 FLOW.append(row)
 for k in ['age','sex','educ','bmi','smoke','diabetes','chronic','country']:
  p=dd.drop_duplicates('id');MISS.append({'cohort':c,'scope':'persons_with_any_composite_interval','field':k,'missing_n':sup(p[k].isna().sum()),'denominator_n':row['any_observable_interval_n'],'overlap':'nonexclusive'})
 # Track exactly the frozen active risk set; count all and baseline-complete-case risk sets.
 for scope,di in [('baseline_ADL_zero',d),('baseline_covariates_complete',d.dropna(subset=[k for k in f.COV if k!='interval']))]:
  active=f.adl(h,c,w).reindex(di.index).eq(0);initial=int(active.sum());term={};lastvalid={}
  for v in range(w+1,f.MAXW[c]+1):
   status=f.num(h,f'r{v}iwstat').reindex(di.index)
   future=pd.concat([f.num(h,f'r{k}iwstat').isin([1,4]) for k in range(v+1,f.MAXW[c]+1)],axis=1).any(axis=1).reindex(di.index) if v<f.MAXW[c] else pd.Series(False,index=di.index)
   bad=status.eq(5)&future;dead=status.eq(5)&~bad;a=f.adl(h,c,v).reindex(di.index)
   valid=dead|(status.eq(1)&a.notna());keep=active&valid&~bad
   masks={'death_alive_conflict':active&bad,'alive_or_interview_ADL_unknown':active&~bad&~valid&status.isin([1,4]),'status_unavailable_or_other':active&~bad&~valid&~status.isin([1,4]),'first_ADL':keep&~dead&a.eq(1),'death_first':keep&dead,'continue_event_free':keep&~dead&a.eq(0)}
   vals={k:int(m.sum()) for k,m in masks.items()};assert sum(vals.values())==int(active.sum())
   CENS.append({'cohort':c,'population':scope,'wave':v,'year':f.YEARS[c][v],'entering_risk_n':sup(active.sum()),**partition(vals)})
   for reason,m in masks.items():
    if reason!='continue_event_free':
     for ix in di.index[m]:term[di.loc[ix,'id']]=reason
   for ix in di.index[keep]:lastvalid[di.loc[ix,'id']]=v
   active=masks['continue_event_free']
  for ix in di.index[active]:term[di.loc[ix,'id']]='administrative_end'
  assert len(term)==initial
  if scope=='baseline_covariates_complete':
   assert set(lastvalid)==zids
   cats={k:sum(term[i]==k for i in zids) for k in ['first_ADL','death_first','death_alive_conflict','alive_or_interview_ADL_unknown','status_unavailable_or_other','administrative_end']}
   assert cats['first_ADL']+cats['death_first']==int(z.event.sum())
   SUMMARY.append({'cohort':c,'population':'primary_model','persons_n':sup(len(zids)),**partition(cats)})
   for group in [0,1]:
    gi=set(z.loc[z.low_EWGSOP2.eq(bool(group)),'id']);tot=len(gi)
    q={'event':sum(term[i] in ['first_ADL','death_first'] for i in gi),'early_censor':sum(term[i] not in ['first_ADL','death_first','administrative_end'] for i in gi),'administrative_end':sum(term[i]=='administrative_end' for i in gi)}
    SUMMARY.append({'cohort':c,'population':'primary_model_by_grip','low_grip':group,'persons_n':sup(tot),**partition(q)})
 return z

def glm_diagnostics(dd,c,kind,analysis,exposure):
 z=dd.dropna(subset=f.COV+[exposure,'event']).copy();z['x']=z[exposure].astype(float)
 old=OLD[(OLD.cohort==c)&(OLD.outcome==kind)&(OLD.analysis==analysis)].iloc[0]
 r={'cohort':c,'outcome':kind,'analysis':analysis,'persons_n':sup(z.id.nunique()),'intervals_n':sup(len(z)),'original_status':old.status}
 if old.status!='OK':r['status']='ORIGINAL_NOT_ESTIMABLE';DIAG.append(r);return
 form='event ~ x + age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'+(' + C(country)' if c=='SHARE' else '')
 with warnings.catch_warnings(record=True) as ws:
  fit=smf.glm(form,data=z,family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster',cov_kwds={'groups':z.id},maxiter=150)
 X=fit.model.exog;rank,cond=design_condition(X);pred=np.asarray(fit.predict());delta=max(abs(float(fit.params['x'])-old.log_effect),abs(float(fit.bse['x'])-old.se))
 assert z.id.nunique()==int(old.persons_n) and len(z)==int(old.intervals_n) and int(z.event.sum())==int(old.events_n)
 r.update(status='OK' if fit.converged else 'NOT_CONVERGED',converged=bool(fit.converged),iterations=fit.fit_history.get('iteration'),parameter_count=X.shape[1],design_rank=rank,standardized_condition_number=cond,score_max_abs_per_row=float(np.max(np.abs(fit.model.score(fit.params)))/len(z)),covariance_finite=bool(np.isfinite(fit.cov_params()).all().all()),probability_finite=bool(np.isfinite(pred).all()),predicted_min=float(pred.min()),predicted_max=float(pred.max()),effect_beta_se_max_abs_delta=delta,events_per_parameter=float(z.event.sum()/X.shape[1]),warning_types=';'.join(sorted({type(x.message).__name__ for x in ws})))
 assert delta<1e-7,'Frozen model reproduction differs'
 DIAG.append(r)
 if analysis=='primary':
  z['pred']=pred
  for (interval,grp),g in z.groupby(['interval','low_EWGSOP2']):
   ev=int(g.event.sum());n=len(g);okay=min(ev,n-ev)>=10
   CAL.append({'cohort':c,'outcome':kind,'interval':int(interval),'low_grip':int(grp),'intervals_n':sup(n),'events_n':sup(ev) if ev<10 or n-ev>=10 else 'SUPPRESSED_COMPLEMENT','observed_rate':float(g.event.mean()) if okay else 'SUPPRESSED','mean_predicted':float(g.pred.mean()) if okay else 'SUPPRESSED','interpretation':'apparent_in_sample_not_external_validation'})
  pr={'cohort':c,'outcome':kind,'diagnostic':'grip_by_categorical_interval_cluster_Wald','replaces_primary':False}
  try:
   with warnings.catch_warnings(record=True) as pw:
    aug=smf.glm(form+' + x:C(interval)',data=z,family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster',cov_kwds={'groups':z.id},maxiter=150)
   ix=[i for i,nm in enumerate(aug.params.index) if 'x:C(interval)' in nm or ('C(interval)' in nm and ':x' in nm)]
   mat=np.asarray(aug.cov_params())[np.ix_(ix,ix)];b=aug.params.iloc[ix].to_numpy();erank=np.linalg.matrix_rank(mat);finite=np.isfinite(mat).all()
   cells=z.groupby(['interval','low_EWGSOP2']).event.agg(['sum','count'])
   zero_cell=bool(((cells['sum']==0)|(cells['count']==cells['sum'])).any())
   pr.update(converged=bool(aug.converged),interaction_df=len(ix),cov_rank=int(erank),sparse_group_interval_present=bool(cells['sum'].lt(10).any()),zero_event_or_nonevent_cell=zero_cell,max_abs_interaction_coefficient=float(np.max(np.abs(b))),max_interaction_se=float(np.sqrt(np.diag(mat)).max()))
   if zero_cell and np.max(np.abs(b))>10:
    pr['status']='NOT_RELIABLE_QUASI_SEPARATION';pr['p']='NOT_REPORTED'
   elif aug.converged and finite and erank==len(ix) and np.linalg.eigvalsh((mat+mat.T)/2).min()>0:
    W=float(b@np.linalg.solve(mat,b));pr.update(status='DIAGNOSTIC_ONLY',wald_chi2=W,p=float(chi2.sf(W,len(ix))))
   else:pr['status']='NOT_RELIABLE_NUMERICAL_DIAGNOSTIC'
  except Exception as e:pr.update(status='DIAGNOSTIC_ERROR',error_type=type(e).__name__)
  PH.append(pr)

def cif_eval(beta,mats,gradient=False):
 n,p=mats[0].shape;P=2*p;S0=np.ones(n);F=np.zeros((n,2));gS=np.zeros((n,P));gF=np.zeros((n,2,P));out=[]
 for mat in mats:
  probs=softmax(np.column_stack([np.zeros(n),mat@beta]),axis=1)
  if gradient:
   gp=np.empty((n,3,P))
   for s in range(3):
    for j in range(2):gp[:,s,j*p:(j+1)*p]=((probs[:,s]*((s==j+1)-probs[:,j+1]))[:,None]*mat)
   for j in range(2):gF[:,j,:]+=gS*probs[:,j+1,None]+S0[:,None]*gp[:,j+1,:]
   gS=gS*probs[:,0,None]+S0[:,None]*gp[:,0,:]
  F+=S0[:,None]*probs[:,1:];S0*=probs[:,0]
  assert np.max(np.abs(S0+F.sum(axis=1)-1))<1e-10
  out.append((F.mean(axis=0).copy(),gF.mean(axis=0).copy() if gradient else None))
 return out

def cif_model(dd,c):
 z=dd.dropna(subset=f.COV+['state']).copy();z['x']=z.low_EWGSOP2.astype(int)
 if any(int((z.state.eq(s)&z.x.eq(v)).sum())<10 for s in [1,2] for v in [0,1]):
  VALID.append({'cohort':c,'status':'NOT_ESTIMABLE_SMALL_TRANSITION_CELL'});return
 form='x + age + C(sex) + C(educ) + bmi + smoke + diabetes + chronic + C(interval)'+(' + C(country)' if c=='SHARE' else '')
 X=patsy.dmatrix(form,z,return_type='dataframe');transform={k:(float(z[k].mean()),float(z[k].std())) for k in ['age','bmi','chronic'] if z[k].std()>0}
 for k,(mu,sd) in transform.items():X[k]=(X[k]-mu)/sd
 with warnings.catch_warnings(record=True) as ws:fit=sm.MNLogit(z.state,X).fit(method='bfgs',disp=False,maxiter=600,cov_type='cluster',cov_kwds={'groups':z.id})
 rank,cond=design_condition(X);P=X.shape[1];B=np.asarray(fit.params);V=np.asarray(fit.cov_params());V=(V+V.T)/2;eig=np.linalg.eigvalsh(V)
 names=[str(t[1]) for t in fit.cov_params().index];assert names==list(X.columns)*2
 valid={'cohort':c,'status':'OK','converged':bool(fit.mle_retvals['converged']),'parameter_count':B.size,'design_rank':rank,'design_columns':P,'standardized_condition_number':cond,'cluster_cov_min_eigenvalue':float(eig.min()),'covariance_finite':bool(np.isfinite(V).all()),'score_max_abs_per_row':float(np.max(np.abs(fit.model.score(B.ravel(order='F')))))/len(z),'warning_types':';'.join(sorted({type(t.message).__name__ for t in ws}))}
 if not fit.mle_retvals['converged'] or eig.min()<-1e-8*max(1.,eig.max()):valid['status']='NOT_ESTIMABLE_COVARIANCE_OR_CONVERGENCE';VALID.append(valid);return
 persons=z.drop_duplicates('id');intervals=sorted(z.interval.unique());maxerr=0.;maxpoint=0.
 for val in [0,1]:
  mats=[]
  for it in intervals:
   temp=persons.copy();temp['x']=val;temp['interval']=it
   mat=pd.DataFrame(patsy.build_design_matrices([X.design_info],temp)[0],columns=X.columns)
   for k,(mu,sd) in transform.items():mat[k]=(mat[k]-mu)/sd
   mats.append(np.asarray(mat))
  vals=cif_eval(B,mats,True)
  # Independent directional finite differences, no random participant sampling.
  for off in [0,1,2]:
   direction=np.sin(np.arange(B.size)+1+off);direction/=np.linalg.norm(direction);db=direction.reshape(B.shape,order='F');eps=1e-5
   fd=(cif_eval(B+eps*db,mats)[-1][0]-cif_eval(B-eps*db,mats)[-1][0])/(2*eps)
   analytic=vals[-1][1]@direction;maxerr=max(maxerr,float(np.max(np.abs(fd-analytic))))
  prev=OLDMS[(OLDMS.cohort==c)&(OLDMS.transition=='standardized_CIF_last_wave')&(OLDMS.low_grip==val)].iloc[0]
  maxpoint=max(maxpoint,abs(vals[-1][0][0]-prev.ADL_CIF),abs(vals[-1][0][1]-prev.death_CIF))
  for it,(point,grad) in zip(intervals,vals):
   for j,label in [(0,'incident_ADL'),(1,'death_before_observed_ADL')]:
    q=float(point[j]);var=float(grad[j]@V@grad[j]);assert var>=-1e-12;se=np.sqrt(max(0,var));ci=expit(logit(q)+np.array([-1,1])*1.96*se/(q*(1-q)))
    CIF.append({'cohort':c,'wave':int(f.BASE[c]+it),'year':f.YEARS[c][int(f.BASE[c]+it)],'years_from_baseline':f.YEARS[c][int(f.BASE[c]+it)]-f.YEARS[c][f.BASE[c]],'low_grip':val,'outcome':label,'standardization_n':sup(len(persons)),'cif':q,'se_delta':float(se),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'interval_method':'cluster_sandwich_delta_logit_pointwise95','standardization':'fixed_empirical_complete_case_baseline_distribution'})
 valid.update(gradient_directional_max_abs_error=maxerr,original_CIF_max_abs_delta=maxpoint)
 assert maxerr<1e-6 and maxpoint<1e-7
 VALID.append(valid)

def save():
 for name,rows in [('model_diagnostics',DIAG),('PH_diagnostics',PH),('apparent_calibration',CAL),('CIF_intervals',CIF),('enrollment_flow',FLOW),('wave_censoring',CENS),('missing_covariates',MISS),('CIF_validation',VALID),('terminal_followup',SUMMARY)]:
  out=pd.DataFrame(rows)
  # Avoid differencing overlapping populations to recover small exclusions.
  if name=='wave_censoring':out=out[out.population.eq('baseline_ADL_zero')]
  if name=='terminal_followup' and len(out):
   grouped=out[out.population.eq('primary_model_by_grip')]
   blocked=set(grouped.loc[grouped[['event','early_censor','administrative_end']].astype(str).isin(['<10','SUPPRESSED_COMPLEMENT']).any(axis=1),'cohort'])
   out=out[~(out.population.eq('primary_model_by_grip')&out.cohort.isin(blocked))]
  out.to_csv(O/(name+'.csv'),index=False)

if __name__=='__main__':
 for c in ['CHARLS','MHAS','SHARE']:
  print(c,'loading authorized non-HRS inputs',flush=True)
  h,d,_=f.load(c);dd,_=f.intervals(h,d,c,'composite');flow_audit(h,d,c,dd);ex=f.early_exclusion(h,d,c)
  for kind in ['composite','death']:
   dd,_=f.intervals(h,d,c,kind)
   for analysis,exposure,sub in [('primary','low_EWGSOP2',dd),('SDOC','low_SDOC',dd),('continuous_5kg','grip5',dd),('exclude_first_year_death','low_EWGSOP2',dd[~dd.id.isin(ex)])]:
    glm_diagnostics(sub,c,kind,analysis,exposure)
   if kind=='composite':cif_model(dd,c)
  save();print(c,'CIF/flow/diagnostics aggregates saved',flush=True)
  del h,d,dd
 (O/'execution_receipt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'implementation_lock_sha256':LOCK['sha256'],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cohorts':['CHARLS','MHAS','SHARE'],'HRS_read':False,'individual_rows_exported':False,'CIF_inference':'conditional fixed empirical standardization; cluster sandwich delta logit pointwise95','input_manifest':f.MANIFEST,'packages':{'numpy':np.__version__,'pandas':pd.__version__,'statsmodels':sm.__version__}},ensure_ascii=False,indent=2))
 print('CLOSEOUT_COMPUTATION_COMPLETE',flush=True)
