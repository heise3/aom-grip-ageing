"""Approved v1.2 execution. Original algorithms imported; affected cohorts only; aggregate export."""
from pathlib import Path
import sys,json,hashlib,datetime,importlib.util,ast,warnings
import numpy as np,pandas as pd,patsy
from scipy.special import expit,logit
from scipy.stats import t,spearmanr
import qc_core as q
import posthoc_qc as ph
f=q.f;O=q.O;R=q.R;V1=R/'outputs/A_formal_v1_20261027';V11=q.OLD;CLOSE=R/'outputs/A_closeout_v1_20261027'
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
TAG=ph.TAG
pre=json.loads((O/'QC_resolved_preflight_decision.json').read_text())
assert all(x['status']!='HOLD_SOURCE_CONFLICT' for x in pre['decisions'])
assert [x['cohort'] for x in pre['decisions'] if x['status']=='REFIT_AFFECTED_MODULES']==['CHARLS','ELSA']
def mod(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
cl=mod(CLOSE/'code/run_closeout.py','closeout');cl.O=O
refpath=V11/'frozen_standard_population_65_84.csv';refhash=hashlib.sha256(refpath.read_bytes()).hexdigest();assert refhash=='19f4176a2801da7a5d77b17fb927813c9ed75f6e3910bde123cba50d5297c457'
ref=pd.read_csv(refpath);refs={lab:{(int(r.age_group[:2]),int(r.sex)):getattr(r,col) for r in ref.itertuples()} for lab,col in [('pooled_five_cohorts','pooled_standard_share'),('WHO_2000_2025','WHO_standard_share')]}
# Extract only the reviewed calculator, never execute old reference-construction top level.
support=[];tree=ast.parse((V11/'code/standardize.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calc');exec(compile(ast.Module(body=[node],type_ignores=[]),'inherited_standardize_calc','exec'))
def records(path):return pd.read_csv(path).to_dict('records')
def combine(old,rows,affected):
 old=pd.read_csv(old);keep=old[~old.cohort.isin(affected)] if 'cohort' in old else old.iloc[0:0]
 return pd.concat([pd.DataFrame(rows),keep],ignore_index=True,sort=False)
def released(df):
 """Conservative cross-version protection: mask changed-cohort person/interval counts across all tables."""
 df=df.copy()
 if 'cohort' in df:
  affected=df.cohort.isin(['CHARLS','ELSA'])
  for col in df:
   if col=='n' or col.endswith('_n') or col in ['first_ADL','death_first','death_alive_conflict','alive_or_interview_ADL_unknown','status_unavailable_or_other','administrative_end','event','early_censor','continue_event_free','events_per_parameter','observed_rate','low_grip_proportion','fraction_clipped']:
    df[col]=df[col].astype(object);df.loc[affected&df[col].notna(),col]='SUPPRESSED_CROSS_VERSION'
 return df
exports={};models=[];multi=[];descr=[];audit=[];std=[];flow=[];lower_rows=[];qc_counts=[]
for c in ['CHARLS','ELSA']:
 print(c,'loading approved QC data',flush=True);h,d,fb=q.qc_load(c)
 qq,base,bv,flags,bad,conflict=q.measurement(c);assert not (base&conflict).any()
 flow.append({'cohort':c,'baseline_wave':f.BASE[c],'eligible_grip_n':f.suppress(len(d)),'positive_weight_n':f.suppress(d.weight.notna().sum()),'baseline_ADL_zero_n':f.suppress(f.adl(h,c,f.BASE[c]).reindex(d.index).eq(0).sum()),'source_psu_fallback_n':f.suppress(fb)})
 descr.extend([r for r in f.describe(h,d,c) if r['measure'] in ['EWGSOP2','SDOC']])
 dg=d[d.weight.notna()].copy();dg['age5']=(dg.age//5*5).astype(int);dg['sex']=dg.sex.astype(int);hu=h.loc[h.weight.notna(),['psu','stratum','design_note']].copy()
 oldstd=pd.read_csv(V11/'standardized_low_grip.csv')
 for country,g in dg.groupby('country_name'):
  full=oldstd[(oldstd.cohort==c)&(oldstd.country==country)&(oldstd.age_scope=='65plus')].copy();assert full.status.eq('NOT_ESTIMABLE_EMPTY_STANDARD_STRATUM').all();full['n']=len(g);full['post_hoc']=TAG;std+=full.to_dict('records')
  for label,rf in refs.items():std.append(calc(hu,g[g.age5.between(65,80)],rf,c,country,label,'65-84'))
 if c=='ELSA':
  print('ELSA description complete; no mortality model admitted',flush=True);del h,d;continue
 dd,_=f.intervals(h,d,c);cl.flow_audit(h,d,c,dd);ex=f.early_exclusion(h,d,c)
 for kind in ['composite','death']:
  dd,aa=f.intervals(h,d,c,kind);audit+=aa
  for label,exposure,sub in [('primary','low_EWGSOP2',dd),('SDOC','low_SDOC',dd),('continuous_5kg','grip5',dd),('exclude_first_year_death','low_EWGSOP2',dd[~dd.id.isin(ex)])]:
   row=f.fitmodel(sub,c,kind,label,exposure);models.append(row);cl.OLD=pd.DataFrame(models);cl.glm_diagnostics(sub,c,kind,label,exposure)
  if kind=='composite':
   multi+=f.multistate(dd,c);cl.OLDMS=pd.DataFrame(multi);cl.cif_model(dd,c)
 print('CHARLS frozen-specification models and diagnostics complete',flush=True)
 primary=next(r for r in models if r['cohort']==c and r['analysis']=='primary' and r['outcome']=='composite');ph.PRIMARY_REFERENCE[c]=pd.Series(primary)
 z,X,sc,orig=ph.ipcw(h,d,c);X.design_info=patsy.dmatrix(ph.FORM,z,return_type='dataframe').design_info
 ph.absolute_risk(z,X,sc,orig,c);ph.baseline_table(h,d,c,z)
 alld=ph.all_baseline(h,c);dd,_=f.intervals(h,alld[alld.eligible.eq('yes')],c);r=f.fitmodel(dd,c,'composite','FNIH','low_FNIH');r['post_hoc']=TAG;ph.MODELS.append(r);ph.interaction(h,alld,c)
 # Approved alternative 1.20m lower bound, no new upper bound and no other model changes.
 height=f.num(qq,'r1mheight');extra=height.lt(1.2-q.TOL)&~bad;ids=set(qq.loc[extra,'id']);h2=h.copy();h2.loc[h2.id.isin(ids),'r1mbmi']=np.nan
 d2=ph.all_baseline(h2,c);d2=d2[d2.eligible.eq('yes')];dd2,_=f.intervals(h2,d2,c);r=f.fitmodel(dd2,c,'composite','height_min_1_20m');r['post_hoc']=TAG;lower_rows.append(r)
 qc_counts.append({'cohort':c,'rule':'additional_height_below1_20_after_1_00_QC','age65_n':q.sup((base&bv&extra).sum()),'outcome_independent':True})
 print('CHARLS IPCW, absolute risk, interaction, FNIH and 1.20m models complete',flush=True)
 del h,d,h2,d2,dd,dd2,alld,z,X
allmodels=combine(V1/'A_models.csv',models,['CHARLS']);exports['A_models.csv']=allmodels
exports['A_multistate.csv']=combine(V1/'A_multistate.csv',multi,['CHARLS'])
old_desc=pd.read_csv(V1/'A_weighted_descriptions.csv');old_desc=old_desc[old_desc.measure.isin(['EWGSOP2','SDOC'])&~old_desc.cohort.isin(['CHARLS','ELSA'])];exports['A_weighted_descriptions.csv']=pd.concat([pd.DataFrame(descr),old_desc],ignore_index=True)
exports['A_flow.csv']=combine(V1/'A_flow.csv',flow,['CHARLS','ELSA']);exports['A_followup_audit.csv']=combine(V1/'A_followup_audit.csv',audit,['CHARLS'])
metas=[]
for kind in ['composite','death']:
 for label in ['primary','SDOC','continuous_5kg','exclude_first_year_death']:
  ss=allmodels[(allmodels.outcome==kind)&(allmodels.analysis==label)];metas.append(f.meta(ss.to_dict('records'),kind+'_'+label))
 ss=allmodels[(allmodels.outcome==kind)&(allmodels.analysis=='primary')&allmodels.cohort.isin(['CHARLS','ELSA','MHAS'])];metas.append(f.meta(ss.to_dict('records'),kind+'_measured_BMI'))
for r in metas:r['PI_df']=r['k']-2 if r['k']>=3 else np.nan
exports['A_meta.csv']=pd.DataFrame(metas)
others=allmodels[(allmodels.outcome=='composite')&(allmodels.analysis=='primary')&allmodels.cohort.isin(['MHAS','SHARE'])].to_dict('records');lr=f.meta(lower_rows+others,'composite_height_min_CHARLS1_20m');lr['PI_df']=lr['k']-2 if lr['k']>=3 else np.nan
exports['height_1_20_sensitivity.csv']=pd.DataFrame(lower_rows+[dict(lr,cohort='POOLED')]);pd.DataFrame(qc_counts).to_csv(O/'QC_height1_20_additional_counts.csv',index=False)
for file,rows in [('posthoc_models',ph.MODELS),('IPCW_weight_distribution',ph.WEIGHTS),('IPCW_bootstrap_diagnostics',ph.BOOT),('absolute_risk_9year',ph.RISK),('effect_modification',ph.INTER),('effect_modification_flow',ph.FLOW),('numerical_checks',ph.CHECK),('baseline_inclusion_descriptions',ph.BASE)]:exports[file+'.csv']=combine(V11/(file+'.csv'),rows,['CHARLS'])
pm=[]
for table,labels in [('posthoc_models.csv',['IPCW','FNIH']),('effect_modification.csv',['interaction_HR_ratio','eligible_HR','ineligible_HR'])]:
 for label in labels:
  ss=exports[table];mm=f.meta(ss[ss.analysis==label].to_dict('records'),label);mm['post_hoc']=TAG;mm['PI_df']=mm['k']-2 if mm['k']>=3 else np.nan
  if table.startswith('effect') and mm['status']=='OK':mm['p']=float(2*t.sf(abs(mm['log_effect']/mm['se']),mm['k']-1))
  pm.append(mm)
exports['posthoc_meta.csv']=pd.DataFrame(pm)
for file,rows in [('model_diagnostics',cl.DIAG),('PH_diagnostics',cl.PH),('apparent_calibration',cl.CAL),('CIF_intervals',cl.CIF),('enrollment_flow',cl.FLOW),('wave_censoring',cl.CENS),('missing_covariates',cl.MISS),('CIF_validation',cl.VALID),('terminal_followup',cl.SUMMARY)]:
 if file=='wave_censoring':rows=[r for r in rows if r['population']=='baseline_ADL_zero']
 if file=='terminal_followup':rows=[r for r in rows if r['population']=='primary_model']
 exports[file+'.csv']=combine(CLOSE/(file+'.csv'),rows,['CHARLS'])
exports['standardized_low_grip.csv']=combine(V11/'standardized_low_grip.csv',std,['CHARLS','ELSA'])
oldsup=pd.read_csv(V11/'standardization_support.csv');oldsup=oldsup[(~oldsup.cohort.isin(['CHARLS','ELSA']))|oldsup.age_scope.eq('65plus')];exports['standardization_support.csv']=pd.concat([pd.DataFrame(support),oldsup],ignore_index=True)
ss=exports['standardized_low_grip.csv'];wide=ss[ss.age_scope.eq('65-84')&ss.status.eq('OK')].pivot(index='country',columns='standard',values='proportion').dropna();rho,_=spearmanr(wide['WHO_2000_2025'],wide['pooled_five_cohorts']);rank=wide.rank(ascending=False);wide['WHO_rank']=rank['WHO_2000_2025'];wide['pooled_rank']=rank['pooled_five_cohorts'];wide['rank_difference']=wide.WHO_rank-wide.pooled_rank;wide.to_csv(O/'standard_population_rank_comparison.csv')
(O/'standard_population_rank_summary.json').write_text(json.dumps({'countries_n':len(wide),'Spearman_rho':float(rho),'same_exact_order':bool((wide.rank_difference==0).all()),'max_absolute_rank_shift':float(wide.rank_difference.abs().max()),'scope':'65-84 common publishable countries','reference_sha256_unchanged':refhash},indent=2))
for file,df in exports.items():
 if 'post_hoc' in df and 'cohort' in df:df.loc[df.cohort.isin(['CHARLS','ELSA']),'post_hoc']=TAG
 released(df).to_csv(O/file,index=False)
assert hashlib.sha256(refpath.read_bytes()).hexdigest()==refhash
(O/'execution_receipt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'HRS_read':False,'UKB_used':False,'individual_rows_written':False,'refitted_prognosis_cohorts':['CHARLS'],'refitted_description_cohorts':['CHARLS','ELSA'],'zero_boundary_no_refit':['MHAS'],'self_reported_BMI_reused':['SHARE','KLoSA'],'reference_sha256':refhash,'input_manifest':f.MANIFEST,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'posthoc_adapter_only_changes':'current primary reference in IPCW; output directory; revision tag; source functions and bootstrap settings unchanged'},ensure_ascii=False,indent=2))
print('QC_COMPUTATION_COMPLETE',flush=True)
