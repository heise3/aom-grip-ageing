"""A_REPORT: descriptive summaries + authorized crude models and I2 CIs only. No adjusted refits."""
from pathlib import Path
import sys,json,hashlib,csv,datetime,warnings
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from scipy.stats import chi2
from scipy.optimize import brentq
O=Path(__file__).resolve().parents[1];R=O.parents[1];V=R/'outputs/A_QC_v1_2_20261027';sys.path.insert(0,str(V/'code'));import qc_core as qc
f=qc.f
assert hashlib.sha256((O/'execution_contract.md').read_bytes()).hexdigest()==json.loads((O/'execution_contract_lock.json').read_text())['sha256']
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
class ConsistencyStop(Exception):pass
def require(ok,msg):
 if not ok:raise ConsistencyStop(msg)
def forbid(*a,**k):raise RuntimeError('ADJUSTED_MODEL_REFIT_FORBIDDEN')
f.fitmodel=forbid;f.multistate=forbid
# Only previously recorded non-HRS source files may be reopened. Verify file versions before reads.
inputs={}
for p,key in [(R/'outputs/A_formal_v1_20261027/run_receipt.json','manifest'),(R/'outputs/A_amendment_v1_1_20261027/execution_receipt.json','input_manifest'),(V/'QC_resolved_preflight_decision.json','input_manifest'),(V/'execution_receipt.json','input_manifest')]:
 for x in json.loads(p.read_text()).get(key,[]):inputs[str(Path(x['path']))]=x
reader=f.read
def frozen_read(p,cols):
 pp=Path(p);known=inputs.get(str(pp));require(known is not None,'Source absent from frozen input receipts: '+pp.name)
 st=pp.stat();require(st.st_size==known['size'] and st.st_mtime_ns==known['mtime_ns'],'Frozen source metadata changed: '+pp.name)
 return reader(p,cols)
f.read=frozen_read
checks=[];cell_lineage=[]
def readcsv(name):return pd.read_csv(V/name,dtype=str,keep_default_na=False)
def extract(name,newname,mask=None,cols=None):
 df=readcsv(name);sel=df if mask is None else df.loc[mask(df)];cols=list(sel.columns) if cols is None else cols
 cols=[x for x in cols if x not in ['pi_low','pi_high','PI_df']];out=sel[cols].copy();out.to_csv(O/'source_tables'/newname,index=False)
 for j,(ix,row) in enumerate(sel.iterrows(),2):
  for col in cols:cell_lineage.append({'output_file':'source_tables/'+newname,'output_row':j,'output_column':col,'source_file':str((V/name).relative_to(R)),'source_row':int(ix)+2,'source_column':col,'expected':row[col]})
 return out
adj=extract('A_models.csv','adjusted_primary.csv',lambda d:d.outcome.eq('composite')&d.analysis.eq('primary')&d.cohort.isin(['CHARLS','MHAS','SHARE']))
fn=extract('posthoc_models.csv','adjusted_FNIH.csv',lambda d:d.analysis.eq('FNIH'))
ma=extract('A_meta.csv','adjusted_primary_meta.csv',lambda d:d.analysis.eq('composite_primary'))
fm=extract('posthoc_meta.csv','adjusted_FNIH_meta.csv',lambda d:d.analysis.eq('FNIH'))
extract('posthoc_meta.csv','interaction_summary.csv',lambda d:d.analysis.eq('interaction_HR_ratio'))
extract('absolute_risk_9year.csv','absolute_risk_9year.csv');extract('standardized_low_grip.csv','standardized_low_grip.csv')
extract('QC_ordered_counts_resolved_before_models.csv','QC_ordered_counts.csv');extract('enrollment_flow.csv','enrollment_flow.csv')
# Validate user-specified headline values before any individual data access or new fit.
for table,expected in [(ma,('1.621','1.233','2.132')),(fm,('1.466','0.991','2.168'))]:
 require(tuple(f'{float(table.iloc[0][x]):.3f}' for x in ['effect','ci_low','ci_high'])==expected,'T7 requested headline differs from v1.2')
ii=readcsv('posthoc_meta.csv');ii=ii[ii.analysis.eq('interaction_HR_ratio')].iloc[0];require(f'{float(ii.effect):.3f}'=='0.900' and f'{float(ii.p):.3f}'=='0.306','T7 interaction headline differs')
checks.append({'check':'user_headline_values','status':'PASS','details':'Extracted v1.2 values round exactly to all user-specified values'})
def sup(n):return '<10' if int(n)<10 else str(int(n))
def numstr(x):return '' if not np.isfinite(x) else repr(float(x))
def count(c,n):return 'SUPPRESSED_CROSS_VERSION' if c in ['CHARLS','ELSA'] else sup(n)
continuous=['age','bmi','grip','chronic'];cats=['sex','educ','smoke','diabetes','country','eligible']
label={'age':'年龄（岁）','sex':'性别','educ':'教育（协调原编码）','bmi':'BMI（kg/m²）','smoke':'当前吸烟','diabetes':'糖尿病','chronic':'六项非糖尿病核心慢病数','country':'国家','grip':'握力（kg）','eligible':'可观察标签资格','interval':'访视间隔（时间项）'}
baseold=readcsv('baseline_inclusion_descriptions.csv')
T1=[];S1=[];FUP=[];FLOW=[];CRUDE=[];CRUDE_RAW=[];WAVES=[];PART=[];aggregate_checks=[];fits=[]
def stats_rows(d,c,scope,group,weighted,vars):
 rows=[];ww=d.weight if weighted else pd.Series(1.,index=d.index)
 for var in vars:
  if var=='interval':rows.append({'cohort':c,'scope':scope,'group':group,'variable':var,'label':label[var],'type':'not_baseline','level':'','n':'NOT_APPLICABLE','missing_n':'NOT_APPLICABLE','display':'不适用：模型时间项'});continue
  xx=d[var];valid=xx.notna();n=int(valid.sum());missing=int((~valid).sum());w=ww[valid].to_numpy(float);x=xx[valid]
  common={'cohort':c,'scope':scope,'group':group,'variable':var,'label':label[var],'n':count(c,n),'missing_n':count(c,missing),'weighting':f.WEIGHTS[c] if weighted else 'unweighted','percentage_denominator':'variable_nonmissing','status':'OK'}
  if var in continuous:
   if n<10:rows.append(dict(common,type='continuous',level='',mean='<10',sd='<10',median='<10',q1='<10',q3='<10',display='<10：抑制',status='SUPPRESSED_SMALL_CELL'));continue
   if weighted:
    vals=x.to_numpy(float);mu=np.average(vals,weights=w);sd=np.sqrt(np.average((vals-mu)**2,weights=w));order=np.argsort(vals);a=vals[order];cs=np.cumsum(w[order]);qt=[a[min(np.searchsorted(cs,p*cs[-1],side='left'),len(a)-1)] for p in [.25,.5,.75]]
   else:mu=float(x.mean());sd=float(x.std());qt=x.quantile([.25,.5,.75]).to_numpy()
   means=[numstr(mu),numstr(sd)]
   # Existing overall means/SD are extracted rather than regenerated; check rounded display agrees.
   if not weighted and group=='All':
    old=baseold[baseold.cohort.eq(c)&baseold.included_model.eq('True')]
    if len(old) and var+'_mean' in old:
     old=old.iloc[0];require(f'{mu:.2f}'==f'{float(old[var+"_mean"]):.2f}' and f'{sd:.2f}'==f'{float(old[var+"_sd"]):.2f}','T7 baseline summary differs: '+c+'/'+var)
     means=[old[var+'_mean'],old[var+'_sd']]
   rows.append(dict(common,type='continuous',level='',mean=means[0],sd=means[1],median=numstr(qt[1]),q1=numstr(qt[0]),q3=numstr(qt[2]),display=f'{float(means[0]):.2f} ({float(means[1]):.2f}); {qt[1]:.2f} [{qt[0]:.2f}, {qt[2]:.2f}]'))
  else:
   for level in sorted(x.unique(),key=str):
    m=x.eq(level);nn=int(m.sum());pc=100*float(w[m].sum()/w.sum()) if len(w) else np.nan;ll=str(level)
    if var=='sex':ll={1:'男',2:'女'}.get(level,ll)
    if var in ['smoke','diabetes']:ll={0:'否',1:'是'}.get(level,ll)
    if var=='eligible':ll='符合可观察条件' if level=='yes' else ll
    if var=='country':ll=str(d.loc[x.index[m],'country_name'].iloc[0])
    protected=c in ['CHARLS','ELSA'] or nn<10 or n-nn in range(1,10)
    cn=count(c,nn);pct='SUPPRESSED_COMPLEMENT' if protected else numstr(pc)
    rows.append(dict(common,type='categorical',level=ll,category_n=cn,percent=pct,display=(cn+'（比例抑制）') if protected else f'{nn} ({pc:.1f}%)',status='SUPPRESSED_CROSS_VERSION' if c in ['CHARLS','ELSA'] else 'SUPPRESSED_SMALL_CELL_OR_COMPLEMENT' if protected else 'OK'))
 return rows

def cross_suppress(rows):
 # Keep original <10 threshold; complementary protection across total/group partitions.
 df=pd.DataFrame(rows)
 for (c,v),gg in df[df.type.eq('categorical')].groupby(['cohort','variable']):
  if gg.status.ne('OK').any():
   ix=gg.index
   for j in ix:
    if df.loc[j,'status']=='OK':df.loc[j,'category_n']='SUPPRESSED_COMPLEMENT';df.loc[j,'status']='SUPPRESSED_COMPLEMENT'
   df.loc[ix,'percent']='SUPPRESSED_COMPLEMENT';df.loc[ix,'display']=df.loc[ix,'category_n'].astype(str)+'（比例抑制）'
 return df

def terminals(h,d,c,z):
 last=z.sort_values('interval').groupby('id',sort=False).tail(1).copy();out=[]
 for _,g in last.iterrows():
  w=f.BASE[c]+int(g.interval);reason='administrative_end' if not g.event and w==f.MAXW[c] else 'first_ADL' if g.event and g.state==1 else 'death_first' if g.event else 'early_censor';detail=reason
  if reason=='early_censor':
   ix=d.index[d.id.eq(g.id)][0];v=w+1;s=f.num(h,f'r{v}iwstat').loc[ix];future=any(f.num(h,f'r{k}iwstat').loc[ix] in [1,4] for k in range(v+1,f.MAXW[c]+1))
   detail='death_alive_conflict' if s==5 and future else 'alive_or_interview_ADL_unknown' if s in [1,4] else 'status_unavailable_or_other'
  out.append({'id':g.id,'low_grip':int(g.low_EWGSOP2),'time':f.YEARS[c][w]-f.YEARS[c][f.BASE[c]],'event':int(g.event),'reason':reason,'detail':detail})
 return pd.DataFrame(out)
def reverse_quantiles(g):
 survival=1.;out={.25:'NR',.5:'NR',.75:'NR'}
 for tm in sorted(g.time.unique()):
  nr=int(g.time.ge(tm).sum());ne=int((g.time.eq(tm)&g.event.eq(0)).sum());survival*=1-ne/nr
  for p in out:
   if out[p]=='NR' and 1-survival>=p-1e-12:out[p]=str(int(tm))
 return out

def fit_crude(z,c,exposure,definition,adjusted):
 zz=z.copy();zz['x']=zz[exposure].astype(float);r={'cohort':c,'definition':definition,'outcome':'composite','model':'cloglog; x + C(interval)','sample':'same_final_complete_case_set_as_v1.2_adjusted','weighting':'unweighted','SE':'individual_cluster_robust','adjusted_HR':adjusted.effect,'adjusted_ci_low':adjusted.ci_low,'adjusted_ci_high':adjusted.ci_high,'persons_n':count(c,zz.id.nunique()),'events_n':count(c,zz.event.sum()),'post_hoc':'报告补充（STROBE 16a / 合并异质性），非预设'}
 if any(zz.loc[zz.x.eq(v),'event'].sum()<10 for v in [0,1]):r['status']='NOT_ESTIMABLE_SMALL_EVENT_CELL';return r,None
 with warnings.catch_warnings(record=True) as ws:fit=smf.glm('event ~ x + C(interval)',data=zz,family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster',cov_kwds={'groups':zz.id},maxiter=150)
 b=float(fit.params['x']);se=float(fit.bse['x']);okay=fit.converged and np.isfinite(se) and se>0
 r.update(status='OK' if okay else 'NOT_CONVERGED',crude_HR=numstr(np.exp(b)),crude_ci_low=numstr(np.exp(b-1.96*se)),crude_ci_high=numstr(np.exp(b+1.96*se)),crude_logHR=numstr(b),crude_SE=numstr(se),crude_p=numstr(fit.pvalues['x']))
 fits.append({'cohort':c,'definition':definition,'adjusted_refit':False,'formula':'event ~ x + C(interval)','converged':bool(fit.converged),'rank':int(np.linalg.matrix_rank(fit.model.exog)),'columns':fit.model.exog.shape[1],'warnings':';'.join(sorted(set(type(x.message).__name__ for x in ws)))})
 raw={'cohort':c,'status':r['status'],'log_effect':b,'se':se,'effect':float(np.exp(b)),'ci_low':float(np.exp(b-1.96*se)),'ci_high':float(np.exp(b+1.96*se)),'definition':definition}
 return r,raw

def main():
 for c in ['CHARLS','MHAS','SHARE','ELSA','KLoSA']:
  print(c,'reconstructing frozen analysis/description sets; no adjusted fit',flush=True);h,d,_=qc.qc_load(c)
  w=f.BASE[c];age=f.num(h,f'r{w}agey').combine_first(f.num(h,f'r{w}agey_e'));inter=f.num(h,f'r{w}iwstat').eq(1);der=f.derive(h,c,f.SPEC[c][1],w,f.SPEC[c][3]);desc=d[d.weight.notna()].copy()
  oldflow=readcsv('A_flow.csv');of=oldflow[oldflow.cohort.eq(c)].iloc[0]
  for key,n in [('eligible_grip_n',len(d)),('positive_weight_n',len(desc)),('baseline_ADL_zero_n',f.adl(h,c,w).reindex(d.index).eq(0).sum())]:
   if of[key].replace('.','',1).isdigit():require(int(float(of[key]))==int(n),'T7 frozen flow differs: '+c+'/'+key)
  stages=[('source_harmonized_records',len(h)),('baseline_interviewed',inter.sum()),('baseline_age65', (inter&age.ge(65)).sum()),('sex_BMI_grip_available_after_QC',len(der)),('confirmed_eligible_with_grip',len(d)),('description_positive_weight',len(desc)),('baseline_ADL0',f.adl(h,c,w).reindex(d.index).eq(0).sum())]
  for stage,n in stages:FLOW.append({'cohort':c,'stage':stage,'n':count(c,n) if stage not in ['source_harmonized_records','baseline_interviewed','baseline_age65'] else sup(n),'branch':'description' if stage=='description_positive_weight' else 'prognosis' if stage=='baseline_ADL0' else 'common','note':'source records span multiple waves' if stage=='source_harmonized_records' else ''})
  S1.extend(stats_rows(desc,c,'weighted_description_eligible_positive_weight','All',True,['age','sex','bmi','grip']))
  if c not in ['CHARLS','MHAS','SHARE']:
   FLOW.append({'cohort':c,'stage':'primary_prognosis','n':'NOT_INCLUDED','branch':'prognosis','note':'MORTALITY_UNAVAILABLE' if c=='ELSA' else 'ADL_CONSTRUCT_NOT_EQUIVALENT'});continue
  dd,_=f.intervals(h,d,c,'composite');z=dd.dropna(subset=f.COV+['low_EWGSOP2','event']).copy();ids=set(z.id);g=d[d.id.isin(ids)].copy();require(g.id.is_unique,'Duplicate baseline identity in '+c)
  ar=adj[adj.cohort.eq(c)].iloc[0]
  for key,n in [('persons_n',len(g)),('intervals_n',len(z)),('events_n',z.event.sum())]:
   if ar[key].replace('.','',1).isdigit():require(int(float(ar[key]))==int(n),'T7 primary sample count differs: '+c+'/'+key)
  for stage,n in [('any_observable_interval',dd.id.nunique()),('primary_complete_case',len(g))]:FLOW.append({'cohort':c,'stage':stage,'n':count(c,n),'branch':'prognosis','note':''})
  vars=['age','sex','educ','bmi','smoke','diabetes','chronic','country','interval','grip','eligible']
  for grp,ss in [('Normal',g[~g.low_EWGSOP2]),('Low',g[g.low_EWGSOP2]),('All',g)]:T1.extend(stats_rows(ss,c,'primary_complete_case_ADL0',grp,False,vars))
  tt=terminals(h,d,c,z);require(tt.id.nunique()==len(g) and tt.event.sum()==z.event.sum(),'T7 terminal count mismatch: '+c)
  origterm=readcsv('terminal_followup.csv');origterm=origterm[origterm.cohort.eq(c)&origterm.population.eq('primary_model')].iloc[0]
  for key in ['first_ADL','death_first','death_alive_conflict','alive_or_interview_ADL_unknown','status_unavailable_or_other','administrative_end']:
   n=tt.detail.eq(key).sum();sv=origterm[key]
   if sv.replace('.','',1).isdigit():require(int(float(sv))==int(n),'T7 terminal reason count differs: '+c+'/'+key)
  wavepairs=';'.join(f'W{k}:{f.YEARS[c][k]}' for k in range(w,f.MAXW[c]+1));WAVES.append({'cohort':c,'waves_years':wavepairs,'nominal_max_years':f.YEARS[c][f.MAXW[c]]-f.YEARS[c][w]})
  for grp,ss in [('Normal',tt[tt.low_grip.eq(0)]),('Low',tt[tt.low_grip.eq(1)]),('All',tt)]:
   qq=reverse_quantiles(ss);row={'cohort':c,'group':grp,'n':count(c,len(ss)),'reverse_KM_median_years':qq[.5],'reverse_KM_q1_years':qq[.25],'reverse_KM_q3_years':qq[.75],'reverse_KM_IQR_width_years':str(int(qq[.75])-int(qq[.25])) if qq[.75]!='NR' and qq[.25]!='NR' else 'NR','max_followup_years':str(int(ss.time.max())),'total_nominal_observation_person_years':str(int(ss.time.sum())),'composite_events_n':count(c,ss.event.sum()),'first_ADL_n':count(c,ss.reason.eq('first_ADL').sum()),'death_before_observed_ADL_n':count(c,ss.reason.eq('death_first').sum()),'censored_n':count(c,ss.event.eq(0).sum()),'early_censor_n':count(c,ss.reason.eq('early_censor').sum()),'administrative_end_n':count(c,ss.reason.eq('administrative_end').sum()),'status_unavailable_or_other_n':count(c,ss.detail.eq('status_unavailable_or_other').sum()),'alive_ADL_unknown_n':count(c,ss.detail.eq('alive_or_interview_ADL_unknown').sum()),'death_alive_conflict_n':count(c,ss.detail.eq('death_alive_conflict').sum()),'waves_years':wavepairs,'time_basis':'NOMINAL_WAVE_YEARS_NOT_EXACT_PERSON_TIME','weighting':'unweighted'};FUP.append(row)
  z['low_FNIH']=z.grip/z.bmi<np.where(z.sex.eq(1),1.,.56)
  for cut,expo,arow in [('EWGSOP2','low_EWGSOP2',ar),('FNIH','low_FNIH',fn[fn.cohort.eq(c)].iloc[0])]:
   r,raw=fit_crude(z,c,expo,cut,arow);CRUDE.append(r)
   if raw is not None:CRUDE_RAW.append(raw)
  aggregate_checks.append({'cohort':c,'published_sample_flow_terminal_counts':'PASS','baseline_existing_mean_SD_display':'PASS','source_versions':'PASS','adjusted_model_refit':False})
  print(c,'descriptive and crude-effect aggregates complete',flush=True)
  del h,d,g,z,dd,tt
 # Complementary censor-count suppression across overlapping grouped and total partitions.
 follow=pd.DataFrame(FUP)
 for c,ix in follow.groupby('cohort').groups.items():
  cols=[x for x in follow if x.endswith('_n') or x=='n']
  # v1.2 suppressed small partitions; do not restore by using the overlapping grouped view.
  if follow.loc[ix,cols].astype(str).eq('<10').any().any():
   for col in cols:
    if col!='n':follow.loc[ix,col]=follow.loc[ix,col].where(follow.loc[ix,col].eq('<10'),'SUPPRESSED_COMPLEMENT')
 follow.to_csv(O/'followup_summary.csv',index=False)
 t1=cross_suppress(T1);s1=cross_suppress(S1);t1.to_csv(O/'table1_long.csv',index=False);s1.to_csv(O/'tableS1.csv',index=False)
 for c,gg in t1.groupby('cohort'):
  out=[]
  for (v,lev),bb in gg.fillna('').groupby(['variable','level'],sort=False):
   row={'variable':v,'label':bb.label.iloc[0],'level':lev,'statistic':'均值(SD); 中位数[Q1,Q3]' if bb.type.iloc[0]=='continuous' else 'n(%)' if bb.type.iloc[0]=='categorical' else '不适用'}
   for gp in ['Normal','Low','All']:
    xx=bb[bb.group.eq(gp)];row[gp]=xx.display.iloc[0] if len(xx) else '<10：抑制';row[gp+'_missing_n']=xx.missing_n.iloc[0] if len(xx) else '<10'
   out.append(row)
  pd.DataFrame(out).to_csv(O/f'table1_{c}.csv',index=False)
 pd.DataFrame(FLOW).to_csv(O/'flow_counts.csv',index=False);pd.DataFrame(WAVES).to_csv(O/'followup_waves.csv',index=False)
 for cut,meta in [('EWGSOP2',ma.iloc[0]),('FNIH',fm.iloc[0])]:
  new=f.meta([r for r in CRUDE_RAW if r['definition']==cut],cut+'_crude')
  CRUDE.append({'cohort':'POOLED','definition':cut,'outcome':'composite','model':'REML_modified_Hartung_Knapp','sample':'same_frozen_complete_case_sets','weighting':'inverse_sampling_variance_random_effects','status':new['status'],'crude_HR':numstr(new.get('effect',np.nan)),'crude_ci_low':numstr(new.get('ci_low',np.nan)),'crude_ci_high':numstr(new.get('ci_high',np.nan)),'crude_logHR':numstr(new.get('log_effect',np.nan)),'crude_SE':numstr(new.get('se',np.nan)),'adjusted_HR':meta.effect,'adjusted_ci_low':meta.ci_low,'adjusted_ci_high':meta.ci_high,'post_hoc':'报告补充（STROBE 16a / 合并异质性），非预设','k':new['k']})
 pd.DataFrame(CRUDE).to_csv(O/'crude_vs_adjusted.csv',index=False);pd.DataFrame(fits).to_csv(O/'crude_model_diagnostics.csv',index=False)
 hetero=[]
 for cut,rows,meta in [('EWGSOP2',adj,ma.iloc[0]),('FNIH',fn,fm.iloc[0])]:
  y=rows.log_effect.astype(float).to_numpy();v=rows.se.astype(float).to_numpy()**2;k=len(y);w=1/v;vtyp=(k-1)/(w.sum()-(w@w)/w.sum())
  def qfun(tau):
   a=1/(v+tau);mu=np.sum(a*y)/sum(a);return float(np.sum(a*(y-mu)**2))
  def bound(target):
   if qfun(0)<=target:return 0.
   upper=1.
   while qfun(upper)>target:upper*=2
   return brentq(lambda x:qfun(x)-target,0,upper,xtol=1e-13)
  lo=bound(chi2.ppf(.975,k-1));hi=bound(chi2.ppf(.025,k-1));cil=100*lo/(lo+vtyp);cih=100*hi/(hi+vtyp)
  hetero.append({'definition':cut,'k':meta.k,'model':'random_effects','tau2_estimator':'REML','pooled_CI':'modified_Hartung_Knapp_scale_at_least1','Q':meta.Q,'Q_p':meta.Q_p,'I2_percent':meta.I2,'I2_ci_low':numstr(cil),'I2_ci_high':numstr(cih),'tau2':meta.tau2,'existing_values_origin':'提取自 v1.2','I2_CI_origin':'报告补充（STROBE 16a / 合并异质性），非预设','I2_CI_method':'Q_profile_tau2_inversion_then_typical_variance_transform','I2_point_method':'original_Cochran_Q_not_replaced','prediction_interval':'NOT_REPORTED_K3','reference':'https://wviechtb.github.io/metafor/reference/confint.html'})
 pd.DataFrame(hetero).to_csv(O/'heterogeneity.csv',index=False)
 pd.DataFrame(aggregate_checks).to_csv(O/'analysis_set_checks.csv',index=False)
 (O/'source_cell_lineage.json').write_text(json.dumps(cell_lineage,ensure_ascii=False,indent=2))
 (O/'execution_receipt.json').write_text(json.dumps({'date':'2026-10-27','adjusted_models_refitted':0,'crude_models_fit':len(fits),'individual_rows_written':False,'HRS_read':False,'input_manifest':f.MANIFEST,'contract_sha256':hashlib.sha256((O/'execution_contract.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
 (O/'followup_method_note.md').write_text('样本为v1.2主复合结局最终完整案例；按EWGSOP2分组，不加调查权重。\n事件取首次可判定事件波次端点，提前删失取最后有效波次端点，行政截止取末次纳入波次。\n反向KM将原删失作为事件、原复合事件作为删失；分位数未达到记NR。\n时间使用冻结的名义波次年份；总观测人年为个体名义时长之和，不是精确实际人年。\n死亡指未先观察到ADL者的死亡；提前删失保留状态缺失、ADL未知及矛盾状态的原分类。\n')
 print('REPORT_COMPUTATION_COMPLETE; NO_ADJUSTED_REFITS',flush=True)
if __name__=='__main__':
 try:main()
 except ConsistencyStop as e:
  (O/'T7_STOP.json').write_text(json.dumps({'status':'STOP','reason':str(e),'automatic_correction':False},ensure_ascii=False,indent=2));print('T7_STOP:',str(e),flush=True);sys.exit(2)
