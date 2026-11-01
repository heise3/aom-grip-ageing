"""Frozen pooled-sample and WHO standardization; no extrapolation across empty cells."""
from pathlib import Path
import sys,json,hashlib,datetime,warnings
import pandas as pd,numpy as np
from scipy.special import expit,logit
from scipy.stats import t,spearmanr
O=Path(__file__).resolve().parents[1];R=O.parents[1];sys.path.insert(0,str(R/'work/a_formal'));import run_formal as f
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
assert hashlib.sha256((O/'A_amendment_v1.1_appendix_01.md').read_bytes()).hexdigest()==json.loads((O/'appendix_01_lock.json').read_text())['sha256']
COHORTS=['CHARLS','ELSA','MHAS','SHARE','KLoSA'];data={};pooled={};WHO={65:2.96,70:2.21,75:1.52,80:.91,85:.44,90:.15,95:.04,100:.005}
for c in COHORTS:
 print(c,'loading description data',flush=True);h,d,_=f.load(c);d=d[d.weight.notna()].copy();d['age5']=(d.age//5*5).astype(int);d['sex']=d.sex.astype(int)
 for k,n in d.groupby(['age5','sex']).size().items():pooled[tuple(map(int,k))]=pooled.get(tuple(map(int,k)),0)+int(n)
 data[c]=(h.loc[h.weight.notna(),['psu','stratum','design_note']].copy(),d[['country_name','age5','sex','weight','low_EWGSOP2']].copy());del h,d
total=sum(pooled.values());restricted={k:n for k,n in pooled.items() if 65<=k[0]<=80};den=sum(restricted.values());alpha={k:n/den for k,n in restricted.items()};alphafull={k:n/total for k,n in pooled.items()}
whoden=sum(WHO[a] for a in [65,70,75,80]);whorest={(a,s):WHO[a]/whoden/2 for a in [65,70,75,80] for s in [1,2]};whofull={(a,s):WHO[a]/sum(WHO.values())/2 for a in WHO for s in [1,2]}
reference=[{'age_group':f'{a}-{a+4}','sex':s,'pooled_count':n,'pooled_standard_share':alpha[(a,s)],'WHO_standard_share':whorest[(a,s)]} for (a,s),n in sorted(restricted.items())]
assert min(restricted.values())>=10
pd.DataFrame(reference).to_csv(O/'frozen_standard_population_65_84.csv',index=False)
pd.DataFrame([{'age_group':f'{a}-{a+4}','sex':s,'n':f.suppress(n),'share':n/total if n>=10 else 'SUPPRESSED'} for (a,s),n in sorted(pooled.items())]).to_csv(O/'frozen_standard_population_65plus_suppressed.csv',index=False)
fullhash=hashlib.sha256(json.dumps(sorted((a,s,n) for (a,s),n in pooled.items())).encode()).hexdigest()
lock={'locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_standardized_outcome_estimation':True,'full_reference_n':total,'restricted_reference_n':den,'full_unsuppressed_structure_sha256':fullhash,'standard65_84_sha256':hashlib.sha256((O/'frozen_standard_population_65_84.csv').read_bytes()).hexdigest(),'HRS_will_not_change_reference':True,'WHO_version':'World Standard 2000-2025; WHO2001 DP31','WHO_url':'https://seer.cancer.gov/stdpopulations/world.who.html','nonHRS_input_manifest':f.MANIFEST}
(O/'standard_population_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
rows=[];support=[]
def calc(h,g,ref,c,country,label,scope):
 grouped=g.groupby(['age5','sex']);avail=set(grouped.groups);missing=set(ref)-avail
 row={'cohort':c,'country':country,'standard':label,'age_scope':scope,'n':f.suppress(len(g)),'post_hoc':'post hoc, added after reviewing v1 results'}
 support.append({'cohort':c,'country':country,'standard':label,'age_scope':scope,'missing_strata':';'.join(f'{a}-{a+4}:sex{s}' for a,s in sorted(missing)),'missing_reference_mass':sum(ref[k] for k in missing)})
 if missing:row['status']='NOT_ESTIMABLE_EMPTY_STANDARD_STRATUM';return row
 pos=int(g.low_EWGSOP2.sum())
 if min(pos,len(g)-pos)<10:row['status']='SUPPRESSED_SMALL_OUTCOME_CELL';return row
 p=0.;u=pd.Series(0.,index=h.index)
 for k,z in grouped:
  if k not in ref:continue
  w=z.weight.to_numpy();y=z.low_EWGSOP2.to_numpy().astype(float);pr=float(w@y/w.sum());p+=ref[k]*pr;u.loc[z.index]=ref[k]*w*(y-pr)/w.sum()
 frame=h[['psu','stratum']].copy();frame['u']=u;scores=frame.groupby(['stratum','psu']).u.sum();var=0.;df=0;sing=[]
 for _,ss in scores.groupby(level=0):
  m=len(ss)
  if m>1:var+=m/(m-1)*float(((ss-ss.mean())**2).sum());df+=m-1
  else:sing.extend(ss.tolist())
 if sing:var+=float(np.square(np.asarray(sing)-scores.mean()).sum())
 se=float(np.sqrt(var));crit=t.ppf(.975,max(1,df));ci=expit(logit(p)+np.array([-1,1])*crit*se/(p*(1-p)))
 row.update(status='OK',proportion=p,ci_low=float(ci[0]),ci_high=float(ci[1]),se=se,design_df=df,design_note=h.design_note.iloc[0],reference_fixed=True)
 return row
for c,(h,d) in data.items():
 for country,g in d.groupby('country_name'):
  for label,ref in [('pooled_five_cohorts',alphafull),('WHO_2000_2025',whofull)]:
   q=g.copy()
   if label.startswith('WHO'):q.loc[q.age5>=100,'age5']=100
   rows.append(calc(h,q,ref,c,country,label,'65plus'))
  g=g[g.age5.between(65,80)]
  for label,ref in [('pooled_five_cohorts',alpha),('WHO_2000_2025',whorest)]:rows.append(calc(h,g,ref,c,country,label,'65-84'))
out=pd.DataFrame(rows);out.to_csv(O/'standardized_low_grip.csv',index=False);pd.DataFrame(support).to_csv(O/'standardization_support.csv',index=False)
wide=out[(out.age_scope=='65-84')&(out.status=='OK')].pivot(index='country',columns='standard',values='proportion').dropna()
if len(wide)>=3:
 rho,p=spearmanr(wide.iloc[:,0],wide.iloc[:,1]);rank=wide.rank(ascending=False,method='average');wide.columns=['WHO_proportion','pooled_proportion'];wide['WHO_rank']=rank.iloc[:,0];wide['pooled_rank']=rank.iloc[:,1];wide['rank_difference']=wide.WHO_rank-wide.pooled_rank
 result={'status':'OK','countries_n':len(wide),'Spearman_rho':float(rho),'p_value_reported':False,'same_exact_order':bool((wide.rank_difference==0).all()),'max_absolute_rank_shift':float(wide.rank_difference.abs().max()),'country_set':list(wide.index),'scope':'65-84 only; common publishable countries; descriptive ranking'}
else:result={'status':'NOT_ESTIMABLE_TOO_FEW_COMMON_COUNTRIES','countries_n':len(wide)}
wide.to_csv(O/'standard_population_rank_comparison.csv');(O/'standard_population_rank_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
(O/'standardization_execution_receipt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'HRS_read':False,'participant_output':False,'age_scope_change_user_authorized':True,'all_raw_outcomes_in_memory_only':True},indent=2))
print('STANDARDIZATION_COMPLETE',flush=True)
