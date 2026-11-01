"""Authorized non-HRS K1 only. Explicit input allowlist. No individual output."""
from pathlib import Path
import json, re, hashlib, datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/A_B_validation_v2_20261026'
SPEC={
 'CHARLS':('H_CHARLS_D_Data.dta','ID',[1,2,3],'mbmi'),
 'ELSA':('h_elsa_g3.dta','idauniq',[2,4,6],'mbmi'),
 'MHAS':('H_MHAS_c2.dta','unhhidnp',[3],'mbmi'),
 'SHARE':('H_SHARE_f2.dta','mergeid',[1,2,4,5,6,7,8],'bmi'),
 'KLoSA':('H_KLoSA_e2.dta','pid',list(range(1,9)),'bmi')}
def num(h,col):
 return pd.to_numeric(h[col],errors='coerce') if col in h else pd.Series(np.nan,index=h.index)
def bn(h,col):return num(h,col).where(lambda x:x.isin([0,1]))
def suppress(n):return '<10' if n<10 else int(n)
def load(cohort):
 name,idcol,waves,bmi=SPEC[cohort]
 schema=json.loads((ROOT/'work/nonHRS_selected_schema.json').read_text())[name]
 p=Path(schema['path'])
 assert p.name==name and not re.search(r'hrs|rand',str(p),re.I)
 suffix=['agey','agey_e','iwstat',bmi,'hibpe','diabe','hearte','stroke','hrtatte','hchole','dyslipe']+[f'{hand}grip{t}' for hand in ['l','r'] for t in range(1,4)]
 cols=[c for c in [idcol,'ragender','country']+[f'r{w}{s}' for w in waves for s in suffix] if c in schema['labels']]
 h=pd.read_stata(p,columns=cols,convert_categoricals=False)
 assert h[idcol].notna().all() and not h[idcol].duplicated().any(), 'ID not unique'
 return h,idcol,waves,bmi
def derive(h,cohort,idcol,w,bmi):
 d=pd.DataFrame(index=h.index);d['id']=h[idcol];d['wave']=w
 d['country']=num(h,'country') if cohort=='SHARE' else cohort
 d['age']=num(h,f'r{w}agey').combine_first(num(h,f'r{w}agey_e'))
 d['sex']=num(h,'ragender');d['bmi']=num(h,f'r{w}{bmi}').where(lambda x:x>0)
 grips=pd.DataFrame({f'{hand}{t}':num(h,f'r{w}{hand}grip{t}').where(lambda x:x.between(0,100)) for hand in ['l','r'] for t in range(1,4)})
 d['grip']=grips.max(axis=1)
 q=lambda s:bn(h,f'r{w}{s}')
 d['nondiabetes_comorb']=pd.concat([q('hibpe'),q('hrtatte' if cohort=='MHAS' else 'hearte'),q('stroke'),q('hchole'),q('dyslipe')],axis=1).eq(1).any(axis=1)
 d['any_comorb']=d.nondiabetes_comorb|q('diabe').eq(1)
 d['low_EWGSOP2']=d.grip<np.where(d.sex.eq(1),27,16)
 d['low_SDOC']=d.grip<np.where(d.sex.eq(1),35.5,20)
 ok=d.age.ge(65)&d.sex.isin([1,2])&d.bmi.notna()&d.grip.notna()&num(h,f'r{w}iwstat').eq(1)
 return d.loc[ok].copy()
def eligibility(d,lower=27,upper=30):
 x=d.copy();x['eligible']=np.where(x.bmi>=upper,'yes',np.where(x.bmi<lower,'no',np.where(x.any_comorb,'yes','unknown')))
 x['strict_eligible']=(x.bmi>=upper)|((x.bmi>=lower)&x.nondiabetes_comorb)
 return x
def main():
 rows=[];flow=[];gate=[];qc=[]
 for cohort in SPEC:
  h,idcol,waves,bmi=load(cohort)
  ds=[derive(h,cohort,idcol,w,bmi) for w in waves]
  d=pd.concat(ds).sort_values('wave',kind='stable').drop_duplicates('id',keep='first')
  assert not d.id.duplicated().any()
  for w,x in zip(waves,ds):flow.append({'cohort':cohort,'wave':w,'age65_bmi_grip_sex_n':suppress(len(x)),'unique_first_visit_n':suppress(int(d.wave.eq(w).sum()))})
  for label,lower,upper in [('Wegovy',27,30)]+([('tirzepatide_China',24,28)] if cohort=='CHARLS' else []):
   x=eligibility(d,lower,upper)
   gate.append({'cohort':cohort,'label':label,'eligible_proxy_n':suppress(int(x.eligible.eq('yes').sum())),'strict_lower_n':suppress(int(x.strict_eligible.sum())),'possible_upper_n':suppress(int(x.bmi.ge(lower).sum())),'evaluated_n':suppress(len(x))})
   for cut in ['EWGSOP2','SDOC']:
    for state in ['yes','no','unknown']:
     for low in [False,True]:
      rows.append({'cohort':cohort,'label':label,'cutpoint':cut,'eligibility':state,'low_grip':int(low),'n':suppress(int((x.eligible.eq(state)&x['low_'+cut].eq(low)).sum()))})
  qc.append({'cohort':cohort,'input_basename':SPEC[cohort][0],'rows':suppress(len(h)),'duplicate_id':False,'grip_aggregation':'max_valid_raw_trials_both_hands','participant_rows_written':False})
  print(cohort,'completed aggregate counts',flush=True)
 for name,data in [('A_K1_crosscounts.csv',rows),('A_K1_flow.csv',flow),('A_K1_gate_counts.csv',gate)]:pd.DataFrame(data).to_csv(OUT/name,index=False,encoding='utf-8-sig')
 primary=[g for g in gate if g['label']=='Wegovy']
 # Conservatively count suppressed cells as zero for proving threshold.
 total=sum(g['strict_lower_n'] if isinstance(g['strict_lower_n'],int) else 0 for g in primary)
 ncohort=sum(isinstance(g['strict_lower_n'],int) and g['strict_lower_n']>=10 for g in primary)
 result={'status':'PASS' if total>=3000 and ncohort>=4 else 'HOLD','strict_lower_total_at_least':total,'cohorts_with_confirmed_at_least_10':ncohort,'HRS':'NOT_READ_USER_RUN_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'qc':qc}
 (OUT/'A_K1_gate_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
