"""Measurement-only preflight and frozen QC adapter. No HRS inputs; no outcome-driven selection."""
from pathlib import Path
import sys,json,hashlib,datetime,warnings
import numpy as np,pandas as pd
from functools import lru_cache
O=Path(__file__).resolve().parents[1];R=O.parents[1];OLD=R/'outputs/A_amendment_v1_1_20261027'
sys.path.insert(0,str(R/'work/a_formal'));import run_formal as f
assert hashlib.sha256((O/'A_amendment_v1.2.md').read_bytes()).hexdigest()==json.loads((O/'approval_lock.json').read_text())['sha256']
assert hashlib.sha256((O/'A_amendment_v1.2_appendix_01.md').read_bytes()).hexdigest()==json.loads((O/'appendix_01_lock.json').read_text())['sha256']
MEASURED={'CHARLS':(1,150,False),'ELSA':(6,130,True),'MHAS':(3,150,True)}
TOL=1e-6
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
def sup(n):return 0 if int(n)==0 else '<10' if int(n)<10 else int(n)
@lru_cache(maxsize=8)
def measurement(c,lower=1.0):
 w,upper,inclusive=MEASURED[c];name,idcol,_,_=f.SPEC[c];sc=f.SCHEMA[name]
 cols=[idcol,'ID_w1',f'r{w}agey',f'r{w}agey_e',f'r{w}iwstat',f'r{w}mheight',f'r{w}mweight',f'r{w}mbmi',f'r{w}htcomp',f'r{w}wtcomp']
 q=f.read(sc['path'],[x for x in cols if x in sc['labels']]);q['id']=q[idcol].astype(str)
 age=f.num(q,f'r{w}agey').combine_first(f.num(q,f'r{w}agey_e'));base=age.ge(65)&f.num(q,f'r{w}iwstat').eq(1)
 h=f.num(q,f'r{w}mheight');wt=f.num(q,f'r{w}mweight');b=f.num(q,f'r{w}mbmi');bv=b.gt(0)&np.isfinite(b)
 flags={'height_below_min':h.notna()&h.lt(lower-TOL),'height_above210':h.gt(2.1+TOL),'weight_nonpositive':wt.notna()&wt.le(0),'weight_over_limit':wt.gt(upper+TOL) if inclusive else wt.ge(upper-TOL)}
 flags['BMI_present_measure_missing']=bv&(~np.isfinite(h)|~np.isfinite(wt));calc=wt/h**2
 flags['BMI_formula_conflict']=bv&h.gt(0)&wt.gt(0)&np.isfinite(calc)&((b-calc).abs()>1e-6+1e-5*calc.abs())
 flags['completion_conflict']=bv&(f.num(q,f'r{w}htcomp').eq(0)|f.num(q,f'r{w}wtcomp').eq(0))
 flags['source_link_or_unit_conflict']=pd.Series(False,index=q.index)
 flags['explicit_not_completed']=pd.Series(False,index=q.index)
 if c=='CHARLS':
  path=next(x['path'] for x in f.INV if x['path'].endswith('/2011charls/biomarkers.dta'))
  with pd.io.stata.StataReader(path) as reader:labels=reader.variable_labels()
  raw=f.read(path,[x for x in ['ID','qi002','ql002','pi001','pl001','pl002'] if x in labels]);assert raw.ID.is_unique
  raw=raw.set_index('ID').reindex(q.ID_w1.astype(str));raw.index=q.index
  rh=raw.qi002.where(raw.qi002.between(1,2),(raw.qi002/100).where(raw.qi002.between(3,200)))
  flags['source_link_or_unit_conflict']=bv&(~np.isfinite(rh)|~np.isfinite(raw.ql002)|((rh-h).abs()>1e-6+1e-5*h.abs())|((raw.ql002-wt).abs()>1e-6+1e-5*wt.abs()))
  flags['explicit_not_completed']=raw.pi001.eq(5)|raw.pl001.eq(5)|raw.pl002.eq(5)
  if 'pi001' in raw:flags['completion_conflict']|=bv&raw.pi001.eq(5)
  if 'pl001' in raw:flags['completion_conflict']|=bv&raw.pl001.eq(5)&~flags['weight_over_limit']
  if 'pl002' in raw:flags['completion_conflict']|=bv&raw.pl002.eq(5)
 bad=pd.concat([flags[k] for k in ['height_below_min','height_above210','weight_nonpositive','weight_over_limit','explicit_not_completed']],axis=1).any(axis=1)
 conflict=pd.concat([flags[k] for k in ['BMI_present_measure_missing','BMI_formula_conflict','completion_conflict','source_link_or_unit_conflict']],axis=1).any(axis=1)
 conflict &= ~bad
 return q,base,bv,flags,bad,conflict

def qc_load(c,lower=1.):
 """Apply measurement validity before every original eligibility derive, without changing original code."""
 if c not in MEASURED:return f.load(c)
 q,base,bv,flags,bad,conflict=measurement(c,lower);assert not (base&conflict).any(),'Unresolved measurement-source conflict'
 invalid_ids=set(q.loc[bad,'id']);original=f.derive
 def derive_qc(h,cohort,idcol,w,bmi):
  if cohort==c:
   h=h.copy();mask=h[idcol].astype(str).isin(invalid_ids);h.loc[mask,f'r{w}{bmi}']=np.nan
  return original(h,cohort,idcol,w,bmi)
 f.derive=derive_qc
 try:h,d,a=f.load(c)
 finally:f.derive=original
 h.loc[h.id.isin(invalid_ids),f'r{f.BASE[c]}mbmi']=np.nan
 return h,d,a

if __name__=='__main__':
 rows=[];flows=[];decisions=[]
 for c,(w,u,inc) in MEASURED.items():
  q,base,bv,flags,bad,conflict=measurement(c)
  for name,m in flags.items():rows.append({'cohort':c,'baseline_wave':w,'rule':name,'all_age65_interviewed_count':sup((base&m).sum()),'previously_positive_BMI_count':sup((base&bv&m).sum()),'overlap':'nonexclusive','height_min':1.,'weight_max':u,'weight_max_included':inc})
  remaining=base.copy();flow={'cohort':c,'interviewed_age65_n':sup(base.sum()),'existing_unavailable_BMI_n':sup((base&~bv).sum())};remaining&=bv
  for k in ['height_below_min','height_above210','weight_nonpositive','weight_over_limit','explicit_not_completed']:
   flow[k+'_excluded_n']=sup((remaining&flags[k]).sum());remaining&=~flags[k]
  flow['source_conflict_n']=sup((remaining&conflict).sum());flow['new_QC_excluded_n']=sup((base&bv&bad).sum());flow['retained_valid_BMI_n']= 'SUPPRESSED_COMPLEMENT' if 0<(base&bv&bad).sum()<10 else sup((remaining&~conflict).sum());flows.append(flow)
  decision={'cohort':c,'boundary_cases_present':bool((base&bad).any()),'new_exclusion_present':bool((base&bv&bad).any()),'source_conflict_present':bool((base&conflict).any()),'status':'HOLD_SOURCE_CONFLICT' if (base&conflict).any() else 'REFIT_AFFECTED_MODULES' if (base&bad).any() else 'ZERO_CASES_REUSE_NO_REFIT'};decisions.append(decision)
 pd.DataFrame(rows).to_csv(O/'QC_rule_counts_resolved_before_models.csv',index=False);pd.DataFrame(flows).to_csv(O/'QC_ordered_counts_resolved_before_models.csv',index=False)
 (O/'QC_resolved_preflight_decision.json').write_text(json.dumps({'date':'2026-10-27','outcome_fields_read':False,'new_outcome_models_run':False,'decisions':decisions,'input_manifest':f.MANIFEST},ensure_ascii=False,indent=2))
 print(pd.DataFrame(flows).to_string(index=False));print(json.dumps(decisions,ensure_ascii=False))
