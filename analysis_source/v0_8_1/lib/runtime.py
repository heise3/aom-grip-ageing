"""Bound local non-HRS reader. Imported metadata only; data access is explicit."""
from pathlib import Path
import json,re,sys,os
from functools import lru_cache
import numpy as np,pandas as pd
import frozen_core as f
import frozen_qc as qc

B=Path(__file__).resolve().parents[1]
ALLOWED={'CHARLS','ELSA','MHAS','SHARE','KLoSA'}
READS=[]
class SafeStop(RuntimeError):pass

def require(ok,code):
    if not bool(ok):raise SafeStop(code)

def configure():
    f.SCHEMA=json.loads((B/'resources/nonHRS_schema.json').read_text())
    f.INV=json.loads((B/'resources/nonHRS_inventory.json').read_text())
    f.read=read;f.extend_followup=bridge
    qc.measurement.cache_clear()

def sources():return json.loads((B/'resources/input_sources.json').read_text())

def validate_source(p):
    p=Path(p);name=p.name
    require(not re.search(r'hrs|rand',name,re.I),'HRS_RAND_SOURCE_FORBIDDEN')
    known={str(Path(x['path'])):x for x in sources()}
    require(str(p) in known,'SOURCE_OUTSIDE_FROZEN_ALLOWLIST')
    require(p.is_file(),'MISSING_INPUT:'+name)
    require(not re.search(r'hrs|rand',p.resolve().name,re.I),'HRS_RAND_TARGET_FORBIDDEN')
    st=p.stat();k=known[str(p)]
    require(st.st_size==k['size'] and st.st_mtime_ns==k['mtime_ns'],'SOURCE_VERSION_MISMATCH:'+name)
    return p

def install_data_audit_guard():
    """Covers StataReader metadata opens as well as the explicit data reader."""
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        try:p=Path(os.fsdecode(args[0]))
        except (TypeError,ValueError):return
        if p.suffix.lower() in ['.dta','.sav','.rds','.sas7bdat']:
            require(p.suffix.lower()=='.dta','UNAPPROVED_DATA_FORMAT')
            validate_source(p)
    sys.addaudithook(audit)

def read(p,cols):
    p=validate_source(p)
    # These analyses do not read calendar death/interview dates.
    cols=[x for x in cols if not re.match(r'^(radyear|radmonth|r\d+(iwy|iwm|iwindy|iwindm))$',x,re.I)]
    READS.append({'basename':p.name,'columns':list(cols),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
    return pd.read_stata(p,columns=cols,convert_categoricals=False)

def bridge(h,c):
    require(c in ALLOWED,'COHORT_FORBIDDEN')
    sc=json.loads((B/'resources/wave_extensions.json').read_text())
    if c=='CHARLS':
        a=read(sc['/2020charls/Health_Status_and_Functioning.dta']['path'],['ID','db001','db003','db005','db007','db009']).set_index('ID')
        z=read(sc['/2020charls/Sample_Infor.dta']['path'],['ID','died']).set_index('ID')
        require(a.index.is_unique and z.index.is_unique,'NONUNIQUE_CHARLS_BRIDGE')
        a=a.reindex(h.ID);z=z.reindex(h.ID);a.index=h.index;z.index=h.index
        h['r5iwstat']=np.where(z.died.eq(1),5,np.where(z.died.eq(0)&a.notna().any(axis=1),1,np.where(z.died.eq(0),4,np.nan)))
        for raw,s in [('db001','dressa'),('db003','batha'),('db005','eata'),('db007','beda'),('db009','toilta')]:h['r5'+s]=a[raw].map({1:0,2:1,3:1,4:1})
        f.MAXW[c]=5;f.YEARS[c][5]=2020
    elif c=='MHAS':
        names=['h13_21','h16a_21','h17a_21','h18a_21','h19a_21']
        a=read(sc['/Core survey Data/sect_a_c_d_e_pc_f_h_i_2021.dta']['path'],['unhhid','np']+names).set_index(['unhhid','np'])
        z=read(sc['/Master Follow-up File/master_follow_up_file_2021.dta']['path'],['unhhid','np','fallecido_21','tipent_21']).set_index(['unhhid','np'])
        require(a.index.is_unique and z.index.is_unique,'NONUNIQUE_MHAS_BRIDGE')
        ix=pd.MultiIndex.from_frame(h[['unhhid','np']]);a=a.reindex(ix);z=z.reindex(ix);a.index=h.index;z.index=h.index
        h['r6iwstat']=np.where(z.fallecido_21.isin([1,2])|z.tipent_21.eq(5),5,np.where(z.tipent_21.isin([1,2,3,4]),1,np.where(z.fallecido_21.eq(0),4,np.nan)))
        for raw,s in zip(names,['dressa','batha','eata','beda','toilta']):h['r6'+s]=a[raw].map({1:1,2:0,6:1})
        f.MAXW[c]=6;f.YEARS[c][6]=2021
    elif c=='SHARE':
        p=next(x['path'] for x in f.INV if x['path'].endswith('/sharew9_rel9-0-0_ph.dta'));q=next(x['path'] for x in f.INV if x['path'].endswith('/sharew9_rel9-0-0_cv_r.dta'))
        names=['ph049d1','ph049d3','ph049d4','ph049d5','ph049d6']
        a=read(p,['mergeid']+names).set_index('mergeid');z=read(q,['mergeid','deceased','interview']).set_index('mergeid')
        require(a.index.is_unique and z.index.is_unique,'NONUNIQUE_SHARE_BRIDGE')
        a=a.reindex(h.mergeid);z=z.reindex(h.mergeid);a.index=h.index;z.index=h.index
        h['r9iwstat']=np.where(z.deceased.eq(1)|z.interview.eq(2),5,np.where(z.interview.eq(1),1,np.where(z.deceased.eq(0),4,np.nan)))
        for raw,s in zip(names,['dressa','batha','eata','beda','toilta']):h['r9'+s]=a[raw].where(a[raw].isin([0,1]))
        f.MAXW[c]=9;f.YEARS[c][9]=2022
    # ELSA is descriptive only. KLoSA retains the frozen W4-W8 death chain.
    return h.copy()

@lru_cache(maxsize=1)
def load(c):
    require(c in ALLOWED,'COHORT_FORBIDDEN')
    h,d,_=qc.qc_load(c)
    require(d.id.is_unique,'NONINDEPENDENT_BASELINE_PEOPLE')
    d['low_FNIH']=d.grip/d.bmi<np.where(d.sex.eq(1),1.,.56)
    d['cohort']=c
    return h,d
