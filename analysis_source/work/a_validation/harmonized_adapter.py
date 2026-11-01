"""No I/O: documented harmonized fields -> nullable K1 components.

HRS invocation is exclusively for the human-run loader. This module can be
tested using wholly synthetic inputs and used for authorized non-HRS cohorts.
"""
import numpy as np
import pandas as pd
from k1_common_engine import s,b,bor,ADL6,ADL5,COMMON6

ADLMAP={'bath':'batha','dress':'dressa','eat':'eata','bed':'beda','toilet':'toilta','room':'walkra'}
HELPMAP={'bath':'bathehlp','dress':'dresshlp','eat':'eathlp','bed':'bedhlp','toilet':'toilethlp','room':'walkhlp'}
IADLMAP={'meals':'mealsa','money':'moneya','meds':'medsa','shop':'shopa','phone':'phonea','housework':'housewka'}
CHRONMAP={'htn':'hibpe','cancer':'cancre','lung':'lunge','heart':'hearte','stroke':'stroke','arthritis':'arthre'}
SUFFIXES=list(set(list(ADLMAP.values())+list(HELPMAP.values())+list(IADLMAP.values())+list(CHRONMAP.values())+['agey','agey_e','diabe','rxdiab','rxdiabi','rxdiabo','nhmliv','iwstat','respe','hrtatte','angine','conhrtfe','alzhe','demene','alzdeme','memrye','kidneye','psyche','urinai','urinaf','urincgh2y','urinurg2y','urinpad','imrc','dlrc','imrc8','dlrc8','orient','orient_m','mbmi','bmi','smoken','fall','fall1y','fall_s','lbronn','loxygen']))

def harmonized_columns(waves):
    return ['ragender','raeducl','raeduc','raedyrs','ID','ID_w1','idauniq','mergeid','unhhid','unhhidnp','np','hhidpn']+[f'r{w}{t}' for w in waves for t in SUFFIXES]

def first(d,cols):
    x=pd.Series(np.nan,index=d.index,dtype=float)
    for col in cols:x=x.combine_first(s(d,col))
    return x

def canonical(h,cohort,w,year,idcol):
    d=pd.DataFrame(index=h.index)
    d['person']=h[idcol].astype(str);d['cohort']=cohort;d['wave']=w;d['year']=year
    f=lambda suffix:s(h,f'r{w}{suffix}')
    q=lambda suffix:b(h,f'r{w}{suffix}')
    d['age']=first(h,[f'r{w}agey',f'r{w}agey_e']).where(lambda x:x.ge(0))
    d['sex']=s(h,'ragender').where(lambda x:x.isin([1,2]))
    d['educ']=s(h,'raeducl').where(lambda x:x.isin([1,2,3]))
    d['diabetes']=q('diabe')
    d['med']=bor(q('rxdiab'),q('rxdiabi')) if cohort=='CHARLS' else q('rxdiab')
    if cohort=='HRS':d['med']=bor(q('rxdiabo'),q('rxdiabi'))
    d['insulin']=q('rxdiabi')
    if f'r{w}nhmliv' in h:
        d['community']=1-q('nhmliv')
    else:
        # Audited household/community sampling; no institutional field in these waves.
        d['community']=1.
    d['ltc']=1-d.community
    for k,t in ADLMAP.items():
        d['adl_'+k]=q(t)
        # Receipt of personal assistance confirms dependence. No difficulty
        # confirms no dependency. Not receiving assistance is not proof of ability.
        dep=pd.Series(np.nan,index=d.index,dtype=float)
        dep.loc[q(t).eq(0)]=0;dep.loc[q(HELPMAP[k]).eq(1)]=1
        d['dep_'+k]=dep
        d['next_adl_'+k]=b(h,f'r{w+1}{t}')
    for k,t in IADLMAP.items():d['iadl_'+k]=q(t)
    for k,t in CHRONMAP.items():d['c_'+k]=q(t)
    if cohort=='MHAS':d['c_lung']=q('respe');d['c_heart']=q('hrtatte')
    d['c_chf']=q('conhrtfe');d['c_mi']=q('hrtatte')
    d['c_cad']=bor(q('hrtatte'),q('angine'))
    if cohort!='MHAS':
        for k in ['chf','mi','cad']:
            d.loc[d.c_heart.eq(0)&d['c_'+k].isna(),'c_'+k]=0
    d['c_ckd_egfr']=np.nan
    d['c_active_cancer']=np.where(d.c_cancer.eq(0),0,np.nan)
    d['c_clinical_depression']=np.nan
    d['c_emphysema']=np.where(d.c_lung.eq(0),0,np.nan)
    d['c_ui_any']=q('urinai')
    if cohort=='MHAS':
        # Codebook explicitly says urge incontinence, despite short variable label.
        d['c_ui_any']=bor(q('urincgh2y'),q('urinurg2y'))
    if cohort=='SHARE':d['c_ui_any']=np.where(q('urinpad').eq(1),1,np.nan)
    d['c_ui_weekly']=np.where(d.c_ui_any.eq(0),0,np.nan)
    d['cog_diagnosis']=bor(q('alzhe'),q('demene'),q('alzdeme'))
    # A negative dementia diagnosis does not rule out mild cognitive impairment.
    d['clinical_cog_any']=np.where(d.cog_diagnosis.eq(1),1,np.nan)
    d['clinical_cog_modsev']=np.nan
    d['dialysis']=np.nan
    d['endstage_chf']=np.where(d.c_chf.eq(0),0,np.nan)
    d['oxygen_lung']=np.where(d.c_lung.eq(0),0,np.nan)
    d['metastatic_uncontrolled_cancer']=np.where(d.c_cancer.eq(0),0,np.nan)
    for k,t in [('cog_immediate','imrc8' if cohort=='MHAS' else 'imrc'),('cog_delayed','dlrc8' if cohort=='MHAS' else 'dlrc'),('cog_orientation','orient_m' if cohort=='MHAS' else 'orient')]:
        hi=(3 if cohort=='MHAS' else 4) if k=='cog_orientation' else (8 if cohort=='MHAS' else 10)
        d[k]=f(t).where(lambda x:x.between(0,hi))
    d['baseline_fall']=q('fall_s' if cohort=='SHARE' else 'fall')
    if cohort=='ELSA':d['baseline_fall']=d.baseline_fall.combine_first(q('fall1y'))
    d['c_baseline_fall']=d.baseline_fall
    d['next_fall']=b(h,f'r{w+1}'+('fall_s' if cohort=='SHARE' else 'fall'))
    if cohort=='ELSA':
        d['next_fall']=d.next_fall.combine_first(b(h,f'r{w+1}fall1y'))
    d['next_status']=s(h,f'r{w+1}iwstat')
    d['bmi']=first(h,[f'r{w}mbmi',f'r{w}bmi']).where(lambda x:x.gt(0))
    d['smoke']=q('smoken')
    return d

def egfr_2009(creat,age,sex):
    """CKD-EPI 2009, mg/dL; China non-Black equation. No cystatin-C substitution."""
    cr=pd.to_numeric(creat,errors='coerce').where(lambda x:x.gt(0))
    female=sex.eq(2);k=np.where(female,.7,.9);a=np.where(female,-.329,-.411)
    ratio=cr/k
    return 141*np.minimum(ratio,1)**a*np.maximum(ratio,1)**(-1.209)*(.993**age)*np.where(female,1.018,1)
