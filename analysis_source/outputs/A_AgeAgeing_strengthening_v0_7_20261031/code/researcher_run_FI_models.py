"""Independent non-HRS researcher runner: FI versions, validity, grip/mortality models.

Never invoke on participant data through an online AI workflow. No HRS/RAND reads.
"""
from pathlib import Path
import argparse,json,sys,re,hashlib,warnings
import numpy as np,pandas as pd
from extension_math import fi_score,fit_cloglog,effect,NoValidResult
O=Path(__file__).resolve().parents[1];R=O.parents[1]
sys.path.insert(0,str(R/'outputs/A_QC_v1_2_20261027/code'))
import qc_core as qc
f=qc.f
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True,choices=['CHARLS','MHAS','SHARE','ELSA','KLoSA']);ap.add_argument('--output',required=True);ap.add_argument('--independent-researcher',action='store_true');ap.add_argument('--provider-min-cell',type=int,required=True);ap.add_argument('--common-items-json')
    a=ap.parse_args();c=a.cohort;dest=Path(a.output)
    if not a.independent_researcher:raise RuntimeError('Independent researcher execution required')
    if dest.exists():raise RuntimeError('New output directory required')
    if a.provider_min_cell<1 or (c=='ELSA' and a.provider_min_cell<3):raise RuntimeError('Provider threshold not satisfied')
    path=O/'protocol/FI_candidate_pool_v0_7.json';pool=json.loads(path.read_text());pl=json.loads((O/'protocol/FI_candidate_pool_lock.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pl['sha256']:raise RuntimeError('FI pool changed')
    items=[r for r in pool['items'] if r['cohort']==c]
    known={};manifest=[]
    for pp,k in [(R/'outputs/A_formal_v1_20261027/run_receipt.json','manifest'),(R/'outputs/A_amendment_v1_1_20261027/execution_receipt.json','input_manifest'),(R/'outputs/A_QC_v1_2_20261027/execution_receipt.json','input_manifest')]:
        for z in json.loads(pp.read_text()).get(k,[]):known[str(Path(z['path']))]=z
    def read(p,cols):
        p=Path(p);k=known.get(str(p));st=p.stat()
        if re.search('hrs|rand',p.name,re.I) or k is None:raise RuntimeError('Source outside frozen non-HRS allowlist')
        if st.st_size!=k['size'] or st.st_mtime_ns!=k['mtime_ns']:raise RuntimeError('Source version mismatch')
        manifest.append({'basename':p.name,'columns':cols,'size':st.st_size,'mtime_ns':st.st_mtime_ns})
        return pd.read_stata(p,columns=cols,convert_categoricals=False)
    f.read=read
    def forbid(*a,**k):raise RuntimeError('Date-based mortality exclusion forbidden')
    f.early_exclusion=forbid
    h,d,_=qc.qc_load(c)
    sc=f.SCHEMA[f.SPEC[c][0]];idcol=f.SPEC[c][1]
    raw=read(sc['path'],[idcol]+[r['field'] for r in items]);raw['id']=raw[idcol].astype(str);raw=raw.set_index('id').reindex(d.id)
    scores=[];screen=[];age=d.age.to_numpy(float)
    for it in items:
        x=pd.to_numeric(raw[it['field']],errors='coerce').map({float(k):v for k,v in it['codes'].items()}).to_numpy(float);ok=np.isfinite(x)
        missing=1-ok.mean();prev=float(np.mean(x[ok]>0)) if ok.any() else np.nan
        slope=float(np.cov(age[ok],x[ok],ddof=0)[0,1]/np.var(age[ok])) if ok.sum()>1 and np.var(age[ok])>0 else np.nan
        selected=bool(.01<=prev<=.8 and missing<=.1 and slope>0)
        screen.append({'cohort':c,'concept':it['concept'],'field':it['field'],'n':len(d),'missing_n':int((~ok).sum()),'missing_fraction':missing,'prevalence':prev,'age_slope':slope,'selected_main':selected,'selected_noMob':selected and not it['mobility_strength'],'common_equiv':it['common_equiv']});scores.append(x)
    S=np.column_stack(scores);selected=[i for i,r in enumerate(screen) if r['selected_main']]
    noMob=[i for i,r in enumerate(screen) if r['selected_noMob']]
    common=None
    if a.common_items_json:
        contract=json.loads(Path(a.common_items_json).read_text())
        if set(contract['cohorts'])!=set(['CHARLS','ELSA','MHAS','SHARE','KLoSA']) or not contract['all_empirical_screens_verified']:raise RuntimeError('FI-common requires verified five-cohort screen intersection')
        common=[i for i in selected if items[i]['common_equiv'] and items[i]['concept'] in contract['common_concepts']]
        if len(common)!=len(contract['common_concepts']):raise RuntimeError('Common item set fails local equivalence/empirical gate')
    validity=[];status=[];models=[];tertiles=[]
    for version,ix in [('main',selected),('noMob',noMob),('common',common)]:
        nitems=len(ix) if ix is not None else 0
        if ix is None or not nitems or (version!='common' and nitems<30):
            status.append({'cohort':c,'FI_version':version,'item_n':nitems,'status':'PENDING_COMMON_INTERSECTION' if ix is None else 'INSUFFICIENT_ITEMS'});continue
        # User explicitly permits <30 for FI-common; generic fi_score's30-item gate
        # therefore does not apply to this exception.
        q=S[:,ix];observed=np.isfinite(q).sum(1);value=np.divide(np.nansum(q,axis=1),observed,out=np.full(len(d),np.nan),where=observed>0);value[observed<np.ceil(.8*nitems)]=np.nan
        dd=d.copy();dd['FI']=value/.1;dd['FI01']=value/.1
        cuts=np.quantile(value[np.isfinite(value)],[1/3,2/3]) if np.isfinite(value).any() else [np.nan,np.nan]
        for group,mask in [('All',np.ones(len(d),bool)),('Men',d.sex.eq(1).to_numpy()),('Women',d.sex.eq(2).to_numpy())]+[(str(g),np.floor(d.age/5).eq(g).to_numpy()) for g in sorted(np.floor(d.age/5).unique())]:
            x=value[mask];xx=x[np.isfinite(x)]
            validity.append({'cohort':c,'FI_version':version,'group':group if group in ['All','Men','Women'] else f'{int(float(group)*5)}-{int(float(group)*5+4)}','n':len(x),'valid_n':len(xx),'missing_n':len(x)-len(xx),'mean':float(np.mean(xx)) if len(xx) else np.nan,'median':float(np.median(xx)) if len(xx) else np.nan,'P99':float(np.quantile(xx,.99)) if len(xx) else np.nan,'max':'SUPPRESSED_PROVIDER_UNIT_RECORD_RULE' if c=='ELSA' else float(np.max(xx)) if len(xx) else np.nan})
        status.append({'cohort':c,'FI_version':version,'item_n':nitems,'valid_FI_n':int(np.isfinite(value).sum()),'missing_FI_n':int((~np.isfinite(value)).sum()),'status':'AVAILABLE_LT30_EXPLICIT_COMMON_EXCEPTION' if nitems<30 else 'AVAILABLE','tertile_cut1':cuts[0],'tertile_cut2':cuts[1]})
        if c=='ELSA':continue # T5 fails complete death coverage; do not fit survival models.
        kinds=['death'] if c=='KLoSA' else ['composite','death']
        for kind in kinds:
            panel,_=f.intervals(h,dd,c,kind);z=panel.dropna(subset=f.COV+['low_EWGSOP2','event','FI']).copy()
            tasks=[('grip_adjusted_FI',z,True)]
            if np.isfinite(cuts).all() and cuts[0]<cuts[1]:
                which=pd.cut(z.FI*.1,[-np.inf,cuts[0],cuts[1],np.inf],labels=[1,2,3],right=True)
                tasks.extend((f'grip_FI_tertile{k}',z.loc[which.eq(k)],False) for k in [1,2,3])
            else:models.append({'cohort':c,'FI_version':version,'outcome':kind,'analysis':'tertiles','status':'NO_VALID_RESULT_TIED_OR_MISSING_CUTPOINTS'})
            for label,zz,addfi in tasks:
                try:r=effect(fit_cloglog(zz,'low_EWGSOP2',c,add_fi=addfi));models.append(dict(cohort=c,FI_version=version,outcome=kind,analysis=label,n=zz.id.nunique(),events_n=int(zz.event.sum()),**r,status='OK'))
                except NoValidResult as e:models.append({'cohort':c,'FI_version':version,'outcome':kind,'analysis':label,'status':'NO_VALID_RESULT','reason':str(e)})
            if kind=='death':
                try:r=effect(fit_cloglog(z,'FI01',c,continuous_exposure=True));models.append(dict(cohort=c,FI_version=version,outcome='death',analysis='FI_per0.1_validity',n=z.id.nunique(),events_n=int(z.event.sum()),**r,status='OK'))
                except NoValidResult as e:models.append({'cohort':c,'FI_version':version,'outcome':'death','analysis':'FI_per0.1_validity','status':'NO_VALID_RESULT','reason':str(e)})
    dest.mkdir(parents=True)
    def release(rows,name):
        if not rows:return
        frame=pd.DataFrame(rows);frame['post_hoc']='T2 replacement2026-10-31';ncols=[k for k in frame if k=='n' or k.endswith('_n')];small=False
        for k in ncols:
            x=pd.to_numeric(frame[k],errors='coerce');hit=x.gt(0)&x.lt(a.provider_min_cell)
            if hit.any():frame[k]=frame[k].astype(object);frame.loc[hit,k]=f'<{a.provider_min_cell}';small=True
        if small:
            for k in ncols:
                frame[k]=frame[k].astype(object);hit=~frame[k].astype(str).str.startswith('<');frame.loc[hit,k]='SUPPRESSED_COMPLEMENT_PROVIDER'
        frame.to_csv(dest/name,index=False)
    release(screen,'FI_item_screen.csv');release(status,'FI_versions_status.csv');release(validity,'FI_validity.csv');release(models,'FI_models.csv')
    (dest/'receipt.json').write_text(json.dumps({'status':'INDEPENDENT_RESEARCHER_RUN','cohort':c,'individual_rows_written':False,'HRS_files_opened':0,'FI_pool_sha256':pl['sha256'],'FI_common_supplied':common is not None,'source_manifest':manifest},ensure_ascii=False,indent=2))
    print('FI aggregate return prepared; unavailable versions remain explicitly labelled')

if __name__=='__main__':
    try:main()
    except Exception as e:print('STOP:',type(e).__name__,'; no participant content emitted');sys.exit(2)
