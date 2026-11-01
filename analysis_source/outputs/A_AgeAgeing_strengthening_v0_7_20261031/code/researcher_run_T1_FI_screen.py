"""INDEPENDENT RESEARCHER ONLY for MHAS/SHARE/ELSA/KLoSA.

The assistant must not invoke this runner on participant files. It outputs aggregate
T1 tables/flow and predeclared FI item screens only, never participant rows or IDs.
No models are fitted. Existing HRS/RAND HRS files are explicitly denied.
"""
from pathlib import Path
import argparse,json,sys,re,datetime,hashlib,warnings
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[1];R=O.parents[1]
sys.path.insert(0,str(R/'outputs/A_QC_v1_2_20261027/code'))
import qc_core as qc
f=qc.f
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True,choices=['MHAS','SHARE','ELSA','KLoSA']);ap.add_argument('--independent-researcher',action='store_true');ap.add_argument('--output',required=True);ap.add_argument('--provider-min-cell',type=int,required=True)
    args=ap.parse_args()
    if not args.independent_researcher:raise RuntimeError('Researcher must run independently, outside online AI tools')
    c=args.cohort;threshold=args.provider_min_cell
    if c=='ELSA' and threshold<3:raise RuntimeError('UKDS EUL minimum threshold3 must be retained')
    if threshold<1:raise RuntimeError('Use1 when no fixed provider small-cell threshold applies; verify accepted agreement')
    target=Path(args.output)
    if target.exists():raise RuntimeError('Use a new return directory; do not overwrite')
    target.mkdir(parents=True)
    poolpath=O/'protocol/FI_candidate_pool_v0_7.json';pool=json.loads(poolpath.read_text())
    if hashlib.sha256(poolpath.read_bytes()).hexdigest()!=json.loads((O/'protocol/FI_candidate_pool_lock.json').read_text())['sha256']:raise RuntimeError('FI pool lock mismatch')
    inputs={}
    for pp,key in [(R/'outputs/A_formal_v1_20261027/run_receipt.json','manifest'),(R/'outputs/A_amendment_v1_1_20261027/execution_receipt.json','input_manifest'),(R/'outputs/A_QC_v1_2_20261027/execution_receipt.json','input_manifest')]:
        for row in json.loads(pp.read_text()).get(key,[]):inputs[str(Path(row['path']))]=row
    manifest=[]
    def read(p,cols):
        p=Path(p)
        if re.search('hrs|rand',p.name,re.I):raise RuntimeError('HRS/RAND source forbidden')
        known=inputs.get(str(p));st=p.stat()
        if known is None or st.st_size!=known['size'] or st.st_mtime_ns!=known['mtime_ns']:raise RuntimeError('Frozen non-HRS source metadata mismatch')
        manifest.append({'basename':p.name,'columns':cols,'size':st.st_size,'mtime_ns':st.st_mtime_ns})
        return pd.read_stata(p,columns=cols,convert_categoricals=False)
    f.read=read
    def forbid(*a,**k):raise RuntimeError('No models or date-based exclusion allowed in this reporting/item-screen runner')
    f.fitmodel=forbid;f.multistate=forbid;f.early_exclusion=forbid
    h,d,_=qc.qc_load(c);w=f.BASE[c]
    # The prior loader also binds design and approved wave extensions. This runner does
    # not evaluate any interview/death-date exclusion; only the approved status chain.
    age=f.num(h,f'r{w}agey').combine_first(f.num(h,f'r{w}agey_e'));inter=f.num(h,f'r{w}iwstat').eq(1)
    flow=[{'cohort':c,'stage':s,'n':int(n),'version':'QCv2_only'} for s,n in [('source_harmonized_records',len(h)),('baseline_interviewed',inter.sum()),('baseline_age65',(inter&age.ge(65)).sum()),('sex_BMI_grip_available_after_QC',len(f.derive(h,c,f.SPEC[c][1],w,f.SPEC[c][3]))),('confirmed_eligible_with_grip',len(d)),('description_positive_weight',d.weight.notna().sum()),('baseline_ADL0',f.adl(h,c,w).reindex(d.index).eq(0).sum())]]
    tables=[];fig3=[]
    if c in ['MHAS','SHARE']:
        dd,_=f.intervals(h,d,c,'composite');z=dd.dropna(subset=f.COV+['low_EWGSOP2','event']);base=d[d.id.isin(set(z.id))]
        flow.extend([{'cohort':c,'stage':'any_observable_interval','n':dd.id.nunique(),'version':'QCv2_only'},{'cohort':c,'stage':'primary_complete_case','n':len(base),'version':'QCv2_only'}])
        for gp,sub in [('Normal',base[~base.low_EWGSOP2]),('Low',base[base.low_EWGSOP2]),('All',base)]:
            for var in ['age','sex','educ','bmi','smoke','diabetes','chronic','country','grip','eligible']:
                a=sub[var];v=a.dropna();common={'cohort':c,'group':gp,'variable':var,'n':len(v),'missing_n':int(a.isna().sum()),'weighting':'unweighted'}
                if var in ['age','bmi','grip','chronic']:
                    q=v.quantile([.25,.5,.75]);tables.append(dict(common,type='continuous',level='',mean=float(v.mean()),SD=float(v.std()),q1=q.iloc[0],median=q.iloc[1],q3=q.iloc[2]))
                else:
                    for lev in sorted(base[var].dropna().unique(),key=str):
                        nn=int(v.eq(lev).sum());ll=str(lev)
                        if var=='country':ll=str(base.loc[base.country.eq(lev),'country_name'].iloc[0])
                        tables.append(dict(common,type='categorical',level=ll,category_n=nn,percent=100*nn/len(v)))
        for gp,sub in [('Normal',z[~z.low_EWGSOP2]),('Low',z[z.low_EWGSOP2]),('All',z)]:fig3.append({'cohort':c,'group':gp,'n':sub.id.nunique(),'events_n':int(sub.event.sum()),'version':'QCv2_only'})
    # Metadata-fixed FI candidate screen, all eligible baseline people, no outcomes used.
    items=[i for i in pool['items'] if i['cohort']==c]
    sc=f.SCHEMA[f.SPEC[c][0]];idcol=f.SPEC[c][1]
    raw=read(sc['path'],[idcol]+[i['field'] for i in items]);raw['id']=raw[idcol].astype(str);raw=raw.set_index('id').reindex(d.id)
    screens=[];age=d.age.to_numpy(float)
    for it in items:
        x=pd.to_numeric(raw[it['field']],errors='coerce').map({float(k):v for k,v in it['codes'].items()}).to_numpy(float)
        ok=np.isfinite(x);n=int(ok.sum());missing=1-n/len(d);prev=float((x[ok]>0).mean()) if n else np.nan
        slope=float(np.cov(age[ok],x[ok],ddof=0)[0,1]/np.var(age[ok])) if n>1 and np.var(age[ok])>0 else np.nan
        keep=bool(n and .01<=prev<=.80 and missing<=.10 and slope>0)
        screens.append({'cohort':c,'concept':it['concept'],'field':it['field'],'domain':it['domain'],'n':len(d),'nonmissing_n':n,'missing_n':len(d)-n,'missing_fraction':missing,'prevalence_score_positive':prev,'mean_score':float(np.nanmean(x)) if n else np.nan,'age_slope':slope,'selected_main':keep,'selected_noMob':keep and not it['mobility_strength'],'common_equiv':it['common_equiv']})
    def release(rows,name):
        if not rows:return
        frame=pd.DataFrame(rows);ncols=[k for k in frame if k=='n' or k.endswith('_n')];small=False
        for col in ncols:
            x=pd.to_numeric(frame[col],errors='coerce');hit=x.gt(0)&x.lt(threshold)
            if hit.any():small=True;frame[col]=frame[col].astype(object);frame.loc[hit,col]=f'<{threshold}'
        if small:
            # Protect all complementary count partitions and percentages in this table.
            for col in ncols:
                frame[col]=frame[col].astype(object);mask=~frame[col].astype(str).str.startswith('<');frame.loc[mask,col]='SUPPRESSED_COMPLEMENT_PROVIDER'
            for col in ['percent','prevalence_score_positive','missing_fraction']:
                if col in frame:frame[col]='SUPPRESSED_COMPLEMENT_PROVIDER'
        frame.to_csv(target/name,index=False)
    release(flow,'fig1_QCv2_counts.csv');release(tables,'table1_QCv2_long.csv');release(fig3,'fig3a_QCv2_counts.csv');release(screens,'FI_item_screen.csv')
    (target/'receipt.json').write_text(json.dumps({'status':'INDEPENDENT_RESEARCHER_AGGREGATES','cohort':c,'models_fit':0,'individual_rows_written':False,'HRS_files_opened':0,'provider_min_cell':threshold,'FI_pool_sha256':hashlib.sha256(poolpath.read_bytes()).hexdigest(),'source_versions_match':True,'manifest':manifest},ensure_ascii=False,indent=2))
    print('Aggregate reporting and FI-screen return complete; review disclosure before sharing')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('STOP:',type(e).__name__,'; no participant record or exception details emitted');sys.exit(2)
