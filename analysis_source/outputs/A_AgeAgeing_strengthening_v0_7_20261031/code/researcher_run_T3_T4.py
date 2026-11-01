"""Independent researcher-run MHAS/SHARE v0.7 T3/T4. Assistant must not run on microdata."""
from pathlib import Path
import argparse,json,re,sys,hashlib,warnings
import numpy as np,pandas as pd
from extension_math import fit_cloglog,effect,standardized_risk,risk_prediction,auc_ipcw,first_followup_landmark,NoValidResult
O=Path(__file__).resolve().parents[1];R=O.parents[1]
sys.path.insert(0,str(R/'outputs/A_QC_v1_2_20261027/code'))
import qc_core as qc
f=qc.f
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True,choices=['MHAS','SHARE']);ap.add_argument('--output',required=True);ap.add_argument('--independent-researcher',action='store_true');ap.add_argument('--provider-min-cell',type=int,required=True)
    args=ap.parse_args();c=args.cohort;dest=Path(args.output)
    if not args.independent_researcher:raise RuntimeError('Independent researcher execution required')
    if args.provider_min_cell<1:raise RuntimeError('Verify accepted provider output threshold; use1 if no fixed threshold applies')
    if dest.exists():raise RuntimeError('New output directory required')
    lock=json.loads((O/'protocol/method_extensions_lock.json').read_text())
    if hashlib.sha256((O/'protocol/method_extensions_v0_7_locked.md').read_bytes()).hexdigest()!=lock['sha256']:raise RuntimeError('Method lock mismatch')
    known={}
    for pp,k in [(R/'outputs/A_formal_v1_20261027/run_receipt.json','manifest'),(R/'outputs/A_amendment_v1_1_20261027/execution_receipt.json','input_manifest'),(R/'outputs/A_QC_v1_2_20261027/execution_receipt.json','input_manifest')]:
        for x in json.loads(pp.read_text()).get(k,[]):known[str(Path(x['path']))]=x
    manifest=[]
    def read(p,cols):
        p=Path(p)
        if re.search('hrs|rand',p.name,re.I):raise RuntimeError('HRS/RAND file forbidden')
        x=known.get(str(p));st=p.stat()
        if x is None or st.st_size!=x['size'] or st.st_mtime_ns!=x['mtime_ns']:raise RuntimeError('Frozen source metadata mismatch')
        # Neither the new wave-lag nor other modules require calendar death/interview dates.
        cols=[a for a in cols if not re.match(r'^(radyear|radmonth|r\d+(iwy|iwm|iwindy|iwindm))$',a,re.I)]
        manifest.append({'basename':p.name,'columns':cols,'size':st.st_size,'mtime_ns':st.st_mtime_ns})
        return pd.read_stata(p,columns=cols,convert_categoricals=False)
    f.read=read
    def bridge(h,cohort):
        if cohort!=c:raise RuntimeError('Wrong cohort')
        sc=json.loads((R/'work/a_formal/new_wave_schema.json').read_text())
        if c=='MHAS':
            names=['h13_21','h16a_21','h17a_21','h18a_21','h19a_21']
            a=read(sc['/Core survey Data/sect_a_c_d_e_pc_f_h_i_2021.dta']['path'],['unhhid','np']+names).set_index(['unhhid','np'])
            b=read(sc['/Master Follow-up File/master_follow_up_file_2021.dta']['path'],['unhhid','np','fallecido_21','tipent_21']).set_index(['unhhid','np'])
            if not a.index.is_unique or not b.index.is_unique:raise RuntimeError('Nonunique approved extension linkage')
            ix=pd.MultiIndex.from_frame(h[['unhhid','np']]);a=a.reindex(ix);b=b.reindex(ix);a.index=h.index;b.index=h.index
            h['r6iwstat']=np.where(b.fallecido_21.isin([1,2])|b.tipent_21.eq(5),5,np.where(b.tipent_21.isin([1,2,3,4]),1,np.where(b.fallecido_21.eq(0),4,np.nan)))
            for raw,s in zip(names,['dressa','batha','eata','beda','toilta']):h['r6'+s]=a[raw].map({1:1,2:0,6:1})
            f.MAXW[c]=6;f.YEARS[c][6]=2021
        else:
            p=next(x['path'] for x in f.INV if x['path'].endswith('/sharew9_rel9-0-0_ph.dta'));q=next(x['path'] for x in f.INV if x['path'].endswith('/sharew9_rel9-0-0_cv_r.dta'))
            names=['ph049d1','ph049d3','ph049d4','ph049d5','ph049d6']
            a=read(p,['mergeid']+names).set_index('mergeid');b=read(q,['mergeid','deceased','interview']).set_index('mergeid')
            if not a.index.is_unique or not b.index.is_unique:raise RuntimeError('Nonunique approved extension linkage')
            a=a.reindex(h.mergeid);b=b.reindex(h.mergeid);a.index=h.index;b.index=h.index
            h['r9iwstat']=np.where(b.deceased.eq(1)|b.interview.eq(2),5,np.where(b.interview.eq(1),1,np.where(b.deceased.eq(0),4,np.nan)))
            for raw,s in zip(names,['dressa','batha','eata','beda','toilta']):h['r9'+s]=a[raw].where(a[raw].isin([0,1]))
            f.MAXW[c]=9;f.YEARS[c][9]=2022
        return h.copy()
    f.extend_followup=bridge
    def forbid(*a,**k):raise RuntimeError('Date-based exclusion forbidden')
    f.early_exclusion=forbid
    h,d,_=qc.qc_load(c);dd,_=f.intervals(h,d,c,'composite');z=dd.dropna(subset=f.COV+['low_EWGSOP2','event']).copy()
    z['low_FNIH']=z.grip/z.bmi<np.where(z.sex.eq(1),1.,.56)
    w=f.BASE[c];baseyear=f.YEARS[c][w]
    bounds={int(i):(f.YEARS[c][w+int(i)-1]-baseyear,f.YEARS[c][w+int(i)]-baseyear) for i in z.interval.unique()}
    modelrows=[];riskrows=[];prevalences=[];aucrows=[];checks=[]
    old=pd.read_csv(R/'outputs/A_QC_v1_2_20261027/A_models.csv');old=old[(old.cohort==c)&(old.outcome=='composite')&(old.analysis=='primary')].iloc[0]
    for definition in ['EWGSOP2','SDOC','FNIH']:
        expo='low_'+definition
        try:
            m=fit_cloglog(z,expo,c);ee=effect(m)
            if definition=='EWGSOP2':
                if abs(ee['log_effect']-float(old.log_effect))>=1e-6 or abs(ee['se']-float(old.se))>=1e-6:raise RuntimeError('Primary model does not reproduce frozen source')
                checks.append({'cohort':c,'check':'primary beta and robust SE reproduce QCv2','status':'PASS'})
            modelrows.append(dict(cohort=c,analysis='primary_definition',definition=definition,**ee,status='OK'))
            riskrows.extend(dict(cohort=c,definition=definition,years=9,**rr,status='OK') for rr in standardized_risk(m,bounds))
        except NoValidResult as e:modelrows.append({'cohort':c,'analysis':'primary_definition','definition':definition,'status':'NO_VALID_RESULT','reason':str(e)})
        d[expo]=d.grip/d.bmi<np.where(d.sex.eq(1),1.,.56) if definition=='FNIH' else d[expo]
        d['cohort']=c;f.suppress=lambda n:int(n)
        pp=f.domain_est(h,d,pd.Series(True,index=d.index),d[expo]);prevalences.append(dict(cohort=c,definition=definition,scope='eligible positive-weight candidates age>=65',**pp))
    lag=first_followup_landmark(z)
    try:modelrows.append(dict(cohort=c,analysis='first_followup_landmark',definition='EWGSOP2',**effect(fit_cloglog(lag,'low_EWGSOP2',c)),status='OK'))
    except NoValidResult as e:modelrows.append({'cohort':c,'analysis':'first_followup_landmark','definition':'EWGSOP2','status':'NO_VALID_RESULT','reason':str(e)})
    def aucs(q,definition):
        mb=fit_cloglog(q,'low_'+definition,c,prediction_base=True,without_exposure=True);ma=fit_cloglog(q,'low_'+definition,c,prediction_base=True)
        terminal=q.sort_values('interval').groupby('id',sort=False).tail(1);p=q.drop_duplicates('id').set_index('id').reindex(terminal.id).reset_index();p['x']=p['low_'+definition].astype(float)
        rb,_=risk_prediction(mb,p,bounds);ra,_=risk_prediction(ma,p,bounds);tt=np.array([bounds[k][1] for k in terminal.interval]);ev=terminal.event.to_numpy(int)
        b=auc_ipcw(rb,tt,ev);a=auc_ipcw(ra,tt,ev);return b,a,a-b
    ids=z.id.unique();groups={i:g for i,g in z.groupby('id',sort=False)}
    for di,definition in enumerate(['EWGSOP2','SDOC','FNIH']):
        row={'cohort':c,'definition':definition,'n':len(ids),'method':'nominal-wave cumulative_dynamic_IPCW_AUC','performance':'apparent_not_external_validation','bootstrap_requested':500,'seed':20261031+di}
        try:base,added,delta=aucs(z,definition)
        except NoValidResult as e:row.update(status='NO_VALID_RESULT',reason=str(e));aucrows.append(row);continue
        rng=np.random.default_rng(20261031+di);vals=[];fail={}
        for rep in range(500):
            parts=[]
            for j,i in enumerate(rng.choice(ids,len(ids),replace=True)):
                a=groups[i].copy();a['id']=str(j);parts.append(a)
            try:vals.append(aucs(pd.concat(parts,ignore_index=True),definition)[2])
            except Exception as e:fail[type(e).__name__]=fail.get(type(e).__name__,0)+1
            if (rep+1)%50==0:print(c,definition,'participant bootstrap',rep+1,'/500',flush=True)
        row.update(base_AUC=base,added_AUC=added,increment=delta,bootstrap_successful=len(vals),failure_types=json.dumps(fail))
        if len(vals)>=450:
            ci=np.quantile(vals,[.025,.975]);row.update(ci_low=float(ci[0]),ci_high=float(ci[1]),status='OK')
        else:row.update(status='POINT_ESTIMATE_ONLY_NO_VALID_BOOTSTRAP_CI')
        aucrows.append(row)
    # Provider threshold and complements are applied to released counts, not model formulas.
    def release(rows,name):
        frame=pd.DataFrame(rows);frame['post_hoc']='v0.7 post hoc reporting/definition/landmark extension'
        ncols=[x for x in frame if x=='n' or x.endswith('_n')];small=False
        for col in ncols:
            q=pd.to_numeric(frame[col],errors='coerce');hit=q.gt(0)&q.lt(args.provider_min_cell)
            if hit.any():frame[col]=frame[col].astype(object);frame.loc[hit,col]=f'<{args.provider_min_cell}';small=True
        if small:
            for col in ncols:
                frame[col]=frame[col].astype(object);q=~frame[col].astype(str).str.startswith('<');frame.loc[q,col]='SUPPRESSED_COMPLEMENT_PROVIDER'
        frame.to_csv(dest/name,index=False)
    dest.mkdir(parents=True)
    release(modelrows,'T3_T4_models.csv');release(riskrows,'definition_risks.csv');release(prevalences,'definition_prevalences.csv');release(aucrows,'definition_AUC.csv');release(checks,'frozen_reproduction.csv')
    (dest/'receipt.json').write_text(json.dumps({'status':'INDEPENDENT_RESEARCHER_RUN','cohort':c,'methods_sha256':lock['sha256'],'calendar_date_columns_read':False,'individual_rows_written':False,'HRS_files_opened':0,'input_manifest':manifest},ensure_ascii=False,indent=2))
    print('Aggregate return written; verify provider disclosure before returning it')

if __name__=='__main__':
    try:main()
    except Exception as e:print('STOP:',type(e).__name__,'; record content deliberately not printed');sys.exit(2)
