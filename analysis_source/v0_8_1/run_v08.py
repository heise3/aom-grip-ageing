#!/usr/bin/env python3
"""Independent-researcher R1/R2 add-on. Do not execute real mode in an AI session."""
from pathlib import Path
import argparse,contextlib,csv,datetime,hashlib,io,json,os,re,subprocess,sys,warnings,zipfile
from zoneinfo import ZoneInfo
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'lib'))

def atomic(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)

def verify_bundle():
    manifest=json.loads((B/'PACKAGE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((B/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('PACKAGE_HASH_MISMATCH:'+name)
    return hashlib.sha256((B/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest()

def expected_count_contract():
    from count_contract import validate_expected_counts
    return validate_expected_counts(B/'resources/expected_primary_counts.csv')

def engineering_trace(error):
    # Code locations only: do not export args, messages, locals or source lines.
    frames=[];tb=error.__traceback__
    while tb is not None:
        frames.append({'code_file':Path(tb.tb_frame.f_code.co_filename).name,
                       'function':tb.tb_frame.f_code.co_name,'line':tb.tb_lineno})
        tb=tb.tb_next
    return {'exception_type':type(error).__name__,'frames':frames,
            'exception_messages_exported':False,'frame_locals_exported':False}

def dependencies(config):
    import importlib
    result={k:importlib.import_module(k).__version__ for k in config['dependencies']}
    if result!=config['dependencies']:raise RuntimeError('DEPENDENCY_VERSION_MISMATCH')
    return result

def audit_nonhrs():
    def guard(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        try:name=Path(os.fsdecode(args[0])).name
        except (TypeError,ValueError):return
        if re.search(r'(^|[^a-z])(?:hrs|rand)(?:[^a-z]|$)',name,re.I):
            raise RuntimeError('HRS_RAND_FILE_OPEN_FORBIDDEN')
    sys.addaudithook(guard)

def worker(c,run,config):
    import numpy as np,pandas as pd
    import runtime as rt,analysis as a
    from extension_math import fit_cloglog,effect,NoValidResult
    from r1_math import paired_bootstrap
    rt.require(c in config['requested_cohorts'],'NONREQUESTED_COHORT')
    audit_nonhrs();rt.configure();rt.install_data_audit_guard()
    output=run/'return';output.mkdir(exist_ok=True)
    h,d=rt.load(c);panel,_=rt.f.intervals(h,d,c,'composite')
    z=panel.dropna(subset=rt.f.COV+['low_EWGSOP2','event']).copy()
    frozen=expected_count_contract()[c]
    n=int(z.id.nunique());ev=int(z.event.sum())
    rt.require(n==frozen['n'] and ev==frozen['events_n'],'PRIMARY_SAMPLE_OR_EVENTS_CHANGED:'+c)
    # Independent people, not person-wave rows. Preserve baseline design indices.
    g=d[d.id.isin(z.id)].copy()
    rt.require(g.id.is_unique and len(g)==n,'PRIMARY_PERSON_BINDING_FAILED:'+c)
    low=int(g.low_EWGSOP2.sum())
    rt.require(low==frozen['Low_n'],'PRIMARY_LOW_GRIP_COUNT_CHANGED:'+c)
    prev=a.prevalences(h,g,c)
    valid_weight_n=int(g.weight.notna().sum())
    prev['prognostic_n']=n;prev['positive_weight_n']=valid_weight_n;prev['excluded_weight_n']=n-valid_weight_n
    prev['population_match_status']='EXACT_PRIMARY_PEOPLE' if n==valid_weight_n else 'WEIGHT_VALID_SUBSET_DIFFERS'
    prev['scope']='primary baseline-ADL0 complete cases; inherited positive-weight restriction'
    prev['analysis_origin']='post hoc v0.8 prognosis-domain flags'
    prev.to_csv(output/f'R2_prevalence_{c}.csv',index=False)
    if c=='SHARE':
        screen,S,_=a.screen_fi(h,d,c)
        expected=pd.read_csv(B/'resources/expected_SHARE_FI_screen.csv')
        cols=['concept','selected_main','selected_noMob','common_equiv']
        rt.require(screen[cols].reset_index(drop=True).equals(expected[cols].reset_index(drop=True)),'FI_SCREEN_CHANGED')
        value,k,status=a.fi_versions(S,screen,[])['main']
        rt.require(k==36 and value is not None,'FROZEN_SHARE_FI_NOT_36_ITEMS')
        zz=z.copy();fi_map=pd.Series(value/.1,index=d.id)
        zz['FI']=zz.id.map(fi_map);zz=zz.dropna(subset=['FI']).copy()
        rt.require(zz.id.nunique()==5787 and int(zz.event.sum())==2569,'R1_FI_COMPLETE_CASE_SAMPLE_CHANGED')
        m1=fit_cloglog(zz,'low_EWGSOP2',c,add_fi=True);e1=effect(m1)
        old=pd.read_csv(B/'resources/v07_T2_models.csv')
        old=old[old.cohort.eq('SHARE')&old.FI_version.eq('main')&old.outcome.eq('composite')&old.analysis.eq('grip_adjusted_FI')].iloc[0]
        for key in ['log_effect','se','effect','ci_low','ci_high']:
            rt.require(abs(e1[key]-float(old[key]))<1e-6,'FROZEN_FI_MODEL_REPRODUCTION_FAILED:'+key)
        m0=fit_cloglog(zz,'low_EWGSOP2',c);e0=effect(m0)
        row={'cohort':c,'n':5787,'events_n':2569,'FI_items':36,'status':'OK','analysis_origin':'post hoc after v0.7 review',
            'sample':'identical FI-complete primary people and person-wave rows','survey_weighting':'none',
            'reference_HR':e0['effect'],'reference_ci_low':e0['ci_low'],'reference_ci_high':e0['ci_high'],
            'reference_logHR':e0['log_effect'],'reference_cluster_SE':e0['se'],
            'FI_HR':e1['effect'],'FI_ci_low':e1['ci_low'],'FI_ci_high':e1['ci_high'],
            'FI_logHR':e1['log_effect'],'FI_cluster_SE':e1['se'],'FI_reproduction':'PASS',
            'relative_change_definition':'100*(beta_noFI-beta_FI)/beta_noFI; descriptive, not mediated proportion'}
        try:
            row.update(paired_bootstrap(m0,m1,run/'_checkpoints/R1_paired_bootstrap.json',
                reps=config['bootstrap_replicates'],min_valid=config['bootstrap_min_valid'],seed=config['R1_seed'],
                tolerance=config['R1_zero_logHR_tolerance'],progress=lambda a,b:print('SHARE R1 bootstrap',a,'/',b,flush=True)))
        except NoValidResult as e:
            row.update(relative_change_status=str(e),relative_logHR_change_percent=np.nan,relative_change_ci_low=np.nan,relative_change_ci_high=np.nan,
                bootstrap_requested=500,bootstrap_successful=0)
        pd.DataFrame([row]).to_csv(output/'R1_SHARE_same_sample.csv',index=False)
    # File/column metadata only in the return receipt; never rows or identifiers.
    atomic(run/f'{c}_done.json',{'cohort':c,'status':'COMPLETE','primary_n':n,'primary_events':ev,
        'HRS_files_read':0,'individual_rows_written':False,'calendar_date_columns_read':False,
        'input_versions':[{'basename':r['basename'],'size':r['size'],'mtime_ns':r['mtime_ns']} for r in rt.READS]})

def assemble(run,config,digest):
    import pandas as pd
    out=run/'return';table=pd.read_csv(B/'resources/R2_definition_comparison_PARTIAL.csv')
    prevalences=pd.concat([pd.read_csv(out/f'R2_prevalence_{c}.csv') for c in config['requested_cohorts']],ignore_index=True)
    for i,r in table.iterrows():
        q=prevalences[prevalences.cohort.eq(r.cohort)&prevalences.definition.eq(r.definition)]
        if len(q)!=1:raise RuntimeError('R2_NONUNIQUE_DOMAIN_BINDING')
        a=q.iloc[0]
        if int(a.prognostic_n)!=int(r.prognostic_n):raise RuntimeError('R2_RISK_PREVALENCE_DENOMINATOR_CHANGED')
        for old,new in [('flagged_per100','flagged_per100'),('ci_low_per100','flagged_ci_low_per100'),('ci_high_per100','flagged_ci_high_per100')]:
            table.loc[i,new]=a.get(old,float('nan'))
        table.loc[i,'prevalence_status']=a.status
        for k in ['positive_weight_n','excluded_weight_n','population_match_status','positive_n','weight','design_df','singleton_strata']:
            table.loc[i,k]=a.get(k,float('nan'))
    table.to_csv(out/'R2_definition_comparison.csv',index=False)
    for name in ['R3_SHARE_FI_validity.csv','S1_analysis_registry_v0_6_to_v0_8.csv','method_extension_v0_8.md','method_lock.json']:
        (out/name).write_bytes((B/'resources'/name).read_bytes())
    r1=pd.read_csv(out/'R1_SHARE_same_sample.csv').iloc[0]
    same=table.population_match_status.eq('EXACT_PRIMARY_PEOPLE').all();prevalence_ok=table.prevalence_status.eq('OK').all()
    state={'status':'COMPLETE' if same and prevalence_ok and r1.relative_change_status=='OK' else 'COMPLETE_WITH_DECLARED_UNAVAILABLE_BRANCHES',
        'R1':str(r1.relative_change_status),'R2':'COMPLETE' if prevalence_ok else 'PARTIAL_PREVALENCE_UNAVAILABLE',
        'main_fig4_population_gate':'PASS' if same and prevalence_ok else 'WAIT_FOR_AUTHOR_DOMAIN_DECISION_OR_VALID_CI',
        'individual_rows_written':False,'HRS_files_read':0,'HRS_scripts_run':0,'all_extensions_post_hoc':True,
        'primary_models_modified':False,'risk_AUC_models_refit':False,'package_manifest_sha256':digest,
        'method_sha256':json.loads((B/'resources/method_lock.json').read_text())['sha256'],
        'original_review_source_sha256':config['original_review_source_sha256']}
    registry=pd.read_csv(out/'S1_analysis_registry_v0_6_to_v0_8.csv')
    registry['status_at_package_creation']=registry.status
    for key,value in [('V08-R1',state['R1']),('V08-R2',state['R2']),('V08-R5','FIGURE_READY' if same and prevalence_ok else state['main_fig4_population_gate'])]:
        registry.loc[registry.analysis_id.eq(key),'status']=value
    registry.to_csv(out/'S1_analysis_registry_v0_6_to_v0_8.csv',index=False)
    atomic(out/'RUN_STATUS.json',state)
    atomic(out/'RUN_RECEIPT.json',{'package_manifest_sha256':digest,'dependency_versions':config['dependencies'],
        'steps':[json.loads((run/f'{c}_done.json').read_text()) for c in config['requested_cohorts']],
        'models_requested':'SHARE same-sample reference plus FI reproduction and paired500 coefficient bootstrap only',
        'participant_rows_exported':False})
    # Main figure and supplemental figure consume summary tables only.
    if same and prevalence_ok:
        subprocess.run([sys.executable,str(B/'render_fig4_v08.py'),'--table',str(out/'R2_definition_comparison.csv'),'--out',str(run/'figures')],check=True)
    pack(run)
    return state

def pack(run):
    folder=run/'return';folder.mkdir(exist_ok=True)
    files=[p for p in folder.iterdir() if p.is_file() and p.name!='RETURN_MANIFEST.csv']
    forbidden={'id','pid','mergeid','hhidpn','idauniq','unhhid','source_row','person'}
    for p in files:
        if p.suffix not in ['.csv','.md','.json']:raise RuntimeError('NONAGGREGATE_RETURN_FORMAT')
        if p.suffix=='.csv':
            with p.open() as f:header=next(csv.reader(f),[])
            if forbidden&{x.lower() for x in header}:raise RuntimeError('PARTICIPANT_COLUMN_IN_RETURN')
    with (folder/'RETURN_MANIFEST.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','sha256','bytes']);w.writeheader()
        for p in sorted(files):w.writerow({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    with zipfile.ZipFile(run/'RETURN_PACKAGE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.iterdir()):z.write(p,p.name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--worker',choices=['CHARLS','MHAS','SHARE']);parser.add_argument('--run-dir')
    args=parser.parse_args();digest=verify_bundle();expected_count_contract();config=json.loads((B/'config.json').read_text());dependencies(config)
    if args.self_test:
        subprocess.run([sys.executable,str(B/'tests/test_synthetic_v08.py')],check=True);return
    if args.check:
        print('Bundle, pinned software and complete frozen count contract PASS. No participant source opened or statted.');return
    if args.worker:
        try:
            with warnings.catch_warnings(record=True),contextlib.redirect_stderr(io.StringIO()):worker(args.worker,Path(args.run_dir),config)
        except BaseException as e:
            import runtime as rt
            code=str(e) if isinstance(e,(rt.SafeStop,)) else type(e).__name__
            diagnostic={'status':'STOPPED','cohort':args.worker,'error_code':code,**engineering_trace(e)}
            atomic(Path(args.run_dir)/f'{args.worker}_FAILED.json',diagnostic)
            atomic(Path(args.run_dir)/'return/ENGINEERING_DIAGNOSTIC.json',diagnostic)
            print('Worker stopped:',args.worker,code,flush=True);raise SystemExit(2)
        return
    if not args.run:parser.error('Real execution requires --run, in the independent researcher terminal.')
    audit_nonhrs()
    runs=B/'runs';runs.mkdir(exist_ok=True);signature=hashlib.sha256(json.dumps({'bundle':digest,'config':config},sort_keys=True).encode()).hexdigest()
    prior=sorted(runs.glob('RUN_*'));run=prior[-1] if prior else runs/('RUN_'+datetime.datetime.now(ZoneInfo(config['timezone'])).strftime('%Y%m%d_%H%M%S'))
    if run.exists():
        control=json.loads((run/'CONTROL.json').read_text())
        if control['signature']!=signature:raise RuntimeError('PACKAGE_CHANGED_USE_NEW_DIRECTORY')
        if control.get('status')=='COMPLETE':print('Already complete:',run/'RETURN_PACKAGE.zip');return
    else:run.mkdir();atomic(run/'CONTROL.json',{'signature':signature,'status':'IN_PROGRESS'})
    lock=run/'RUNNING.lock'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError('RUN_ALREADY_ACTIVE')
    lock.write_text(str(os.getpid()));(run/'_checkpoints').mkdir(exist_ok=True);(run/'return').mkdir(exist_ok=True)
    try:
        import runtime as rt
        for x in rt.sources():rt.validate_source(x['path'])  # metadata only before independent workers
        for c in config['requested_cohorts']:
            if (run/f'{c}_done.json').exists():print('Resume: skip',c,flush=True);continue
            print('Starting',c,flush=True)
            p=subprocess.run([sys.executable,'-u',str(B/'run_v08.py'),'--worker',c,'--run-dir',str(run)])
            if p.returncode:
                error=run/f'{c}_FAILED.json';code=json.loads(error.read_text())['error_code'] if error.exists() else 'INTERRUPTED'
                raise rt.SafeStop(code)
        state=assemble(run,config,digest);atomic(run/'CONTROL.json',{'signature':signature,'status':'COMPLETE'})
        print(state['status'],'Return only:',run/'RETURN_PACKAGE.zip',flush=True)
        print('Figures (summary-only):',run/'figures',flush=True)
    except BaseException as e:
        import runtime as rt
        code=str(e) if isinstance(e,rt.SafeStop) else type(e).__name__
        atomic(run/'return/PARENT_ENGINEERING_DIAGNOSTIC.json',engineering_trace(e))
        atomic(run/'return/RUN_STATUS.json',{'status':'STOPPED_PARTIAL','error_code':code,'biological_negative':False,'HRS_files_read':0,'individual_rows_written':False})
        pack(run);atomic(run/'CONTROL.json',{'signature':signature,'status':'STOPPED','error_code':code})
        print('STOPPED:',code,'; return aggregate diagnostic ZIP:',run/'RETURN_PACKAGE.zip',flush=True);raise SystemExit(2)
    finally:
        if lock.exists():lock.unlink()

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,ValueError) as e:
        print('PRECHECK STOP:',str(e) if isinstance(e,RuntimeError) else type(e).__name__);raise SystemExit(2)
