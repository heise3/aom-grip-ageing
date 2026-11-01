"""Provider-threshold disclosure treatment for aggregate returns, never data rows."""
import numpy as np,pandas as pd

FORBIDDEN={'id','pid','mergeid','hhidpn','idauniq','unhhid','source_row','person'}
DESCRIPTIVE={'mean','sd','SD','median','q1','q3','P99','max','mean_score','age_slope','women_mean','men_mean','difference_women_minus_men'}
RATIOS={'percent','proportion','flagged_per100','ci_low_per100','ci_high_per100','prevalence','prevalence_score_positive','missing_fraction'}
NONPARTICIPANT_COUNTS={'item_n','selected_item_n','candidate_item_n','bootstrap_n','replicate_n','parameter_n'}

def release(df,thresholds):
    df=df.copy()
    if FORBIDDEN&set(df.columns):raise ValueError('Individual-column output forbidden')
    if df.empty:return df
    if 'cohort' not in df:df['cohort']='ALL'
    for cohort,ix in df.groupby('cohort',dropna=False).groups.items():
        k=int(thresholds.get(str(cohort),1))
        if k<=1:continue
        counts=[c for c in df if (c=='n' or c.endswith('_n')) and c not in NONPARTICIPANT_COUNTS]
        hits=pd.Series(False,index=ix)
        for col in counts:
            v=pd.to_numeric(df.loc[ix,col],errors='coerce');mask=v.gt(0)&v.lt(k)
            if mask.any():
                df[col]=df[col].astype(object);df.loc[v.index[mask],col]=f'<{k}';hits|=mask
        if not hits.any():continue
        # For category partitions protect all categories/groups of the affected variable;
        # otherwise protect this table's count partitions and directly derived statistics.
        affected=ix
        if 'variable' in df:
            affected=df.loc[ix].index[df.loc[ix,'variable'].isin(df.loc[hits.index[hits],'variable'])]
        for col in counts:
            v=df.loc[affected,col];numeric=pd.to_numeric(v,errors='coerce');protect=numeric.notna()&numeric.ne(0)
            df[col]=df[col].astype(object);df.loc[v.index[protect],col]='SUPPRESSED_COMPLEMENT_PROVIDER'
        for col in RATIOS|DESCRIPTIVE:
            if col in df:
                df[col]=df[col].astype(object);df.loc[affected,col]='SUPPRESSED_PROVIDER'
        if 'category_n' in df:
            # Group totals and means cannot restore the protected category because all
            # category counts/percentages for that variable across groups are hidden.
            pass
    return df
