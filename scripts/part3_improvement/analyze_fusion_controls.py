"""Analyze verified main fusion, history controls and frozen development candidates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from _common import ROOT,repo_path,checksum,write_csv,write_json
from _video_review import read_json,read_rows,contact_sheet,verify_preview
from part3_improvement.run_supplementary_ablations import load_sources,describe,reference_specs,specifications


def load_variant(folder):
    meta=read_json(folder/'metadata.json')
    arrays=[np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('depths','valid','history_weight')]
    return meta,arrays


def check_row(row,saved):
    for k,v in row.items():
        if v is None:
            if saved[k]!='': raise ValueError('Unexpected defined '+k)
        elif isinstance(v,(int,float)) and not isinstance(v,bool):
            if not np.isclose(v,float(saved[k]),rtol=1e-10,atol=1e-12): raise ValueError('Changed '+k)
        elif str(v)!=saved[k]: raise ValueError('Changed '+k)


def seal(out,sources,extra):
    write_json(out/'quantitative_review.json',{'status':'quantitative_complete_visual_pending',
        'source_sha256':{p.relative_to(ROOT).as_posix():checksum(p) for p in sources},
        'implementation_sha256':{p:checksum(ROOT/p) for p in ('scripts/part3_improvement/analyze_fusion_controls.py',
            'scripts/_video_review.py','scripts/part3_improvement/run_supplementary_ablations.py','src/p3_depth/evaluation/temporal.py')},
        'scope':'Verified frozen/reference and supplementary artifact hashes, recomputed all diagnostic scores and output departures; no model inference or prediction changes.',
        'limitations':'No video depth GT. T, departure and history weights are diagnostics. Source timing sums are not a repeated end-to-end benchmark. Visual conclusions require separate actual inspection.',
        **extra,'artifacts_sha256':{p.relative_to(out).as_posix():checksum(p) for p in out.rglob('*') if p.is_file()}})


def candidate_analysis(source,out,frames,roi,windows):
    path=ROOT/'outputs/part3_improvement/selection/selection.json'; selection=read_json(path)
    if selection['selected_candidate']!='alpha070': raise ValueError('Selected candidate changed')
    if checksum(ROOT/selection['review_path'])!=selection['review_sha256']: raise ValueError('Selection review changed')
    rows=[]; arrays=[]; masks=[]; sources=[path,ROOT/selection['review_path'],ROOT/'outputs/part3_improvement/selection/fusion_frozen.yaml']
    fixed=None
    for candidate,entries in selection['candidates'].items():
        if len(entries)!=1 or entries[0]['clip_id']!='dev_tokyo_100': raise ValueError('Unexpected candidate set')
        entry=entries[0]; suite=ROOT/entry['suite']
        for name,key in (('metadata.json','metadata_sha256'),('verification.json','verification_sha256'),('comparison.csv','comparison_sha256')):
            if checksum(suite/name)!=entry[key]: raise ValueError('Candidate source changed')
            sources.append(suite/name)
        s=load_sources(source['cache'],source['baseline'],source['flow'],suite)
        params=dict(s['sm']['config']['parameters']); alpha=params.pop('history_weight')
        if fixed is not None and params!=fixed: raise ValueError('Candidates change more than alpha')
        fixed=params
        m,(d,v,w)=load_variant(suite/'E4')
        spec={'id':'E4_recursive','variant':'E4','history_source':'recursive','confidence_terms':['photo','flow','depth','edge']}
        row,scores,_=describe(d,v,w,spec,s,m['fusion_seconds'],True)
        if scores!=read_json(suite/'E4/metrics.json'): raise ValueError('Candidate metrics do not recompute')
        rows.append({'candidate':candidate,'alpha':alpha,**row}); arrays.append(d); masks.append(v)
        verify_preview(source['cache'],suite/'E4',s['times'],s['limits'])
    out.mkdir(parents=True); (out/'figures').mkdir(); write_csv(out/'comparison.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    axes[0].plot([r['alpha'] for r in rows],[r['capped_normalized_l1_full_reference'] for r in rows],'o-')
    axes[0].set(xlabel='Maximum history weight alpha',ylabel='T',title='Temporal variation (not accuracy)')
    axes[1].plot([r['alpha'] for r in rows],[r['mean_departure_from_DA_normalized'] for r in rows],'o-')
    axes[1].set(xlabel='Maximum history weight alpha',ylabel='Mean |E4-DA| / DA span',title='Departure from current DA (not error)')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Tokyo dev | only alpha changes | frozen alpha=0.70 retained')
    fig.savefig(out/'figures/tradeoff.png',dpi=180); plt.close(fig)
    for win in windows:
        contact_sheet(frames,arrays,masks,[source['limits']]*3,[r['candidate'] for r in rows],source['times'],win,
                      out/f'figures/roi_{win[0]}_{win[-1]}.png',roi=roi)
    seal(out,sources,{'selection_unchanged':True,'selected_candidate':'alpha070','parameters_other_than_alpha':fixed,
                     'roi_xyxy':roi,'windows':windows,'selection':'Fixed moving-subject windows also used for main/history review; no new candidate selection.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip',choices=('dev_tokyo_100','course_rollercoaster_70'),default='course_rollercoaster_70')
    p.add_argument('--output-root',help='Optional new root containing main/, history/ and candidates/ analyses')
    a=p.parse_args(); dev=a.clip=='dev_tokyo_100'; split='dev' if dev else 'test'
    cache=ROOT/'outputs/shared/video_cache'/a.clip; base=ROOT/'outputs/part1_baseline/dav2_relative'/split/a.clip
    flow=ROOT/'outputs/part3_improvement/flow'/split/a.clip
    suite=ROOT/('outputs/part3_improvement/dev_candidates/alpha070' if dev else 'outputs/part3_improvement/test')/a.clip
    history=ROOT/'outputs/part3_improvement/supplementary'/split/a.clip/'history'
    root=repo_path(a.output_root) if a.output_root else None
    out=root/'main' if root else suite/'analysis'; hout=root/'history' if root else history/'analysis'
    cout=root/'candidates' if root else ROOT/'outputs/part3_improvement/selection/analysis'
    if any(x.exists() for x in [out,hout]+([cout] if dev else [])): raise FileExistsError('Preserve prior analyses; use a new --output-root')
    source=load_sources(cache,base,flow,suite); hm=read_json(history/'metadata.json'); receipt=read_json(history/'verification.json')
    if (hm['experiment']!='history' or receipt['status']!='verified' or receipt['supplement_metadata_sha256']!=checksum(history/'metadata.json')
        or hm['source_suite_sha256']!=checksum(suite/'metadata.json') or hm['parameters']!=source['sm']['config']['parameters']):
        raise ValueError('History source/configuration mismatch')
    for name,digest in hm['artifacts_sha256'].items():
        if checksum(history/name)!=digest: raise ValueError('History artifact changed: '+name)
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    roi=[200,60,390,350] if dev else [100,60,450,260]
    windows=[list(range(28,34)),list(range(60,66))] if dev else [list(range(14,20)),list(range(32,38)),list(range(64,70))]
    overview=np.linspace(0,len(frames)-1,6).round().astype(int).tolist()
    originals={}; rows=[]; all_pairs=[]; history_rows=[]
    for spec in reference_specs('history'):
        folder=suite/spec['variant']; meta,arrays=load_variant(folder); d,v,w=arrays
        row,scores,pairs=describe(d,v,w,spec,source,meta['fusion_seconds'],True)
        if scores!=read_json(folder/'metrics.json'): raise ValueError('Main diagnostics differ: '+spec['id'])
        originals[spec['variant']]=arrays; history_rows.append(row)
        ref=source['fields']['common_mask']; selected=ref & v[1:]
        partitions={label:selected & mask for label,mask in (('DA_positive',source['base'][1:]>0),('DA_zero',source['base'][1:]==0))}
        rows.append({**row,**{'mean_history_weight_'+k:float(w[1:][m].astype(np.float64).mean()) if m.any() else None for k,m in partitions.items()}})
        all_pairs.extend({'variant':spec['variant'],**r} for r in pairs)
        verify_preview(cache,folder,source['times'],source['limits'])
    pairs_by_variant={}
    for spec in specifications('history'):
        folder=history/spec['id']; meta,arrays=load_variant(folder); d,v,w=arrays
        row,scores,_=describe(d,v,w,spec,source,meta['fusion_seconds'],False)
        if scores!=read_json(folder/'metrics.json'): raise ValueError('Pairwise diagnostics differ')
        history_rows.append(row); pairs_by_variant[spec['variant']]=arrays
    saved=read_rows(history/'comparison.csv')
    if len(saved)!=len(history_rows): raise ValueError('Incomplete history table')
    for row,old in zip(history_rows,saved): check_row(row,old)
    contrasts=[]
    for v in ('E1','E2','E3','E4'):
        r=next(r for r in history_rows if r['id']==v+'_recursive'); b=next(r for r in history_rows if r['id']==v+'_pairwise')
        key='capped_normalized_l1_full_reference'
        contrasts.append({'variant':v,'recursive_T':r[key],'pairwise_T':b[key],
            'recursive_T_reduction_percent':100*(1-r[key]/b[key]),
            'recursive_departure':r['mean_departure_from_DA_normalized'],'pairwise_departure':b['mean_departure_from_DA_normalized'],
            'recursive_mean_weight':r['mean_history_weight_on_reference'],'pairwise_mean_weight':b['mean_history_weight_on_reference']})
    out.mkdir(parents=True); (out/'figures').mkdir(); hout.mkdir(parents=True); (hout/'figures').mkdir()
    write_csv(out/'comparison.csv',rows); write_csv(out/'per_pair.csv',all_pairs)
    write_csv(hout/'comparison.csv',history_rows); write_csv(hout/'contrasts.csv',contrasts)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=[r['variant'] for r in rows]
    axes[0].bar(labels,[r['capped_normalized_l1_full_reference'] for r in rows]); axes[0].set(title='Flow-aligned temporal variation',ylabel='T (not accuracy)')
    axes[1].bar(labels,[r['mean_departure_from_DA_normalized'] for r in rows]); axes[1].set(title='Departure from raw DA',ylabel='Mean |output-DA| / DA span (not error)')
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.suptitle(a.clip+' | identical reference pixels and fixed DA span')
    fig.savefig(out/'figures/main_tradeoff.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained'); x=np.arange(4)
    for ax,key,title in zip(axes,('T','departure'),('Temporal variation','Departure from raw DA')):
        ax.bar(x-.18,[r['pairwise_'+key] for r in contrasts],.36,label='pairwise')
        ax.bar(x+.18,[r['recursive_'+key] for r in contrasts],.36,label='recursive')
        ax.set(title=title,xticks=x,xticklabels=['E1','E2','E3','E4']); ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.suptitle(a.clip+' | history source control; parameters fixed, resulting E4 weights may change')
    fig.savefig(hout/'figures/history_tradeoff.png',dpi=180); plt.close(fig)
    contact_sheet(frames,[originals[v][0] for v in labels],[originals[v][1] for v in labels],
                  [source['limits']]*5,labels,source['times'],overview,out/'figures/overview.png')
    for win in windows:
        contact_sheet(frames,[originals[v][0] for v in labels],[originals[v][1] for v in labels],
            [source['limits']]*5,labels,source['times'],win,out/f'figures/roi_{win[0]}_{win[-1]}.png',roi=roi)
    # One matched continuous window per variant keeps the renewed review compact.
    for v in ('E1','E2','E3','E4'):
        win=windows[0]; rec=originals[v]; pair=pairs_by_variant[v]
        contact_sheet(frames,[rec[0],pair[0]],[rec[1],pair[1]],[source['limits']]*2,[v+'_recursive',v+'_pairwise'],
                      source['times'],win,hout/f'figures/{v}_{win[0]}_{win[-1]}.png',roi=roi)
    sources=[p/'metadata.json' for p in (cache,base,flow,suite,history)]+[suite/'verification.json',history/'verification.json']
    seal(out,sources,{'clip':a.clip,'roi_xyxy':roi,'overview_frames':overview,'windows':windows,'parameters':source['sm']['config']['parameters'],
        'selection':'Uniform overview plus fixed moving-subject windows; post-hoc diagnostic selection.',
        'weight_partitions':'DA positive/zero values are prediction-defined diagnostic partitions, not semantic foreground/background masks.'})
    seal(hout,sources,{'clip':a.clip,'roi_xyxy':roi,'windows':[windows[0]],'parameters':source['sm']['config']['parameters'],
        'selection':'One renewed matched continuous window per variant; older broader visual records remain preserved.',
        'history_control':'Only previous raw versus previous fused input changes. The E4 weight formula and parameters remain fixed; depth confidence depends on history, so numeric weights need not match.'})
    if dev: candidate_analysis(source,cout,frames,roi,windows)
    print({'clip':a.clip,'main':str(out),'history':str(hout),'candidates':str(cout) if dev else None},flush=True)


if __name__=='__main__': main()
