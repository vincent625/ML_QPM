#!/usr/bin/env python3
"""Offline, roster-reconciled cohort screen using the plugin's exact QC rules.

Never modifies a reference bank, reparses instrument files, sums overlapping
archive read totals, or infers a QPM root cause from an outcome flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from qpm_reference_bank import resolve_reference_bank,file_sha256 as sha,MANIFEST
from qpm_reference_qualification import (load_policy,build_reference_qualification,
    classify_barcode_outcomes,lane_tokens)

VERSION="all-csa-cohort-screen-1.0.0"
MANUAL_DEFAULT=Path(__file__).resolve().parent.parent / 'private_data' / 'manual_reports'
ISSUES=['dropout','partial_imbalance','overrepresented']


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [safe(v) for v in value]
    if isinstance(value,np.generic):return safe(value.item())
    if value is None or value is pd.NA or value is pd.NaT:return None
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False))


def text(value):
    if value is None or pd.isna(value):return ''
    value=str(value).strip()
    return '' if value.lower() in {'none','nan','null','n/a'} else value


def unique(values):return list(dict.fromkeys(text(v) for v in values if text(v)))


def script_json(payload):
    return json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<',r'\u003c').replace('>',r'\u003e').replace('&',r'\u0026')


def assay_context(row):
    """Use recorded package/panel identities; never guess an assay from run name."""
    packages,panels=[],[]
    for field in ['global.analysis_regions_file','global.analysis_fusions_file']:
        raw=text(row.get(field)).replace('\\','/')
        if not raw:continue
        parts=raw.split('/')
        if 'AssayImport' in parts:
            position=parts.index('AssayImport')+1
            if position<len(parts):packages.append(parts[position])
        panels.append(parts[-1])
    packages,panels=unique(packages),unique(panels)
    return dict(assay_panel_label=' | '.join(packages or panels) or 'Not recorded',
        assay_label_basis='recorded_AssayImport_package' if packages else 'recorded_panel_filename' if panels else 'not_recorded',
        assay_type=text(row.get('global.analysis_application_type')) or 'Not recorded',
        panel_files=' | '.join(panels),panel_technology=text(row.get('global.analysis_panel_technology')),
        workflow_script=text(row.get('planned_run.object_script_name')),run_mode=text(row.get('planned_run.object_run_type')))


def attach_manual(core,bars,cases):
    """Reuse identical payload/scope; append only a genuinely new observation."""
    core,bars=core.copy(),bars.copy()
    mappings=[]
    extras=[]
    extra_bars=[]
    for case in cases:
        project=Path(case['project'])
        incoming=pd.read_parquet(project/'csa_ml_dataset.parquet')
        target=incoming[incoming.observation_id.eq(case['observation_id'])]
        if len(target)!=1:raise ValueError('Manual target identity ambiguous')
        row=target.iloc[0]
        matches=core[core.archive_sha256.eq(case['archive_sha256'])&core.chip_lane.astype(str).eq(str(row.chip_lane))]
        if len(matches)>1:raise ValueError('Manual payload maps to multiple same-lane primaries')
        if len(matches):
            mapped=matches.iloc[0]
            if mapped.physical_run_id!=row.physical_run_id:raise ValueError('Identical manual payload has inconsistent physical identity')
            if pd.notna(mapped.total_reads) and pd.notna(row.total_reads):assert np.isclose(mapped.total_reads,row.total_reads)
            observation=mapped.observation_id
            action='already_represented_payload_and_lane'
        elif row.observation_id in set(core.observation_id):
            # A new export of the same raw scope must not silently replace a
            # more complete sealed primary or create an independent run.
            observation=row.observation_id
            action='same_raw_scope_new_export_not_counted_twice'
        else:
            if any(r['observation_id']==row.observation_id for r in extras):raise ValueError('Duplicate new manual scope')
            extras.append(row.to_dict())
            target_bars=pd.read_csv(project/'qpm_failure_examples/dna_rna_barcode_counts.csv')
            target_bars=target_bars[target_bars.observation_id.eq(row.observation_id)]
            extra_bars.append(target_bars)
            observation=row.observation_id
            action='added_new_manual_observation'
        mappings.append(dict(input_filename=case['input_filename'],input_path=case['input_path'],
            input_sha256=case['input_sha256'],payload_sha256=case['archive_sha256'],observation_id=observation,
            physical_run_id=row.physical_run_id,instrument=case['hostname'],chip_lane=row.chip_lane,
            action=action,project=str(project),qpm_report=str(Path(case['report_dir'])/'qpm_dropout_report.html')))
    if extras:core=pd.concat([core,pd.DataFrame(extras)],ignore_index=True)
    if extra_bars:bars=pd.concat([bars,*extra_bars],ignore_index=True)
    assert not core.observation_id.duplicated().any()
    assert not core.duplicated(['physical_run_id','chip_lane']).any()
    return core,bars,pd.DataFrame(mappings)


def barcode_material_summary(classified,observations):
    summaries=[]
    for observation in observations:
        group=classified[classified.observation_id.eq(observation)]
        record={'observation_id':observation}
        for material in ['DNA','RNA']:
            rows=group[group.material_type.eq(material)&group.is_control.eq(False).fillna(False)]
            stem=material.lower()
            record[stem+'_noncontrol_barcode_rows']=len(rows)
            record[stem+'_median_reads']=float(rows.read_count.median()) if rows.read_count.notna().any() else np.nan
            for kind in [*ISSUES,'unresolved']:
                selected=rows[rows.outcome_class.eq(kind)]
                record[stem+'_'+kind+'_count']=len(selected)
                record[stem+'_'+kind+'_orders']=', '.join(f'{float(v):g}' for v in selected.planned_order if pd.notna(v)) if 'planned_order' in selected else ''
        summaries.append(record)
    return pd.DataFrame(summaries)


def low_read_text(row):
    reads='unknown' if pd.isna(row.total_reads_value) else f'{row.total_reads_value/1e6:.2f}M'
    threshold='unknown' if pd.isna(row.total_reads_threshold) else f'{row.total_reads_threshold/1e6:g}M'
    return f'Lanes {row.chip_lane}: {reads} / {threshold} minimum ({row.total_read_criterion_status})'


def summarize_runs(observations):
    rows=[]
    for physical,group in observations.groupby('physical_run_id',sort=False):
        assert group.instrument_id.nunique()==1
        dropout=group.dropout_barcode_count.gt(0).any()
        imbalance=(group.partial_imbalance_barcode_count+group.overrepresented_barcode_count).gt(0).any()
        low=group.total_read_criterion_status.eq('below_threshold').any()
        unknown_reads=group.total_read_criterion_status.eq('unresolved').any()
        unknown_bars=group.barcode_screen_reasons.fillna('').map(lambda v:any(r and r!='sample_dropout_or_imbalance' for r in v.split(';'))).any()
        incomplete=bool(unknown_reads or unknown_bars)
        flagged=bool(dropout or imbalance or low)
        lanes=[set(lane_tokens(v) or ()) for v in group.chip_lane]
        overlaps=any(a&b for i,a in enumerate(lanes) for b in lanes[i+1:])
        known_lanes=sorted(set().union(*lanes)) if lanes else []
        names=unique(group.run_name)
        dates=unique(group.run_start_time)
        chips=unique(group.chip_type)
        assays=unique(group.assay_panel_label)
        flags=[name for name,yes in [('Dropout',dropout),('Imbalance',imbalance),('Low reads',low)] if yes]
        row=dict(physical_run_id=physical,run_date=dates[0][:10] if dates else 'Not recorded',
            run_start_time=' | '.join(dates),instrument=' | '.join(unique(group.instrument)),instrument_id=group.instrument_id.iloc[0],
            run_name=names[0] if names else 'Not recorded',all_run_names=' | '.join(names),chip_type=' | '.join(chips),
            assay_panel=' | '.join(assays),assay_type=' | '.join(unique(group.assay_type)),
            workflow=' | '.join(unique(group.workflow_script)),retained_observation_scopes=len(group),
            available_lane_tokens='_'.join(map(str,known_lanes)),overlapping_analyzed_scopes=overlaps,
            flags='; '.join(flags),has_dropout=bool(dropout),has_imbalance=bool(imbalance),has_low_read_scope=bool(low),flagged=flagged,
            screen_incomplete=incomplete,read_screen_unresolved=bool(unknown_reads),barcode_screen_incomplete=bool(unknown_bars),
            review_status='flagged_with_unresolved_fields' if flagged and incomplete else 'flagged' if flagged else
                          'unresolved_no_known_flag' if incomplete else 'no_flag_under_current_rules',
            lowest_scope_read_fraction=float(group.reads_fraction_of_threshold.min()) if group.reads_fraction_of_threshold.notna().any() else np.nan,
            reads_by_scope=' | '.join(low_read_text(r) for r in group.sort_values('chip_lane').itertuples()),
            observation_ids=' | '.join(group.observation_id),
            qpm_reports=' | '.join(unique(group.qpm_report)),
            read_scope_warning='No cross-archive sum. A low scope does not by itself establish low total yield over all physically active lanes.')
        for material in ['dna','rna']:
            for kind in ISSUES:row[material+'_'+kind+'_barcode_scope_calls']=int(group[material+'_'+kind+'_count'].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['run_date','instrument','run_name'],ascending=[False,True,True]).reset_index(drop=True)


def make_roster(bank,core,manual):
    transfers=pd.read_csv(bank/'csa_transfer_manifest.csv')
    audit=pd.read_csv(bank/'archive_parse_audit.csv')
    source=pd.read_csv(bank/'source_archive_status.csv')
    inventory=pd.read_csv(bank/'csa_inventory.csv').set_index('archive_id')
    source_ids=set(audit.archive_id)
    legacy=transfers[~transfers.archive_id.isin(source_ids)]
    rows=[]
    represented=set(core.archive_id)
    hashes={key:' | '.join(group.observation_id) for key,group in core.groupby('archive_sha256')}
    for r in legacy.itertuples():
        inv=inventory.loc[r.archive_id]
        rows.append(dict(source_group='earlier_collected_archive',archive_id=r.archive_id,filename=inv.filename,
            local_path=r.local_path,payload_sha256=r.sha256,source_instrument_folder='',
            disposition='retained_primary' if r.archive_id in represented else 'same_content_as_primary' if r.sha256 in hashes else 'unretained_earlier_export',
            represented_observation_ids=hashes.get(r.sha256,''),parse_error_type=''))
    duplicates=pd.read_csv(bank/'duplicate_archives.csv').set_index('archive_id')
    for r in source.itertuples():
        represented_obs=hashes.get(r.sha256,'')
        if not represented_obs and r.archive_id in duplicates.index:represented_obs=duplicates.loc[r.archive_id,'observation_id']
        rows.append(dict(source_group='expanded_2082_archive_collection',archive_id=r.archive_id,filename=r.filename,
            local_path=r.local_path,payload_sha256=r.sha256,source_instrument_folder=r.source_instrument_folder,
            disposition=r.reference_disposition,represented_observation_ids=represented_obs,
            parse_error_type=text(getattr(r,'error_type',''))))
    for r in manual.itertuples():
        rows.append(dict(source_group='manually_supplied_CSA',archive_id='manual_wrapper_'+r.input_sha256[:20],filename=r.input_filename,
            local_path=r.input_path,payload_sha256=r.payload_sha256,source_instrument_folder='',disposition=r.action,
            represented_observation_ids=r.observation_id,parse_error_type=''))
    result=pd.DataFrame(rows)
    assert len(result)==len(legacy)+len(source)+len(manual)
    assert not result.duplicated(['source_group','local_path']).any()
    return result


def create(project,manual_root,output):
    began=time.monotonic()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    bank=resolve_reference_bank(project)
    manifest=json.loads((bank/MANIFEST).read_text())
    required=['csa_ml_dataset.parquet','qpm_failure_examples/dna_rna_barcode_counts.csv','csa_inventory.csv',
        'csa_transfer_manifest.csv','csa_reference_qualification.csv','reference_qualification_policy.json','instrument_reference_coverage.csv']
    for name in required:assert sha(bank/name)==manifest['artifacts'][name]['sha256']
    source_hashes={str(bank/name):sha(bank/name) for name in required}
    for name in ['source_archive_status.csv','archive_parse_audit.csv','duplicate_archives.csv']:source_hashes[str(bank/name)]=sha(bank/name)
    original_core=pd.read_parquet(bank/'csa_ml_dataset.parquet')
    original_bars=pd.read_csv(bank/'qpm_failure_examples/dna_rna_barcode_counts.csv')
    policy=load_policy(bank)
    fresh=build_reference_qualification(original_core,original_bars,policy).sort_values('observation_id').reset_index(drop=True)
    frozen=pd.read_csv(bank/'csa_reference_qualification.csv').sort_values('observation_id').reset_index(drop=True)
    comparable=['observation_id','total_read_criterion_status','dropout_barcode_count','partial_imbalance_barcode_count',
        'overrepresented_barcode_count','barcode_screen_status','qualification_status']
    pd.testing.assert_frame_equal(fresh[comparable],frozen[comparable],check_dtype=False)
    cases=[]
    if manual_root is not None:
        assert json.loads((manual_root/'validation.json').read_text())['status']=='passed'
        cases=json.loads((manual_root/'cases.json').read_text())
        source_hashes[str(manual_root/'cases.json')]=sha(manual_root/'cases.json')
        for case in cases:
            assert sha(Path(case['input_path']))==case['input_sha256']
            p=Path(case['project'])
            for name in ['csa_ml_dataset.parquet','qpm_failure_examples/dna_rna_barcode_counts.csv']:
                source_hashes[str(p/name)]=sha(p/name)
    core,bars,manual=attach_manual(original_core,original_bars,cases)
    assert set(bars.observation_id)<=set(core.observation_id)
    qualified=build_reference_qualification(core,bars,policy)
    classified=classify_barcode_outcomes(bars,policy)
    metadata=[]
    inventory=pd.read_csv(bank/'csa_inventory.csv').set_index('archive_id')
    hosts=pd.read_csv(bank/'instrument_reference_coverage.csv').set_index('instrument_id').hostname.to_dict()
    manual_by_obs=manual.set_index('observation_id').to_dict('index') if len(manual) else {}
    qpm_by_phys=manual.groupby('physical_run_id').qpm_report.first().to_dict() if len(manual) else {}
    for row in core.to_dict('records'):
        manual_record=manual_by_obs.get(row['observation_id'])
        if row['archive_id'] in inventory.index:
            inv=inventory.loc[row['archive_id']]
            name,filename=text(inv.run_name),text(inv.filename)
        elif manual_record:
            filename=manual_record['input_filename'];name=re.sub(r'(?i)\.ChipLane.*$','',filename)
        else:raise ValueError('Primary archive missing run-name inventory')
        instrument=text(row.get('hostname_for_report')) or text(hosts.get(row['instrument_id']))
        if not instrument and manual_record:instrument=manual_record['instrument']
        metadata.append(dict(observation_id=row['observation_id'],run_name=name,filename=filename,
            run_start_time=text(row.get('run_start_time')),instrument=instrument or 'Not recorded',
            archive_sha256=row['archive_sha256'],analysis_id=row['analysis_id'],
            mean_read_length_bp=row.get('mean_read_length_bp'),raw_read_accuracy_pct=row.get('raw_read_accuracy_pct'),
            qpm_report=qpm_by_phys.get(row['physical_run_id'],''),**assay_context(row)))
    obs=qualified.merge(pd.DataFrame(metadata),on='observation_id',validate='one_to_one')
    material=barcode_material_summary(classified,obs.observation_id)
    obs=obs.merge(material,on='observation_id',validate='one_to_one')
    obs['has_dropout']=obs.dropout_barcode_count.gt(0)
    obs['has_imbalance']=(obs.partial_imbalance_barcode_count+obs.overrepresented_barcode_count).gt(0)
    obs['has_low_read_scope']=obs.total_read_criterion_status.eq('below_threshold')
    obs['flagged']=obs[['has_dropout','has_imbalance','has_low_read_scope']].any(axis=1)
    assert int(obs.flagged.sum())==int(obs.observation_qualification_status.eq('failed').sum())
    runs=summarize_runs(obs)
    barcode_columns=['observation_id','physical_run_id','archive_id','barcode_id','merged_pool_id','material_type','planned_order',
        'is_control','read_count','relative_to_material_median','dropout_threshold_reads','outcome_class','path_status','count_status','source']
    barcode_export=classified.reindex(columns=barcode_columns)
    roster=make_roster(bank,core,manual)
    exceptions=roster[roster.disposition.eq('failed')].copy()
    def group_summary(field):
        return runs.groupby(field,dropna=False).agg(physical_runs=('physical_run_id','size'),flagged_runs=('flagged','sum'),
            dropout_runs=('has_dropout','sum'),imbalance_runs=('has_imbalance','sum'),low_read_scope_runs=('has_low_read_scope','sum'),
            screen_incomplete_runs=('screen_incomplete','sum')).reset_index().sort_values('flagged_runs',ascending=False)
    instrument_summary=group_summary('instrument');chip_summary=group_summary('chip_type')
    frames={'all_runs.csv':runs,'flagged_runs.csv':runs[runs.flagged],'all_observations.csv':obs,
        'flagged_observations.csv':obs[obs.flagged],'barcode_calls.csv':barcode_export,
        'flagged_barcode_calls.csv':barcode_export[barcode_export.outcome_class.isin(ISSUES)],
        'source_archive_roster.csv':roster,'unparsed_archive_exceptions.csv':exceptions,'manual_CSA_reconciliation.csv':manual,
        'instrument_summary.csv':instrument_summary,'chip_summary.csv':chip_summary}
    for name,frame in frames.items():frame.to_csv(output/name,index=False)
    bars_by_obs={key:part.to_dict('records') for key,part in barcode_export.groupby('observation_id',sort=False)}
    obs_by_run={key:part.to_dict('records') for key,part in obs.groupby('physical_run_id',sort=False)}
    details={physical:[dict(**record,barcodes=bars_by_obs.get(record['observation_id'],[])) for record in records] for physical,records in obs_by_run.items()}
    stats=dict(source_archive_entries=len(roster),source_groups=roster.source_group.value_counts().to_dict(),
        primary_observations=len(obs),physical_runs=len(runs),instruments=runs.instrument_id.nunique(),
        flagged_physical_runs=int(runs.flagged.sum()),dropout_physical_runs=int(runs.has_dropout.sum()),
        imbalance_physical_runs=int(runs.has_imbalance.sum()),low_read_scope_physical_runs=int(runs.has_low_read_scope.sum()),
        flagged_observations=int(obs.flagged.sum()),dropout_observations=int(obs.has_dropout.sum()),
        imbalance_observations=int(obs.has_imbalance.sum()),low_read_observations=int(obs.has_low_read_scope.sum()),
        unresolved_read_observations=int(obs.total_read_criterion_status.eq('unresolved').sum()),
        incomplete_screen_physical_runs=int(runs.screen_incomplete.sum()),
        unresolved_no_known_flag_runs=int(runs.review_status.eq('unresolved_no_known_flag').sum()),
        no_rule_flag_complete_screen_runs=int(runs.review_status.eq('no_flag_under_current_rules').sum()),
        unparsed_source_archives=len(exceptions),manual_new_observations=int(manual.action.eq('added_new_manual_observation').sum()),
        date_min=runs.run_date.min(),date_max=runs.run_date.max())
    payload=dict(version=VERSION,generated_utc=datetime.now(timezone.utc).isoformat(),stats=stats,
        policy=policy,runs=runs.to_dict('records'),details=details,
        instrument_summary=instrument_summary.to_dict('records'),chip_summary=chip_summary.to_dict('records'),
        exceptions=exceptions[['filename','source_instrument_folder','parse_error_type','disposition']].to_dict('records'))
    write_json(output/'report_data.json',payload)
    write_json(output/'summary.json',stats)
    (output/'screen_all_csa_runs.executed.py').write_bytes(Path(__file__).read_bytes())
    names=[*frames,'report_data.json','summary.json','screen_all_csa_runs.executed.py']
    write_json(output/'screen_validation.json',dict(status='passed',version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        source_bank=str(bank),source_manifest_sha256=sha(bank/MANIFEST),input_sha256=source_hashes,
        original_frozen_screen_reproduced=True,unique_primary_observations=True,unique_physical_run_table=True,
        archived_totals_never_summed=True,material_types_screened_separately=True,manual_payloads_reconciled=True,
        missing_data_never_labeled_pass=True,source_roster_reconciled=True,
        code_sha256=sha(Path(__file__)),policy=policy,execution_seconds=time.monotonic()-began,
        stats=stats,artifact_sha256={name:sha(output/name) for name in names},
        limitations=['Raw-run timestamps are retained as recorded/assumed UTC; instrument time zones are not independently verified.',
        'Physical-run low-read flag means at least one analyzed CSA scope is below its criterion, not a reconstructed whole-chip total.',
        'Barcode labels are apparent relative-outcome flags, not confirmed sample failures or QPM root causes.',
        'Assay display names are not guaranteed available; recorded assay type and package/panel identities are labeled explicitly.',
        'Unparsed archives remain unresolved and do not contribute invented run or assay metadata.']))
    print(json.dumps(safe(stats),indent=2),flush=True)


HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>All CSA run quality screen</title><style>
:root{--ink:#264653;--teal:#218c80;--gold:#d8b25c;--coral:#df7356;--paper:#f7f9f3}body{font:15px system-ui;line-height:1.45;color:var(--ink);background:var(--paper);margin:25px}h1{font-size:30px}section{background:white;padding:20px;border-radius:9px;margin:16px 0}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{padding:13px 22px;background:white;border-top:4px solid var(--teal);border-radius:6px}.card strong{display:block;font-size:28px}.warn{border-left:5px solid var(--gold);background:#fff8e8;padding:12px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}input,select,button{font:inherit;color:var(--ink);padding:7px;border:1px solid #bacdc5;border-radius:5px;background:white}select{max-width:420px}input[type=search]{min-width:300px}button,a{cursor:pointer}a,.link{color:var(--teal)}.link{border:0;padding:0;text-align:left;background:none;white-space:normal;font:inherit}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:9px;vertical-align:top;border-bottom:1px solid #e0e7df}th{position:sticky;top:0;background:#edf3eb;z-index:1}td.runname{min-width:260px;max-width:370px;overflow-wrap:anywhere}td.assay{min-width:200px;max-width:340px;overflow-wrap:anywhere}td.reads{min-width:230px}.scroll{overflow:auto;max-height:72vh}.badge{display:inline-block;padding:2px 6px;margin:2px;border-radius:4px;background:#fff0e7;color:#904023;font-weight:600}.unknown{background:#fff6db;color:#765a15}.quiet{color:#61736d;font-size:12px}.pages{display:flex;gap:12px;align-items:center;flex-wrap:wrap}details{margin:13px 0;padding:12px;border:1px solid #dbe5da;border-radius:6px}summary{cursor:pointer;font-weight:600}.detailtable td:first-child{font-weight:600;width:210px}.downloads a{margin-right:15px}.ratio{font-variant-numeric:tabular-nums}.summarytable{max-width:1100px}#run-details{scroll-margin-top:10px}.empty{padding:35px;text-align:center}.small{font-size:12px}
</style></head><body><h1>CSA cohort screen: dropout, imbalance and low reads</h1>
<p id="subtitle"></p><div class="cards" id="cards"></div>
<p class="warn">Flags overlap: category counts must not be added. One row below is one physical run. <strong>Low reads means at least one retained analyzed-lane scope is below its chip/lane cutoff.</strong> No archive totals are added across scopes; an individual low lane does not prove low whole-chip yield when other lanes are missing or perform differently.</p>
<section><h2>Run list</h2><div class="controls"><input id="search" type="search" placeholder="Search run, date, instrument, assay or ID"><select id="status"><option value="flagged">All flagged runs</option><option value="dropout">Dropout</option><option value="imbalance">Imbalance</option><option value="low">Low reads</option><option value="both">Barcode issue AND low reads</option><option value="incomplete">Any incomplete screen</option><option value="unresolved">Unresolved, no known flag</option><option value="no_flag">No flag under current rules</option><option value="all">All physical runs</option></select><select id="instrument"><option value="">All instruments</option></select><select id="chip"><option value="">All chip types</option></select><select id="assay"><option value="">All assay/panel labels</option></select></div>
<div class="controls"><label>From <input id="from" type="date"></label><label>To <input id="to" type="date"></label><select id="sort"><option value="date">Newest first</option><option value="ratio">Lowest read/threshold fraction first</option><option value="instrument">Instrument</option></select><select id="page-size"><option>50</option><option>100</option><option>250</option></select><button id="reset">Reset</button><button id="download">Download filtered list</button></div>
<p id="counter"></p><div class="scroll"><table><thead><tr><th>Recorded date</th><th>Instrument / chip</th><th>Run — click for details</th><th>Assay / panel; application type</th><th>Flags</th><th>DNA D/P/H</th><th>RNA D/P/H</th><th>Reads / minimum by analyzed scope</th></tr></thead><tbody id="rows"></tbody></table></div><div class="pages"><button id="previous">Previous</button><span id="page"></span><button id="next">Next</button></div>
<p class="quiet">D = dropout, P = partial low imbalance, H = overrepresentation. Counts are barcode-scope calls across retained observations, not unique biological samples across lanes. Run dates retain the parser's recorded/assumed UTC timestamp; instrument time zones are unverified. “Assay/panel” is the recorded package or panel filename, not an invented commercial assay display name.</p></section>
<section id="run-details" hidden></section>
<section><h2>Screening rules</h2><table class="summarytable"><thead><tr><th>Chip</th><th>1 analyzed lane</th><th>2 lanes</th><th>3 lanes</th><th>4 lanes</th></tr></thead><tbody><tr><td>GX5</td><td>12.5M</td><td>25M</td><td>37.5M</td><td>50M</td></tr><tr><td>GX7 / GX8</td><td>25M</td><td>50M</td><td>75M</td><td>100M</td></tr><tr><td>GX9</td><td>50M</td><td>100M</td><td>150M</td><td>200M</td></tr></tbody></table>
<p>Below the cutoff flags low reads; equality passes. The numerator and lane count must describe the same retained CSA analyzed scope. Unknown or suspiciously duplicated numerators are unresolved. These are user-defined investigation thresholds, not manufacturer certification.</p>
<p>DNA and RNA are evaluated separately; controls are excluded. Dropout: reads below max(100, 15% of within-material median). Partial low imbalance: below 40% of that median, after dropout classification. High imbalance: above 2.5× the median among non-low surviving candidates. Fewer than two samples in a present material cannot establish a passing imbalance screen. Missing counts are never replaced with zero. Labels are apparent outcome flags, not confirmed sample loss.</p>
<p>This report screens outcomes, not QPM root causes. Existing current QPM reports are linked where available; no full waveform root-cause scan was run for every flagged run. Read length and accuracy, where recorded, are shown without inventing additional low-performance cutoffs. No models, baselines or instrument settings were changed.</p></section>
<section><h2>Coverage by instrument</h2><div class="scroll" style="max-height:450px" id="instrument-summary"></div><h2>Coverage by chip</h2><div id="chip-summary"></div></section>
<section><h2>Unparsed archive exceptions</h2><p id="exception-note"></p><div class="scroll" style="max-height:350px" id="exceptions"></div></section>
<section class="downloads"><h2>Downloads and provenance</h2><p><a href="flagged_runs.csv">All flagged physical runs</a><a href="all_runs.csv">Every physical run</a><a href="flagged_observations.csv">Flagged analyzed scopes</a><a href="all_observations.csv">Every analyzed scope</a><a href="flagged_barcode_calls.csv">Every flagged barcode</a><a href="barcode_calls.csv">All barcode calls</a></p><p><a href="source_archive_roster.csv">Complete archive roster / disposition</a><a href="unparsed_archive_exceptions.csv">Unparsed archives</a><a href="manual_CSA_reconciliation.csv">Manual CSA deduplication</a><a href="screen_validation.json">Policies, input hashes and validation</a></p><p id="provenance" class="small"></p></section>
<script id="payload" type="application/json">__DATA__</script><script>
"use strict";
const D=JSON.parse(document.getElementById('payload').textContent),all=D.runs;let page=0,filtered=[];
const el=id=>document.getElementById(id),str=x=>x==null?'Not recorded':String(x),num=x=>x==null?'unknown':Number(x).toLocaleString(undefined,{maximumFractionDigits:2});
function node(tag,value,cls){const n=document.createElement(tag);if(value!=null)n.textContent=value;if(cls)n.className=cls;return n}
function addOptions(id,values){for(const v of [...new Set(values)].sort()){const o=node('option',v);o.value=v;el(id).append(o)}}
function table(rows,fields){const t=node('table'),head=node('thead'),tr=node('tr');for(const [key,label] of fields)tr.append(node('th',label));head.append(tr);t.append(head);const b=node('tbody');for(const r of rows){const row=node('tr');for(const [key,label] of fields)row.append(node('td',str(r[key])));b.append(row)}t.append(b);return t}
function badges(r){const n=node('td');for(const f of r.flags.split('; ').filter(Boolean))n.append(node('span',f,'badge'));if(r.screen_incomplete)n.append(node('span','Incomplete fields','badge unknown'));if(!r.flagged&&!r.screen_incomplete)n.append(node('span','No rule flag','quiet'));return n}
function detail(id){const r=all.find(x=>x.physical_run_id===id);if(!r)return;const panel=el('run-details');panel.replaceChildren();panel.hidden=false;panel.append(node('h2',r.run_name));const fields=[['Physical run',r.physical_run_id],['All retained run names',r.all_run_names],['Recorded start',r.run_start_time],['Instrument / chip',r.instrument+' / '+r.chip_type],['Assay / panel identities',r.assay_panel],['Application type',r.assay_type],['Workflow',r.workflow],['Flags / screen status',(r.flags||'No known flag')+' / '+r.review_status],['Available lane tokens',r.available_lane_tokens],['Scope warning',r.read_scope_warning]];const meta=node('table',null,'detailtable');for(const [k,v] of fields){const tr=node('tr');tr.append(node('td',k),node('td',str(v)));meta.append(tr)}panel.append(meta);
 if(r.qpm_reports){for(const href of r.qpm_reports.split(' | ')){const p=node('p'),a=node('a','Open existing QPM diagnostic report');a.href=href;p.append(a);panel.append(p)}}else panel.append(node('p','No current saved QPM root-cause report is linked for this run. Outcome flags alone do not identify the failing workflow step.','warn'));
 for(const o of D.details[id]){const box=node('details');box.open=true;box.append(node('summary','Analyzed lanes '+o.chip_lane+' · '+o.total_read_criterion_status+' · '+num(o.total_reads_value)+' reads / '+num(o.total_reads_threshold)+' minimum'));const p=node('p',o.run_name+' | '+o.assay_panel_label+' | '+o.assay_type);box.append(p,node('p','Assay label source: '+o.assay_label_basis+'; panels: '+(o.panel_files||'not recorded')));box.append(node('p','Barcode screen: '+o.barcode_screen_status+'; reasons: '+(o.barcode_screen_reasons||'none')+'; read scope: '+o.scope_validation));box.append(node('p','Mean raw read length: '+num(o.mean_read_length_bp)+' bp; raw read accuracy: '+num(o.raw_read_accuracy_pct)+'%. Total reads provenance: '+o.total_reads_source_file+' :: '+o.total_reads_source_field));const data=o.barcodes.map(b=>({...b,read_count:num(b.read_count),relative_to_material_median:b.relative_to_material_median==null?'unknown':Number(b.relative_to_material_median).toFixed(3),is_control:b.is_control==null?'unknown':b.is_control?'yes':'no'})).sort((a,b)=>str(a.material_type).localeCompare(str(b.material_type))||Number(a.planned_order)-Number(b.planned_order));box.append(data.length?table(data,[['planned_order','Planned order'],['material_type','Material'],['is_control','Control'],['read_count','Assigned reads'],['relative_to_material_median','Reads / material median'],['outcome_class','Outcome'],['barcode_id','Hashed barcode ID']]):node('p','No usable planned barcode rows; unresolved, not passed.','warn'));panel.append(box)}panel.scrollIntoView({behavior:'smooth',block:'start'});}
function draw(){const size=Number(el('page-size').value);page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(filtered.length/size)-1)));const body=el('rows');body.replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){const tr=node('tr');tr.append(node('td',r.run_date),node('td',r.instrument+' / '+r.chip_type));const title=node('td',null,'runname'),button=node('button',r.run_name,'link');button.onclick=()=>{location.hash=r.physical_run_id;detail(r.physical_run_id)};title.append(button,node('div',r.physical_run_id+' · '+r.retained_observation_scopes+' analyzed scopes','quiet'));tr.append(title);const a=node('td',r.assay_panel,'assay');a.append(node('div',r.assay_type,'quiet'));tr.append(a,badges(r));for(const material of ['dna','rna'])tr.append(node('td',[r[material+'_dropout_barcode_scope_calls'],r[material+'_partial_imbalance_barcode_scope_calls'],r[material+'_overrepresented_barcode_scope_calls']].join(' / ')));tr.append(node('td',r.reads_by_scope,'reads'));body.append(tr)}el('counter').textContent=filtered.length+' matching physical runs · '+all.length+' total (not archive copies)';el('page').textContent='Page '+(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/size));el('previous').disabled=page===0;el('next').disabled=(page+1)*size>=filtered.length;}
function filter(reset=true){
 if(reset)page=0;
 const query=el('search').value.toLowerCase().trim(),status=el('status').value;
 filtered=all.filter(r=>{
  if(el('instrument').value&&r.instrument!==el('instrument').value)return false;
  if(el('chip').value&&r.chip_type!==el('chip').value)return false;
  if(el('assay').value&&r.assay_panel!==el('assay').value)return false;
  if(el('from').value&&r.run_date<el('from').value)return false;
  if(el('to').value&&r.run_date>el('to').value)return false;
  if(query&&![r.all_run_names,r.run_date,r.instrument,r.chip_type,r.assay_panel,r.assay_type,r.physical_run_id,r.flags,r.observation_ids].join(' ').toLowerCase().includes(query))return false;
  return status==='all'||(status==='flagged'&&r.flagged)||(status==='dropout'&&r.has_dropout)||
   (status==='imbalance'&&r.has_imbalance)||(status==='low'&&r.has_low_read_scope)||
   (status==='both'&&(r.has_dropout||r.has_imbalance)&&r.has_low_read_scope)||
   (status==='incomplete'&&r.screen_incomplete)||(status==='unresolved'&&r.review_status==='unresolved_no_known_flag')||
   (status==='no_flag'&&r.review_status==='no_flag_under_current_rules');
 });
 filtered.sort((a,b)=>{
  if(el('sort').value==='ratio')return (a.lowest_scope_read_fraction==null?Infinity:a.lowest_scope_read_fraction)-(b.lowest_scope_read_fraction==null?Infinity:b.lowest_scope_read_fraction);
  if(el('sort').value==='instrument')return a.instrument.localeCompare(b.instrument)||b.run_date.localeCompare(a.run_date);
  return b.run_date.localeCompare(a.run_date)||a.instrument.localeCompare(b.instrument);
 });
 draw();
}
el('subtitle').textContent=D.stats.primary_observations+' retained analyzed-scope observations · '+D.stats.physical_runs+' physical runs · '+D.stats.instruments+' instruments · '+D.stats.date_min+' to '+D.stats.date_max;
el('cards').replaceChildren();for(const [key,label] of [['flagged_physical_runs','Flagged physical runs'],['dropout_physical_runs','With dropout'],['imbalance_physical_runs','With imbalance'],['low_read_scope_physical_runs','With a low-read scope'],['unresolved_no_known_flag_runs','Unresolved, no known flag']]){const box=node('div',null,'card');box.append(node('strong',num(D.stats[key])),node('span',label));el('cards').append(box)}
addOptions('instrument',all.map(r=>r.instrument));addOptions('chip',all.map(r=>r.chip_type));addOptions('assay',all.map(r=>r.assay_panel));
for(const id of ['search','status','instrument','chip','assay','from','to','sort','page-size'])el(id).addEventListener(id==='search'?'input':'change',()=>filter());
el('previous').onclick=()=>{page--;draw()};el('next').onclick=()=>{page++;draw()};el('reset').onclick=()=>{for(const id of ['search','instrument','chip','assay','from','to'])el(id).value='';el('status').value='flagged';el('sort').value='date';filter()};
el('download').onclick=()=>{const fields=['run_date','instrument','chip_type','run_name','assay_panel','assay_type','flags','review_status','reads_by_scope','physical_run_id'];const quote=x=>'"'+str(x).replaceAll('"','""')+'"';const csv=[fields.join(','),...filtered.map(r=>fields.map(k=>quote(r[k])).join(','))].join(String.fromCharCode(13,10));const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=u;a.download='filtered_CSA_runs.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
el('instrument-summary').append(table(D.instrument_summary,[['instrument','Instrument'],['physical_runs','Runs'],['flagged_runs','Flagged'],['dropout_runs','Dropout'],['imbalance_runs','Imbalance'],['low_read_scope_runs','Low reads'],['screen_incomplete_runs','Incomplete']]));el('chip-summary').append(table(D.chip_summary,[['chip_type','Chip'],['physical_runs','Runs'],['flagged_runs','Flagged'],['dropout_runs','Dropout'],['imbalance_runs','Imbalance'],['low_read_scope_runs','Low reads'],['screen_incomplete_runs','Incomplete']]));
el('exception-note').textContent=D.stats.unparsed_source_archives+' downloaded archives could not be parsed and remain unresolved. They are listed here, not counted as normal or assigned invented run identities. Source folders are collection locations, not verified instrument identities.';el('exceptions').append(table(D.exceptions,[['filename','Archive filename'],['source_instrument_folder','Source folder (unverified origin)'],['parse_error_type','Parse error class'],['disposition','Disposition']]));el('provenance').textContent='Report '+D.version+' · generated '+D.generated_utc+' · '+D.stats.source_archive_entries+' source archive entries reconciled (includes duplicate exports and manual wrappers). No network requests, external scripts, or AI summaries.';
window.addEventListener('hashchange',()=>detail(location.hash.slice(1)));filter();if(location.hash)detail(location.hash.slice(1));
</script></body></html>'''


def render(output):
    validation=json.loads((output/'screen_validation.json').read_text())
    assert validation['status']=='passed'
    for name,value in validation['artifact_sha256'].items():assert sha(output/name)==value
    payload=json.loads((output/'report_data.json').read_text())
    encoded=script_json(payload)
    report=output/'all_csa_run_screen.html'
    content=HTML.replace('__DATA__',encoded)
    # The complete flagged list is readable even if a previewer disables JS.
    # When scripts run, pagination replaces it rather than duplicating rows.
    static=[]
    for row in payload['runs']:
        if not row['flagged']:continue
        values=[row['run_date'],row['instrument']+' / '+row['chip_type'],row['run_name'],
            row['assay_panel']+' | '+row['assay_type'],row['flags']]
        values += [' / '.join(str(row[m+'_'+k+'_barcode_scope_calls']) for k in ISSUES) for m in ['dna','rna']]
        values += [row['reads_by_scope']]
        static.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    stats=payload['stats']
    subtitle=f"{stats['primary_observations']} retained analyzed-scope observations · {stats['physical_runs']} physical runs · {stats['instruments']} instruments · {stats['date_min']} to {stats['date_max']}"
    content=content.replace('<p id="subtitle"></p>','<p id="subtitle">'+html.escape(subtitle)+'</p>')
    cards=''.join(f'<div class="card"><strong>{stats[key]:,}</strong><span>{label}</span></div>' for key,label in [
        ('flagged_physical_runs','Flagged physical runs'),('dropout_physical_runs','With dropout'),('imbalance_physical_runs','With imbalance'),
        ('low_read_scope_physical_runs','With a low-read scope'),('unresolved_no_known_flag_runs','Unresolved, no known flag')])
    content=content.replace('<div class="cards" id="cards"></div>','<div class="cards" id="cards">'+cards+'</div>')
    content=content.replace('<tbody id="rows"></tbody>','<tbody id="rows">'+''.join(static)+'</tbody>')
    content=content.replace('<p id="counter"></p>',f'<p id="counter">{len(static)} flagged physical runs. Enable JavaScript for filters and expandable details; the full flagged list and CSVs remain available without it.</p>')
    content=content.replace('<p id="exception-note"></p>',f'<p id="exception-note">{stats["unparsed_source_archives"]} downloaded archives remain unresolved, not normal.</p>')
    report.write_text(content)
    write_json(output/'render_validation.json',dict(status='passed',screen_validation_sha256=sha(output/'screen_validation.json'),
        renderer_sha256=sha(Path(__file__)),report_sha256=sha(report),report=str(report),
        no_external_resources=True,all_physical_runs_available_to_filters=True,static_flagged_rows=len(static)))
    print(json.dumps(dict(report=str(report),flagged_physical_runs=payload['stats']['flagged_physical_runs'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--manual-reports',type=Path,default=MANUAL_DEFAULT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['data','render','all'],default='all')
    a=p.parse_args()
    if a.stage in ['data','all']:create(a.project,a.manual_reports,a.output)
    if a.stage in ['render','all']:render(a.output)
