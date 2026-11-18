"""Reaggregate saved predictive diagnostics; do not evaluate any predictor."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
 if out.exists():raise FileExistsError('Choose a new output directory')
 index=read(ROOT/'DIAGNOSTIC_INDEX.json');summaries=[];residual=[];inputs=[]
 for block in ['C','D']:
  paths={}
  for name,rec in index['catalogs'][block].items():
   rel=PurePosixPath(rec['path']);p=(ROOT/rel).resolve()
   if rel.is_absolute() or '..' in rel.parts or not p.is_relative_to(ROOT) or sha(p)!=rec['sha256']:raise ValueError('Diagnostic source must match the selected package index')
   paths[name]=p;inputs.append(rec)
  predictive=read(paths['predictive_diagnostics.json'])['rows'];assert len(predictive)==80
  for context in ['Hidden5','Coarse2','Coarse8','Full5']:
   for fraction in ['f010','f100']:
    for representation in ['Raw','CorrectedPhysics']:
     rows=[r for r in predictive if (r['labels']['context_label'],r['labels']['fraction_tag'],r['labels']['representation'])==(context,fraction,representation)]
     assert sorted(r['labels']['seed'] for r in rows)==list(range(3101,3106))
     for grouping,count in [('momentum_groups',7),('physical_groups',42)]:
      assert all(len(r[grouping]['groups'])==count for r in rows)
      for group in range(count):
       values=[next(g for g in r[grouping]['groups'] if g['group_id']==group) for r in rows]
       assert len({g['count'] for g in values})==1
       metrics=[k for k in values[0] if k not in ['group_id','count']]
       for metric in metrics:
        vals=[float(g[metric]) for g in values]
        summaries.append({'catalog':block,'context':context,'fraction_tag':fraction,'representation':representation,'grouping':grouping,'group_id':group,'events':values[0]['count'],'metric':metric,'five_seed_mean':statistics.mean(vals),'seed_sample_sd':statistics.stdev(vals),'seed_minimum':min(vals),'seed_maximum':max(vals),**{f'seed_{r["labels"]["seed"]}':v for r,v in zip(rows,vals)}})
  raw=read(paths['residual_diagnostics.json'])['rows'];assert len(raw)==4
  for row in raw:
   for units,val in row.items():
    if isinstance(val,dict) and 'groups' in val:
     assert len(val['groups'])==42
     for group in val['groups']:residual.append({'catalog':block,'representation':row['representation'],'fraction_tag':row['fraction_tag'],'units':units,**group})
 out.mkdir(parents=True)
 def write(name,records):
  keys=list(dict.fromkeys(k for row in records for k in row))
  with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
 write('predictive_group_seed_summaries.csv',summaries);write('residual_group_summaries.csv',residual)
 quoted=[r for r in summaries if r['context']=='Hidden5' and r['fraction_tag']=='f100' and r['grouping']=='momentum_groups' and r['group_id'] in [0,6] and r['metric'] in ['median_coverage','predictive_iqr_coverage']]
 result={'schema':'saved-predictive-residual-aggregate-verification/v1','status':'PASS_SAVED_DIAGNOSTICS','generated_utc':datetime.now(timezone.utc).isoformat(),'saved_predictive_model_rows':160,'saved_residual_representation_fraction_rows':8,'all_seed_values_retained':True,'summary_rows':len(summaries),'residual_group_rows':len(residual),'manuscript_regional_diagnostic_rows':quoted,'inputs':inputs,'no_predictive_inference_or_event_quantiles_computed':True,'interpretation':'Descriptive means and sample SD across the same five paired optimizer seeds; no new interval or causal claim. Existing per-seed diagnostic frequencies are reaggregated, not recomputed from omitted event arrays.'}
 (out/'DIAGNOSTIC_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'status':result['status'],'predictive_rows':len(summaries),'residual_rows':len(residual)}))
if __name__=='__main__':main()
