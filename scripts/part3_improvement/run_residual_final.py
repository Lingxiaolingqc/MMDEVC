"""Freeze a development-selected proposal or evaluate completed frozen refits."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual.locking import study_lock
from p3_depth.residual.selection import freeze_selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/residual_study.json')
    parser.add_argument('--action',choices=['freeze','evaluate'],required=True)
    parser.add_argument('--proposal',help='Development-selected final specification, required only for freeze')
    args=parser.parse_args();study=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    with study_lock(ROOT/study['output']):
        if args.action=='freeze':
            if not args.proposal:raise ValueError('--proposal required')
            proposal=json.loads((ROOT/args.proposal).read_text(encoding='utf-8'))
            frozen,digest=freeze_selection(ROOT,study,proposal)
            from p3_depth.residual.jobs import build_job,save_job
            split=json.loads((ROOT/study['split']).read_text(encoding='utf-8'))
            for row in frozen['refits']:
                job=build_job(study,split,'R0','refit',row['seed'],row['run_id'],overrides=row['candidate'],
                    feature_hw=row['feature_hw'],fitting_ids=frozen['development_ids'],training_overrides=row['training'],
                    frozen_selection_sha256=digest)
                print(save_job(ROOT,study,job),flush=True)
            print('PASS frozen methods/durations; no test evaluation performed',flush=True)
        else:
            from p3_depth.residual.final_evaluation import evaluate_final
            evaluate_final(ROOT,study)


if __name__=='__main__':main()
