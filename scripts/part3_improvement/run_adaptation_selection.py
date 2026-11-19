"""Complete only the predeclared independent Residual and LoRA selections."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.adaptation_study.common import load
from p3_depth.residual.locking import study_lock
from p3_depth.residual.records import write_json, now, append_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-current-study', action='store_true',
                        help='Wait for an already running study process before continuing the fixed selection sequence.')
    args = parser.parse_args()
    study, _ = load(ROOT)
    out = ROOT / study['output']
    with study_lock(out / 'selection_orchestrator'):
        if args.wait_for_current_study:
            while True:
                try:
                    with study_lock(out):
                        break
                except RuntimeError:
                    time.sleep(20)
        def command(stage, script, *options):
            value = {'status': 'running', 'stage': stage, 'at': now(),
                     'scope': 'Independent selections only; no comparison between selected methods, size studies, all795 refits or new654 evaluations.'}
            write_json(out / 'selection_progress.json', value)
            argv = [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/part3_improvement' / script), *map(str, options)]
            append_json(out / 'selection_commands.jsonl', {**value, 'argv': argv})
            print('STAGE ' + stage, flush=True)
            result = subprocess.run(argv, cwd=ROOT)
            append_json(out / 'selection_commands.jsonl', {**value, 'status': 'complete' if result.returncode == 0 else 'failed',
                                                          'returncode': result.returncode, 'finished_at': now()})
            if result.returncode:
                write_json(out / 'selection_progress.json', {**value, 'status': 'failed', 'returncode': result.returncode})
                raise SystemExit(result.returncode)
        stage_cache = out / 'stage_cache/development'
        if not all((stage_cache / (name + '.npy')).exists() for name in ('init', 'mid', 'late')):
            command('Prepare frozen stage features', 'run_fusion_ablations.py', '--prepare')
        if not (out / 'residual_frozen.json').exists():
            command('Residual factorial', 'run_fusion_ablations.py', '--names', *study['screening'], '--seeds', *study['confirmation_seeds'], '--resume')
            command('Residual spatial controls', 'run_fusion_ablations.py', '--names', *study['controls'], '--seeds', study['screening_seed'], '--resume')
            command('Residual provisional analysis', 'analyze_fusion_ablations.py')
            rows = json.loads((out / 'analysis/factorial/review.json').read_text())['summary']
            rows = [r for r in rows if r['method'] in ['S0', *study['screening']]]
            assert all(r['seeds'] == study['confirmation_seeds'] for r in rows)
            best = min(r['abs_rel'] for r in rows)
            winner = min([r for r in rows if r['abs_rel'] <= best * (1 + study['selection']['relative_tie'])],
                         key=lambda r: (r['parameters'], r['abs_rel']))
            if len(study['candidates'][winner['method']]['branches']) == 3:
                command('Residual zero-middle control', 'run_fusion_ablations.py', '--names', winner['method'] + '_zero_mid',
                        '--phase', 'zero_middle', '--seeds', *study['confirmation_seeds'], '--resume')
            command('Residual selection freeze', 'analyze_fusion_ablations.py', '--freeze')
        if not (out / 'lora_frozen.json').exists():
            preflight = out / 'engineering/lora/finite_rays_v2_diverse/result.json'
            if not preflight.exists():
                command('LoRA diverse numerical preflight', 'preflight_lora_study.py',
                        '--tag', 'finite_rays_v2_diverse', '--steps', 200)
            checked = json.loads(preflight.read_text())
            assert checked['status'] == 'pass' and checked['numerical_protocol'] == 'finite_rays_v2'
            assert len(checked['rows']) == 4 and all(len(r['records']) == 200 for r in checked['rows'])
            command('LoRA engineering smoke', 'run_lora_study.py', '--names', 'L_D_r4_lr1e4', 'L_E_r4_lr1e4', '--smoke', '--resume')
            command('LoRA bounded search', 'run_lora_study.py', '--resume')
            if not (out / 'lora_shortlist.json').exists():
                command('LoRA shortlist', 'analyze_lora_study.py', '--select')
            shortlist = json.loads((out / 'lora_shortlist.json').read_text())['methods']
            command('LoRA repeated-seed confirmation', 'run_lora_study.py', '--names', *shortlist,
                    '--phase', 'confirm', '--seeds', *[s for s in study['confirmation_seeds'] if s != study['screening_seed']], '--resume')
            command('LoRA selection freeze', 'analyze_lora_study.py', '--freeze')
        for family in ('factorial', 'lora'):
            command(f'{family} figures', 'plot_adaptation_study.py', '--family', family)
        pairs = [('S0', name) for name in study['screening']]
        pairs += [('F2', 'P2'), ('F3', 'P3'), ('P2', 'G2'), ('P3', 'G3'),
                  ('F2', 'F3'), ('P2', 'P3'), ('G2', 'G3'), ('C2', 'P2'), ('C3', 'P3')]
        winner = json.loads((out / 'residual_frozen.json').read_text())['method']
        if len(study['candidates'][winner]['branches']) == 3:
            pairs.append((winner + '_zero_mid', winner))
        for before, after in pairs:
            command(f'Residual cases {before} to {after}', 'render_adaptation_cases.py',
                    '--family', 'residual', '--before', before, '--after', after)
        winner = json.loads((out / 'lora_frozen.json').read_text())['method']
        command('LoRA cases against foundation', 'render_adaptation_cases.py', '--family', 'lora', '--before', 'B0', '--after', winner)
        write_json(out / 'selection_progress.json', {'status': 'experiments_complete', 'at': now(),
            'remaining': 'Human/assistant visual review, compact archival report, scoped commit and push.',
            'scope': 'Only independent selection. No comparison, data-size runs, all795 refits or new654 evaluations executed.'})
        print('PASS independent selections; review/archive remains', flush=True)


if __name__ == '__main__':
    main()
