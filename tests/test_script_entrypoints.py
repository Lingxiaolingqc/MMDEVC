"""Exercise standalone commands after moving scripts into course-part directories."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointsTests(unittest.TestCase):
    def command(self, relative, *args):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        return subprocess.run(
            [sys.executable, str(ROOT / relative), *args],
            cwd=ROOT.parent, env=env, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=60,
        )

    def test_every_executable_resolves_imports_outside_repo_cwd(self):
        for path in sorted((ROOT / 'scripts').rglob('*.py')):
            if path.name.startswith('_'):
                continue
            with self.subTest(script=path.relative_to(ROOT).as_posix()):
                result = self.command(path.relative_to(ROOT), '--help')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_entries_reject_the_other_model_family(self):
        for entry, wrong_model in (
            ('scripts/part1_baseline/run_framewise.py', 'vda_relative'),
            ('scripts/part2_models/run_video.py', 'dav2_relative'),
        ):
            with self.subTest(entry=entry):
                result = self.command(entry, '--model', wrong_model,
                                      '--cache', 'unused', '--output-dir', 'unused')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
