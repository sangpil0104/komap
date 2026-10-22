"""CPU-only regression checks for command serialization and isolated resume."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import worker
from common import new_attempt, read_json, write_json
from prepare_resume import import_saved_work


class ResumeFix(unittest.TestCase):
    def test_real_subprocess_receives_exact_logged_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output with spaces.json'
            script = root / 'echo argv.py'
            script.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n')
            argv = [Path(sys.executable), '-B', script, output, '--fold', 2, 'literal $(not_a_shell)', 'two words']
            worker.call(argv, root, 'probe')
            self.assertEqual(read_json(root / 'probe_command.json'), list(map(str, argv)))
            self.assertEqual(read_json(output), ['--fold', '2', 'literal $(not_a_shell)', 'two words'])

    def test_b4_and_hr_valid_and_oof_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, fold in [('b4', None), ('hr1', None), ('hr2', None), ('hr1', 2), ('hr2', 4)]:
                with self.subTest(family=family, fold=fold), patch('worker.subprocess.run') as run:
                    worker.expert_predict(root / 'view', root / 'output', family, root, fold)
                    argv = run.call_args.args[0]
                    self.assertTrue(all(isinstance(v, str) for v in argv))
                    self.assertEqual(read_json(root / f'{family}_{fold or "valid"}_command.json'), argv)
                    if fold:
                        self.assertEqual(argv[argv.index('--fold')+1], str(fold))

    def test_failed_child_is_still_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.CalledProcessError):
                worker.call([Path(sys.executable), '-c', 'raise SystemExit(7)'], Path(directory), 'failure')

    def test_import_preserves_old_attempts_and_copies_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / 'old'; destination = Path(directory) / 'new'
            for run in [origin, destination]:
                run.mkdir()
                for name in ['completed', 'jobs', 'field_cache', 'evaluations']:
                    (run / name).mkdir()
            record = dict(id='done', kind='train', checkpoint='/old/checkpoint.pt', artifacts={})
            write_json(origin / 'completed/done.json', record)
            for task in ['done', 'interrupted']:
                attempt = origin / 'jobs' / task / 'attempt_1'; attempt.mkdir(parents=True)
                (attempt / 'saved.txt').write_text('original')
            cache = origin / 'evaluations/key'; cache.mkdir()
            write_json(cache / 'complete.json', dict(files_sha256={}))
            (cache / 'score.txt').write_text('original cache')
            partial = origin / 'evaluations/incomplete'; partial.mkdir()
            (partial / 'partial.txt').write_text('unfinished')
            receipt = import_saved_work(origin, destination)
            self.assertEqual(receipt['completed_training'], 1)
            self.assertEqual(read_json(destination / 'completed/done.json'), record)
            self.assertEqual((destination / 'completed/done.json').read_bytes(), (origin / 'completed/done.json').read_bytes())
            attempt = new_attempt(destination, 'interrupted')
            self.assertEqual(attempt.name, 'attempt_2')
            self.assertFalse(attempt.is_symlink())
            (attempt / 'saved.txt').write_text('new')
            self.assertEqual((origin / 'jobs/interrupted/attempt_1/saved.txt').read_text(), 'original')
            (destination / 'evaluations/key/score.txt').write_text('new cache')
            self.assertEqual((cache / 'score.txt').read_text(), 'original cache')
            self.assertFalse((destination / 'evaluations/incomplete').exists())
            with self.assertRaises(FileExistsError):
                import_saved_work(origin, destination)


if __name__ == '__main__':
    unittest.main()
