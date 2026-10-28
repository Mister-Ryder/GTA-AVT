import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).with_name('v12_gdf_joint_supervisor.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.c = dict(main_completion_receipt=str(self.root/'main.json'), old_queue_state=str(self.root/'queue.json'))
        self.main = dict(status='complete', exit_code=0, successful_updates=122000, native_cases=8, test_opened=False)
        self.jobs = []
        for name in sorted(module.OLD_JOBS):
            arm = name.split('20261032_')[1]
            run = self.root/name
            run.mkdir()
            (run/'pipeline.json').write_text(json.dumps(dict(status='complete', completed_successful_updates=42000)))
            metrics = run/'metrics'/('native_VAL_'+arm)
            metrics.mkdir(parents=True)
            (metrics/'metrics.json').write_text(json.dumps(dict(split='VAL', per_case=[{}]*8)))
            self.jobs.append(dict(name=name, arm=arm, status='complete', exit_code=0,
                                 process=dict(pid=99999999, start_ticks='0'), run=str(run)))
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        Path(self.c['main_completion_receipt']).write_text(json.dumps(self.main))
        Path(self.c['old_queue_state']).write_text(json.dumps(dict(jobs=self.jobs)))

    def test_requires_all_native_evaluations(self):
        self.assertTrue(module.barrier(self.c)[0])
        self.jobs[0]['status'] = 'running'
        self.save()
        self.assertFalse(module.barrier(self.c)[0])

    def test_missing_main_receipt_waits(self):
        Path(self.c['main_completion_receipt']).unlink()
        self.assertFalse(module.barrier(self.c)[0])

    def test_main_training_alone_does_not_release(self):
        self.main['native_cases'] = 7
        self.save()
        self.assertFalse(module.barrier(self.c)[0])

    def test_failure_never_restarts_old_trial(self):
        self.jobs[1]['status'] = 'failed'
        self.save()
        with self.assertRaisesRegex(RuntimeError, 'no automatic handoff'):
            module.barrier(self.c)

    def test_live_old_process_prevents_release(self):
        with patch.object(module, 'alive', return_value=True):
            self.assertFalse(module.barrier(self.c)[0])

    def test_missing_or_wrong_old_job_rejected(self):
        self.jobs.pop()
        self.save()
        with self.assertRaises(ValueError):
            module.barrier(self.c)

    def test_test_split_rejected(self):
        j = self.jobs[0]
        path = Path(j['run'])/'metrics'/('native_VAL_'+j['arm'])/'metrics.json'
        path.write_text(json.dumps(dict(split='TEST', per_case=[{}]*8)))
        with self.assertRaises(ValueError):
            module.barrier(self.c)


if __name__ == '__main__':
    unittest.main()
