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


class ResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.path=self.root/'resource_gate.json'
        self.config=dict(resource_gate_file=str(self.path))
        self.gate=dict(schema='GTA-AVT-resource-gate-1',requested_gpu_count=2,allow_revision_start=False)
        self.save()

    def tearDown(self):self.tmp.cleanup()
    def save(self):self.path.write_text(json.dumps(self.gate))

    def test_old_endpoints_complete_does_not_bypass_operator_hold(self):
        self.assertFalse(module.resource_permission(self.config,['0','1','2'])[0])
        self.assertFalse(module.resource_permission(self.config,['0','1'])[0])

    def test_operator_release_and_actual_two_cards_both_required(self):
        self.gate['allow_revision_start']=True;self.save()
        self.assertFalse(module.resource_permission(self.config,['0','1','2'])[0])
        self.assertTrue(module.resource_permission(self.config,['0','1'])[0])

    def test_missing_or_invalid_gate_is_never_permission(self):
        self.assertFalse(module.resource_permission({},['0','1'])[0])
        self.path.unlink();self.assertFalse(module.resource_permission(self.config,['0','1'])[0])
        self.gate['requested_gpu_count']=3;self.save()
        with self.assertRaises(ValueError):module.resource_permission(self.config,['0','1','2'])

    def waiting(self,status='waiting_resource_change',stages=None):
        state=dict(status=status,stages=[] if stages is None else stages,supervisor=dict(pid=12345,start_ticks='99'))
        (self.root/'state.json').write_text(json.dumps(state))
        (self.root/'supervisor.lock').write_text('12345')

    def test_live_waiting_supervisor_cannot_be_duplicated(self):
        self.waiting()
        with patch.object(module,'alive',return_value=True),self.assertRaises(RuntimeError):
            module.acquire_waiting_lock(self.root,True)
        self.assertEqual((self.root/'supervisor.lock').read_text(),'12345')

    def test_started_revision_is_not_a_waiting_resume(self):
        self.waiting(status='paired_training',stages=[dict(pid=2)])
        with patch.object(module,'alive',return_value=False),self.assertRaises(ValueError):
            module.acquire_waiting_lock(self.root,True)

    def test_dead_childfree_waiting_state_resumes_and_preserves_evidence(self):
        self.waiting()
        with patch.object(module,'alive',return_value=False):
            previous=module.acquire_waiting_lock(self.root,True)
        self.assertEqual(previous['supervisor']['pid'],12345)
        self.assertEqual((self.root/'supervisor.lock').read_text(),str(__import__('os').getpid()))
        self.assertEqual(len(list(self.root.glob('supervisor.lock.before_resource_change_*'))),1)
        self.assertEqual(len(list(self.root.glob('waiting_state_before_resource_change_*.json'))),1)

    def test_missing_lock_does_not_silently_reinitialize_previous_state(self):
        self.waiting();(self.root/'supervisor.lock').unlink()
        with self.assertRaises(RuntimeError):module.acquire_waiting_lock(self.root,True)

    def test_waiting_resume_claim_serializes_lock_archive_and_creation(self):
        self.waiting()
        original=module._acquire_waiting_lock_unserialized
        def nested_attempt(control,resume):
            with self.assertRaises(FileExistsError):module.acquire_waiting_lock(control,True)
            return original(control,resume)
        with patch.object(module,'alive',return_value=False),patch.object(module,'_acquire_waiting_lock_unserialized',side_effect=nested_attempt):
            module.acquire_waiting_lock(self.root,True)
        self.assertFalse((self.root/'waiting_resume.claim').exists())


if __name__ == '__main__':
    unittest.main()
