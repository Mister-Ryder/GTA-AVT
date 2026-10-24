"""Small synthetic/default routing readout checks; no actual endpoints."""
import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def main():
    here=Path(__file__).parent;path=here/'calibre_routing_readout.py'
    ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    m=load(path,'routing_readout_synthetic');base=m.module(here/'calibre_basis_readout.py',m.BASE_SHA,'routing_math_synthetic')
    fixture=load(here/'boundary_field_readout_checks.py','routing_fixture');ids,old,ignored=fixture.fixture()
    rows={'C0':old['C0'],'A_only':old['A_only'],'AR':old['AB']}
    for arm in m.ARMS:
        for r in rows[arm]:r['oracle']=dict(available_radius_sites=100,unknown_sites=64**3-100)
    interventions=[]
    for r in rows['AR']:
        other=copy.deepcopy(r['metrics']);other['dice']-=.002
        interventions.append(dict(case_id=r['case_id'],source=r['source'],slot=r['slot'],stratum=r['stratum'],primary=r['primary'],
            normal=copy.deepcopy(r['metrics']),permuted=other,radius_moments=dict(count=100,sum=200.,sum_squares=500.),
            zR_changed=True,mean_abs_logit_response=.01,binary_changed_voxels=10))
    good=m.analyze_seed({'seed':m.SEEDS[0]},rows,interventions,ids,base)
    assert good['original_primary_screen_passed'] is True and good['CT_uniform_transfer_safety']['passed'] is True
    assert good['SOURCE_safety']['passed'] is True and good['capacity_continue'] is True
    assert abs(good['mechanisms']['own_A_half_ring_radius']['mean_correct_minus_permuted_Dice_pp']-.2)<1e-12
    # A positive aggregate and8 positive cases cannot hide a negative source.
    bad=copy.deepcopy(interventions)
    for r in bad:
        effect=-.001 if r['case_id'].startswith('Rider_') else .003
        r['permuted']['dice']=r['normal']['dice']-effect
    stopped=m.analyze_seed({'seed':m.SEEDS[0]},rows,bad,ids,base)
    assert stopped['original_primary_screen_passed'] is True and stopped['CT_uniform_transfer_safety']['passed'] is True
    assert stopped['SOURCE_safety']['source_groups']['Rider']['passed'] is False and stopped['capacity_continue'] is False
    missing=copy.deepcopy(interventions)
    for r in missing:
        if r['case_id']==ids[-1]:r['zR_changed']=False
    unsupported=m.analyze_seed({'seed':m.SEEDS[0]},rows,missing,ids,base)
    rider=unsupported['SOURCE_safety']['source_groups']['Rider']
    assert rider['case_equal_correct_minus_half_ring_Dice_pp'] is None and rider['passed'] is False
    assert rider['expected_case_ids']==ids[-4:] and rider['supported_case_ids']==ids[-4:-1]
    assert unsupported['capacity_continue'] is not True
    mechanism=copy.deepcopy(good['mechanisms']['own_A_half_ring_radius'])
    for cid,effect in zip(ids[-4:],[-.125,.125,-.125,.125]):mechanism['per_case'][cid]['correct_minus_permuted_Dice_pp']=effect
    zero=m.SOURCE_safety(mechanism,ids,base)['source_groups']['Rider']
    assert zero['case_equal_correct_minus_half_ring_Dice_pp']==0. and zero['passed'] is True
    first,second=copy.deepcopy(rows['AR'][0]),copy.deepcopy(rows['AR'][1])
    first['metrics'].update(fine_nodes=1,fine_covered=1,fine_recall=1.)
    second['metrics'].update(fine_nodes=99,fine_covered=0,fine_recall=0.)
    assert base.aggregate([first,second])['fine_recall']==.01
    # All three seeds retained; later seeds cannot rescue a source failure.
    bank=dict(probe_cases=ids,fit_cases=['fit'+str(i) for i in range(24)])
    proof={'normalization.json':'same'}
    real_sha=m.sha
    with patch.object(m,'sha',side_effect=lambda p:m.BANK_SHA if str(p)=='bank' else real_sha(p)),patch.object(m,'read',return_value=bank),patch.object(m,'load_seed',side_effect=[
        ({'seed':m.SEEDS[0]},rows,bad,proof),({'seed':m.SEEDS[1]},rows,interventions,proof),({'seed':m.SEEDS[2]},rows,interventions,proof)]):
        result=m.analyze(['one','two','three'],'bank')
    assert result['all3_original_primary_screens_passed'] is True and result['all3_CT_uniform_transfer_safety_passed'] is True
    assert result['all3_SOURCE_safety_passed'] is False and result['capacity_continue'] is False and result['primary_failure_cannot_be_rescued']
    with tempfile.TemporaryDirectory(prefix='routing_readout_smoke_') as td:
        out=Path(td)/'output';args=type('Args',(),dict(inputs=['one','two','three'],bank='bank',output=out))()
        with patch.object(m,'analyze',return_value=result):m.execute(args)
        for name,count in [('casewise.csv',540),('mechanism_cases.csv',36),('source_safety.csv',9)]:
            with (out/name).open(encoding='utf-8') as stream:assert len(list(csv.DictReader(stream)))==count
        rendered=(out/'readout.txt').read_text(encoding='utf-8')
        assert 'SOURCE=False' in rendered and '全部检查合取继续' in rendered and '两者合取继续' not in rendered
        assert json.loads((out/'analysis.json').read_bytes())['coefficient_shape']==[3,64]
    try:m.metadata(Path('no-input-access'),{'schema':'GTA-AVT-calibre-basis-coadapt-pilot-v1'},bank,base)
    except ValueError:pass
    else:raise AssertionError('Old basis metadata accepted')
    worker=m.module(here/'calibre_routing_pilot.py',m.WORKER_SHA,'routing_recipe_synthetic')
    assert worker.RECIPE['gate_parameters']==576 and worker.RECIPE['trainable_parameters']==115458
    assert m.sha(here/'calibre_routing_stage_model.py')==m.MODEL_SHA
    run=subprocess.run([sys.executable,'-B','-I','-S',str(path)],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['status']=='inert'
    print(json.dumps(dict(status='PASS',groups=7,actual_endpoints_read=False,original_math_reused=True,
        source_support_null=True,source_no_rescue=True,complete_CSV_rows=[540,36,9])))


if __name__=='__main__':main()
