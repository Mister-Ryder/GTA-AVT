"""One temporary declared-JSON fixture and watermarked figure set; no real data."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

EXTRACTED_BLOCK_SHA256 = {
    "COMPARISONS": "2884e78bbca317950e83b1d447342870b1bb8858b4a4077ab4cc89335f6afaa1",
    "METRICS": "b108cdbf1c9f3d7a7f150ee2230b6da3574f00639bdb72a3f76c6a7567eef656",
    "GROUPS": "7d86f4939942f0fc30c8d6f7f26ff179d0eb40fec8bde7c114a34c6b8f091611",
    "GROUP_LABELS": "435410cee432604808015ed70ea5ea74434d54c5874ad69b1c42169bb0e97ad1",
    "caption": "153adafaa01bcccfcc65f33808b1975c79601a11a78fcaa2e9b30de31e82603e",
    "record": "9b715dab795c95e098ba6be974c61803555121a6271f5d10dd31600506fc4626",
    "case_figure": "c4566dfed599303096d1f22330135789df68ff8de4a53c5f247da642730ab687",
    "draw_effect": "8aaab2b364ca36fb2ff0b8fb30281a73cf5b078bc091ef7d52c59226df0b1721",
    "metric_figure": "c7d4655180969c8fefef55710e36c5b6c50f88d99938d929aa0f461ea1f0d8e2",
    "geometry_figure": "469a58e23e842d0e084f491a8788bae28412526d7dc46afdb9e5646895869518"
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def reject(action):
    try:
        action()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError('Invalid plotted metadata was accepted')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True); parser.add_argument('--reader-directory', required=True)
    parser.add_argument('--preview-directory'); parser.add_argument('--output')
    args = parser.parse_args()
    target = Path(__file__).with_name('plot_minimal4_portable.py')
    text = target.read_text(encoding='utf-8'); tree = ast.parse(text, feature_version=(3, 8)); lines = target.read_bytes().splitlines(keepends=True)
    blocks = {}
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
        if name in EXTRACTED_BLOCK_SHA256:
            blocks[name] = hashlib.sha256(b''.join(lines[node.lineno - 1:node.end_lineno])).hexdigest()
    assert blocks == EXTRACTED_BLOCK_SHA256
    # Fixture generation may bootstrap; no plotting/admission function may do so.
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('paired', 'analyze_case_rows') for n in ast.walk(tree))
    guard = "import importlib.abc,sys,runpy\nclass G(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in ('numpy','matplotlib','torch','SimpleITK'):raise RuntimeError('heavy import forbidden')\nsys.meta_path.insert(0,G());sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    response = subprocess.run([sys.executable, '-c', guard, str(target)], capture_output=True, text=True)
    assert response.returncode == 0 and json.loads(response.stdout)['state'] == 'plan_only'
    plot = load(target, 'portable4_plot_fixture'); reader = plot.load_reader(args.reader_directory)
    fixture_tool = load(Path(args.reader_directory) / 'verify_minimal4_portable.py', 'portable4_raw_fixture')
    checks = ['Python3.8/stdlb default with no sources/heavy imports; six exact plotting functions/four constants; plot contains no bootstrap call']
    preview = Path(args.preview_directory).absolute() if args.preview_directory else None
    if preview is not None:
        assert not preview.exists() and preview.resolve() == preview
    with tempfile.TemporaryDirectory(prefix='GTA_AVT_PORTABLE4_PLOT_SYNTHETIC_') as tmp:
        root = Path(tmp); declaration, path = fixture_tool.fixture(reader, args.repository, root, 'val')
        # This creates one synthetic paired artifact; it is not part of plotting.
        result, _ = reader.analyze_files(path, reader.digest(path), args.repository)
        paired = root / 'paired.json'; write(paired, result)
        inputs = lambda: plot.actual_inputs(paired, plot.sha(paired), path, plot.sha(path), args.repository, args.reader_directory)
        admitted, cases, binding, same_reader, bound = inputs()
        assert len(admitted['casewise']) == 32 and all(len(x) == 8 for x in cases.values())
        assert binding['author_private_marker_FILE_sha256'] is None and not admitted['author_server_or_generic_export_verified']
        checks.append('complete declared4VAL8/rawFILE hashes/null metrics/support reconcile, author marker alwaysnull; no source-level upgrade')
        for change in (
            lambda r: r['casewise'][0].__setitem__('dice', 99.),
            lambda r: r['comparisons']['MAIN_A_minus_C']['cases'][0]['delta_after_minus_before'].__setitem__('dice', 99.),
            lambda r: r['comparisons']['MAIN_A_minus_C']['scopes']['all']['geometry']['radius_le_2mm']['case_support_and_missing'][1].__setitem__('after_GT_nodes', -1),
            lambda r: r['comparisons']['MAIN_A_minus_C']['scopes']['all']['metrics']['dice'].__setitem__('ci95_lower', 1000.),
            lambda r: r.__setitem__('author_server_or_generic_export_verified', True),
            lambda r: r.__setitem__('source_marker_FILE_sha256', 'f' * 64),
            lambda r: r['comparisons']['MAIN_A_minus_C'].__setitem__('after_slot', 'ablations_A_soft_platform_bridge'),
        ):
            bad = copy.deepcopy(result); change(bad); write(paired, bad); reject(inputs)
        write(paired, result)
        reject(lambda: plot.actual_inputs(paired, '0' * 64, path, plot.sha(path), args.repository, args.reader_directory))
        reject(lambda: plot.actual_inputs(paired, plot.sha(paired), path, '0' * 64, args.repository, args.reader_directory))
        checks.append('FILEhash mismatch, case/delta/support/interval drift, cross-role swap and false author-attestation rejected')
        first = declaration['case_ids'][0]; metric = root / 'evaluations/main_C_concat' / first / 'metrics.json'
        original = metric.read_bytes(); metric.write_bytes(original + b'\n'); reject(inputs); metric.write_bytes(original)
        reject(lambda: plot.actual_inputs(paired, plot.sha(paired), path, plot.sha(path), args.repository, args.reader_directory, root / 'absent_future_selection.json', 'f' * 64))
        checks.append('raw metric byte drift and futureTESTselection supplied toVAL rejected; TEST separate admission is delegated to exact reader')
        output = preview if preview is not None else root / 'figures'
        manifest = plot.render(admitted, cases, output, binding, args.repository, same_reader, bound, watermark_fixture=True)
        assert len(manifest['files']) == 6 and manifest['source_record_count'] == 64
        assert manifest['bootstrap_recomputed'] is False and manifest['temporary_synthetic_watermark'] is True
        assert manifest['source_marker_FILE_sha256'] is None and manifest['author_server_or_export_attested'] is False
        assert all(row['bytes'] > 8000 for row in manifest['files'])
        mapping = plot.read(output / 'figure_source.json')
        assert sum(row['figure'] == '01_case_Dice_by_training_role' for row in mapping['records']) == 32
        assert sum(row['figure'] == '02_all_metric_paired_changes' for row in mapping['records']) == 20
        assert sum(row['figure'] == '03_GT_geometry_recall_changes' for row in mapping['records']) == 12
        reject(lambda: plot.render(admitted, cases, output, binding, args.repository, same_reader, bound))
        checks.append('one temporary syntheticVAL3PNG+3PDF/64source rows; pp/ml/mm separated; overwrite rejected; fixture generation only may bootstrap')
        if preview is not None:
            (preview / 'SYNTHETIC_ONLY_REMOVE_AFTER_INSPECTION.txt').write_text('Synthetic metadata fixture only. Not results. Remove directory after visual inspection.\n', encoding='utf-8')
    report = dict(schema='GTA-AVT-minimal4-portable-plot-candidate-QA-v1', state='passed', checks=checks,
        temporary_synthetic_metadata_and_watermarked_figures_only=True, genuine_results_generated=False,
        arrays_weights_GPU_server_or_inference_used=False, plotting_bootstrap_recomputed=False,
        synthetic_artifact_generation_bootstrap_used=True, preview_retained_for_inspection=preview is not None,
        files_sha256={p.name: plot.sha(p) for p in (target, Path(__file__))}, reader_sha256=plot.READER_SHA,
        extracted_function_constant_sha256=blocks)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
