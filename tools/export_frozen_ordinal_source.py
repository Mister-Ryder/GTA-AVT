"""Bind and optionally isolate the mapped public strict-frozen39 sources.

Default is source-only inspection: no arrays, checkpoints, model imports, GPU
queries or writes. --execute copies only declared39 bytes into a fresh folder;
its sibling receipt is neither model QA nor an experiment authorization.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
MAP_SHA = '21e35ab772ae791dd3fa30658b1d6a8f8178efb254332935847106523e2e9983'
IDENTITY_SHA = 'f97635bf48440f874c9a6ab066596254f6d000d1c18b351610eb0f651ac93507'
PRIVATE_MAP_SHA = '92b424a4141efd3d6e6184090fbb5b06f44b0c278345caf1391d3e517099ae10'
NEW3 = {'next_stage/frozen_ordinal_stage_model.py', 'next_stage/frozen_ordinal_stage_train.py',
        'next_stage/frozen_ordinal_stage_checks.py'}
PARENT3 = {'next_stage/ordinal_stage_model.py', 'next_stage/ordinal_stage_train.py',
           'next_stage/ordinal_stage_checks.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(root, name):
    rel = PurePosixPath(name)
    require(rel.parts and not rel.is_absolute() and '..' not in rel.parts and
        '\\' not in name and ':' not in name, 'Unsafe scientific source path')
    path = root.joinpath(*rel.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
        'Missing or symlink-traversing source file: ' + name)
    return path


def literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing expected source-identity literal: ' + name)


def bind(project, overlay, manifest, identity):
    raw, identity_raw = manifest.read_bytes(), identity.read_bytes()
    require(sha(raw) == MAP_SHA and sha(identity_raw) == IDENTITY_SHA,
        'Exact public39 manifest and identity bytes required')
    value, identity_value = json.loads(raw), json.loads(identity_raw)
    source = value['source_sha256']
    require(value['schema'] == 'GTA-AVT-frozen-ordinal-scientific-source-map-v1' and
        len(source) == 39 and (NEW3 | PARENT3).issubset(source), 'Exact public frozen39 map required')
    require(identity_value['schema'] == 'GTA-AVT-frozen-ordinal-public-source-identity-v1' and
        identity_value['private_executed_source_manifest_sha256'] == PRIVATE_MAP_SHA and
        identity_value['public_mapped_source_manifest_sha256'] == MAP_SHA and
        identity_value['public_mapped_source_sha256'] == source and
        identity_value['public_mapped_GPU_QA_executed'] is False and
        identity_value['public_mapped_formal_training_executed'] is False and
        identity_value['private_GPU_QA_is_not_public_GPU_QA'] is True,
        'Private/public source identity or unexecuted status differs')
    payload = {}
    for name, expected in source.items():
        origin = overlay if overlay is not None and name in NEW3 else project
        content = ordinary(origin, name).read_bytes()
        require(sha(content) == expected, 'Actual mapped public source differs: ' + name)
        payload[name] = content
    tree = ast.parse(payload['next_stage/frozen_ordinal_stage_train.py'], feature_version=(3, 8))
    base, parent = literal(tree, 'BASE_SOURCE_SHA256'), literal(tree, 'PARENT_ORDINAL_SHA256')
    require(isinstance(base, dict) and len(base) == 33 and isinstance(parent, dict) and
        set(parent) == PARENT3 and set(source) == set(base) | set(parent) | NEW3 and
        all(source.get(name) == digest for name, digest in dict(base, **parent).items()),
        'Mapped trainer base33/parent3 source pins are not closed')
    parent_tree = ast.parse(payload['next_stage/ordinal_stage_train.py'], feature_version=(3, 8))
    require(literal(parent_tree, 'BASE_SOURCE_SHA256') == base,
        'Parent and frozen trainers disagree on inherited public33 bytes')
    return raw, identity_raw, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--overlay', help='Reviewed overlay supplying only the NEW frozen3 files')
    parser.add_argument('--source-manifest')
    parser.add_argument('--identity')
    parser.add_argument('--output', help='Fresh exact39 source folder, with sibling metadata')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    project = Path(args.project).resolve()
    overlay = Path(args.overlay).resolve() if args.overlay else None
    manifest = (Path(args.source_manifest) if args.source_manifest else
        project / 'next_stage/frozen_ordinal_source_manifest_20261024.json').resolve()
    identity = (Path(args.identity) if args.identity else
        project / 'next_stage/frozen_ordinal_source_identity_20261024.json').resolve()
    require(all(path.is_file() and not path.is_symlink() for path in (manifest, identity)),
        'Ordinary public manifest/identity files required')
    raw, identity_raw, payload = bind(project, overlay, manifest, identity)
    result = dict(execute=bool(args.execute), source_binding_checked=True, source_identity_checked=True,
        source_manifest_sha256=MAP_SHA, source_identity_sha256=IDENTITY_SHA, scientific_source_files=39,
        arrays_opened=False, weights_opened=False, models_created=False, Torch_imported=False,
        GPU_queried=False, public_mapped_CPU_model_executed=False, public_mapped_GPU_QA_executed=False,
        formal_training=False, efficacy_established=False, private_receipts_reused=False)
    if not args.execute:
        result['files_written'] = 0
        print(json.dumps(result))
        return
    require(args.output, 'Explicit fresh export output required')
    output = Path(args.output).resolve()
    map_output = output.with_name(output.name + '_source_manifest.json')
    identity_output = output.with_name(output.name + '_source_identity.json')
    receipt = output.with_name(output.name + '_export_receipt.json')
    require(not any(path.exists() for path in (output, map_output, identity_output, receipt)),
        'Fresh immutable export destinations required; no overwrite')
    require(output != project and output not in project.parents and
        (overlay is None or output != overlay and output not in overlay.parents),
        'Export cannot contain or replace an input root')
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payload.items():
        target = output.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        require(sha(target.read_bytes()) == sha(content), 'Exported scientific bytes differ')
    for path, content in ((map_output, raw), (identity_output, identity_raw)):
        with path.open('xb') as stream:
            stream.write(content)
    actual = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
    require(actual == set(payload), 'Export must contain exactly39 scientific files')
    result.update(status='SOURCE_BINDING_ONLY_NOT_MODEL_QA', isolated_scientific_files=39,
        exported_source_sha256={name: sha(content) for name, content in payload.items()},
        exporter_sha256=sha(Path(__file__).read_bytes()), private_GPU_QA_certifies_this_export=False)
    with receipt.open('xb') as stream:
        stream.write((json.dumps(result, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
