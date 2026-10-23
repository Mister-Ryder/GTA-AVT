"""Bind and optionally isolate the exact public ordinal36 source, without a model.

The expanded public repository contains other studies. Only the declared36
files are copied. Default verifies source bytes and prints a metadata plan;
--execute writes a new source folder plus sibling manifest/receipt. No arrays,
weights, runtime receipts, model imports, GPU or training are involved.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
MAP_SHA = '4c6fbfb938fb99a76dcd00118281733d878c5aa27aa9a0523ca912b92b4cdabd'
NEW3 = {'next_stage/ordinal_stage_model.py','next_stage/ordinal_stage_train.py',
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
    p = root.joinpath(*rel.parts)
    require(p.is_file() and not p.is_symlink() and p.resolve()==p,
            'Missing or symlink-traversing scientific file: '+name)
    return p


def bind(project, overlay, manifest):
    raw = manifest.read_bytes()
    require(sha(raw)==MAP_SHA, 'Public ordinal source manifest bytes differ')
    value = json.loads(raw)
    source = value['source_sha256']
    require(value['schema']=='GTA-AVT-ordinal-scientific-source-map-v1' and
        len(source)==36 and NEW3.issubset(source), 'Expected public ordinal36 source map')
    payload, resolved = {}, {}
    for name,digest in source.items():
        origin = overlay if overlay is not None and name in NEW3 else project
        p = ordinary(origin,name); content = p.read_bytes()
        require(sha(content)==digest, 'Actual public source differs: '+name)
        payload[name]=content; resolved[name]=str(p)
    tree = ast.parse(payload['next_stage/ordinal_stage_train.py'].decode('utf-8'))
    base = None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and
                n.id=='BASE_SOURCE_SHA256' for n in node.targets):
            base = ast.literal_eval(node.value)
    require(isinstance(base,dict) and len(base)==33 and set(source)-set(base)==NEW3 and
        all(source.get(n)==h for n,h in base.items()), 'Trainer public base33 pins differ from map')
    return raw,payload,resolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--overlay',help='Optional reviewed overlay for only the three new scientific files')
    p.add_argument('--source-manifest',help='Defaults to project/next_stage/ordinal_source_manifest_20261023.json')
    p.add_argument('--output',help='Fresh isolated36 source folder; manifest/receipt are siblings')
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    project = Path(a.project).resolve()
    overlay = Path(a.overlay).resolve() if a.overlay else None
    manifest = (Path(a.source_manifest) if a.source_manifest else
                project/'next_stage/ordinal_source_manifest_20261023.json').resolve()
    require(manifest.is_file() and not manifest.is_symlink(), 'Regular public source manifest required')
    raw,payload,resolved = bind(project,overlay,manifest)
    plan = dict(execute=bool(a.execute),source_binding_checked=True,scientific_source_files=36,
        source_manifest_sha256=MAP_SHA,arrays_opened=False,weights_opened=False,
        models_created=False,Torch_imported=False,GPU_queried=False,GPU_QA_certified=False,
        formal_training=False,public_mapped_source_GPU_executed=False)
    if not a.execute:
        plan.update(files_written=0)
        print(json.dumps(plan)); return
    require(a.output, 'Explicit fresh --output is required')
    output = Path(a.output).resolve()
    map_output = output.with_name(output.name+'_source_manifest.json')
    receipt = output.with_name(output.name+'_export_receipt.json')
    require(not any(p.exists() for p in (output,map_output,receipt)), 'Fresh immutable export destinations required')
    require(output!=project and output not in project.parents and
        (overlay is None or output!=overlay and output not in overlay.parents), 'Export cannot contain/replace an input root')
    output.mkdir(parents=True,exist_ok=False)
    for name,content in payload.items():
        target = output.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:
            f.write(content)
        require(sha(target.read_bytes())==sha(content), 'Exported source bytes differ')
    with map_output.open('xb') as f:
        f.write(raw)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(actual==set(payload), 'Export does not contain exactly public36')
    plan.update(status='source_binding_only_passed',isolated_scientific_files=36,
        exported_source_sha256={n:sha(b) for n,b in payload.items()},
        source_manifest_output=str(map_output),exporter_sha256=sha(Path(__file__).read_bytes()))
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(plan,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(plan))


if __name__=='__main__':
    main()
