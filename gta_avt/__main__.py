"""Explicit data preparation, training, inference and evaluation entry points."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="GTA-AVT v1")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Index downloaded AVT NRRDs; no automatic new split")
    index.add_argument("--root", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--split-file")
    prepare = sub.add_parser("prepare", help="Prepare CT, labels and TRAIN-only geometric statistics")
    prepare.add_argument("--cases", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--geometry-options", help="JSON options file")
    imported = sub.add_parser("import-cache", help="Import a verified compatible 1mm cache")
    imported.add_argument("--manifest", required=True)
    imported.add_argument("--output", required=True)
    imported.add_argument("--cache-root")
    imported.add_argument("--raw-root")
    imported.add_argument("--gdf-manifest")
    imported.add_argument("--link-mode",choices=("hardlink_or_copy","symlink"),default="hardlink_or_copy")
    for name in ("pretrain", "train", "infer", "evaluate", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                       help="Override a top-level JSON config value; paths use current working directory")
    args = parser.parse_args(argv)
    if args.command == "index":
        from .data import index_avt
        result = index_avt(args.root, args.output, split_file=args.split_file)
    elif args.command == "prepare":
        from .data import prepare_dataset
        options = json.loads(Path(args.geometry_options).read_text(encoding="utf-8-sig")) if args.geometry_options else None
        result = prepare_dataset(args.cases, args.output, geometry_options=options)
    elif args.command == "import-cache":
        from .data import import_prepared_cache
        result = import_prepared_cache(args.manifest,args.output,cache_root=args.cache_root,
            raw_root=args.raw_root,gdf_manifest=args.gdf_manifest,link_mode=args.link_mode)
    else:
        if args.command in ("infer", "evaluate"):
            from . import inference as engine
        else:
            from . import engine
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        for setting in args.set:
            key, separator, value = setting.partition("=")
            if not separator or not key:
                parser.error("--set requires KEY=JSON")
            try:
                config[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                parser.error(str(exc))
        if args.command == "pipeline":
            from .pipeline import run_pipeline
            result = run_pipeline(config)
        else:
            result = getattr(engine, args.command)(config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
