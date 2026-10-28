"""Prepare bounded whole-case Gaussian targets without copying CT caches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gta_avt.data import prepare_bounded_gdf_overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-depth', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = prepare_bounded_gdf_overlay(args.source_manifest, args.output,
                                        chunk_depth=args.chunk_depth, resume=args.resume)
    print(json.dumps(dict(event='bounded_gdf_overlay_complete', cases=len(result['cases']),
                          schema=result['schema'], gdf_target=result['gdf_target']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
