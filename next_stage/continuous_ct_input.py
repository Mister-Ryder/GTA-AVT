"""Small image-only 1mm input path for a future CT-conditioned consumer.

No label, mask, geometry, split or ROI annotation is accepted or discovered.
The original CT algorithms are reused; no A/R predictor/consumer bridge is
implemented here. Import and the default CLI perform no image IO.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

SPACING_MM = 1.0
SCHEMA = 'GTA-AVT-continuous-CT-input-v1'
CODE = Path(__file__).resolve().parents[1] / 'code'


def _algorithms():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    prepare = importlib.import_module('prepare')
    data = importlib.import_module('data')
    common = importlib.import_module('common')
    if any(Path(module.__file__).resolve() != (CODE / name).resolve()
           for module,name in ((prepare,'prepare.py'),(data,'data.py'),(common,'common.py'))):
        raise RuntimeError('Expected this project original prepare/data/common algorithms')
    return prepare, data, common


def _record(record):
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_bytes())
    if not isinstance(record, dict) or record.get('schema') != SCHEMA or record.get('spacing_mm') != SPACING_MM:
        raise ValueError('Explicit image-only 1mm record required')
    return record


def prepare_ct(image_path, case_id, output, reviewed_override=None):
    """Write a fresh image.npy/record.json from one CT file, never a label.

    reviewed_override has the original source-hash-locked intensity-exception
    format. Ambiguous CT intensity remains an error; no new heuristic is used.
    """
    import numpy as np
    import SimpleITK as sitk
    prepare, _, common = _algorithms()
    if not isinstance(case_id, str) or not case_id or any(c in case_id for c in ('/', '\\')):
        raise ValueError('One nonempty logical case ID required')
    image_path = Path(image_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh image-only cache directory')
    image_sha = common.sha256(image_path)
    if reviewed_override is not None and image_sha != reviewed_override['image_sha256']:
        raise ValueError('Reviewed CT-only intensity exception source SHA differs')
    original = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    if original.GetDimension() != 3 or original.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Only one scalar 3D CT image is accepted')
    corrected, intensity_audit = prepare.correct_stored_intensity(original, reviewed_override)
    image = sitk.DICOMOrient(corrected, 'LPS')
    grid = prepare.reference_grid(image, SPACING_MM)
    resampled = sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear,
                             -1024., sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(resampled)
    crop = prepare.image_body_crop(array, SPACING_MM)
    # Preserve original cache quantization BEFORE patch normalization.
    cached = np.clip(np.rint(array[crop]), -1024, 2000).astype(np.int16)
    record = dict(schema=SCHEMA, id=case_id, image=str(image_path), image_sha256=image_sha,
        spacing_mm=SPACING_MM, shape_zyx=list(cached.shape), full_shape_zyx=list(array.shape),
        crop_start_zyx=[s.start for s in crop], crop_stop_zyx=[s.stop for s in crop],
        cache_dir=str(output), resampled_origin=grid.GetOrigin(),
        resampled_direction=grid.GetDirection(), resampled_spacing=grid.GetSpacing(),
        resampled_size=grid.GetSize(), original_origin=original.GetOrigin(),
        original_direction=original.GetDirection(), original_spacing=original.GetSpacing(),
        original_size=original.GetSize(), intensity_audit=intensity_audit,
        crop_definition='largest CT body >-600 HU xy box plus10mm; retain full z',
        label_required=False, geometry_required=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'image.npy').open('xb') as stream:
        np.save(stream, cached, allow_pickle=False)
    record['image_cache_sha256'] = common.sha256(output / 'image.npy')
    with (output / 'record.json').open('xb') as stream:
        stream.write((json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    return record


def load_image(record, mmap_mode='r'):
    """Load only image.npy using a CT-only record; no CachedData/mask access."""
    import numpy as np
    record = _record(record)
    value = np.load(Path(record['cache_dir']) / 'image.npy', mmap_mode=mmap_mode, allow_pickle=False)
    if value.dtype != np.int16 or list(value.shape) != record['shape_zyx']:
        raise ValueError('Image-only cache shape or dtype differs')
    return value


def tile_starts(size, patch, overlap):
    """Original inference.tile_starts arithmetic, without importing Torch."""
    if size <= patch:
        return [0]
    stride = max(1, int(patch * (1. - overlap)))
    values = list(range(0, size - patch + 1, stride))
    if values[-1] != size - patch:
        values.append(size - patch)
    return values


def patch_batches(image, record, patch=128, batch_size=2, overlap=.5):
    """Yield starts, normalized CT N1PPP, and original full-grid coordinates.

    All tiles cover the image-defined crop; there is no GT-jitter or support
    selection. Coordinates retain the historical +patch/2 convention.
    """
    import itertools
    import math
    import numpy as np
    _, data, _ = _algorithms()
    record = _record(record)
    if (not isinstance(patch,int) or patch <= 0 or not isinstance(batch_size,int) or batch_size <= 0 or
            not math.isfinite(overlap) or not 0. <= overlap < 1.):
        raise ValueError('Positive patch/batch and overlap in[0,1) required')
    if list(image.shape) != record['shape_zyx']:
        raise ValueError('CT cache and record shape differ')
    shape = np.asarray(image.shape)
    starts = list(itertools.product(*(tile_starts(int(n), patch, overlap)
                                      for n in np.maximum(shape, patch))))
    for offset in range(0, len(starts), batch_size):
        chosen = starts[offset:offset+batch_size]
        patches, locations = [], []
        for start in chosen:
            slices = tuple(slice(s, min(s+patch,int(n))) for s,n in zip(start,shape))
            sample = np.asarray(image[slices], np.float32)
            sample = np.pad(sample, [(0,patch-n) for n in sample.shape], constant_values=-1024)
            patches.append(data.normalize(sample)[None])
            centre = np.asarray(record['crop_start_zyx']) + np.asarray(start) + patch/2.
            locations.append(2.*centre/np.maximum(np.asarray(record['full_shape_zyx'])-1.,1.)-1.)
        yield chosen, np.ascontiguousarray(patches,dtype=np.float32), np.ascontiguousarray(locations,dtype=np.float32)


def sliding_probability(predict_probability, image, record, patch=128, batch_size=2, overlap=.5):
    """Original Gaussian tile fusion with one explicit CT-only callback.

    callback(normalized_CT, coordinates) -> probability[N,P,P,P] or[N,1,P,P,P].
    The caller owns any model/device/AMP/condition adapter; no A/R bridge or
    Torch model is constructed here. This preserves full-crop tiling/fusion.
    """
    import numpy as np
    shape = np.asarray(image.shape)
    probabilities = np.zeros(tuple(np.maximum(shape,patch)),np.float32)
    weights = np.zeros_like(probabilities)
    coordinates = np.arange(patch,dtype=np.float32)-(patch-1.)/2.
    gaussian = np.exp(-.5*(coordinates/(patch/8.))**2)
    weight = np.maximum(gaussian[:,None,None]*gaussian[None,:,None]*gaussian[None,None,:],1e-5)
    tiles = 0
    for chosen,CT,locations in patch_batches(image,record,patch,batch_size,overlap):
        prediction = np.asarray(predict_probability(CT,locations),dtype=np.float32)
        if prediction.shape == (len(chosen),1,patch,patch,patch):
            prediction = prediction[:,0]
        if (prediction.shape != (len(chosen),patch,patch,patch) or not np.isfinite(prediction).all() or
                (prediction<0.).any() or (prediction>1.).any()):
            raise ValueError('Callback must return finite patch probabilities in[0,1]')
        for start,value in zip(chosen,prediction):
            slices = tuple(slice(s,s+patch) for s in start)
            probabilities[slices] += value*weight
            weights[slices] += weight
        tiles += len(chosen)
    probabilities /= np.maximum(weights,1e-8)
    return probabilities[tuple(slice(0,int(n)) for n in shape)],tiles


def native_probability(probability, record):
    """Recover continuous probability using only the saved original CT header."""
    import numpy as np
    import SimpleITK as sitk
    record = _record(record)
    probability = np.asarray(probability,dtype=np.float32)
    if (list(probability.shape) != record['shape_zyx'] or not np.isfinite(probability).all() or
            (probability<0.).any() or (probability>1.).any()):
        raise ValueError('Finite cropped-grid probabilities in[0,1] required')
    full = np.zeros(record['full_shape_zyx'],np.float32)
    slices = tuple(slice(a,b) for a,b in zip(record['crop_start_zyx'],record['crop_stop_zyx']))
    full[slices] = probability
    grid = sitk.GetImageFromArray(full)
    grid.SetOrigin(record['resampled_origin']); grid.SetDirection(record['resampled_direction'])
    grid.SetSpacing(record['resampled_spacing'])
    reference = sitk.Image([int(n) for n in record['original_size']],sitk.sitkFloat32)
    reference.SetOrigin(record['original_origin']); reference.SetDirection(record['original_direction'])
    reference.SetSpacing(record['original_spacing'])
    return sitk.Resample(grid,reference,sitk.Transform(),sitk.sitkLinear,0.,sitk.sitkFloat32)


def native_prediction(probability, record):
    """Original fixed0.5 threshold, after native-grid probability restoration."""
    return native_probability(probability,record) >= .5


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image');parser.add_argument('--case-id');parser.add_argument('--output')
    parser.add_argument('--intensity-override-json');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert',spacing_mm=SPACING_MM,label_required=False,
            geometry_required=False,CT_predictor_bridge_implemented=False)))
        return
    if not all((args.image,args.case_id,args.output)):
        parser.error('--image/--case-id/--output are required with --execute')
    override=json.loads(Path(args.intensity_override_json).read_bytes()) if args.intensity_override_json else None
    record=prepare_ct(args.image,args.case_id,args.output,override)
    print(json.dumps(dict(status='prepared',case_id=record['id'],spacing_mm=SPACING_MM,
        shape_zyx=record['shape_zyx'],image_cache_sha256=record['image_cache_sha256'])))


if __name__=='__main__':
    main()
