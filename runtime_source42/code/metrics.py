"""Explicit native-grid overlap, surface and centreline definitions in mm."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from geometry import sparse_geometry, RADIUS_EDGES_MM


def binary_metrics(prediction, truth, spacing):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    spacing = np.asarray(spacing, float)
    if p.shape != t.shape or p.ndim != 3 or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError('Expected equal 3D prediction/truth and positive zyx spacing')
    intersection, pn, tn = int((p & t).sum()), int(p.sum()), int(t.sum())
    result = {'dice': 2. * intersection / max(pn + tn, 1),
              'precision': intersection / max(pn, 1), 'recall': intersection / max(tn, 1),
              'fp_voxels': pn - intersection, 'fn_voxels': tn - intersection,
              'fp_ml': (pn - intersection) * float(spacing.prod()) / 1000.,
              'fn_ml': (tn - intersection) * float(spacing.prod()) / 1000.,
              'empty_prediction': pn == 0, 'empty_truth': tn == 0,
              'assd_mm': None, 'hd95_mm': None,
              'assd_definition': 'pooled bidirectional 6-neighbour boundary-voxel distances; not surfel-area weighted',
              'centerline_definition': 'Lee skeleton voxel coverage on native grid; not arc-length weighted'}
    if not pn or not tn:
        result.update(centerline_recall=0. if tn else None, centerline_precision=0. if pn else None, cldice=0.)
        return result
    # Crop union for memory only; retain every foreground prediction including FP.
    bounds = ndi.find_objects((p | t).astype(np.uint8))[0]
    bounds = tuple(slice(max(0, x.start - 1), min(n, x.stop + 1)) for x, n in zip(bounds, p.shape))
    p, t = p[bounds], t[bounds]
    six = ndi.generate_binary_structure(3, 1)
    ps = np.argwhere(p & ~ndi.binary_erosion(p, structure=six, border_value=0)) * spacing
    ts = np.argwhere(t & ~ndi.binary_erosion(t, structure=six, border_value=0)) * spacing
    a, b = cKDTree(ts).query(ps, workers=4)[0], cKDTree(ps).query(ts, workers=4)[0]
    result['assd_mm'] = float((a.sum() + b.sum()) / (len(a) + len(b)))
    result['hd95_mm'] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
    result['asd_prediction_to_truth_mm'] = float(a.mean())
    result['asd_truth_to_prediction_mm'] = float(b.mean())
    gt = sparse_geometry(t, spacing)
    from skimage.morphology import skeletonize
    skp = skeletonize(p, method='lee').astype(bool)
    points = gt['points']
    covered = p[tuple(points.T)]
    recall = float(covered.mean()) if len(points) else None
    precision = float(t[skp].mean()) if skp.any() else None
    result.update(centerline_recall=recall, centerline_precision=precision,
                  cldice=2. * recall * precision / max(recall + precision, 1e-12) if recall is not None and precision is not None else None,
                  truth_skeleton_support=int(len(points)), prediction_skeleton_support=int(skp.sum()),
                  prediction_components=int(ndi.label(p, np.ones((3, 3, 3)))[1]),
                  truth_components=int(ndi.label(t, np.ones((3, 3, 3)))[1]))
    groups = {}
    for index, name in enumerate(('radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm', 'radius_gt_8mm')):
        selected = np.searchsorted(RADIUS_EDGES_MM, gt['radius'], side='left') == index
        groups[name] = {'support_nodes': int(selected.sum()),
                        'centerline_recall': float(covered[selected].mean()) if selected.any() else None}
    for index, name in enumerate(('nonjunction', 'junction_neighbourhood'), 1):
        selected = gt['structure'] == index
        groups[name] = {'support_nodes': int(selected.sum()),
                        'centerline_recall': float(covered[selected].mean()) if selected.any() else None}
    result['geometry_groups'] = groups
    result['geometry_group_scope'] = 'GT-defined centerline recall only; false-positive tradeoff reported globally'
    return result


def aggregate(rows):
    fields = ('dice', 'precision', 'recall', 'fp_ml', 'fn_ml', 'assd_mm', 'hd95_mm',
              'centerline_recall', 'centerline_precision', 'cldice')
    output = {'cases': len(rows), 'statistical_unit': 'case', 'metrics': {}}
    for name in fields:
        values = [x.get(name) for x in rows]
        defined = [x for x in values if x is not None]
        output['metrics'][name] = {'mean': float(np.mean(defined)) if len(defined) == len(values) and defined else None,
                                  'defined_case_count': len(defined),
                                  'std_cases': float(np.std(defined, ddof=1)) if len(defined) > 1 else None}
    return output
