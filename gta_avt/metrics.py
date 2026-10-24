"""Objective case metrics; all reported scores are fractions, not percentages."""
from __future__ import annotations

import math
import numpy as np


def binary_metrics(prediction, target, valid=None, *, centerline=True, spacing_zyx=None):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    valid = np.ones_like(target) if valid is None else np.asarray(valid, bool)
    prediction, target = prediction & valid, target & valid
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    pp, tt = tp + fp, tp + fn
    result = {'tp_voxels': tp, 'fp_voxels': fp, 'fn_voxels': fn,
              'dice': 2*tp/(pp+tt) if pp+tt else 1.0,
              'precision': tp/pp if pp else None, 'recall': tp/tt if tt else None}
    if centerline:
        from skimage.morphology import skeletonize
        # Older scikit-image Lee implementations return uint8 {0,255}; counts
        # and bitwise overlaps must operate on Boolean skeleton membership.
        ps = np.asarray(skeletonize(prediction, method='lee'),dtype=bool) if pp else prediction
        ts = np.asarray(skeletonize(target, method='lee'),dtype=bool) if tt else target
        npred, ntrue = int(ps.sum()), int(ts.sum())
        tprec = float((ps & target).sum()) / npred if npred else None
        trec = float((ts & prediction).sum()) / ntrue if ntrue else None
        cl = (2*tprec*trec/(tprec+trec) if tprec is not None and trec is not None and tprec+trec
              else (1.0 if not pp and not tt else 0.0))
        result.update(cldice=cl, centerline_precision=tprec, centerline_recall=trec,
                      predicted_skeleton_voxels=npred, target_skeleton_voxels=ntrue)
        if spacing_zyx is not None:
            from scipy.ndimage import distance_transform_edt
            spacing=np.asarray(spacing_zyx,dtype=np.float64)
            if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
                raise ValueError('spacing_zyx must contain three positive finite millimeter spacings')
            radius=distance_transform_edt(target,sampling=spacing)
            fine=ts & (radius<=2.)
            count=int(fine.sum())
            result.update(fine_centerline_recall=float((fine & prediction).sum())/count if count else None,
                          fine_centerline_supported_voxels=count,fine_radius_threshold_mm=2.)
    return result


def summarize_cases(rows):
    """Case-equal mean/std/min and the mean of the worst ceil(20%) cases."""
    if not rows:
        raise ValueError('No cases to summarize')
    keys = sorted({k for row in rows for k,v in row.items()
                   if isinstance(v, (int,float)) and not isinstance(v,bool)})
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        result[key] = {'mean': float(np.mean(values)), 'std_population': float(np.std(values)),
                       'minimum': min(values), 'defined_cases': len(values)}
        if key in ('dice', 'cldice', 'centerline_recall', 'fine_centerline_recall', 'precision', 'recall'):
            result[key]['worst20_mean'] = float(np.mean(sorted(values)[:max(1,math.ceil(.2*len(values)))]))
    return result


class AttributeMetrics:
    """Pool sampled patches within each case, retaining radius spatial moments.

    Centered error centers prediction and target separately across all sampled
    GT-emitted cells of a case. Overlapping patches count repeated observations;
    this is a deterministic patch-validation statistic, not whole-volume truth.
    """
    def __init__(self, prior):
        self.prior = float(prior)
        if not 0 < self.prior < 1:
            raise ValueError('TRAIN applicability prior must be inside (0,1)')
        self.cases = {}

    def update(self, output, batch):
        def array(x):
            return x.detach().float().cpu().numpy()
        logits, predicted = array(output['a_logits']), array(output['z_radius'])
        a, radius = array(batch['a']), array(batch['r'])
        av, rv = array(batch['a_valid']).astype(bool), array(batch['r_valid']).astype(bool)
        ids = batch.get('case_id', [str(i) for i in range(len(a))])
        for i, case in enumerate(ids):
            stats = self.cases.setdefault(str(case), dict(a_n=0, a_bce=0., a_prior_bce=0.,
                a_tp=0,a_pred=0,a_truth=0,r_n=0,r_sse=0.,r_prior_sse=0.,r_sum_p=0.,r_sum_t=0.,
                r_sum_p2=0.,r_sum_t2=0.,r_sum_pt=0.,predicted_supported_n=0.,predicted_sum=0.,predicted_sum2=0.))
            ll,aa = logits[i][av[i]].astype(np.float64), a[i][av[i]].astype(np.float64)
            prob = np.exp(-np.logaddexp(0.,-ll))
            support = prob >= .5
            stats['a_n'] += len(aa)
            stats['a_bce'] += float((np.logaddexp(0.,ll)-aa*ll).sum())
            stats['a_prior_bce'] += float((-aa*np.log(self.prior)-(1-aa)*np.log1p(-self.prior)).sum())
            stats['a_tp'] += int((support & (aa==1)).sum())
            stats['a_pred'] += int(support.sum()); stats['a_truth'] += int(aa.sum())
            pp,tt = predicted[i][rv[i]].astype(np.float64), radius[i][rv[i]].astype(np.float64)
            stats['r_n'] += len(tt);stats['r_sse'] += float(((pp-tt)**2).sum())
            stats['r_prior_sse'] += float((tt**2).sum())
            for key,value in (('r_sum_p',pp.sum()),('r_sum_t',tt.sum()),('r_sum_p2',(pp**2).sum()),
                              ('r_sum_t2',(tt**2).sum()),('r_sum_pt',(pp*tt).sum())):
                stats[key] += float(value)
            predicted_supported = predicted[i][av[i]][support].astype(np.float64)
            stats['predicted_supported_n'] += len(predicted_supported)
            stats['predicted_sum'] += float(predicted_supported.sum())
            stats['predicted_sum2'] += float((predicted_supported**2).sum())

    def report(self):
        rows=[]
        for case,s in sorted(self.cases.items()):
            n=s['r_n']
            centered_energy=max(0.,s['r_sum_t2']-s['r_sum_t']**2/n) if n else 0.
            centered_sse=max(0.,s['r_sse']-(s['r_sum_p']-s['r_sum_t'])**2/n) if n else 0.
            pn=s['predicted_supported_n']
            predvar=max(0.,s['predicted_sum2']/pn-(s['predicted_sum']/pn)**2) if pn else None
            rows.append({'case_id':case,'availability_bce':s['a_bce']/s['a_n'] if s['a_n'] else None,
                'constant_availability_bce':s['a_prior_bce']/s['a_n'] if s['a_n'] else None,
                'emission_precision':s['a_tp']/s['a_pred'] if s['a_pred'] else None,
                'emission_recall':s['a_tp']/s['a_truth'] if s['a_truth'] else None,
                'radius_mse':s['r_sse']/n if n else None,
                'constant_radius_mse':s['r_prior_sse']/n if n else None,
                'case_centered_radius_ratio':centered_sse/centered_energy if centered_energy>0 else None,
                'predicted_supported_radius_variance':predvar,'emitted_target_sites':n,
                'predicted_supported_sites':int(pn),
                'predicted_support_fraction':pn/s['a_n'] if s['a_n'] else None})
        summary=summarize_cases(rows)
        def mean(key):
            return summary.get(key,{}).get('mean')
        supported=bool(rows) and all(r['radius_mse'] is not None and r['case_centered_radius_ratio'] is not None for r in rows)
        tests={'all_cases_have_radius_support_and_variation':supported,
               'availability_beats_train_prior':mean('availability_bce') is not None and mean('availability_bce')<mean('constant_availability_bce'),
               'radius_beats_train_mean':mean('radius_mse') is not None and mean('radius_mse')<mean('constant_radius_mse'),
               'case_centered_radius_beats_constant':mean('case_centered_radius_ratio') is not None and mean('case_centered_radius_ratio')<1.,
               'consumer_has_nonconstant_supported_radius':all(r['predicted_supported_radius_variance'] is not None and r['predicted_supported_radius_variance']>0 for r in rows)}
        minimum = all(tests.values())
        return {'scope':'deterministic sampled VAL patches; case-equal summary',
                'per_case':rows,'summary':summary,'readiness':{'passed':minimum,
                'strict_spatial_diagnostic_passed':all(tests.values()),'tests':tests,
                'minimum_gate':'Every VAL case has target variation and nonconstant predicted hard-A support; case-mean applicability BCE and radius MSE beat TRAIN constants, and case-centered radius error beats a constant field. All checks fixed before training.',
                'meaning':'Prerequisite diagnostic only; passing does not establish segmentation benefit.'}}
