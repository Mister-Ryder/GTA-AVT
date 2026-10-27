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
    """Case-equal prediction and consumption-domain prerequisite diagnostics.

    Deployment consumes continuous V, not a threshold mask. Threshold .5 below
    defines a readable high-support diagnostic only. Unknown radius sites have
    no radius-error score; their consumed mass is reported explicitly.
    """
    def __init__(self, prior, *, min_support_dice=.5, min_support_recall=.5):
        self.prior = float(prior)
        if not 0 < self.prior < 1:
            raise ValueError('TRAIN lumen-support prior must be inside (0,1)')
        if not (0 <= min_support_dice <= 1 and 0 <= min_support_recall <= 1):
            raise ValueError('Support readiness floors must be in [0,1]')
        self.min_support_dice, self.min_support_recall = float(min_support_dice), float(min_support_recall)
        self.cases = {}

    def update(self, output, batch):
        def array(x):
            return x.detach().float().cpu().numpy()
        logits, predicted = array(output['support_logits']), array(output['radius_mean'])
        support, radius = array(batch['support']), array(batch['r'])
        valid, measured = array(batch['support_valid']).astype(bool), array(batch['r_valid']).astype(bool)
        if any(value.shape != logits.shape for value in (predicted, support, radius, valid, measured)):
            raise ValueError('Metric attribute/target shapes differ')
        if (measured & ~valid).any():
            raise ValueError('Measured radius cannot include unobserved cells')
        ids = batch.get('case_id', [str(i) for i in range(len(support))])
        for i, case in enumerate(ids):
            s = self.cases.setdefault(str(case), dict(n=0, bce=0., prior_bce=0., p_sum=0., t_sum=0., pt_sum=0.,
                tp=0, pred_n=0, truth_n=0, r_n=0, r_sse=0., r_prior_sse=0., r_sum_p=0., r_sum_t=0.,
                r_sum_t2=0., consumer_n=0, overlap_n=0, overlap_sse=0., overlap_prior_sse=0.,
                overlap_sum=0., overlap_sum2=0., consumer_background_n=0, consumer_unmeasured_fg_n=0,
                measured_soft_mass=0., background_soft_mass=0., soft_radius_sse=0., soft_radius_prior_sse=0.,
                radius_axis_sum=0., unknown_radius_axis_sum=0., fractional_target_n=0, fractional_pred_sum=0.))
            real = valid[i]
            ll, tt = logits[i][real].astype(np.float64), support[i][real].astype(np.float64)
            pp = predicted[i][real].astype(np.float64)
            if not all(np.isfinite(value).all() for value in (ll, tt, pp)):
                raise ValueError('Nonfinite attribute prediction on real cells')
            prob = np.exp(-np.logaddexp(0., -ll))
            clipped = np.clip(pp, -4., 4.)  # Matches the production consumer.
            consumer, truth = prob >= .5, tt >= .5
            mr = measured[i][real]
            target = radius[i][real][mr].astype(np.float64)
            if not np.isfinite(target).all():
                raise ValueError('Nonfinite measured radius target')
            s['n'] += len(tt); s['bce'] += float((np.logaddexp(0., ll) - tt * ll).sum())
            s['prior_bce'] += float((-tt*np.log(self.prior) - (1.-tt)*np.log1p(-self.prior)).sum())
            s['p_sum'] += float(prob.sum()); s['t_sum'] += float(tt.sum()); s['pt_sum'] += float((prob*tt).sum())
            s['tp'] += int((consumer & truth).sum()); s['pred_n'] += int(consumer.sum()); s['truth_n'] += int(truth.sum())
            rp = pp[mr]
            error = (rp-target)**2
            s['r_n'] += len(target); s['r_sse'] += float(error.sum()); s['r_prior_sse'] += float((target**2).sum())
            s['r_sum_p'] += float(rp.sum()); s['r_sum_t'] += float(target.sum()); s['r_sum_t2'] += float((target**2).sum())
            s['consumer_n'] += int(consumer.sum())
            overlap = consumer & mr
            ovp = clipped[overlap]; ovt = radius[i][real][overlap].astype(np.float64)
            s['overlap_n'] += int(overlap.sum()); s['overlap_sse'] += float(((ovp-ovt)**2).sum())
            s['overlap_prior_sse'] += float((ovt**2).sum())
            s['overlap_sum'] += float(ovp.sum()); s['overlap_sum2'] += float((ovp**2).sum())
            background = tt == 0.
            s['consumer_background_n'] += int((consumer & background).sum())
            s['consumer_unmeasured_fg_n'] += int((consumer & ~mr & ~background).sum())
            s['measured_soft_mass'] += float(prob[mr].sum())
            s['background_soft_mass'] += float(prob[background].sum())
            s['soft_radius_sse'] += float((prob[mr] * (clipped[mr]-target)**2).sum())
            s['soft_radius_prior_sse'] += float((prob[mr] * target**2).sum())
            axis = np.abs(prob*clipped)
            s['radius_axis_sum'] += float(axis.sum()); s['unknown_radius_axis_sum'] += float(axis[~mr].sum())
            fractional = (tt > 0.) & (tt < .5)
            s['fractional_target_n'] += int(fractional.sum()); s['fractional_pred_sum'] += float(prob[fractional].sum())

    def report(self):
        rows=[]
        for case, s in sorted(self.cases.items()):
            n, on = s['r_n'], s['overlap_n']
            energy = max(0., s['r_sum_t2'] - s['r_sum_t']**2/n) if n else 0.
            centered = max(0., s['r_sse'] - (s['r_sum_p']-s['r_sum_t'])**2/n) if n else 0.
            ovvar = max(0., s['overlap_sum2']/on-(s['overlap_sum']/on)**2) if on else None
            mass = s['p_sum']; measured_mass = s['measured_soft_mass']
            rows.append(dict(case_id=case,
                support_bce=s['bce']/s['n'] if s['n'] else None,
                constant_support_bce=s['prior_bce']/s['n'] if s['n'] else None,
                support_dice=(2*s['pt_sum']+1e-5)/(s['p_sum']+s['t_sum']+1e-5) if s['n'] else None,
                support_precision=s['tp']/s['pred_n'] if s['pred_n'] else None,
                support_recall=s['tp']/s['truth_n'] if s['truth_n'] else None,
                support_target_sites=s['truth_n'], support_true_positive_sites=s['tp'], predicted_support_sites=s['pred_n'],
                radius_mse=s['r_sse']/n if n else None, constant_radius_mse=s['r_prior_sse']/n if n else None,
                case_centered_radius_ratio=centered/energy if energy > 0 else None,
                measured_radius_sites=n, consumer_sites=s['consumer_n'], consumer_overlap_sites=on,
                consumer_unmeasured_fraction=1-on/s['consumer_n'] if s['consumer_n'] else None,
                consumer_known_background_fraction=s['consumer_background_n']/s['consumer_n'] if s['consumer_n'] else None,
                consumer_unmeasured_foreground_sites=s['consumer_unmeasured_fg_n'],
                consumer_radius_mse=s['overlap_sse']/on if on else None,
                consumer_constant_radius_mse=s['overlap_prior_sse']/on if on else None,
                consumer_radius_variance=ovvar,
                soft_support_mass_on_unmeasured_fraction=1-measured_mass/mass if mass else None,
                soft_support_mass_on_background_fraction=s['background_soft_mass']/mass if mass else None,
                soft_consumed_radius_mse=s['soft_radius_sse']/measured_mass if measured_mass else None,
                soft_consumed_constant_radius_mse=s['soft_radius_prior_sse']/measured_mass if measured_mass else None,
                effective_radius_axis_absmean=s['radius_axis_sum']/s['n'] if s['n'] else None,
                unmeasured_radius_axis_fraction=s['unknown_radius_axis_sum']/s['radius_axis_sum'] if s['radius_axis_sum'] else None,
                fractional_lumen_target_sites=s['fractional_target_n'],
                fractional_lumen_predicted_support_mean=s['fractional_pred_sum']/s['fractional_target_n'] if s['fractional_target_n'] else None))
        summary = summarize_cases(rows)
        def mean(key):
            return summary.get(key, {}).get('mean')
        def below(a, b):
            return mean(a) is not None and mean(b) is not None and mean(a) < mean(b)
        tests = dict(
            every_case_has_anatomical_and_radius_support=bool(rows) and all(
                r['support_target_sites'] > 0 and r['case_centered_radius_ratio'] is not None for r in rows),
            every_case_has_true_support_overlap=bool(rows) and all(r['support_true_positive_sites'] > 0 for r in rows),
            support_dice_floor=mean('support_dice') is not None and mean('support_dice') >= self.min_support_dice,
            support_recall_floor=mean('support_recall') is not None and mean('support_recall') >= self.min_support_recall,
            support_beats_train_prior=below('support_bce', 'constant_support_bce'),
            radius_beats_train_mean=below('radius_mse', 'constant_radius_mse'),
            case_centered_radius_beats_constant=mean('case_centered_radius_ratio') is not None and mean('case_centered_radius_ratio') < 1.,
            every_case_has_measured_consumer_overlap=bool(rows) and all(r['consumer_overlap_sites'] > 0 for r in rows),
            consumed_radius_beats_train_mean=below('consumer_radius_mse', 'consumer_constant_radius_mse'),
            soft_consumed_radius_beats_train_mean=below('soft_consumed_radius_mse', 'soft_consumed_constant_radius_mse'))
        return dict(scope='deterministic sampled VAL patches; case-equal; support threshold .5 is diagnostic only',
            per_case=rows, summary=summary,
            readiness=dict(passed=all(tests.values()), tests=tests,
                min_support_dice=self.min_support_dice, min_support_recall=self.min_support_recall,
                meaning='Necessary prediction/consumer-overlap prerequisites only; not convergence or dynamic Dice efficacy.',
                consumer_definition='Production V and V*clip(radius,-4,4); high-support diagnostic V>=.5. Errors only where radius is measured.'))
