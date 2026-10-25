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
    """Case-equal support and radius-quality diagnostics on sampled patches.

    Radius errors use measured targets only. Consumer overlap additionally
    requires predicted anatomical support and positive relative information.
    No radius truth is invented at unmeasured consumer locations.
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
        logvar = np.clip(array(output['radius_log_variance']), -6., 4.)
        support, radius = array(batch['support']), array(batch['r'])
        valid, measured = array(batch['support_valid']).astype(bool), array(batch['r_valid']).astype(bool)
        if any(value.shape != logits.shape for value in (predicted, logvar, support, radius, valid, measured)):
            raise ValueError('Metric attribute/target shapes differ')
        if (measured & ~valid).any():
            raise ValueError('Measured radius cannot include unobserved cells')
        ids = batch.get('case_id', [str(i) for i in range(len(support))])
        for i, case in enumerate(ids):
            s = self.cases.setdefault(str(case), dict(n=0, bce=0., prior_bce=0., p_sum=0., t_sum=0., pt_sum=0.,
                tp=0, pred_n=0, truth_n=0, r_n=0, r_sse=0., r_prior_sse=0., r_sum_p=0., r_sum_t=0.,
                r_sum_p2=0., r_sum_t2=0., r_sum_pt=0., r_var_sum=0., r_nll_sum=0., q_sum=0.,
                q_error_sum=0., q_square_sum=0., error_square_sum=0., effective_weight_sum=0.,
                consumer_n=0, overlap_n=0, overlap_sse=0., overlap_prior_sse=0.,
                overlap_sum=0., overlap_sum2=0., r_q_sum=0.))
            real = valid[i]
            ll, tt = logits[i][real].astype(np.float64), support[i][real].astype(np.float64)
            pp, vv = predicted[i][real].astype(np.float64), np.exp(logvar[i][real].astype(np.float64))
            if not all(np.isfinite(value).all() for value in (ll, tt, pp, vv)):
                raise ValueError('Nonfinite attribute prediction on real cells')
            prob = np.exp(-np.logaddexp(0., -ll))
            q = np.maximum((1. - vv) / (1. + vv), 0.)
            hard, truth = prob >= .5, tt >= .5
            consumer = hard & (q > 0.)
            mr = measured[i][real]
            target = radius[i][real][mr].astype(np.float64)
            if not np.isfinite(target).all():
                raise ValueError('Nonfinite measured radius target')
            s['n'] += len(tt); s['bce'] += float((np.logaddexp(0., ll) - tt * ll).sum())
            s['prior_bce'] += float((-tt*np.log(self.prior) - (1.-tt)*np.log1p(-self.prior)).sum())
            s['p_sum'] += float(prob.sum()); s['t_sum'] += float(tt.sum()); s['pt_sum'] += float((prob*tt).sum())
            s['tp'] += int((hard & truth).sum()); s['pred_n'] += int(hard.sum()); s['truth_n'] += int(truth.sum())
            rp, rv, rq = pp[mr], vv[mr], q[mr]
            error = (rp-target)**2
            s['r_n'] += len(target); s['r_sse'] += float(error.sum()); s['r_prior_sse'] += float((target**2).sum())
            s['r_var_sum'] += float(rv.sum()); s['r_nll_sum'] += float((.5*(error/rv+np.log(rv))).sum())
            s['r_q_sum'] += float(rq.sum())
            for key, value in (('r_sum_p', rp.sum()), ('r_sum_t', target.sum()), ('r_sum_p2', (rp**2).sum()),
                               ('r_sum_t2', (target**2).sum()), ('r_sum_pt', (rp*target).sum()),
                               ('q_sum', rq.sum()), ('q_error_sum', (rq*error).sum()),
                               ('q_square_sum', (rq**2).sum()), ('error_square_sum', (error**2).sum())):
                s[key] += float(value)
            s['effective_weight_sum'] += float((prob*q).sum())
            s['consumer_n'] += int(consumer.sum())
            overlap = consumer & mr
            ovp = pp[overlap]; ovt = radius[i][real][overlap].astype(np.float64)
            s['overlap_n'] += int(overlap.sum()); s['overlap_sse'] += float(((ovp-ovt)**2).sum())
            s['overlap_prior_sse'] += float((ovt**2).sum())
            s['overlap_sum'] += float(ovp.sum()); s['overlap_sum2'] += float((ovp**2).sum())

    def report(self):
        rows=[]
        for case, s in sorted(self.cases.items()):
            n, on = s['r_n'], s['overlap_n']
            energy = max(0., s['r_sum_t2'] - s['r_sum_t']**2/n) if n else 0.
            centered = max(0., s['r_sse'] - (s['r_sum_p']-s['r_sum_t'])**2/n) if n else 0.
            qvar = max(0., s['q_square_sum'] - s['q_sum']**2/n) if n else 0.
            evar = max(0., s['error_square_sum'] - s['r_sse']**2/n) if n else 0.
            cov = s['q_error_sum'] - s['q_sum']*s['r_sse']/n if n else 0.
            ovvar = max(0., s['overlap_sum2']/on-(s['overlap_sum']/on)**2) if on else None
            rows.append(dict(case_id=case,
                support_bce=s['bce']/s['n'] if s['n'] else None,
                constant_support_bce=s['prior_bce']/s['n'] if s['n'] else None,
                support_dice=(2*s['pt_sum']+1e-5)/(s['p_sum']+s['t_sum']+1e-5) if s['n'] else None,
                support_precision=s['tp']/s['pred_n'] if s['pred_n'] else None,
                support_recall=s['tp']/s['truth_n'] if s['truth_n'] else None,
                support_target_sites=s['truth_n'], support_true_positive_sites=s['tp'], predicted_support_sites=s['pred_n'],
                radius_mse=s['r_sse']/n if n else None, constant_radius_mse=s['r_prior_sse']/n if n else None,
                case_centered_radius_ratio=centered/energy if energy > 0 else None,
                radius_nll=s['r_nll_sum']/n if n else None, predicted_radius_variance=s['r_var_sum']/n if n else None,
                measured_reliability_mean=s['r_q_sum']/n if n else None,
                reliability_error_correlation=cov/np.sqrt(qvar*evar) if qvar*evar > 0 else None,
                measured_radius_sites=n, consumer_sites=s['consumer_n'], consumer_overlap_sites=on,
                consumer_unmeasured_fraction=1-on/s['consumer_n'] if s['consumer_n'] else None,
                consumer_radius_mse=s['overlap_sse']/on if on else None,
                consumer_constant_radius_mse=s['overlap_prior_sse']/on if on else None,
                consumer_radius_variance=ovvar,
                effective_radius_weight_mean=s['effective_weight_sum']/s['n'] if s['n'] else None))
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
            consumed_radius_beats_train_mean=below('consumer_radius_mse', 'consumer_constant_radius_mse'))
        return dict(scope='deterministic sampled VAL patches; case-equal; support threshold .5 is diagnostic only',
            per_case=rows, summary=summary,
            readiness=dict(passed=all(tests.values()), tests=tests,
                min_support_dice=self.min_support_dice, min_support_recall=self.min_support_recall,
                meaning='Prerequisites only; no segmentation efficacy or calibrated uncertainty claim.',
                consumer_definition='Predicted support>=.5 and relative information q>0; error evaluated only on measured overlap.'))
