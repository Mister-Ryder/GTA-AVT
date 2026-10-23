import json,math,statistics as st
from pathlib import Path
COHORTS=("Dongyang","KiTS","Rider")

def require(value, message):
    if not value:
        raise ValueError(message)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def close(a, b, tolerance=1e-12):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)

def corr(x, y):
    vx, vy = (st.pvariance(x), st.pvariance(y))
    return st.mean(((a - st.mean(x)) * (b - st.mean(y)) for a, b in zip(x, y))) / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None

def recompute_semantics(rows, protocol):
    """Independent arithmetic; does not import the producing audit helper."""
    cases = protocol['fit_cases'] + protocol['probe_cases']
    primary = {c: [r for r in rows if r['case_id'] == c and r['slot'] < 8 and r['known_nonempty']] for c in cases}
    fit_complete = all((primary[c] for c in protocol['fit_cases']))
    require(fit_complete, 'Fit24 baseline lacks a case; never silently pool remaining cases')
    baseline = st.mean((st.mean((r['GT_ordinal'] for r in primary[c])) for c in protocol['fit_cases']))
    stats = {}
    cx, cy = ([], [])
    for c in protocol['probe_cases']:
        require(primary[c], 'Probe case has no primary support')
        y = [r['GT_ordinal'] for r in primary[c]]
        p = [r['raw_ordinal'] for r in primary[c]]
        cond = [r['conditional_ordinal'] for r in primary[c]]
        require(all((finite(v) for v in y + p + cond)), 'Undefined primary semantic value')
        stats[c] = dict(n=len(y), mse=st.mean(((a - b) ** 2 for a, b in zip(y, p))), baseline_mse=st.mean(((a - baseline) ** 2 for a in y)), conditional_mse=st.mean(((a - b) ** 2 for a, b in zip(y, cond))), GT_variance=st.pvariance(y), predicted_variance=st.pvariance(p), correlation=corr(y, p))
        cx.extend((a - st.mean(y) for a in y))
        cy.extend((a - st.mean(p) for a in p))
    g = protocol['semantic_gate']
    informative = [c for c, s in stats.items() if s['n'] >= g['minimum_primary_nonempty_per_case'] and s['GT_variance'] >= g['minimum_GT_ordinal_variance'] and (s['predicted_variance'] > g['minimum_raw_ordinal_std'] ** 2)]
    positive = [c for c in informative if stats[c]['correlation'] is not None and stats[c]['correlation'] > 0]
    mse = st.mean((s['mse'] for s in stats.values()))
    baseline_mse = st.mean((s['baseline_mse'] for s in stats.values()))
    reduction = 1 - mse / baseline_mse if baseline_mse > 0 else None
    centered_r = corr(cx, cy)
    conditional = st.mean((s['conditional_mse'] for s in stats.values()))
    sources = {name: dict(model_MSE=st.mean((s['mse'] for c, s in stats.items() if c.startswith(name + '_'))), fit_mean_MSE=st.mean((s['baseline_mse'] for c, s in stats.items() if c.startswith(name + '_')))) for name in COHORTS}
    parts = dict(all_36_cases_and_576_rows_present=len(rows) == 576 and len({(r['case_id'], r['slot']) for r in rows}) == 576 and (len(primary) == 36), all_fit24_cases_support_baseline=fit_complete, all_probe12_cases_primary_support=all((s['n'] >= g['minimum_primary_nonempty_per_case'] for s in stats.values())), macro_MSE_improvement=reduction is not None and reduction >= g['minimum_fractional_MSE_improvement'], minimum_improved_cases=sum((s['mse'] < s['baseline_mse'] for s in stats.values())) >= g['minimum_improved_probe_cases'], all_sources_nonworse=all((s['model_MSE'] <= s['fit_mean_MSE'] for s in sources.values())), minimum_informative_probe_cases=len(informative) >= g['minimum_informative_probe_cases'], minimum_positive_within_case_correlations=len(positive) >= g['minimum_positive_within_case_correlations'], source_informative_support=all((sum((c.startswith(name + '_') for c in informative)) >= g['minimum_informative_probe_cases_per_source'] for name in COHORTS)), case_centered_correlation=centered_r is not None and centered_r >= g['minimum_case_centered_Pearson_r'], conditional_macro_MSE_nonworse=conditional <= baseline_mse)
    return dict(parts=parts, passed=all(parts.values()), fit24_case_equal_GT_ordinal_mean=baseline, probe12_case_equal_MSE=mse, probe12_case_equal_fit_mean_MSE=baseline_mse, probe12_fractional_MSE_improvement=reduction, conditional_macro_MSE=conditional, case_centered_Pearson_r=centered_r, informative_cases=informative, positive_correlation_cases=positive, per_case=stats, source_groups=sources)

def recompute_geometry(value, protocol):
    rows, cases = (value['rows'], value['case_stats'])
    original = protocol['fit_cases'] + protocol['probe_cases']
    require(set(cases) == set(original), 'Geometry case population differs')
    integrity = all((s['support_lost_nodes'] == 0 and s['supported_nodes'] == s['anchor_nodes'] > 0 for s in cases.values()))
    errors, variances = ({}, {})
    for c in original:
        group = [r for r in rows if r['case_id'] == c and r['slot'] < 8]
        require(len(group) == 8, 'Missing geometry primary slot')
        supported = all((r['original_support_nodes'] == r['common_support_nodes'] > 0 and r['lost_support_nodes'] == 0 and (r['all_original_anchors_retained'] is True) for r in group))
        integrity = integrity and supported
        if c not in protocol['probe_cases']:
            continue
        require(supported, 'Probe geometry has lost or undefined original anchors')
        native = [r['original_all_anchor_ordinal'] for r in group]
        iso = [r['resampled_EDT_common_anchor_ordinal'] for r in group]
        require(all((finite(x) and 0 <= x <= 1 for x in native + iso)), 'Invalid ordinal geometry value')
        errors[c] = st.mean(((a - b) ** 2 for a, b in zip(native, iso)))
        variances[c] = st.pvariance(native)
        require(close(errors[c], cases[c]['primary8_ordinal_measurement_MSE']) and close(variances[c], cases[c]['primary8_native_ordinal_variance']), 'Geometry summary/rows differ')
    n, v = (st.mean(errors.values()), st.mean(variances.values()))
    ratio = n / v if v > 0 else None
    parts = dict(probe12_measurement_rows_complete=len(errors) == 12 and len(variances) == 12, probe12_case_macro_MSE_to_within_case_signal_ratio=ratio is not None and ratio <= protocol['measurement_gate']['maximum_error_to_signal_MSE_ratio'])
    return dict(parts=parts, passed=all(parts.values()), input_integrity_passed=bool(integrity), probe12_case_macro_primary8_ordinal_measurement_MSE=n, probe12_case_macro_primary8_native_ordinal_variance=v, error_to_signal_MSE_ratio=ratio, original_anchor_count=sum((s['anchor_nodes'] for s in cases.values())))

def main():
    here=Path(__file__).resolve().parent
    read=lambda n:json.loads((here/n).read_text(encoding='utf-8'))
    protocol=read('audit_definitions.json')
    sem=recompute_semantics(read('TRAIN_head_rows.json'),protocol)
    summary=read('TRAIN_head_summary.json')
    geo=recompute_geometry(read('TRAIN_geometry.json'),protocol)
    expected=read('TRAIN_geometry.json')['measurement_gate']
    require(sem['parts']==summary['semantic_gate']['parts'],'Semantic gate mismatch')
    require(sem['passed'] and geo['passed'] and geo['input_integrity_passed'],'Prerequisite failed')
    require(close(sem['probe12_case_equal_MSE'],summary['probe12_case_equal_MSE']),'MSE mismatch')
    require(close(geo['error_to_signal_MSE_ratio'],expected['error_to_signal_MSE_ratio']),'Geometry ratio mismatch')
    require(geo['original_anchor_count']==55294,'Anchor support mismatch')
    for row in read('engineering_and_input_summary.json'):
        require(row['actual_QA_status']=='passed' and row['successful_optimizer_updates']==2,'Actual QA mismatch')
        require(row['TRAIN_VAL_cache_files_rehashed']==124 and row['native_VAL_files_rehashed']==16 and row['TEST_arrays_opened'] is False,'Byte audit mismatch')
    print(json.dumps(dict(status='passed',TRAIN_rows=576,anchors=55294,semantic_passed=sem['passed'],measurement_passed=geo['passed'],integrity_passed=geo['input_integrity_passed'],models_executed=False,arrays_opened=False,efficacy_claim=False)))

if __name__=='__main__':main()
