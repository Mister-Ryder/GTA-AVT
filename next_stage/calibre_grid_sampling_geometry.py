"""Synthetic, infinite axial-cylinder sampling geometry; default inert.

No real inputs, Torch, model, training or candidate decision. Circle/lattice
closed forms give continuous-phase mean, SD and miss probability. Midpoint
phase grids approximate the error distribution/range, without an image FOV.
"""
import argparse
import json
import math
from pathlib import Path

SPACING_MM = 2.0
RADII_MM = (.5, .75, 1., 1.5, 2., 4., 8., 16.)
PHASE_GRIDS = (256, 512, 1024)


def overlap_area(radius, distance):
    """Closed-form intersection area of two equal-radius disks, FP64 evaluation."""
    if distance >= 2. * radius:
        return 0.
    if distance == 0.:
        return math.pi * radius * radius
    return (2. * radius * radius * math.acos(distance / (2. * radius)) -
            .5 * distance * math.sqrt(4. * radius * radius - distance * distance))


def continuous_phase(radius, spacing=SPACING_MM):
    """E[N²]=sum_k disk_intersection_area(h*k)/h², including k=0.

    Tiling a fundamental phase cell by lattice translations gives E[h²N]=πr².
    The finite overlap sum is an analytic formula, not numerical quadrature.
    Floating-point evaluation can lose a few digits in the variance subtraction.
    """
    area = math.pi * radius * radius
    reach = math.ceil(2. * radius / spacing)
    intersections = [overlap_area(radius, spacing * math.hypot(i, j))
        for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)
        if spacing * math.hypot(i, j) < 2. * radius]
    mean_n = area / (spacing * spacing)
    variance_n = math.fsum(intersections) / (spacing * spacing) - mean_n * mean_n
    if variance_n < 0.:
        raise ArithmeticError('Negative variance from analytic overlap formula')
    sd = spacing * spacing * math.sqrt(variance_n)
    if radius <= spacing / 2.:
        miss = 1. - area / (spacing * spacing)
        miss_reason = 'Disjoint hit disks in the phase torus (tangent boundaries have measure zero)'
    elif radius >= spacing / math.sqrt(2.):
        miss = 0.
        miss_reason = 'Disk radius reaches the square-lattice covering radius'
    else:
        miss = None
        miss_reason = 'No closed-form miss probability implemented for this unused radius interval'
    return dict(true_area_mm2=area, mean_estimated_area_mm2=area,
        SD_estimated_area_mm2=sd, relative_mean_error=0., relative_SD=sd / area,
        complete_miss_probability=miss, miss_formula=miss_reason,
        uniform_phase_measure='Continuous uniform translation over [0,2mm)²')


def phase_samples(radius, size, spacing=SPACING_MM):
    """Deterministic 2D midpoint rule; finite support exactly covers the disk.

    Count vertical lattice sites with floor/ceil rather than constructing an
    image window. Thus every potentially intersecting lattice site is counted.
    """
    import numpy as np
    phase = (np.arange(size, dtype=np.float64) + .5) * spacing / size
    count = np.zeros((size, size), dtype=np.uint16)
    reach = math.ceil(radius / spacing)
    for i in range(-reach, reach + 2):
        dx = spacing * i - phase
        valid = abs(dx) <= radius
        if not valid.any():
            continue
        half_height = np.sqrt(np.maximum(0., radius * radius - dx[valid] * dx[valid]))
        low = np.ceil((phase[None, :] - half_height[:, None]) / spacing)
        high = np.floor((phase[None, :] + half_height[:, None]) / spacing)
        count[valid, :] += np.maximum(0., high - low + 1.).astype(np.uint16)
    estimate = (spacing * spacing) * count.astype(np.float64)
    truth = math.pi * radius * radius
    relative = (estimate - truth) / truth
    return dict(phase_grid_per_axis=size, phases=size * size,
        phase_rule='Uniform deterministic midpoint lattice, not random samples',
        mean_estimated_area_mm2=float(estimate.mean()), SD_estimated_area_mm2=float(estimate.std(ddof=0)),
        sampled_area_range_mm2=[float(estimate.min()), float(estimate.max())],
        sampled_point_count_range=[int(count.min()), int(count.max())],
        mean_relative_error=float(relative.mean()), relative_RMSE=float(np.sqrt(np.mean(relative * relative))),
        mean_absolute_relative_error=float(np.abs(relative).mean()),
        sampled_relative_error_range=[float(relative.min()), float(relative.max())],
        complete_miss_fraction=float((count == 0).mean()))


def analyze():
    results = []
    for radius in RADII_MM:
        exact = continuous_phase(radius)
        samples = [phase_samples(radius, n) for n in PHASE_GRIDS]
        results.append(dict(radius_mm=radius, analytic_continuous_phase=exact,
            numerical_phase_grids=samples, reported_phase_grid=samples[-1],
            last_two_grid_absolute_changes={key: abs(samples[-1][key] - samples[-2][key])
                for key in ('mean_estimated_area_mm2', 'SD_estimated_area_mm2',
                            'mean_absolute_relative_error', 'complete_miss_fraction')}))
    example = dict(radius_mm=1., axis_xy_mm=[1., 1.],
        nearest_four_grid_points_xy_mm=[[0., 0.], [0., 2.], [2., 0.], [2., 2.]],
        nearest_distance_mm=math.sqrt(2.), all_infinite_lattice_points_miss=True,
        point_sample_estimated_area_mm2=0., true_area_mm2=math.pi,
        four_centered_grid_cells_each_intersection_area_mm2=math.pi / 4.,
        each_cell_integrated_support_fraction=math.pi / 16.,
        integral='Each cell centered at a nearest grid point contains one quarter of the disk; exact by symmetry')
    return dict(schema='GTA-AVT-synthetic-calibre-grid-sampling-geometry-v1',
        settings=dict(spacing_mm=SPACING_MM, radii_mm=RADII_MM, phase_grids=PHASE_GRIDS,
            geometry='Infinite cylinder parallel to a lattice axis; analytic disk cross-section',
            estimate='h² times the number of infinite-lattice points in the closed disk',
            image_window_or_FOV=False, random_sampling=False),
        radii=results, analytic_counterexample=example,
        limitations=[
            'Finite midpoint-grid ranges are observed ranges, not certified extrema over every continuous phase; isolated tangent phases can differ.',
            'Phase-grid differences are convergence diagnostics, not rigorous error bounds. Continuous mean/SD/miss use the stated analytic formulas.',
            'No numerical cell integration is performed; the single integrated-cell counterexample is exact by symmetry.',
            'A continuous ideal cylinder is not the cached nearest-node radius-ball field, which has discrete nodes, exclusions, branches and FOV censoring.',
            'This does not establish an AVT bottleneck, true boundary coverage deficit, routing benefit, new success gate or training authorization.'])


def render(result):
    lines = ['纯合成：2mm网格对无限轴向圆柱的相位敏感性。无真实影像、模型、训练或FOV。',
        '连续均匀相位：面积均值解析无偏；SD由圆盘重叠面积有限和解析求得。',
        '下表范围/平均绝对相对误差来自1024²中点相位近似；并非连续相位严格极值或严格误差界。',
        'r(mm) | 真面积/解析均值(mm²) | 解析SD(mm²) | 相对SD(%) | 采样面积范围(mm²) | 平均绝对相对误差(%) | 解析完全漏检率(%)']
    for row in result['radii']:
        a, n = row['analytic_continuous_phase'], row['reported_phase_grid']
        lines.append('%g | %.8g | %.8g | %.5f | %s | %.5f | %.5f' %
            (row['radius_mm'], a['true_area_mm2'], a['SD_estimated_area_mm2'],
             100. * a['relative_SD'], n['sampled_area_range_mm2'],
             100. * n['mean_absolute_relative_error'], 100. * a['complete_miss_probability']))
    lines += ['', '明确反例：r=1mm、管轴在(1,1)mm，最近四点均距sqrt(2)mm，整张无限网格完全漏检；',
        '但四个相邻、以网格点为中心的2mm单元各有π/4mm²圆盘交集，积分支持比例各为π/16≈0.19635。',
        '相位均值无偏不表示单个管段无偏；小半径的相位方差和漏检概率仍可很高。',
        '真实oracle已经做半径球内传播。本计算只说明点采样的理想混叠可能性，不能证明真实AVT属性支撑失败或后继方法有效。',
        '所有近似范围与跨分辨率差见JSON；未进行数值单元积分、拟合、阈值搜索或真实数据访问。']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', synthetic_only=True, files_read=False, writes=False)))
    else:
        if not args.output or Path(args.output).exists():
            parser.error('A fresh explicit --output directory is required')
        result = analyze()
        output = Path(args.output); output.mkdir(parents=True)
        (output / 'analysis.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        (output / 'readout.txt').write_bytes(render(result).encode('utf-8'))
        print(json.dumps(dict(status='complete', synthetic_only=True, radii=RADII_MM, real_inputs_read=False)))
