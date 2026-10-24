"""Image-only CT offset audit for AVT NRRDs; no mask or cohort input.

Mixed raw NRRDs can be HU or HU+1024 and can additionally contain -2000
padding. The heuristic below requires a clear air-intensity signal; ambiguous
images return offset=None and must not silently pass to fixed-HU windowing.
An offset inferred from air is not a substitute for original DICOM calibration.
"""
import numpy as np


def detect_ct_offset(image):
    x=np.asarray(image)
    step=max(1,int(np.ceil((x.size/750000)**(1/3)))) if x.ndim==3 else 1
    sample=x[::step,::step,::step].ravel() if x.ndim==3 else x.ravel()[::max(1,x.size//750000)]
    sample=sample[np.isfinite(sample)].astype(np.float32)
    if not len(sample):raise ValueError('CT has no finite intensities')
    # Common padding values are excluded from the decision, not from the image.
    # Exact -1024/0 can be scan padding in either convention; nearby air remains.
    usable=sample[(sample>-1500)&(sample!=-1024)&(sample!=0)]
    neg=usable[(usable>=-1120)&(usable<=-850)]
    pos=usable[(usable>=-96)&(usable<=174)]
    n=max(len(usable),1)
    def peak(a):
        if not len(a):return {'center':None,'count_32unit_window':0}
        h,edges=np.histogram(a,bins=np.arange(-1200,2209,8))
        smooth=np.convolve(h,np.ones(4,dtype=int),mode='valid');i=int(smooth.argmax())
        return dict(center=float(edges[i]+16),count_32unit_window=int(smooth[i]))
    pn,pp=peak(neg),peak(pos)
    nf,pf=len(neg)/n,len(pos)/n
    a,b=pn['count_32unit_window'],pp['count_32unit_window']
    offset=None;rule='ambiguous_air_signal'
    if nf>=.01:
        offset=0.;rule='nonpadding_negative_air_fraction_ge_1percent'
    elif nf<.0001 and pf>=.01:
        offset=1024.;rule='no_negative_air_and_positive_air_fraction_ge_1percent'
    elif a>=4*max(b,1) and nf>=.001:
        offset=0.;rule='negative_air_peak_dominates_4x'
    elif b>=4*max(a,1) and pf>=.01:
        offset=1024.;rule='positive_air_peak_dominates_4x'
    # A positive soft-tissue mode is not sufficient evidence for stored+1024.
    # Require a spatial exterior near24 peak and appreciable denser anatomy.
    spatial=None
    if x.ndim==3:
        sy=max(1,int(round(x.shape[1]*.05)));sz=max(1,int(round(x.shape[2]*.05)))
        outside=np.concatenate([x[:,:sy,:].ravel()[::11],x[:,-sy:,:].ravel()[::11],
                                x[:,:,:sz].ravel()[::11],x[:,:,-sz:].ravel()[::11]])
        outside=outside[np.isfinite(outside)&(outside>-1500)&(outside!=-1024)&(outside!=0)]
        positive_outside=outside[(outside>=-96)&(outside<=174)]
        outside_peak=peak(positive_outside)
        concentration=outside_peak['count_32unit_window']/max(len(outside),1)
        centre=outside_peak['center']
        dense_fraction=float(np.mean(usable>(centre+250.))) if centre is not None and len(usable) else 0.
        spatial=dict(exterior_32unit_peak=outside_peak,exterior_usable_count=int(len(outside)),
                     exterior_peak_fraction=float(concentration),dense_fraction_above_peak_plus250=dense_fraction)
        shifted_confirmed=(centre is not None and 8.<=centre<=40. and len(outside)>=100
                           and concentration>=.20 and dense_fraction>=.01)
    else:
        shifted_confirmed=False
    if offset==1024. and not shifted_confirmed:
        offset=None;rule='positive_air_without_spatial_peak_and_contrast_support'
    return dict(offset_subtract=offset,rule=rule,requires_review=offset is None,
        negative_air_fraction=float(nf),positive_air_fraction=float(pf),
        negative_air_peak=pn,positive_air_peak=pp,spatial_exterior_support=spatial,
        sample_count=int(len(sample)),decision_sample_count=int(len(usable)),
        raw_quantiles_0_001_01_50_99_999_100=np.quantile(sample,[0,.001,.01,.5,.99,.999,1]).tolist(),
        excluded_decision_values='<=-1500, exact -1024, exact 0; image itself unchanged',
        method='CT-only air histogram heuristic; mixed-offset NRRD audit, not verified DICOM calibration')


def to_hu_estimate(image):
    audit=detect_ct_offset(image)
    if audit['requires_review']:
        raise ValueError('Ambiguous AVT intensity convention; inspect audit before fixed-HU preprocessing: '+str(audit))
    return np.asarray(image,dtype=np.float32)-audit['offset_subtract'],audit
