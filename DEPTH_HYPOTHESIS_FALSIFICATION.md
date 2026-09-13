# Depth Hypothesis Falsification

The question is whether the operational custom result near 2.2 m is already a physical underwater distance (H0), or whether it is an air-equivalent result that still needs approximately `×1.333` (H1).

## Candidate definitions

```text
Z1 = native SVO rectified f × native rectified B / native disparity
Z2 = 1.333 × Z1                         # scalar refractive candidate
Z3 = custom rectified f × custom rectified B / custom disparity
Z4 = 1.333 × Z3                         # custom + extra Snell control only
```

`Z1/Z2` and `Z3/Z4` use different rectified image geometries and therefore different disparities. They are not interchangeable measurements.

## Same-frame numerical evidence

From the 100-frame independent native/custom audit:

| region / statistic | Z1 native | Z2 native×1.333 | Z3 custom | Z4 custom×1.333 |
|---|---:|---:|---:|---:|
| native image-center neighborhood median | 2.0615 m | 2.7480 m | — | — |
| custom image-center neighborhood median | — | — | 2.1989 m | 2.9311 m |
| native central 40% median | 2.0075 m | 2.6761 m | — | — |
| custom central 40% median | — | — | 2.2302 m | 2.9729 m |

The center/central regions are separate coordinate-system regions; they are not asserted to be the same physical rays.

For the per-frame pipeline comparison, the custom/native ratios were:

| paired region | frames | median `Z3/Z2` | median `Z3/Z1` | median `Z4/Z2` |
|---|---:|---:|---:|---:|
| same output-coordinate labels | 95 | 0.8015 | 1.0684 | 1.0684 |
| central area labels | 100 | 0.8333 | 1.1108 | 1.1108 |
| optical-center neighborhoods, explicitly not the same ray | 100 | 0.8208 | 1.0941 | 1.0941 |

These ratios demonstrate why comparing only focal lengths, or comparing different disparities, cannot falsify H0/H1. They do not provide metric GT.

## Evidence table

| evidence | supports H0 | supports H1 | strength / limitation |
|---|---|---|---|
| custom raw focal ratio about 1.35 | weakly compatible | weakly compatible | no shared image geometry or provenance proof |
| custom rectified `f=1423.95` versus native rectified `f=1078.94` | weakly compatible | weakly compatible | disparity and virtual principal points also changed |
| custom `fB/d` and OpenCV Q agree | neither | neither | implementation identity, not absolute truth |
| alpha=0/1 same-correspondence depth invariance | neither | neither | rectification consistency only |
| raw native ×1.333 produces 2.75 m near center | weakly against | weakly compatible | no independent target distance |
| custom ×1.333 produces 2.93–2.97 m | weakly compatible | weakly compatible | it is a hypothesized multiplier, not a measured model |
| calibration medium / housing / target scale | unavailable | unavailable | all required provenance fields are UNKNOWN |
| independent metric GT | unavailable | unavailable | no tape/laser/rod/known target distance found |
| explicit Snell rig model | unavailable | unavailable | port geometry and indices missing |

## Verdict

The falsification result is **insufficient evidence**: neither H0 nor H1 is closed by the repository alone. Operationally, use the custom result without an extra multiplier only as an empirical R2 output, and keep the absolute accuracy claim open. Do not promote Z4 to a physical answer.

Precision and accuracy remain separate: frame-to-frame spread and Q consistency quantify repeatability/internal consistency; they do not establish absolute metric accuracy.
