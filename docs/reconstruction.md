# Reconstruction integrations

The package provides a real `underwater sfm aliked-colmap` workflow. It migrates the legacy `lightglue.ALIKED` + `kornia.feature.match_adalam` behavior into explicit feature, matching, database, and external-process boundaries. ORB + BFMatcher remains available only as a named lightweight fixture backend; it is never a fallback for the ALIKED command.

`integrations/colmap/` documents the lower-level external process contract and makes calibration refinement explicit. The default mode freezes focal length, principal point, and extra parameters; refinement must be requested deliberately. See [runbooks/aliked_adalam_colmap.md](runbooks/aliked_adalam_colmap.md) for the complete workflow and output layout.

The default input view is raw unrectified image data, with keypoint coordinates restored to original image pixels before insertion into COLMAP. Pair generation is bounded to temporal and synchronized stereo pairs; it does not create an accidental all-pairs graph. Feature cache metadata invalidates stale image/config combinations.

For near-planar scenes, `--calibrated-stereo-planar` uses the canonical ZED
left/right extrinsics to triangulate each synchronized pair and estimates
adjacent-frame motion from 3-D correspondences. This deliberately avoids
using a homography-derived incremental initialization as the only seed; a
plane is therefore treated as a valid scene geometry. The generated model is
still a COLMAP-compatible model and is checked with `model_analyzer`, but its
poses are produced by the calibrated-stereo path rather than ordinary
incremental `mapper`.

`reconstruction.metashape` converts COLMAP-style world-to-camera poses to local camera-to-world references and writes explicit `georeferenced=false` and scale-source metadata. `integrations/metashape/` contains import guidance. These coordinates are not GPS/geographic coordinates, and arbitrary SfM scale must not be presented as physical underwater scale.

Ordinary COLMAP reconstruction scale is arbitrary/local unless an actual valid
scale constraint is supplied. The calibrated-stereo planar path supplies the
canonical stereo baseline, so its model scale is metric relative to that
calibration; this does not by itself establish absolute underwater physical
accuracy. Metashape conversion remains a library/export boundary and external
GUI import is separate.
