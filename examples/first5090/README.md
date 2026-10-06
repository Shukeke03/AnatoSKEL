# First real CT model prediction

Fixed held-out case: `sub-gl279_dir-ax`. Model prediction: 138568 vertices, 277197 faces.

OBJ coordinates are RAS millimetres; keep bones.json with the OBJ. metrics.json contains aggregate validation/test results. The case was fixed before optimization, not selected by its output quality. No CT, weights, cache or full result tree is included.

This is an initial research experiment; anatomical pose axes and clinical accuracy have not been validated. Empty predictions, if present, are reported honestly and never replaced by a ground-truth mesh.

Data attribution: [VerSe](https://github.com/anjany/verse), [dataset paper](https://doi.org/10.1016/j.media.2021.102166). This derived example is provided under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
