"""Check the exported air-domain STEP in meters without modifying the source."""
import json
from pathlib import Path
import gmsh
import numpy as np

root = Path(__file__).resolve().parents[1]
gmsh.initialize()
try:
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.occ.importShapes(str(root / "geometry" / "air-copper-cylinder.step"))
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(3)
    assert len(volumes) == 1
    bounds = gmsh.model.getBoundingBox(*volumes[0])
    dimensions = np.array(bounds[3:])-np.array(bounds[:3])
    np.testing.assert_allclose(dimensions, [.105, .105, .070], atol=1e-6, rtol=0)
    record = {"geometry": "Air domain, not metal shell", "units": "m",
              "bounding_box": list(bounds), "dimensions_m": dimensions.tolist(),
              "expected_dimensions_m": [.105, .105, .070], "passed": True}
    (root / "validation" / "step-unit-check.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
finally:
    gmsh.finalize()
