# RESQML v2 (EPC) → VTU Converter (with cell properties)

Convert **RESQML v2** corner-point / parametric grids stored in an **EPC** package into **VTK Unstructured Grid** files (`.vtu`) that open cleanly in **ParaView** (or other VTK-based tools), including **cell-indexed RESQML properties**.

This script uses:
- **resqpy** to read RESQML grids/properties from EPC
- **pyvista/vtk** to write `.vtu`
- **h5py/zipfile/xml** for extracting parametric point geometry stored in HDF5 and referenced from the EPC XML

---

## Features

- Reads all `IjkGridRepresentation` objects in an EPC.
- Exports each grid to its own `*.vtu` file.
- Builds **VTK hexahedral cells** (`VTK_HEXAHEDRON = 12`).
- Attaches **cell properties** (continuous/discrete/categorical) to `ugrid.cell_data`.
- Handles two geometry paths:
  1. **Corner-point geometry** via `grid.corner_points()` (preferred)
  2. **Point3dParametricArray (linear, KnotCount=2)** via HDF5 (fallback)

---

## Output

For each grid found in the EPC:
- Writes: `out_dir/<grid_title>.vtu`
- Adds metadata in `field_data`:
  - `resqpy_grid_title`
  - `resqpy_grid_uuid`
- Adds properties in `cell_data`:
  - Scalar: `(nk, nj, ni) → (nCells,)`
  - Vector/tensor: `(nk, nj, ni, nComp) → (nCells, nComp)`

---

## Installation

```bash
pip install resqpy pyvista vtk numpy h5py
