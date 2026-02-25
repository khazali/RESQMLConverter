#!/usr/bin/env python3
"""
Convert RESQML v2 (EPC) to VTK (.vtu) with cell properties using resqpy + pyvista.

Install:
  pip install resqpy pyvista vtk numpy

Usage:
  python resqml_v2_to_vtk.py input.epc out_dir
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.property as rqp



try:
    import pyvista as pv
except ImportError as e:
    raise SystemExit("pyvista is required. Install with: pip install pyvista vtk") from e


VTK_HEXAHEDRON = 12  # VTK cell type id for hexahedron


import xml.etree.ElementTree as ET

import zipfile
import h5py


def _debug_split_arrays(h5_path: Path, resqml_uuid: str) -> None:
    base = f"/RESQML/{resqml_uuid}"
    keys = [
        "PillarIndices",
        "ColumnsPerSplitCoordinateLine",
        "SplitCoordinateLineIndices",
        "ColumnIndicesForSplitCoordinateLines",
        "PillarGeometryIsDefined",
        "CellGeometryIsDefined",
        "LineKindIndices",
    ]

    with h5py.File(h5_path, "r") as h5:
        print("=== DEBUG split arrays under", base, "===")
        for k in keys:
            p = f"{base}/{k}"
            if p not in h5:
                print(f"{k:35s} MISSING")
                continue

            obj = h5[p]
            if isinstance(obj, h5py.Dataset):
                print(f"{k:35s} DATASET shape={obj.shape} dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                # list children datasets/groups
                kids = list(obj.keys())
                print(f"{k:35s} GROUP children={kids}")
                # also show dataset details for common child names
                for child in kids:
                    child_path = f"{p}/{child}"
                    child_obj = h5[child_path]
                    if isinstance(child_obj, h5py.Dataset):
                        print(f"  - {child:31s} dataset shape={child_obj.shape} dtype={child_obj.dtype}")
                    else:
                        print(f"  - {child:31s} group")
            else:
                print(f"{k:35s} UNKNOWN type={type(obj)}")



def _debug_list_hdf5(h5_path: Path, max_items: int = 50) -> None:
    print("DEBUG: Listing HDF5 content:", h5_path)
    with h5py.File(h5_path, "r") as h5:
        items = []
        def visitor(name, obj):
            items.append(name)
        h5.visititems(lambda name, obj: visitor(name, obj))
        for i, name in enumerate(items[:max_items]):
            print(f"  {i:02d}: /{name}")
        if len(items) > max_items:
            print(f"  ... ({len(items) - max_items} more)")


def _find_hdf5_containing_datasets(epc_path: Path, required_paths: list[str]) -> Path:
    """
    Find an HDF5 file (next to EPC or inside EPC zip) that contains all required dataset paths.
    """
    # Candidates next to EPC
    candidates = list(epc_path.parent.glob("*.h5")) + list(epc_path.parent.glob("*.hdf5"))

    # Also candidates inside the EPC zip
    zip_h5_files: list[tuple[str, Path]] = []
    with zipfile.ZipFile(epc_path, "r") as z:
        h5_names = [n for n in z.namelist() if n.lower().endswith((".h5", ".hdf5"))]
        for name in h5_names:
            out = epc_path.parent / Path(name).name
            if not out.exists():
                z.extract(name, path=epc_path.parent)
                extracted = epc_path.parent / name
                # extracted might be in a subfolder; rename to flat file for convenience
                extracted.rename(out)
            candidates.append(out)

    def has_all_paths(h5_path: Path) -> bool:
        try:
            with h5py.File(h5_path, "r") as h5:
                for p in required_paths:
                    if p not in h5:
                        return False
            return True
        except Exception:
            return False

    for c in candidates:
        if has_all_paths(c):
            return c

    # If nothing matched, print what we checked to help debug
    print("DEBUG: Could not find any HDF5 file containing these paths:")
    for p in required_paths:
        print("  ", p)
    print("DEBUG: Checked these candidate HDF5 files:")
    for c in candidates:
        print("  ", c)

    raise RuntimeError("No HDF5 file found that contains the required datasets.")



def _grid_points_from_point3d_parametric_array(grid, epc_path: Path) -> np.ndarray:
    """
    Build explicit grid points from a resqml20:Point3dParametricArray with KnotCount=2 (linear).

    Handles your case:
      - ControlPoints: (2, nj+1, ni+1, 3)
      - PointParameters: (nk+1, n_coord_lines) with n_coord_lines >= (nj+1)*(ni+1)

    NOTE: If n_coord_lines > (nj+1)*(ni+1), this function uses ONLY the first
          (nj+1)*(ni+1) parameter columns (primary pillars) and ignores split pillars.
    """
    root = grid.root
    points_node = root.find(".//{*}Geometry/{*}Points")
    if points_node is None:
        raise RuntimeError("No Geometry/Points node found.")

    cp_path_node = points_node.find(".//{*}ControlPoints//{*}Coordinates//{*}PathInHdfFile")
    if cp_path_node is None or not cp_path_node.text:
        raise RuntimeError("Could not find ControlPoints/Coordinates/PathInHdfFile.")
    control_points_hdf_path = cp_path_node.text.strip()

    pp_path_node = points_node.find(".//{*}Parameters//{*}Values//{*}PathInHdfFile")
    if pp_path_node is None or not pp_path_node.text:
        raise RuntimeError("Could not find Parameters/Values/PathInHdfFile.")
    point_params_hdf_path = pp_path_node.text.strip()

    # infer resqml uuid from ControlPoints path: /RESQML/<uuid>/ControlPoints
    parts = control_points_hdf_path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "RESQML":
        raise RuntimeError(f"Unexpected ControlPoints path: {control_points_hdf_path}")
    resqml_uuid = parts[1]

    h5_path = _find_hdf5_containing_datasets(
        epc_path,
        required_paths=[control_points_hdf_path, point_params_hdf_path],
    )

    nk, nj, ni = grid.extent_kji
    n_pillars_primary = (nj + 1) * (ni + 1)

    with h5py.File(h5_path, "r") as h5:
        control_points = np.array(h5[control_points_hdf_path])
        point_params = np.array(h5[point_params_hdf_path])

    # KnotCount should be 2 (linear)
    knot_count_node = points_node.find(".//{*}KnotCount")
    knot_count = int(knot_count_node.text) if (knot_count_node is not None and knot_count_node.text) else None
    if knot_count != 2:
        raise RuntimeError(f"This helper currently supports KnotCount=2 (linear). Found KnotCount={knot_count}.")

    # ---- ControlPoints: expect (2, nj+1, ni+1, 3) in your file ----
    if not (control_points.ndim == 4 and control_points.shape[0] == 2 and control_points.shape[3] == 3):
        raise RuntimeError(f"ControlPoints shape not supported by this simplified path: {control_points.shape}")

    if control_points.shape[1] != (nj + 1) or control_points.shape[2] != (ni + 1):
        raise RuntimeError(
            f"ControlPoints shape {control_points.shape} not compatible with (nj+1,ni+1)=({nj+1},{ni+1})"
        )

    p0_grid = control_points[0]  # (nj+1, ni+1, 3)
    p1_grid = control_points[1]  # (nj+1, ni+1, 3)

    # Flatten pillars in (j,i) with i fastest
    p0 = p0_grid.reshape(n_pillars_primary, 3, order="C")
    p1 = p1_grid.reshape(n_pillars_primary, 3, order="C")

    # ---- PointParameters: expect one axis == nk+1 ----
    if point_params.ndim != 2:
        raise RuntimeError(f"Unexpected PointParameters ndim={point_params.ndim}, shape={point_params.shape}")

    if point_params.shape[0] == (nk + 1):
        params_k_line = point_params  # (nk+1, n_coord_lines)
    elif point_params.shape[1] == (nk + 1):
        params_k_line = point_params.T
    else:
        raise RuntimeError(f"PointParameters shape {point_params.shape} does not include nk+1={nk+1} on either axis.")

    n_coord_lines = params_k_line.shape[1]
    if n_coord_lines < n_pillars_primary:
        raise RuntimeError(
            f"PointParameters has only {n_coord_lines} lines, but need at least primary pillars {n_pillars_primary}."
        )

    if n_coord_lines != n_pillars_primary:
        print(
            f"WARNING: PointParameters has {n_coord_lines} lines but primary pillars are {n_pillars_primary}. "
            "Using only the first primary-pillar block and ignoring split pillars."
        )

    # Take primary pillar parameters only
    params_primary = params_k_line[:, :n_pillars_primary]  # (nk+1, n_pillars_primary)

    # Evaluate linear P(t) for each k and pillar
    points = np.empty((nk + 1, nj + 1, ni + 1, 3), dtype=float)

    for k in range(nk + 1):
        t = params_primary[k, :].astype(float)[:, None]  # (n_pillars_primary,1)
        pk = (1.0 - t) * p0 + t * p1                     # (n_pillars_primary,3)
        points[k] = pk.reshape((nj + 1, ni + 1, 3), order="C")

    return points


def _dump_points_xml(grid, max_chars: int = 4000) -> None:
    """Print the Geometry/Points XML snippet and the unique tag names under it."""
    root = grid.root
    points_node = root.find(".//{*}Geometry/{*}Points")
    if points_node is None:
        print("DEBUG: No Geometry/Points node found.")
        return

    xml_txt = ET.tostring(points_node, encoding="unicode")
    print("\n==== DEBUG Geometry/Points XML (truncated) ====")
    print(xml_txt[:max_chars])
    print("==== END DEBUG ====\n")

    # Show unique local tag names
    tags = set()
    for n in points_node.iter():
        tags.add(n.tag.split("}")[-1])
    print("DEBUG: Unique tags under Geometry/Points:")
    print(sorted(tags))


def _text_float(node, tag_suffix: str) -> float:
    """Find first child whose tag ends with tag_suffix and return float(text)."""
    for c in list(node):
        if c.tag.endswith(tag_suffix):
            return float(c.text)
    raise ValueError(f"Missing {tag_suffix} under {node.tag}")

def _read_point3d(node) -> np.ndarray:
    """Read a RESQML Point3d-like node with Coordinate1/2/3."""
    return np.array(
        [
            _text_float(node, "Coordinate1"),
            _text_float(node, "Coordinate2"),
            _text_float(node, "Coordinate3"),
        ],
        dtype=float,
    )

def _lattice_origin_and_offsets_from_grid_xml(grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly extract origin + 3 offset vectors from Geometry/Points/Parameters for IjkBlockGrid-style geometry.

    Fallback heuristic:
      - collect float-like leaf texts under Parameters in document order
      - if >= 12 floats, use first 12 as [origin(3), I(3), J(3), K(3)]
    """
    root = grid.root
    points_node = root.find(".//{*}Geometry/{*}Points")
    if points_node is None:
        raise RuntimeError("No Geometry/Points node found in grid XML.")

    container = None
    # Common cases: Points has one child, either Parameters or Point3dLatticeArray
    children = list(points_node)
    if children:
        if children[0].tag.endswith("Parameters") or children[0].tag.endswith("Point3dLatticeArray"):
            container = children[0]

    if container is None:
        container = points_node.find(".//{*}Parameters")
    if container is None:
        container = points_node.find(".//{*}Point3dLatticeArray")
    if container is None:
        raise RuntimeError(f"Unrecognized Geometry/Points encoding. First child tag: {children[0].tag if children else 'NONE'}")

    # --- Try the "structured" case first: Origin + Offset vectors as grouped triples ---
    def _try_read_grouped_triples() -> tuple[np.ndarray, np.ndarray] | None:
        origin_node = container.find(".//{*}Origin")
        if origin_node is not None:
            # Origin might wrap the coords one level down
            try:
                origin = _read_point3d(origin_node)
            except Exception:
                inner = next((c for c in list(origin_node) if any(cc.tag.endswith(("Coordinate1","Coordinate2","Coordinate3")) for cc in list(c))), None)
                if inner is None:
                    return None
                origin = _read_point3d(inner)

            offset_nodes = container.findall(".//{*}Offset")
            offsets = []
            for on in offset_nodes:
                try:
                    offsets.append(_read_point3d(on))
                except Exception:
                    inner = next(iter(list(on)), None)
                    if inner is not None:
                        try:
                            offsets.append(_read_point3d(inner))
                        except Exception:
                            pass

            # de-dup
            uniq = []
            for v in offsets:
                if not any(np.allclose(v, u) for u in uniq):
                    uniq.append(v)
            offsets = uniq

            if len(offsets) >= 3:
                return origin, np.stack(offsets[:3], axis=0)

        return None

    got = _try_read_grouped_triples()
    if got is not None:
        return got

    # --- Fallback: collect float leaf values in-order from Parameters subtree ---
    floats: list[float] = []
    for n in container.iter():
        # leaf node with text
        if len(list(n)) == 0 and n.text:
            t = n.text.strip()
            # try float parse
            try:
                v = float(t)
            except Exception:
                continue
            # ignore insane values sometimes used as flags? keep it simple for now
            floats.append(v)

    if len(floats) < 12:
        # dump XML to help you see what's there
        _dump_points_xml(grid)
        raise RuntimeError(
            f"Could not extract origin/offsets: found only {len(floats)} float leaf values under {container.tag}. "
            "See DEBUG dump above for the actual Geometry/Points encoding."
        )

    origin = np.array(floats[0:3], dtype=float)
    di = np.array(floats[3:6], dtype=float)
    dj = np.array(floats[6:9], dtype=float)
    dk = np.array(floats[9:12], dtype=float)
    offsets_ijk = np.stack([di, dj, dk], axis=0)

    return origin, offsets_ijk



def _points_from_lattice(extent_kji: tuple[int,int,int], origin: np.ndarray, offsets_ijk: np.ndarray) -> np.ndarray:
    """
    Build points on the (nk+1, nj+1, ni+1) lattice:
      P[k,j,i] = origin + i*di + j*dj + k*dk
    """
    nk, nj, ni = extent_kji
    di, dj, dk = offsets_ijk[0], offsets_ijk[1], offsets_ijk[2]

    ii = np.arange(ni + 1, dtype=float)[None, None, :, None]
    jj = np.arange(nj + 1, dtype=float)[None, :, None, None]
    kk = np.arange(nk + 1, dtype=float)[:, None, None, None]

    pts = origin[None, None, None, :] + ii * di[None, None, None, :] + jj * dj[None, None, None, :] + kk * dk[None, None, None, :]
    return pts  # (nk+1, nj+1, ni+1, 3)

def _cell_corners_from_regular_points(points: np.ndarray, k: int, j: int, i: int) -> np.ndarray:
    """points shape: (nk+1, nj+1, ni+1, 3) -> returns (2,2,2,3) for cell."""
    return points[k:k+2, j:j+2, i:i+2, :]



def _vtk_hex_corner_order(cell_cp_2x2x2: np.ndarray, flip: bool = False) -> np.ndarray:
    """
    Map cell corner points (2,2,2,3) -> VTK hex ordering.
    If flip=True, reverse orientation to make volumes positive (swap winding).
    """
    cp = cell_cp_2x2x2

    if not flip:
        # Standard VTK ordering
        return np.array(
            [
                cp[0, 0, 0],  # 0
                cp[0, 0, 1],  # 1
                cp[0, 1, 1],  # 2
                cp[0, 1, 0],  # 3
                cp[1, 0, 0],  # 4
                cp[1, 0, 1],  # 5
                cp[1, 1, 1],  # 6
                cp[1, 1, 0],  # 7
            ],
            dtype=float,
        )

    # Flipped orientation (swap 1<->3 and 5<->7)
    return np.array(
        [
            cp[0, 0, 0],  # 0
            cp[0, 1, 0],  # 3
            cp[0, 1, 1],  # 2
            cp[0, 0, 1],  # 1
            cp[1, 0, 0],  # 4
            cp[1, 1, 0],  # 7
            cp[1, 1, 1],  # 6
            cp[1, 0, 1],  # 5
        ],
        dtype=float,
    )


def build_vtk_unstructured_grid_from_resqpy_grid(grid: grr.Grid, epc_path: Path, flip_hex: bool = True) -> pv.UnstructuredGrid:
    # Try irregular corner-point geometry first (HDF5)
    cp = None
    regular_points = None

    try:
        # shape: (nk, nj, ni, 2, 2, 2, 3)
        cp = grid.corner_points(cache_cp_array=True)
    except AssertionError:
        # Likely Point3dLatticeArray / RegularGrid geometry
        cp = None

    if cp is None:
        regular_points = _grid_points_from_point3d_parametric_array(grid, epc_path=epc_path)
        nk, nj, ni = grid.extent_kji
        if regular_points.shape[:3] != (nk + 1, nj + 1, ni + 1):
            raise RuntimeError(f"Unexpected points shape from parametric array: {regular_points.shape}")


    nk, nj, ni = grid.extent_kji  # (nk, nj, ni)
    n_cells = nk * nj * ni

    # Prepare points: 8 per cell
    points = np.empty((n_cells * 8, 3), dtype=float)

    # Prepare VTK "cells" connectivity array:
    # For each cell: [8, p0, p1, p2, p3, p4, p5, p6, p7]
    cells = np.empty((n_cells, 9), dtype=np.int64)
    cells[:, 0] = 8

    # VTK cell types
    celltypes = np.full(n_cells, VTK_HEXAHEDRON, dtype=np.uint8)

    # Fill
    c = 0
    p = 0
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                if cp is not None:
                    cell_cp = cp[k, j, i]  # (2,2,2,3)
                else:
                    cell_cp = _cell_corners_from_regular_points(regular_points, k, j, i)

                corners = _vtk_hex_corner_order(cell_cp, flip_hex)  # (8,3)
                points[p : p + 8, :] = corners

                # connectivity references these new points
                cells[c, 1:] = np.arange(p, p + 8, dtype=np.int64)

                c += 1
                p += 8

    # Flatten connectivity for pyvista
    cells_flat = cells.reshape(-1)

    ugrid = pv.UnstructuredGrid(cells_flat, celltypes, points)
    ugrid.field_data["resqpy_grid_title"] = np.array([grid.title], dtype=object)
    ugrid.field_data["resqpy_grid_uuid"] = np.array([str(grid.uuid)], dtype=object)
    return ugrid


def _iter_property_uuids_for_grid(model: rq.Model, grid_uuid) -> List[str]:
    """
    Get property uuids related to this grid.
    Works even if grid.property_collection isn't preloaded.
    """
    uuids = []
    for obj_type in ("ContinuousProperty", "DiscreteProperty", "CategoricalProperty"):
        found = model.uuids(obj_type=obj_type, related_uuid=grid_uuid) or []
        uuids.extend([str(u) for u in found])
    # De-dup preserve order
    seen = set()
    out = []
    for u in uuids:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _safe_vtk_name(name: str) -> str:
    # VTK/ParaView behave better with simple names
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name).strip("_") or "property"


def attach_cell_properties(
    model: rq.Model,
    grid: grr.Grid,
    ugrid: pv.UnstructuredGrid,
    include_inactive: bool = True,
) -> None:
    """
    Attach all *cell-indexed* properties to ugrid.cell_data.

    - cell properties are expected to be shaped (nk, nj, ni) or (nk, nj, ni, ncomp)
    - we flatten in (k, j, i) loop order to match build_vtk_unstructured_grid_from_resqpy_grid()
    """
    # Prefer the grid's property_collection if present, else build one
    pc = getattr(grid, "property_collection", None)
    if pc is None:
        pc = rqp.PropertyCollection(grid)
        pc.populate_from_model(model)

    # We’ll iterate by property uuid to keep it robust.
    prop_uuids = _iter_property_uuids_for_grid(model, grid.uuid)
    if not prop_uuids:
        return

    nk, nj, ni = grid.extent_kji
    n_cells = nk * nj * ni

    inactive = getattr(grid, "inactive", None)  # boolean array (nk,nj,ni) if present
    if inactive is not None and inactive.shape != (nk, nj, ni):
        inactive = None

    for pu in prop_uuids:
        # Try to get metadata (title) from model
        title = model.title(uuid=pu) or pu
        vtk_name = _safe_vtk_name(title)

        try:
            a = pc.single_array_ref(uuid=pu)  # numpy array
        except Exception:
            # Some properties might not be loadable (eg time series / patch / unsupported indexable element)
            continue

        if a is None:
            continue

        # Filter to cell-shaped properties
        # Common: (nk,nj,ni) or (nk,nj,ni,ncomp)
        if a.shape[:3] != (nk, nj, ni):
            continue

        # Optionally mask inactive
        if (not include_inactive) and (inactive is not None):
            # We'll set inactive to NaN (continuous) or a sentinel (discrete)
            if np.issubdtype(a.dtype, np.floating):
                a = a.copy()
                a[inactive] = np.nan
            else:
                a = a.copy()
                a[inactive] = 0

        # Flatten in k-j-i order
        if a.ndim == 3:
            flat = a.reshape(n_cells, order="C")
            ugrid.cell_data[vtk_name] = flat
        elif a.ndim == 4:
            # vector/tensor per cell
            ncomp = a.shape[3]
            flat = a.reshape(n_cells, ncomp, order="C")
            ugrid.cell_data[vtk_name] = flat
        # else: ignore


def convert_epc_to_vtu(epc_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = rq.Model(epc_file=str(epc_path))

    grid_uuids = model.uuids(obj_type="IjkGridRepresentation") or []
    if not grid_uuids:
        raise RuntimeError("No IjkGridRepresentation grids found in the EPC.")

    outputs: List[Path] = []
    for idx, gu in enumerate(grid_uuids, start=1):
        # Load grid object, with properties if available
        grid = grr.any_grid(model, uuid=gu)
        # optional: populate properties if you want
        try:
            grid.find_property_collection()
        except Exception:
            pass
        if grid is None:
            continue

        ugrid = build_vtk_unstructured_grid_from_resqpy_grid(grid, epc_path, flip_hex=True)
        attach_cell_properties(model, grid, ugrid, include_inactive=True)

        title = getattr(grid, "title", None) or model.title(uuid=str(gu)) or f"grid_{idx}"
        safe_title = _safe_vtk_name(title)
        out_path = out_dir / f"{safe_title}.vtu"

        ugrid.save(str(out_path))
        outputs.append(out_path)

    return outputs


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: python resqml_v2_to_vtk.py input.epc out_dir")
        return 2

    epc_path = Path(argv[1]).expanduser().resolve()
    out_dir = Path(argv[2]).expanduser().resolve()

    if not epc_path.exists():
        print(f"ERROR: EPC not found: {epc_path}")
        return 1

    outputs = convert_epc_to_vtu(epc_path, out_dir)
    print("Wrote:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    outputs = convert_epc_to_vtu(Path("BASEGRID_GRID.epc").expanduser().resolve(), Path("out").expanduser().resolve())
    print("Wrote:")
    for p in outputs:
        print(f"  {p}")
