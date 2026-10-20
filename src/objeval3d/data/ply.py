"""A small reader and writer of PLY files, for what this package touches: the BOP models (ASCII, vertices and triangles) and the point
clouds it writes itself (binary, coordinates and colour).

It is written here, and not taken from the `plyfile` package, because `plyfile` is licensed GPL-3.0 and this package is Apache-2.0. The
values read are the ones `plyfile` gives: each token of an ASCII file is converted with the numpy scalar type of its property, exactly as
`plyfile` does (tests/test_ply.py compares the two where `plyfile` is installed).

Supported: the formats `ascii`, `binary_little_endian` and `binary_big_endian`; scalar properties of the eight PLY types; list
properties (the vertex indices of a face), of any length per face. Not supported: anything else in the standard (there is nothing else
in the files of the BOP datasets).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import numpy as np

_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_ORDER = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": "="}


class Prop(NamedTuple):
    name: str
    dtype: str  # numpy code of the value, e.g. 'f4'
    count_dtype: str | None = None  # numpy code of the length, for a list property


class Element(NamedTuple):
    name: str
    count: int
    props: tuple[Prop, ...]

    @property
    def is_list(self) -> bool:
        return any(p.count_dtype for p in self.props)


class Ply(NamedTuple):
    """A parsed file: `elements[name]` is a structured array, or, for an element with a list property, a dict
    {property name: list of 1-D arrays} next to the arrays of its scalar properties."""

    format: str
    elements: dict[str, np.ndarray | dict]


def _header(raw: bytes) -> tuple[str, list[Element], int]:
    end = raw.find(b"end_header")
    if end < 0 or not raw.startswith(b"ply"):
        raise ValueError("not a PLY file")
    nl = raw.find(b"\n", end)
    head = raw[:end].decode("ascii", "replace").splitlines()
    body = nl + 1
    fmt, elements = None, []
    for line in head[1:]:
        tok = line.split()
        if not tok or tok[0] in ("comment", "obj_info"):
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append([tok[1], int(tok[2]), []])
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1][2].append(Prop(tok[4], _TYPES[tok[3]], _TYPES[tok[2]]))
            else:
                elements[-1][2].append(Prop(tok[2], _TYPES[tok[1]]))
    if fmt not in _ORDER:
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, [Element(n, c, tuple(p)) for n, c, p in elements], body


def _scalar_dtype(el: Element, order: str) -> np.dtype:
    return np.dtype([(p.name, order + p.dtype) for p in el.props if not p.count_dtype])


def _read_binary(raw: bytes, pos: int, elements: list[Element], order: str) -> dict:
    out: dict = {}
    for el in elements:
        if not el.is_list:
            dt = _scalar_dtype(el, order)
            out[el.name] = np.frombuffer(raw, dtype=dt, count=el.count, offset=pos).copy()
            pos += dt.itemsize * el.count
            continue
        lists = {p.name: [] for p in el.props if p.count_dtype}
        scalars = {p.name: [] for p in el.props if not p.count_dtype}
        for _ in range(el.count):
            for p in el.props:
                if p.count_dtype:
                    cdt = np.dtype(order + p.count_dtype)
                    n = int(np.frombuffer(raw, dtype=cdt, count=1, offset=pos)[0])
                    pos += cdt.itemsize
                    vdt = np.dtype(order + p.dtype)
                    lists[p.name].append(np.frombuffer(raw, dtype=vdt, count=n, offset=pos).copy())
                    pos += vdt.itemsize * n
                else:
                    vdt = np.dtype(order + p.dtype)
                    scalars[p.name].append(np.frombuffer(raw, dtype=vdt, count=1, offset=pos)[0])
                    pos += vdt.itemsize
        out[el.name] = {**{k: np.asarray(v, dtype=order + _prop(el, k).dtype) for k, v in scalars.items()}, **lists}
    return out


def _prop(el: Element, name: str) -> Prop:
    return next(p for p in el.props if p.name == name)


def _read_ascii(text: str, elements: list[Element]) -> dict:
    lines = iter(text.splitlines())
    out: dict = {}
    for el in elements:
        scalars = {p.name: [] for p in el.props if not p.count_dtype}
        lists = {p.name: [] for p in el.props if p.count_dtype}
        for _ in range(el.count):
            tok = iter(next(lines).split())
            for p in el.props:
                if p.count_dtype:
                    n = int(np.dtype(p.count_dtype).type(next(tok)))
                    lists[p.name].append(np.array([np.dtype(p.dtype).type(next(tok)) for _ in range(n)], dtype=p.dtype))
                else:
                    scalars[p.name].append(np.dtype(p.dtype).type(next(tok)))  # the conversion plyfile makes
        if el.is_list:
            out[el.name] = {**{k: np.asarray(v, dtype=_prop(el, k).dtype) for k, v in scalars.items()}, **lists}
        else:
            arr = np.empty(el.count, dtype=_scalar_dtype(el, "="))
            for k, v in scalars.items():
                arr[k] = np.asarray(v, dtype=_prop(el, k).dtype)
            out[el.name] = arr
    return out


def read_ply(source: str | Path | bytes) -> Ply:
    """Parse a PLY file given by path or by its bytes."""
    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    fmt, elements, body = _header(raw)
    if fmt == "ascii":
        return Ply(fmt, _read_ascii(raw[body:].decode("ascii", "replace"), elements))
    return Ply(fmt, _read_binary(raw, body, elements, _ORDER[fmt]))


def read_mesh(source: str | Path | bytes) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [V,3] float64, faces [F,3] int64) of a PLY mesh; polygons are fan-triangulated."""
    ply = read_ply(source)
    v = ply.elements["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    faces = np.zeros((0, 3), dtype=np.int64)
    face = ply.elements.get("face")
    if face is not None:
        lists = face["vertex_indices"] if "vertex_indices" in face else face["vertex_index"]
        tri = []
        for poly in lists:
            poly = np.asarray(poly, dtype=np.int64)
            for i in range(1, len(poly) - 1):
                tri.append((poly[0], poly[i], poly[i + 1]))
        if tri:
            faces = np.asarray(tri, dtype=np.int64)
    return verts, faces


def write_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """A point cloud as a binary little-endian PLY: float32 coordinates and uint8 colour (grey when none is given)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)
    verts = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))
    buf.write(verts.tobytes())
    path.write_bytes(buf.getvalue())


def read_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(points [N,3] float64, colours [N,3] uint8 or None) of a PLY point cloud."""
    v = read_ply(path).elements["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    cols = None
    if "red" in v.dtype.names:
        cols = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return pts, cols
