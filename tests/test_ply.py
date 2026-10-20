"""The package reads and writes PLY itself (the `plyfile` package is GPL-3.0). Where `plyfile` is installed the two are compared."""

import numpy as np
import pytest

from objeval3d.data.ply import read_cloud, read_mesh, read_ply, write_cloud

ASCII_MESH = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile obj_000001.png
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float texture_u
property float texture_v
element face 3
property list uchar int vertex_indices
end_header
-45.0355 -22.5 -66.936 -0.8845149 -0.4488803 0.1270429 0.296300 0.219800
10.1 0.3333333333 7.77 0 0 1 0.5 0.5
1e-3 2.5E2 -0.1 0 1 0 0 1
3.14159274 2.71828175 1.41421354 1 0 0 1 0
3 0 1 2
3 1 2 3
4 0 1 3 2
"""


@pytest.fixture
def ascii_mesh(tmp_path):
    p = tmp_path / "obj.ply"
    p.write_text(ASCII_MESH)
    return p


def test_ascii_mesh_vertices_faces_and_fan_triangulation(ascii_mesh):
    verts, faces = read_mesh(ascii_mesh)
    assert verts.shape == (4, 3) and verts.dtype == np.float64
    assert faces.dtype == np.int64
    # two triangles and one quad fanned into two
    assert faces.tolist() == [[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 3, 2]]
    # float32 values widened to float64, as the files declare them
    assert verts[0, 0] == float(np.float32("-45.0355"))
    assert verts[1, 1] == float(np.float32("0.3333333333"))


def test_the_bytes_of_a_file_are_read_like_the_file(ascii_mesh):
    a, b = read_mesh(ascii_mesh), read_mesh(ascii_mesh.read_bytes())
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_cloud_roundtrip_is_exact_in_float32(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * 100
    cols = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
    write_cloud(tmp_path / "c.ply", pts, cols)
    p, c = read_cloud(tmp_path / "c.ply")
    assert np.array_equal(p, pts.astype(np.float32).astype(np.float64))
    assert np.array_equal(c, cols)


def test_cloud_without_colours_is_grey(tmp_path):
    write_cloud(tmp_path / "c.ply", np.zeros((3, 3)))
    _, c = read_cloud(tmp_path / "c.ply")
    assert (c == 200).all()


def test_an_empty_cloud_is_a_valid_file(tmp_path):
    write_cloud(tmp_path / "e.ply", np.zeros((0, 3)))
    p, c = read_cloud(tmp_path / "e.ply")
    assert p.shape == (0, 3) and c.shape == (0, 3)


def test_binary_little_and_big_endian_with_a_face_list():
    def make(order_name, order):
        head = (
            f"ply\nformat {order_name} 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        ).encode()
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=order + "f4").tobytes()
        f = np.array([3], dtype="u1").tobytes() + np.array([0, 1, 2], dtype=order + "i4").tobytes()
        return head + v + f

    for name, order in (("binary_little_endian", "<"), ("binary_big_endian", ">")):
        verts, faces = read_mesh(make(name, order))
        assert verts.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert faces.tolist() == [[0, 1, 2]]


def test_not_a_ply_file_is_refused():
    with pytest.raises(ValueError):
        read_ply(b"solid stl\n")


def test_same_values_as_plyfile(ascii_mesh, tmp_path):
    plyfile = pytest.importorskip("plyfile")
    ref = plyfile.PlyData.read(str(ascii_mesh))
    mine = read_ply(ascii_mesh)
    for name in ("x", "y", "z", "nx", "ny", "nz", "texture_u", "texture_v"):
        assert np.array_equal(ref["vertex"].data[name], mine.elements["vertex"][name])
        assert ref["vertex"].data[name].dtype == mine.elements["vertex"][name].dtype
    for a, b in zip(ref["face"].data["vertex_indices"], mine.elements["face"]["vertex_indices"]):
        assert np.array_equal(a, b)
    # a cloud written here is read by plyfile, with the same numbers
    pts = np.random.default_rng(1).normal(size=(50, 3))
    cols = np.random.default_rng(2).integers(0, 256, size=(50, 3)).astype(np.uint8)
    write_cloud(tmp_path / "c.ply", pts, cols)
    v = plyfile.PlyData.read(str(tmp_path / "c.ply"))["vertex"].data
    assert np.array_equal(v["x"], pts[:, 0].astype(np.float32))
    assert np.array_equal(v["red"], cols[:, 0])
