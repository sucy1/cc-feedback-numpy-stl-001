"""Regression tests for robust STL importing.

Covers assembly-style files (multiple solids per file), exporter
quirks (BOM, leading comments/blank lines, indented or missing
``facet normal`` lines) and readable errors for corrupt input.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from stl import mesh, stl

REPO_ROOT = pathlib.Path(__file__).parent.parent

FACET = """facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
"""


def _solid(name, count):
    return 'solid %s\n' % name + FACET * count + 'endsolid %s\n' % name


def _write(tmp_path, content, name='test.stl'):
    path = tmp_path / name
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)
    return str(path)


def test_from_file_merges_multiple_solids(tmp_path, speedups):
    path = _write(
        tmp_path, _solid('part_a', 1) + _solid('part_b', 2) + _solid('part_c', 1)
    )
    m = mesh.Mesh.from_file(path, speedups=speedups)
    assert len(m.data) == 4


def test_from_multi_file_still_splits_solids(tmp_path, speedups):
    path = _write(tmp_path, _solid('part_a', 1) + _solid('part_b', 2))
    solids = list(mesh.Mesh.from_multi_file(path, speedups=speedups))
    assert [s.name for s in solids] == [b'part_a', b'part_b']
    assert [len(s.data) for s in solids] == [1, 2]


def test_ascii_with_utf8_bom(tmp_path, speedups):
    path = _write(tmp_path, b'\xef\xbb\xbf' + _solid('p', 2).encode())
    m = mesh.Mesh.from_file(path, speedups=speedups)
    assert len(m.data) == 2


def test_ascii_with_leading_comments(tmp_path, speedups):
    content = '# exported by some-exporter\n# revision 3\n' + _solid('p', 2)
    m = mesh.Mesh.from_file(_write(tmp_path, content), speedups=speedups)
    assert len(m.data) == 2


def test_ascii_with_leading_blank_lines(tmp_path, speedups):
    m = mesh.Mesh.from_file(
        _write(tmp_path, '\n\n  \n' + _solid('p', 1)), speedups=speedups
    )
    assert len(m.data) == 1


def test_ascii_with_bom_comments_and_blank_lines(tmp_path, speedups):
    content = b'\xef\xbb\xbf# exported by foo\n\n# rev 2\n' + _solid('p', 2).encode()
    m = mesh.Mesh.from_file(_write(tmp_path, content), speedups=speedups)
    assert len(m.data) == 2


def test_ascii_with_indented_facet_and_no_solid_line(tmp_path, speedups):
    content = '  ' + FACET.replace('\n', '\n  ').rstrip() + '\n'
    m = mesh.Mesh.from_file(_write(tmp_path, content), speedups=speedups)
    assert len(m.data) == 1


def test_ascii_missing_facet_normal(tmp_path, speedups):
    content = _solid('p', 1).replace('facet normal 0 0 1\n', '')
    m = mesh.Mesh.from_file(
        _write(tmp_path, content), calculate_normals=False, speedups=speedups
    )
    assert len(m.data) == 1
    # Right-hand rule: counter-clockwise vertices give an outward
    # normal, consistent with Mesh.update_normals()
    assert np.allclose(m.normals[0], [0, 0, 1])


def test_merged_mesh_roundtrip_ascii_and_binary(tmp_path, speedups):
    path = _write(tmp_path, _solid('part_a', 1) + _solid('part_b', 2))
    m = mesh.Mesh.from_file(path, speedups=speedups)
    assert len(m.data) == 3

    ascii_path = str(tmp_path / 'roundtrip_ascii.stl')
    binary_path = str(tmp_path / 'roundtrip_binary.stl')
    m.save(ascii_path, mode=stl.Mode.ASCII)
    m.save(binary_path, mode=stl.Mode.BINARY)

    assert len(mesh.Mesh.from_file(ascii_path, speedups=speedups).data) == 3
    assert len(mesh.Mesh.from_file(binary_path, speedups=speedups).data) == 3


def test_truncated_binary_raises_readable_error(tmp_path, speedups):
    m = mesh.Mesh.from_file(
        _write(tmp_path, _solid('p', 2), name='source.stl'), speedups=speedups
    )
    binary_path = _write(tmp_path, b'', name='truncated.stl')
    m.save(binary_path, mode=stl.Mode.BINARY)

    raw = pathlib.Path(binary_path).read_bytes()
    # Keep one and a half triangle records of the two declared
    truncated = _write(tmp_path, raw[: 84 + 75], name='truncated.stl')

    with pytest.raises(stl.StlFormatError) as excinfo:
        mesh.Mesh.from_file(truncated, speedups=speedups)
    assert 'File too large' not in str(excinfo.value)
    assert 'runcated' in str(excinfo.value)


def test_non_stl_file_raises_readable_error(tmp_path, speedups):
    path = _write(tmp_path, b'x' * 200)
    with pytest.raises(stl.StlFormatError) as excinfo:
        mesh.Mesh.from_file(path, speedups=speedups)
    assert 'File too large' not in str(excinfo.value)
    # StlFormatError stays catchable as AssertionError/ValueError for
    # backwards compatibility
    assert isinstance(excinfo.value, (AssertionError, ValueError))


def _run_parts(path):
    result = subprocess.run(
        [sys.executable, '-m', 'stl.parts', str(path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_parts_cli_multiple_solids(tmp_path, speedups):
    path = _write(tmp_path, _solid('part_a', 1) + _solid('part_b', 2))
    assert _run_parts(path) == [
        {'name': 'part_a', 'faces': 1},
        {'name': 'part_b', 'faces': 2},
    ]


def test_parts_cli_single_solid(tmp_path, speedups):
    path = _write(tmp_path, _solid('only_part', 3))
    assert _run_parts(path) == [{'name': 'only_part', 'faces': 3}]


def test_parts_cli_binary_file(tmp_path, speedups):
    m = mesh.Mesh.from_file(
        _write(tmp_path, _solid('p', 2), name='source.stl'), speedups=speedups
    )
    binary_path = str(tmp_path / 'binary.stl')
    m.save(binary_path, mode=stl.Mode.BINARY)

    (part,) = _run_parts(binary_path)
    assert part['faces'] == 2
