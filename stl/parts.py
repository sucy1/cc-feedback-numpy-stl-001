"""List the parts (solids) of an STL file as JSON.

Usage:
    python -m stl.parts assembly.stl

Writes one JSON array to stdout with an entry per ``solid`` block, in
the order the solids appear in the file::

    [{"name": "part_a", "faces": 1}, {"name": "part_b", "faces": 2}]

Single-solid (and binary) files produce a single entry.
"""

import argparse
import json
import sys
from collections.abc import Iterator

from .mesh import Mesh


def _decode_name(name: 'bytes | str') -> str:
    if isinstance(name, bytes):
        return name.decode('utf-8', errors='replace')
    return name


def iter_parts(filename: str) -> 'Iterator[dict[str, object]]':
    """Yield one ``{'name': ..., 'faces': ...}`` dict per solid."""
    for solid in Mesh.from_multi_file(filename):
        yield {'name': _decode_name(solid.name), 'faces': len(solid.data)}


def main(argv: 'list[str] | None' = None) -> None:
    """Console entry point for ``python -m stl.parts``."""
    parser = argparse.ArgumentParser(
        prog='stl.parts',
        description='List the parts (solids) of an STL file as JSON.',
    )
    parser.add_argument('filename', help='STL file to inspect')
    args = parser.parse_args(argv)

    json.dump(list(iter_parts(args.filename)), sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
