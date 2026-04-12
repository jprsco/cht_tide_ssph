import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cht_tide.read_bca import (
    FlowBoundaryPoint,
    IniStruct,
    Keyword,
    Point,
    Section,
    SfincsBoundary,
)


@pytest.fixture
def bnd_file(tmp_path):
    file_path = tmp_path / "test.bnd"
    points = [
        (-80.00000, 26.00000, "0001"),
        (-81.00000, 27.00000, "0002"),
    ]

    with open(file_path, "w") as f:
        for x, y, name in points:
            f.write(f"{x}\t{y}\n")
    return file_path, points


def test_read_flow_boundary_points(bnd_file):
    bnd_file, expected_points = bnd_file
    boundary = SfincsBoundary().read_flow_boundary_points(bnd_file)

    assert len(boundary.flow_boundary_points) == len(expected_points)
    for real, expected in zip(boundary.flow_boundary_points, expected_points):
        assert real.geometry.x == expected[0]
        assert real.geometry.y == expected[1]
        assert real.name == expected[2]


@pytest.fixture
def bca_file():
    return Path(__file__).parent / "sfincs.bca"


def test_ini_struct_read(bca_file):
    ini_struct = IniStruct(bca_file)
    assert len(ini_struct.section) > 0
    assert ini_struct.section[0].name == "forcing"


def test_section_get_value():
    section = Section()
    section.keyword.append(Keyword(name="TestKey", value="TestValue"))
    assert section.get_value("TestKey") == "TestValue"


def test_point_creation():
    point = Point(x=1.0, y=2.0, name="TestPoint", crs="EPSG:4326")
    assert point.x == 1.0
    assert point.y == 2.0
    assert point.name == "TestPoint"
    assert point.crs == "EPSG:4326"


def test_flow_boundary_point_creation():
    flow_point = FlowBoundaryPoint(
        x=3.0, y=4.0, name="FlowPoint", astro={"M2": [0.39452, 14.38]}
    )
    assert flow_point.geometry.x == 3.0
    assert flow_point.geometry.y == 4.0
    assert flow_point.name == "FlowPoint"
    assert flow_point.astro == {"M2": [0.39452, 14.38]}


@pytest.fixture
def temp_ini_file():
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    temp_file.write("""
    [Section1]
    key1 = value1
    key2 = value2 # Comment
    FOO 2.0 3.0
    BAR 5.0 6.0

    [Section2]
    keyA = valueA
    keyB = valueB
    """)
    temp_file.close()
    yield temp_file.name
    os.remove(temp_file.name)


def test_read_method(temp_ini_file):
    ini = IniStruct(temp_ini_file)

    # Check if sections are correctly parsed
    assert len(ini.section) == 2
    assert ini.section[0].name == "Section1"
    assert ini.section[1].name == "Section2"

    # Check if keywords are correctly parsed
    section1_keywords = {kw.name: kw.value for kw in ini.section[0].keyword}
    assert section1_keywords["key1"] == "value1"
    assert section1_keywords["key2"] == "value2"

    section2_keywords = {kw.name: kw.value for kw in ini.section[1].keyword}
    assert section2_keywords["keyA"] == "valueA"
    assert section2_keywords["keyB"] == "valueB"

    # Check if data is correctly parsed
    expected_data = pd.DataFrame(
        [[2.0, 3.0], [5.0, 6.0]], index=["FOO", "BAR"], columns=[1, 2]
    )
    expected_data.index.name = 0

    pd.testing.assert_frame_equal(
        ini.section[0].data.astype(float), expected_data, check_index_type=False
    )
