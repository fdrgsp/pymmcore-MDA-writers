import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDASequence

from pymmcore_mda_writers import ZarrWriter, MiltiTiffWriter

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def core() -> CMMCorePlus:
    mmc = CMMCorePlus.instance()
    if len(mmc.getLoadedDevices()) < 2:
        mmc.loadSystemConfiguration(str(Path(__file__).parent / "test-config.cfg"))
    return mmc


def test_engine_registration(core: CMMCorePlus, tmp_path: Path, qtbot: "QtBot"):
    mda = MDASequence(
        metadata={"blah": "blah blah blah"},
        stage_positions=[
            (1, 1, 1),
            {
                "x": 2,
                "y": 2,
                "z": 2,
                "sequence": {
                    "grid_plan": {"rows": 2, "columns": 1},
                    "z_plan": {"range": 2, "step": 1},
                },
            },
        ],
        z_plan={"range": 3, "step": 1},
        channels=[{"config": "DAPI", "exposure": 1}],
    )

    writer = ZarrWriter(  # noqa
        tmp_path, "zarr_data", dtype=np.uint16, core=core
    )
    new_engine = MDAEngine(core)
    with qtbot.waitSignal(core.events.mdaEngineRegistered):
        core.register_mda_engine(new_engine)
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda)
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda)
    run1 = zarr.open(tmp_path / "zarr_data_000.zarr")
    arr1 = np.asarray(run1)
    run2 = zarr.open(tmp_path / "zarr_data_001.zarr")
    arr2 = np.asarray(run2)
    assert arr1.shape == (2, 1, 4, 2, 512, 512)  # p c z g x y
    assert arr2.shape == (2, 1, 4, 2, 512, 512)  # p c z g x y

    assert sum(sum(arr1[0, 0, 3, 0])) > 0  # z_plan
    assert sum(sum(arr1[1, 0, 3, 0])) == 0  # z_plan

    assert sum(sum(arr1[0, 0, 0, 1])) == 0  # grid_plan
    assert sum(sum(arr1[1, 0, 0, 0])) > 0  # grid_plan

    attrs = run2.attrs.asdict()
    assert attrs["name"] == "zarr_data_001"
    assert attrs["axis_labels"] == ["p", "c", "z", "g", "y", "x"]
    assert attrs["uid"] == str(mda.uid)
    assert mda == MDASequence(**json.loads(attrs["sequence"]))


def test_tiff_writer(core: CMMCorePlus, tmp_path: Path, qtbot: "QtBot"):
    mda = MDASequence(
        metadata={"blah": "blah blah blah"},
        time_plan={"interval": 0.1, "loops": 2},
        stage_positions=[(1, 1, 1)],
        z_plan={"range": 3, "step": 1},
        channels=[{"config": "DAPI", "exposure": 1}],
    )
    writer = MiltiTiffWriter(str(tmp_path / "mda_data"), core=core)  # noqa

    # run twice to check that we aren't overwriting files
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda).join()
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda).join()

    # check that the correct folders/files were generated
    data_folders = set(tmp_path.glob("mda_data/mda_data*"))
    assert {
        tmp_path / "mda_data" / "mda_data_000",
        tmp_path / "mda_data" / "mda_data_001",
    }.issubset(set(data_folders))
    expected = [
        Path("t000_p000_c000_z000.tiff"),
        Path("t001_p000_c000_z000.tiff"),
        Path("t001_p000_c000_z002.tiff"),
        Path("t001_p000_c000_z001.tiff"),
        Path("t000_p000_c000_z001.tiff"),
        Path("t001_p000_c000_z003.tiff"),
        Path("t000_p000_c000_z002.tiff"),
        Path("t000_p000_c000_z003.tiff"),
    ]
    actual_1 = list((tmp_path / "mda_data" / "mda_data_000").glob("*"))
    actual_2 = list((tmp_path / "mda_data" / "mda_data_001").glob("*"))
    for e in expected:
        assert tmp_path / "mda_data" / "mda_data_000" / e in actual_1
        assert tmp_path / "mda_data" / "mda_data_001" / e in actual_2
    with open(tmp_path / "mda_data" / "mda_data_000" / "useq-sequence.json") as f:
        seq = MDASequence(**json.load(f))
    assert seq == mda


def test_missing_deps():
    with patch("pymmcore_mda_writers._writers.tifffile", None):
        with pytest.raises(ValueError) as e:
            MiltiTiffWriter("blarg")
        assert "requires tifffile to be installed" in str(e)
    with patch("pymmcore_mda_writers._writers.zarr", None):
        with pytest.raises(ValueError) as e:
            ZarrWriter("blarg", np.uint16)
        assert "requires zarr to be installed" in str(e)


def test_disconnect(core: CMMCorePlus, tmp_path: Path, qtbot: "QtBot"):
    mda = MDASequence(
        stage_positions=[(1, 1, 1)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=[{"config": "DAPI", "exposure": 1}],
    )

    writer = MiltiTiffWriter(tmp_path / "mda_data", core)
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda).join()
    writer.disconnect()
    with qtbot.waitSignal(core.mda.events.sequenceFinished):
        core.run_mda(mda)
    data_folders = set(tmp_path.glob("mda_data*"))
    assert len(data_folders) == 1
    # assert writer._onMDAFrame.call_count == 3




