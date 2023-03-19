from __future__ import annotations

__all__ = [
    "BaseMDASequenceWriter",
    "MiltiTiffMDASequenceWriter",
    "ZarrMDASequenceWriter",
]

import contextlib
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import PMDAEngine
from useq import MDAEvent, MDASequence

try:
    import tifffile
except ModuleNotFoundError:
    tifffile = None
try:
    import zarr
except ModuleNotFoundError:
    zarr = None


class BaseMDASequenceWriter:
    """Base class for MDASequence writers.

    Parameters
    ----------
    core : CMMCorePlus, optional
        The core to use. If not specified, the active core will be used
        (or a new one will be created)
    """

    def __init__(self, core: CMMCorePlus = None) -> None:
        self._core = core or CMMCorePlus.instance()
        self._core.mda.events.sequenceStarted.connect(self._onMDAStarted)
        self._core.mda.events.frameReady.connect(self._onMDAFrame)
        # TODO add paused and finished events

    def _onMDAStarted(self, sequence: MDASequence):
        ...

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent):
        ...  # pragma: no cover

    def _disconnect(self, engine: PMDAEngine):
        engine.events.sequenceStarted.disconnect(self._onMDAStarted)
        engine.events.frameReady.disconnect(self._onMDAFrame)

    def disconnect(self):
        "Disconnect this writer from processing any more events"
        self._core.mda.events.sequenceStarted.disconnect(self._onMDAStarted)
        self._core.mda.events.frameReady.disconnect(self._onMDAFrame)

    @staticmethod
    def get_unique_folder(
        folder_path: str | Path,
        folder_name: str = "",
        suffix: str | Path = None,
        create: bool = False,
    ) -> Path:
        """
        Get a unique folder name in the form: '{folder_path/folder_name or folder_path.stem}_{iii}'.

        Parameters
        ----------
        folder_path : str or Path
            The folder path in which to put data.
        folder_name : str
            The folder name to use. If not given, the name of the folder_path.
        suffix : str or Path
            If given, to be used as the path suffix. e.g. `.zarr`
        create : bool, default False
            Whether to create the folder.
        '"""
        folder = Path(folder_path).resolve()
        stem = folder_name or str(folder.stem)

        def new_path(i):
            path = folder / f"{stem}_{i:03d}"
            return path.with_suffix(suffix) if suffix else path

        i = 0
        path = new_path(i)
        while path.exists():
            i += 1
            path = new_path(i)
        if create:
            path.mkdir(parents=True)
        return path


class MiltiTiffMDASequenceWriter(BaseMDASequenceWriter):
    def __init__(
        self,
        folder_path: str | Path | None = None,
        file_name: str = "",
        core: CMMCorePlus = None,
    ) -> None:
        """Write each frame from a MDASequence as a separate tiff file.

        e.g. if the sequence is `tpcz`, then the files will be named:
        `t000_p000_c000_z000.tif`, `t000_p000_c000_z001.tif`, etc.

        Parameters
        ----------
        folder_path : str or Path, optional
            The folder path in which to put data. If None, no data will be saved.
        file_name : str, optional
            The folder name to use. If not given, the name of the folder_path.
        core : CMMCorePlus, optional
            The core to use. If not given, the default core will be used.
        """
        if tifffile is None:
            raise ValueError(
                "This writer requires tifffile to be installed. "
                "Try: `pip install tifffile`"
            )
        super().__init__(core)
        self.folder_path = folder_path
        self.file_name = file_name
        self._path: Path | None = None

    def sequence_axis_order(self, sequence: MDASequence) -> tuple[str]:
        """Get the axis order using only axes that are present in events."""
        # hacky way to drop unncessary parts of the axis order
        # e.g. drop the `p` in `tpcz` if there is only one position
        # TODO: add a better implementation upstream in useq
        event = next(sequence.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in sequence.axis_order if a in event_axes)

    def event_to_index(
        self, axis_order: Sequence[str], event: MDAEvent
    ) -> tuple[int, ...]:
        return tuple(event.index[a] for a in axis_order)

    def _onMDAStarted(self, sequence: MDASequence) -> None:
        if self.folder_path is None:
            return
        self._path = self.get_unique_folder(
            self.folder_path, self.file_name, create=True
        )
        self._axis_order = self.sequence_axis_order(sequence)
        with open(self._path / "useq-sequence.json", "w") as f:
            f.write(sequence.json())

    def _onMDAFrame(self, img: np.ndarray, event: MDASequence) -> None:
        if self.folder_path is None:
            return
        index = self.event_to_index(self._axis_order, event)
        name = (
            "_".join(
                [
                    self._axis_order[i] + f"{index[i]}".zfill(3)
                    for i in range(len(index))
                ]
            )
            + ".tiff"
        )
        tifffile.imwrite(self._path / name, img)


class ZarrMDASequenceWriter(BaseMDASequenceWriter):
    """Write the MDASequence data to a zarr store.

    Parameters
    ----------
    folder_path : Path or str, optional
        The path to the zarr store. If None, no data will be saved.
    file_name : str, optional
        The name of the zarr store. If not given, the name of the folder_path.
    core : CMMCorePlus, optional
        The core to use. If not given, the default core will be used.
    """

    def __init__(
        self,
        folder_path: Path | str | None = None,
        file_name: str = "",
        core: CMMCorePlus = None,
    ) -> None:
        super().__init__(core)

        self.folder_path = folder_path
        self.file_name = file_name
        self._zarr: zarr.Array | None = None

    def _onMDAStarted(self, sequence: MDASequence):
        if self.folder_path is None:
            return
        self._zarr = None
        _shape, _axis_labels = self._determine_zarr_shape_and_axis_labels(sequence)
        _path = self.get_unique_folder(
            self.folder_path, self.file_name or "exp", create=True, suffix=".zarr"
        )
        dtype = f"uint{self._core.getImageBitDepth()}"
        self._create_zarr(sequence, _path, _shape, dtype, _axis_labels)

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent):
        if self.folder_path is None:
            return
        self._populate_zarr(img, event)

    def _get_axis_labels(self, sequence: MDASequence) -> tuple[list[str], bool]:
        # sourcery skip: use-next
        main_seq_axis = list(sequence.used_axes)

        if not sequence.stage_positions:
            return main_seq_axis, False

        sub_seq_axis: list | None = None
        for p in sequence.stage_positions:
            if p.sequence:
                sub_seq_axis = list(p.sequence.used_axes)
                break

        if sub_seq_axis:
            for i in sub_seq_axis:
                if i not in main_seq_axis:
                    main_seq_axis.append(i)

        return main_seq_axis, bool(sub_seq_axis)

    def _determine_zarr_shape_and_axis_labels(
        self, sequence: MDASequence
    ) -> tuple[Path, str, tuple, str, list[str]]:
        """Determine the zarr info to then create the zarr array from the sequence."""
        axis_labels, pos_sequence = self._get_axis_labels(sequence)
        array_shape = [sequence.sizes[k] or 1 for k in axis_labels]

        if pos_sequence:
            for p in sequence.stage_positions:
                if not p.sequence:
                    continue
                array_shape = self._update_array_shape(
                    p.sequence, array_shape, axis_labels
                )

        yx_shape = [self._core.getImageHeight(), self._core.getImageWidth()]
        _shape = array_shape + yx_shape

        return _shape, axis_labels

    def _update_array_shape(
        self, sequence: MDASequence, array_shape: list[int], axis_labels: list[str]
    ) -> list[int] | None:
        """Update the array shape to fit the sub sequence."""
        for ax in "cgzt":
            with contextlib.suppress(ValueError):
                axes_shape = sequence.sizes[ax]  # type: ignore
                index = axis_labels.index(ax)
                array_shape[index] = max(array_shape[index], axes_shape)
        return array_shape

    def _create_zarr(
        self,
        sequence: MDASequence,
        path: Path,
        shape: list[int],
        dtype: npt.DTypeLike,
        axis_labels: list[str],
    ):
        """Create the zarr array."""
        self._zarr = zarr.open(f"{path}", shape=shape, dtype=dtype, mode="w")
        self._zarr.attrs["name"] = path.stem
        self._zarr._attrs["uid"] = str(sequence.uid)
        self._zarr.attrs["axis_labels"] = axis_labels + ["y", "x"]
        self._zarr.attrs["sequence"] = sequence.json()

    def _populate_zarr(self, image: np.ndarray, event: MDAEvent) -> None:
        """Populate the zarr array with the image data."""
        axis_order, _ = self._get_axis_labels(event.sequence)

        # the index of this event in the full zarr array
        im_idx: tuple[int, ...] = ()
        for k in axis_order:
            try:
                im_idx += (event.index[k],)
            # if axis not in event.index
            # e.g. if we have bot a position sequence grid and a single position
            except KeyError:
                im_idx += (0,)

        self._zarr[im_idx] = image
