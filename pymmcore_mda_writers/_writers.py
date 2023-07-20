from __future__ import annotations

__all__ = [
    "BaseWriter",
    "MiltiTiffWriter",
    "ZarrWriter",
]

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent, MDASequence
from pymmcore_plus.mda import PMDAEngine

try:
    import tifffile
except ModuleNotFoundError:
    tifffile = None
try:
    import zarr
except ModuleNotFoundError:
    zarr = None


class BaseWriter:
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
        self._enabled: bool = False

    @property
    def enabled(self) -> bool:
        """Whether this writer is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable this writer."""
        self._enabled = value

    def _onMDAStarted(self, sequence: MDASequence):
        ...

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent):
        ...  # pragma: no cover

    def _get_axis_labels(self, sequence: MDASequence) -> tuple[str, ...]:
        """Get the axis labels using only axes that are present in events."""
        # axis main sequence
        main_seq_axis = list(sequence.used_axes)
        if not sequence.stage_positions:
            return main_seq_axis
        # axes from sub sequences
        sub_seq_axis: list = []
        for p in sequence.stage_positions:
            if p.sequence is not None:
                sub_seq_axis.extend(
                    [ax for ax in p.sequence.used_axes if ax not in main_seq_axis]
                )
        return tuple(main_seq_axis + sub_seq_axis)

    # def _disconnect(self) -> None:
    #     "Disconnect this writer from processing any more events"
    #     self._core.mda.events.sequenceStarted.disconnect(self._onMDAStarted)
    #     self._core.mda.events.frameReady.disconnect(self._onMDAFrame)

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


class MiltiTiffWriter(BaseWriter):
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

    def _event_to_index(
        self, axis_order: Sequence[str], event: MDAEvent
    ) -> tuple[int, ...]:
        """Return the index of each axes for this event."""
        return tuple(event.index[a] for a in axis_order)

    def _onMDAStarted(self, sequence: MDASequence) -> None:
        if not self._enabled or self.folder_path is None:
            return

        # create unique folder path
        self._path = self.get_unique_folder(
            self.folder_path, self.file_name, create=True
        )

        # if the sequence has multiple positions, create a folder for each position
        if len(sequence.stage_positions) > 1:
            for p in range(len(sequence.stage_positions)):
                pos_path = self._path / f"pos_{p:03d}"
                pos_path.mkdir()
        else:
            pos_path = self._path / "pos_000"
            pos_path.mkdir()

        # save the sequence info as json
        with open(self._path / "useq-sequence.json", "w") as f:
            f.write(sequence.json())

        # get the axis order
        self._axis_order = self._get_axis_labels(sequence)

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent) -> None:
        """Save the image as a tiff file at every core frameReady signal."""
        if not self._enabled or self.folder_path is None:
            return

        index = self._event_to_index(self._axis_order, event)
        name = (
            "_".join(
                [
                    self._axis_order[i] + f"{index[i]}".zfill(3)
                    for i in range(len(index))
                ]
            )
            + ".tiff"
        )

        pos_path = self._path / f"pos_{event.index.get('p', 0):03d}"
        tifffile.imwrite(pos_path / name, img)


class ZarrWriter(BaseWriter):
    """Write the MDASequence data to a zarr store.

    Parameters
    ----------
    folder_path : Path or str, optional
        The path to the zarr store. If None, no data will be saved.
    file_name : str, optional
        The name of the zarr store. If not given, the name of the folder_path.
    dtype : np.dtype, optional
        The dtype to use. If not given, the dtype of the first frame will be used.
    core : CMMCorePlus, optional
        The core to use. If not given, the default core will be used.
    """

    def __init__(
        self,
        folder_path: Path | str | None = None,
        file_name: str = "",
        dtype: npt.DTypeLike | None = None,
        core: CMMCorePlus = None,
    ) -> None:
        if zarr is None:
            raise ValueError(
                "This writer requires zarr to be installed. Try: `pip install zarr`"
            )
        super().__init__(core)

        self.folder_path = folder_path
        self.file_name = file_name
        self._dtype = dtype
        self._zarr: zarr.Array | None = None
        self._axis_labels: tuple[str, ...] | None = None

    def _onMDAStarted(self, sequence: MDASequence) -> None:
        if not self._enabled or self.folder_path is None:
            return

        self._axis_labels = self._get_axis_labels(sequence)
        _shape = self._get_array_shape(sequence)
        _path = self.get_unique_folder(
            self.folder_path, self.file_name or "exp", create=True, suffix=".zarr"
        )
        _dtype = self._dtype or f"uint{self._core.getImageBitDepth()}"
        self._zarr = self._create_zarr(sequence, _path, _shape, _dtype)

    def _get_array_shape(self, sequence: MDASequence) -> tuple[int, ...]:
        """Retun the shape for the zarr array.

        Update the array shape to also fit the sub sequence
        """
        # main sequence array shape
        array_shape = [(sequence.sizes[k] or 1) for k in self._axis_labels]
        # update array shape with sub sequence info
        for p in sequence.stage_positions:
            if not p.sequence:
                continue
            for ax in p.sequence.used_axes:
                axes_shape = p.sequence.sizes[ax]
                index = self._axis_labels.index(ax)
                array_shape[index] = max(array_shape[index], axes_shape)
        yx_shape = [self._core.getImageHeight(), self._core.getImageWidth()]
        return array_shape + yx_shape

    def _create_zarr(
        self, sequence: MDASequence, path: Path, shape: list[int], dtype: npt.DTypeLike
    ) -> zarr.Array:
        """Create the zarr array."""
        chunk_size = [1] * len(shape[:-2]) + shape[-2:]
        z = zarr.open_array(
            f"{path}", shape=shape, dtype=dtype, mode="w", chunks=chunk_size
        )
        z.attrs["name"] = path.stem
        z._attrs["uid"] = str(sequence.uid)
        z.attrs["axis_labels"] = list(self._axis_labels) + ["y", "x"]
        z.attrs["sequence"] = sequence.json()
        # TODO: add OME metadata with ome-types
        return z

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent) -> None:
        if not self._enabled or self.folder_path is None:
            return
        self._populate_zarr(img, event)

    def _populate_zarr(self, image: np.ndarray, event: MDAEvent) -> None:
        """Populate the zarr array with the image data."""
        axis_order = self._get_axis_labels(event.sequence)

        # the index of this event in the full zarr array
        im_idx: tuple[int, ...] = ()
        for k in axis_order:
            # using try/except in the case the axis is not in event.index
            # e.g. if in the sequence we have both a position sequence grid and
            # a single position
            try:
                im_idx += (event.index[k],)
            except KeyError:
                im_idx += (0,)

        self._zarr[im_idx] = image
