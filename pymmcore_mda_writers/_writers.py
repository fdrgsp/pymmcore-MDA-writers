from __future__ import annotations

__all__ = [
    "BaseWriter",
    "SimpleMultiFileTiffWriter",
    "ZarrWriter",
    "ZarrNapariMicromanagerWriter",
]
from pathlib import Path
from typing import Sequence, Tuple, Union, Any, cast

import numpy as np
import numpy.typing as npt
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import PMDAEngine
from useq import MDAEvent, MDASequence
from useq import NoGrid

try:
    import tifffile
except ModuleNotFoundError:
    tifffile = None
try:
    import zarr
except ModuleNotFoundError:
    zarr = None


class BaseWriter:
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
        folder_path: Union[str, Path],
        folder_name: str = "",
        suffix: Union[str, Path] = None,
        create: bool = False,
    ) -> Path:
        """
        Get a unique foldername of the form '{folder_path/folder_name}_{i}

        Parameters
        ----------
        folder_path : str or Path
            The folder path in which to put data
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

    @staticmethod
    def sequence_axis_order(seq: MDASequence) -> Tuple[str]:
        """Get the axis order using only axes that are present in events."""
        # hacky way to drop unncessary parts of the axis order
        # e.g. drop the `p` in `tpcz` if there is only one position
        # TODO: add a better implementation upstream in useq
        event = next(seq.iter_events())
        event_axes = list(event.index.keys())
        return tuple(a for a in seq.axis_order if a in event_axes)

    @staticmethod
    def event_to_index(axis_order: Sequence[str], event: MDAEvent) -> Tuple[int, ...]:
        return tuple(event.index[a] for a in axis_order)


class SimpleMultiFileTiffWriter(BaseWriter):
    def __init__(
        self,
        data_folder_path: Union[str, Path] = "",
        data_folder_name: str = "",
        core: CMMCorePlus = None,
    ) -> None:
        if tifffile is None:
            raise ValueError(
                "This writer requires tifffile to be installed. "
                "Try: `pip install tifffile`"
            )
        super().__init__(core)
        self._data_folder_path = data_folder_path
        self._data_folder_name = data_folder_name

    def _onMDAStarted(self, sequence: MDASequence) -> None:
        if not self._data_folder_path:
            return
        self._path = self.get_unique_folder(
            self._data_folder_path, self._data_folder_name, create=True
        )
        self._axis_order = self.sequence_axis_order(sequence)
        with open(self._path / "useq-sequence.json", "w") as f:
            f.write(sequence.json())

    def _onMDAFrame(self, img: np.ndarray, event: MDASequence) -> None:
        if not self._data_folder_path:
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


class ZarrWriter(BaseWriter):
    def __init__(
        self,
        store_name: Union[str, Path],
        img_shape: Tuple[int, int],
        dtype: npt.DTypeLike,
        core: CMMCorePlus = None,
    ):
        """
        Parameters
        ----------
        store_name : str
            Should accept .format(run=INT)
        img_shape : (int, int)
        dtype : numpy dtype
        core : CMMCorePlus, optional
            If not given the current core instance will be used.
        """
        if zarr is None:
            raise ValueError(
                "This writer requires zarr to be installed. Try: `pip install zarr`"
            )
        super().__init__(core)

        self._store_name = str(store_name)
        self._img_shape = img_shape
        self._dtype = dtype

    def _onMDAStarted(self, sequence: MDASequence):
        self._axis_order = self.sequence_axis_order(sequence)

        name = self.get_unique_folder(self._store_name, suffix=".zarr")
        assert isinstance(name, (Path, str))
        self._z = zarr.open(
            name,
            # self._store_name.format(run=self._run_number),
            mode="w",
            shape=sequence.shape + self._img_shape,
            dtype=self._dtype,
        )
        self._z.attrs["axis_order"] = f"{sequence.axis_order}yx"
        self._z.attrs["useq-sequence"] = sequence.json()

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent):
        self._z[self.event_to_index(self._axis_order, event)] = img


class ZarrNapariMicromanagerWriter(BaseWriter):
    def __init__(
        self, path: Path | str = "", file_name: str = "", core: CMMCorePlus = None
    ) -> None:
        super().__init__(core)

        self._path = path
        self._file_name = file_name

        self._zarr_list: list[zarr.Array] = []

    def _onMDAStarted(self, sequence: MDASequence):
        if not self._path:
            return
        self._zarr_list.clear()
        _path, _name, _shape, dtype, axis_labels = self._determine_zarr_info(sequence)
        self._create_zarr(sequence, _path, _name, _shape, dtype, axis_labels)

    def _onMDAFrame(self, img: np.ndarray, event: MDAEvent):
        if not self._path:
            return
        self._populate_zarr(img, event)

    def _get_axis_labels(self, sequence: MDASequence) -> tuple[list[str], bool]:
        # sourcery skip: use-next
        main_seq_axis = list(sequence.used_axes)

        if not sequence.stage_positions:
            return main_seq_axis, False

        sub_seq_axis: list | None = None
        for p in sequence.stage_positions:
            if p.sequence:  # type: ignore
                sub_seq_axis = list(p.sequence.used_axes)  # type: ignore
                break

        if sub_seq_axis:
            main_seq_axis.extend(sub_seq_axis)

        return main_seq_axis, bool(sub_seq_axis)

    def _determine_zarr_info(
        self, sequence: MDASequence
    ) -> tuple[Path, str, tuple, str, list[str]]:
        """Determine the zarr info to then create the zarr array from the sequence."""
        _folder_name = self._file_name or "exp"
        _path = self.get_unique_folder(self._path, _folder_name, create=True)
        _name = _path.stem
        axis_labels, pos_sequence = self._get_axis_labels(sequence)
        array_shape = [sequence.sizes[k] or 1 for k in axis_labels]

        if pos_sequence:
            for p in sequence.stage_positions:
                if not p.sequence:  # type: ignore
                    continue
                pos_g_shape = p.sequence.sizes["g"]  # type: ignore
                index = axis_labels.index("g")
                array_shape[index] = max(array_shape[index], pos_g_shape)

        yx_shape = [self._core.getImageHeight(), self._core.getImageWidth()]
        _shape = array_shape + yx_shape
        dtype = f"uint{self._core.getImageBitDepth()}"

        return _path, _name, _shape, dtype, axis_labels

    def _create_zarr(
        self,
        sequence: MDASequence,
        _path: Path,
        _name: str,
        shape: list[int],
        dtype: npt.DTypeLike,
        axis_labels: list[str],
    ):
        """Create the zarr array."""
        z = zarr.open(f"{_path}/{_name}.zarr", shape=shape, dtype=dtype, mode="w")
        z.attrs["name"] = _name
        z._attrs["uid"] = str(sequence.uid)
        z.attrs["axis_labels"] = axis_labels
        z.attrs["sequence"] = sequence.json()
        self._zarr_list.append(z)

    def _populate_zarr(self, image: np.ndarray, event: MDAEvent) -> None:
        """Populate the zarr array with the image data."""
        axis_order, _ = self._get_axis_labels(event.sequence)

        _next_name = self.get_unique_folder(self._path, self._file_name or "exp").stem
        _num = int(_next_name[-3:]) - 1
        _name = f"{_next_name[:-3]}{_num:03d}"

        # the index of this event in the full zarr array
        im_idx: tuple[int, ...] = ()
        for k in axis_order:
            try:
                im_idx += (event.index[k],)
            # if axis not in event.index
            # e.g. if we have bot a position sequence grid and a single position
            except KeyError:
                im_idx += (0,)

        for i in self._zarr_list:
            i = cast(zarr.Array, i)
            if (i.attrs["name"], i.attrs["uid"]) == (_name, str(event.sequence.uid)):
                i[im_idx] = image
                break
