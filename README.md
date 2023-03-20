# pymmcore-MDA-writers

[![License](https://img.shields.io/pypi/l/pymmcore-MDA-writers.svg?color=green)](https://github.com/ianhi/pymmcore-MDA-writers/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pymmcore-MDA-writers.svg?color=green)](https://pypi.org/project/pymmcore-MDA-writers)
[![Python Version](https://img.shields.io/pypi/pyversions/pymmcore-MDA-writers.svg?color=green)](https://python.org)
[![Test](https://github.com/ianhi/pymmcore-MDA-writers/actions/workflows/ci.yml/badge.svg)](https://github.com/ianhi/mpl-interactions/actions/)
[![codecov](https://codecov.io/gh/ianhi/pymmcore-MDA-writers/branch/main/graph/badge.svg)](https://codecov.io/gh/ianhi/pymmcore-MDA-writers)

This package provides writers for [pymmcore-plus](https://pymmcore-plus.readthedocs.io). Currently provided are:

1. A simple multifile tiff writer to store each frame from a MDASequence as a separate tiff file - can be loaded with `tifffile`
2. A simple zarr writer to store the MDASequence data to a zarr store - not ome-zarr

```bash
pip install pymmcore-mda-writers
```

(This will require a minimum of `useq-schema>=0.1.5` which has not yet been released. You can install a working version with `pip install git+https://github.com/pymmcore-plus/useq-schema`)


All you need to add to your script is:
```python
# tiff writer
writer = MiltiTiffMDASequenceWriter(folder_path="data/tiff_writer_example", file_name="run")

# zarr writer
writer = ZarrMDASequenceWriter(folder_path="data/zarr_writer_example", file_name="run")
```

for a complete example see the examples folder.
