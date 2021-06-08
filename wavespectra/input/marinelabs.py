import copy
import datetime
import glob
import os
from collections import OrderedDict

import numpy as np
import xarray as xr
from dateutil import parser

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.misc import interp_spec
from wavespectra.specdataset import SpecDataset


def read_marinelabs(filename_or_fileglob, toff=0):
    """Read spectra from Triaxys wave buoy ASCII files.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying one or
          more files to read.
        - toff (float): time-zone offset from UTC in hours.

    Returns:
        - dset (SpecDataset): spectra dataset object read from ww3 file.

    Note:
        - frequencies and directions from first file are used as reference
          to interpolate spectra from other files in case they differ.

    """
    txys = Marinelabs(filename_or_fileglob, toff)
    txys.run()
    return txys.dset


class Marinelabs(object):
    def __init__(self, filename_or_fileglob, toff=0):
        """Read wave spectra file from MARINELABS buoy.

        Args:
            - filename_or_fileglob (str, list): filename or fileglob
              specifying files to read.
            - toff (float): time offset in hours to account for
              time zone differences.

        Returns:
            - dset (SpecDataset) wavespectra SpecDataset instance.

        Remark:
            - frequencies and directions from first file are used as reference
              to interpolate spectra from other files in case they differ.

        """
        self._filename_or_fileglob = filename_or_fileglob
        self.toff = toff
        self.stream = None
        self.is_dir = None
        self.time_list = []
        self.spec_list = []
        self.header_keys = header_keys = [
            "buoy_id",
            "serial_num",
            "is_dir",
            "time",
            "nf",
            "nd",
            "f0",
            "df",
            "fmin",
            "fmax",
            "ddir",
        ]

    def run(self):
        for ind, self.filename in enumerate(self.filenames):
            print(self.filename)
            self.open()
            self.read_header()
            if ind == 0:
                self.interp_freq = copy.deepcopy(self.freqs)
                self.interp_dir = copy.deepcopy(self.dirs)

            if self.header.get("nf")>0:
                self.read_data()
            self.close()
        self.construct_dataset()

    def open(self):
        self.stream = open(self.filename, "r")

    def close(self):
        if self.stream and not self.stream.closed:
            self.stream.close()

    def read_header(self):
        self.header = {k: None for k in self.header_keys}
        self.header.update(is_dir=False)
        while True:
            line = self.stream.readline()
            if "Buoy ID" in line:
                self.header.update(buoy_id=str(line.split(":")[1]))
            if "Serial Number" in line:
                self.header.update(serial_num=int(line.split(":")[1]))
            if "Start time" in line:
                time = parser.parse(
                    line.split(":")[1].split("(")[0].strip()
                ) - datetime.timedelta(hours=self.toff)
                self.header.update(time=time)
            if "Number of frequencies" in line:
                self.header.update(nf=int(line.split(":")[1]))

                # self.header.update(df=0.004882813)
                # self.header.update(f0=0.048828125)
                # fmax = self.header['f0'] + self.header['df'] * (self.header['nf']-1)
                # self.header.update(fmax=fmax)
                # self.header.update(fmin=self.header['f0'])
            if "Minimum frequency" in line:
                f0 = float(line.split(":")[1])
                self.header.update(f0=f0)
                self.header.update(fmin=f0)
            if "Frequency spacing" in line:
                self.header.update(df=float(line.split(":")[1]))
                fmax = self.header['f0'] + self.header['df'] * (self.header['nf']-1)
                self.header.update(fmax=fmax)
            if "Number of directions" in line:
                self.header.update(nd=int(line.split(":")[1]))
                self.header.update(is_dir=True)
            if "Columns" in line:
                d = line.split(":")[1].split("to")
                dmin = float(d[0])
                dmax = float((d[1].split()[0]))
                ddir = (dmax-dmin)/(self.header['nd']-1)
                self.header.update(ddir=ddir)
            if "Rows" in line:
                f = line.split(":")[1].split("to")
                fmin = float(f[0])
                fmax = float((f[1].split()[0]))
                df = (fmax-fmin)/(self.header['nf']-1)
                self.header.update(df=df,f0=fmin,fmax=fmax)
                break
            if not line or "Column 2" in line:
                break

        if not self.header.get("time"):
            raise IOError("Cannot parse time")
        if self.is_dir is not None and self.is_dir != self.header.get("is_dir"):
            raise IOError("Cannot merge spectra 2D and spectra 1D")
        self.is_dir = self.header.get("is_dir")

    def _append_spectrum(self):
        """Append spectra after ensuring same spectral basis."""
        self.spec_list.append(
            interp_spec(
                inspec=self.spec_data,
                infreq=self.freqs,
                indir=self.dirs,
                outfreq=self.interp_freq,
                outdir=self.interp_dir,
            )
        )
        self.time_list.append(self.header.get("time"))

    def read_data(self):
        try:
            self.spec_data = np.zeros((len(self.freqs), len(self.dirs)))
            for i in range(self.header.get("nf")):
                row = np.array(list(map(float, self.stream.readline().split(","))))
                if self.header.get("is_dir"):
                    self.spec_data[i, :] = row / (180 / np.pi )
                else:
                    self.spec_data[i, :] = row[-1]

            self._append_spectrum()

        except ValueError as err:
            raise ValueError("Cannot read {}:\n{}".format(self.filename, err))

    def construct_dataset(self):
        self.dset = xr.DataArray(
            data=self.spec_list, coords=self.coords, dims=self.dims, name=attrs.SPECNAME
        ).to_dataset()
        set_spec_attributes(self.dset)
        if not self.is_dir:
            self.dset = self.dset.isel(drop=True, **{attrs.DIRNAME: 0})
            self.dset[attrs.SPECNAME].attrs.update(units="m^{2}.s")

    @property
    def dims(self):
        return (attrs.TIMENAME, attrs.FREQNAME, attrs.DIRNAME)

    @property
    def coords(self):
        _coords = OrderedDict(
            (
                (attrs.TIMENAME, self.time_list),
                (attrs.FREQNAME, self.interp_freq),
                (attrs.DIRNAME, self.interp_dir),
            )
        )
        return _coords

    @property
    def dirs(self):
        ddir = self.header.get("ddir")
        if ddir:
            return list(np.arange(0.0, 360.0, ddir))
        else:
            return [0.0]

    @property
    def freqs(self):
        try:
            f0, df, nf = self.header["f0"], self.header["df"], self.header["nf"]
            return list(np.arange(f0, f0 + df * nf, df))
        except Exception as exc:
            raise IOError("Not enough info to parse frequencies:\n{}".format(exc))

    @property
    def filenames(self):
        if isinstance(self._filename_or_fileglob, list):
            filenames = sorted(self._filename_or_fileglob)
        elif isinstance(self._filename_or_fileglob, str):
            filenames = sorted(glob.glob(self._filename_or_fileglob))
        if not filenames:
            raise ValueError("No file located in {}".format(self._filename_or_fileglob))
        return filenames


if __name__ == "__main__":

    # 1D files
    dset_1d = read_marinelabs("D:/PRIMED/Stations/106/uvic_01 - 20181115 to 20190203/spectra/20181115T220000Z/nondirectional_spectrum.csv")

    # 2D files
    # dset_2d = read_marinelabs("D:/PRIMED/Stations/106/uvic_01 - 20181115 to 20190203/spectra/20181115T220000Z/directional_spectrum.csv")

    import matplotlib.pyplot as plt
    from wavespectra.specdataset import SpecDataset

    # print(dset_1d)
    print(dset_1d)
    dset_1d = SpecDataset(dset_1d)
    print(dset_1d)

    # dset_2d.spec.plot.contourf( as_log10=False)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # dset_1d.spec.hs().plot(label="1D")
    # dset_2d.spec.hs().plot(label="2D")
    # plt.legend()
    # plt.show()
