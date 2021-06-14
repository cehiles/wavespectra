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


def read_formatb(filename_or_fileglob, toff=0):
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
    fmtb = FormatB(filename_or_fileglob, toff)
    fmtb.run()
    return fmtb.dset


class FormatB(object):
    def __init__(self, filename_or_fileglob, toff=0):
        """Read wave spectra file from TRIAXYS buoy.

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
        self.parameters = None
        self.time_list = []
        self.spec_list = []
        self.header_keys = header_keys = [
            "is_formatb",
            "station_type",
            "station_name",
            "station_ID",
            "latitude",
            "longitude",
            "depth",
            "time",
            "length_of_recording",
            "sampling_frequency",
            "quality_code",
            "num_addl_params",
            "num_wave_heights",
            "num_wave_periods",
            "num_spec_estimates",
            "is_dir",
            "ddir",
        ]
        #
        # "is_dir",
        # "time",
        # "nf",
        # "f0",
        # "df",
        # "fmin",
        # "fmax",
        # ,
        #
        # self,
        # filename,
        # freqs=None,
        # dirs=None,
        # x=None,
        # y=None,
        # time=False,
        # id="Swan Spectrum",
        # dirorder=False,
        # append=False,
        # tabfile=None,

    def run(self):
        for ind, self.filename in enumerate(self.filenames):
            self.open()

            ii = 0
            while True:
                try:
                    self.read_header()
                except IOError:
                    if ii>0:
                        break
                    else:
                        raise IOError("Not FormatB Spectra file.")

                self.read_parameters()
                self.read_data(ind)
                ii = ii+1

            self.close()
        self.construct_dataset()

    def open(self):
        self.stream = open(self.filename, "r")

    def close(self):
        if self.stream and not self.stream.closed:
            self.stream.close()

    def read_header(self):
        self.header = {k: None for k in self.header_keys}

        # --------------------------------------------------------------------
        # READ LINE 1
        # --------------------------------------------------------------------
        line = self.stream.readline()
        if line[0:10].strip() in self.get_wave_instrument_type_codes():
            self.header.update(is_formatb=True)
            self.header.update(station_type=line[0:10].strip())
        self.header.update(station_name=line[15:35].strip())
        self.header.update(station_ID=line[40:50].strip())

        if not self.header.get("is_formatb"):
            raise IOError("Not FormatB Spectra file.")

        # --------------------------------------------------------------------
        # READ LINE 2
        # --------------------------------------------------------------------
        line = self.stream.readline()
        self.header.update(latitude=float(line[0:10].strip()))
        self.header.update(longitude=float(line[10:20].strip()))
        self.header.update(depth=float(line[20:28].strip()))

        yyyy = int(line[28:33].strip())
        mm = int(line[33:35].strip())
        dd = int(line[35:37].strip())
        if line[37:40].strip()=='':
            HH = 0
        else:
            HH = int(line[37:40].strip())

        if line[40:42].strip()=='':
            MM = 0
        else:
            MM = int(line[40:42].strip())

        time = datetime.datetime(yyyy,mm,dd,HH,MM) - \
               datetime.timedelta(hours=self.toff)
        # print(time)
        self.header.update(time=time)

        self.header.update(length_of_recording=float(line[42:50].strip()))
        self.header.update(sampling_frequency=float(line[50:62].strip()))
        self.header.update(quality_code=int(line[62:66].strip()))
        self.header.update(num_addl_params=int(line[66:70].strip()))
        self.header.update(num_wave_heights=int(line[70:73].strip()))
        self.header.update(num_wave_heights=int(line[70:73].strip()))
        self.header.update(num_wave_periods=int(line[73:76].strip()))
        self.header.update(num_spec_estimates=int(line[76:80].strip()))


        if not self.header.get("time"):
            raise IOError("Cannot parse time")
        # if self.is_dir is not None and self.is_dir != self.header.get("is_dir"):
        #     raise IOError("Cannot merge spectra 2D and spectra 1D")
        # self.is_dir = self.header.get("is_dir")


    def read_parameters(self):

        # --------------------------------------------------------------------
        # READ ADDITIONAL (METOCEAN) PARAMETERS
        # --------------------------------------------------------------------
        params = {}
        c = 80

        for ii in range(1,self.header['num_addl_params']+1):
            if c == 80:
                line = self.stream.readline()
                c = 0
            val = line[c:c+12]
            key = line[c+12:c+16]
            if key in params.keys():
                key = key +'.1'
            params[key] = val
            c = c+16

        c = 80
        num_wave_params = self.header['num_wave_heights'] + \
                          self.header['num_wave_periods']
        for ii in range(0,num_wave_params):
            if c == 80:
                line = self.stream.readline()
                c = 0
            val = line[c:c+6]
            key = line[c+6:c+10]
            if key in params.keys():
                key = key +'.1'
            params[key] = val
            c = c+10

        # print(params)
        self.parameters = params


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

    def read_data(self,ind):
        try:
            self.spec_data = np.zeros((self.header['num_spec_estimates'],1))
            self.freqs = np.zeros(self.header['num_spec_estimates'])
            self.bandwidth = np.zeros(self.header['num_spec_estimates'])
            c = 80
            for ii in range(0,self.header['num_spec_estimates']):
                if c+12 > 80:
                    line = self.stream.readline()
                    c = 0
                self.freqs[ii] = float(line[c:c+12])
                self.bandwidth[ii] = float(line[c+12:c+24])
                self.spec_data[ii] = float(line[c+24:c+36])
                c = c+36
            if ind == 0:
                self.interp_freq = copy.deepcopy(self.freqs)
                self.interp_dir = copy.deepcopy(self.dirs)
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

    def get_wave_instrument_type_codes(self):
        # https://meds-sdmm.dfo-mpo.gc.ca/isdm-gdsi/waves-vagues/formats-eng.html#QC
        WITC = [
        '12', #MSC Non-directional ODAS buoy. 12m Discus
        '3D', #MSC Non-directional ODAS buoy. 3m Discus
        '6N', #MSC Non-directional ODAS buoy. 6m NOMAD
        'AE', #MSC Non-directional ODAS buoy. (6m NOMAD, 12m Discus, 3m Discus or 1.7m Watchkeeper)
        'AW', #MSC buoy data with bad Watchman payload. (Truncated spectra, VCAR=VWH$, VTPK=VTP$)
        'EN', #Directional Buoy, Endeco
        'HX', #Hexoid buoy
        'KG', #Kelk Pressure cell
        'MI', #Miros Radar
        'PC', #Pressure cell
        'ST', #Staff gauge
        'SW', #Swartz gauge
        'TG', #Toga buoy
        'TR', #Directional buoy, TriAxys
        'WC', #Directional buoy, WAVEC information processing system (Datawell)
        'WD', #Directional Waverider buoy, standard information processing system (Datawell)
        'WK', #MSC Non-directional ODAS buoy. 1.7m Watchkeeper
        'WP', #Non-directional Waverider buoy, WRIPS system (Datawell)
        'WR', #Non-directional Waverider buoy, standard system (Datawell)
        ]

        return WITC

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
            return list(np.arange(0.0, 360.0 + ddir, ddir))
        else:
            return [0.0]

    # @property
    # def freqs(self):
    #     try:
    #         f0, df, nf = self.header["f0"], self.header["df"], self.header["nf"]
    #         return list(np.arange(f0, f0 + df * nf, df))
    #     except Exception as exc:
    #         raise IOError("Not enough info to parse frequencies:\n{}".format(exc))

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
    dset_1d = read_triaxys("NONDIRSPEC/2018??????0000.NONDIRSPEC")

    # 2D files
    dset_2d = read_triaxys("DIRSPEC/2018??????0000.DIRSPEC")

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    dset_1d.spec.hs().plot(label="1D")
    dset_2d.spec.hs().plot(label="2D")
    plt.legend()
