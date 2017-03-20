import datetime
import xarray as xr
import numpy as np
from attributes import *

  
class SwanSpecFile(object):
    def __init__(self,filename,freqs=None,dirs=None,x=None,y=None,time=False,id='Swan Spectrum',dirorder=False,append=False):
        self.times=False
        self.filename=filename
        self.buf=None
        try:
            if freqs:#Writable file
                self.freqs=np.array(freqs)
                self.dirs=np.array(dirs)
                self.x=np.array(x)
                self.y=np.array(y)
                if time:self.times=[]
                self.f=open('w')
                self.f.write(self.fileHeader(locations,time,id))
            else:
                self.f=open(filename,'r+' if append else 'r')
                header=self._readhdr('SWAN')
                while True:
                    if not self._readhdr('$'):break
                if self._readhdr('TIME'):
                    self._readhdr('1')
                    self.times=[]
                self.x=[]
                self.y=[]
                for ip in self._readhdr('LONLAT',True):
                    xy=map(float,ip.split())
                    self.x.append(xy[0])
                    self.y.append(xy[1])
                self.x=np.array(self.x)
                self.y=np.array(self.y)
                self.freqs=np.array(map(float,self._readhdr('AFREQ',True)))
                self.dirs=np.array(map(float,self._readhdr('NDIR',True)))
                self._readhdr('QUANT',True)
                self.f.readline()
                self.f.readline()
        except exp:
            raise 'File error with %s [%s]' % (filename,exp)
        if dirorder:
            self.dirmap=list(numpy.argsort(self.dirs % 360.))
            self.dirs=self.dirs[self.dirmap] % 360.
        else:
            self.dirmap=False
        lons = np.unique(self.x)
        lats = np.unique(self.y)
        self.is_grid=(len(lons)*len(lats) == len(self.x))

    def _readhdr(self,keyword,numspec=False):
        if not self.buf:self.buf=self.f.readline()
        if self.buf.find(keyword)>=0:
            if numspec:
                line=self.f.readline()
                n=int(line[0:min(len(line),20)])
                self.buf=[self.f.readline() for i in range(0,n)]
            rtn=self.buf
            self.buf=None
        else:
            rtn=False
        return rtn

    def read(self):
        if not self.f:return None
        import copy
        if isinstance(self.times,list):
            line=self.f.readline()
            if line:
                ttime=datetime.datetime.strptime(line[0:15],'%Y%m%d.%H%M%S')
                self.times.append(ttime)
            else:
                return None
        Sout=[]
        for ip,pp in enumerate(self.x):
            Snew=np.nan*np.zeros((len(self.freqs),len(self.dirs)))
            if self._readhdr('NODATA'):
                pass
            else:
                if self._readhdr('ZERO'):
                    Snew=np.zeros((len(self.f),len(self.dirs)))
                elif self._readhdr('FACTOR'):
                    fac=float(self.f.readline())
                    for i,f in enumerate(self.freqs):
                        line=self.f.readline()
                        lsplit=line.split()
                        try:
                            Snew[i,:]=map(float,lsplit)
                        except:
                            pass
                    Snew*=fac
                    if self.dirmap:
                        Snew=Snew[:,self.dirmap]
            Sout.append(Snew)
        return Sout

    def scan(self,time):
        nf=len(self.S.freqs)+1
        tstr=time.strftime('%Y%m%d.%H%M%S')
        i=0
        while True:
            line=self.f.readline()
            if not line:
                return -1
            elif line[:15]==tstr:
                self.f.seek(-len(line),1)
                return i/nf
            i+=1

    def readall(self):
        while True:
            sset=self.read()
            if sset:
                yield sset
            else:
                break
            
    def writeHeader(self,time=False,str1='',str2=''):
        strout='SWAN   1\n$   '+str1+'\n$   '+str2+'\n'
        if (time):strout+='TIME\n1\n'
        np=len(xyloc)
        strout+='LONLAT\n'+str(np)+'\n'
        for i,loc in enumerate(self.x):
            strout+='%f %f\n' % (loc,self.y[i])
        strout += 'AFREQ\n%d\n' % (len(self.freqs))
        for freq in self.freqs:strout+='%f\n' % (freq)
        
        strout+='NDIR\n%d\n' % (len(self.dirs))
        for dir in self.dirs:strout+='%f\n' % (dir)  
        
        strout+='QUANT\n1\nVaDens\nm2/Hz/degr\n-99\tException value\n'
        self.f.write(strout)
    
    def writeSpectum(self,S,time=None):
        if time:
            self.times.append(time)
        if not isinstance(S,list):S=[S]
        if len(S)<>len(self.locations):raise ValueError('Number of spectra must equal number of locations')
        for spec in S:
            if not spec:
                self.f.write('NODATA\n')
            else:
                if not isinstance(spec,SwanSpectrum):spec=SwanSpectrum(spec.freqs,spec.dirs,spec.S)
                self.f.write(spec.fileRecord())
            
    def readSpectrum(self):
        if self.S.any():
            fac = self.S.max()/9998
            if fac<0:return 'NODATA\n'
            strout='FACTOR\n'+str(fac)+'\n'
            for row in self.S:
                strout+=(self.fmt % tuple(row/fac)) + '\n'
            return strout
        else:
            return 'NODATA\n'

    def close(self):
        if self.f:self.f.close()
        self.f=False
        self.times=[]
        
    
def to_swan(self,filename,id='Swan Spectrum',append=False):
    f=SwanSpecFile(filename,freqs=self.freq,dirs=self.dir,locations=locations,append=append)
    for t in self.times:
        f.write()
    f.close()
            
                
def read_swan(filename, dirorder=True):
    from spectra import SpecDataset
    """
    Read Spectra off SWAN ASCII file
    - dirorder :: If True reorder spectra read from file so that directions are sorted
    Returns:
    - dset :: SpecArray
    """
    swanfile=SwanSpecFile(filename)
    times=swanfile.times
    lons=swanfile.x
    lats=swanfile.y
    sites=np.arange(len(lons))+1
    freqs=swanfile.freqs
    dirs=swanfile.dirs
    
    spec_list=swanfile.readall()
    if swanfile.is_grid:
# Looks like gridded data, grid DataArray accordingly
        arr = np.array([s for s in spec_list]).reshape(len(times), len(lons), len(lats), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=np.swapaxes(arr, 1, 2),
            coords=OrderedDict(((TIMENAME, times), (LATNAME, lats), (LONNAME, lons), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, LATNAME, LONNAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
            ).to_dataset()
    else:
        # Keep it with sites dimension
        dset = xr.DataArray(
            data=np.array([s for s in spec_list]).reshape(len(times), len(sites), len(freqs), len(dirs)),
            coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
        ).to_dataset()
        dset[LATNAME] = xr.DataArray(data=lats, coords={SITENAME: sites}, dims=[SITENAME])
        dset[LONNAME] = xr.DataArray(data=lons, coords={SITENAME: sites}, dims=[SITENAME])
    return SpecDataset(dset)

    
    