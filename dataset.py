import torch
import numpy as np
from PIL import Image
from pathlib import Path
from zipfile import ZipFile
import sys, re, os, pickle, shutil


class RawDatasetArchive():
    """Loads a zip file containing (a part of) the raw dataset and
    provides member functions for further data processing.

    (adapted from https://github.com/GabrielMajeri/nyuv2-python-toolbox)
    """
    def __init__(self, zip_path):
        self.zip = ZipFile(zip_path)
        self.frames = self.synchroniseFrames(self.zip.namelist())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def extract_frame(self, frame, path=None):
        """Extracts a synchronised frame of depth and color images.

        The frame parameter must be a pair of depth and color maps from
        the archive. Optionally the path of an extraction directory can be given.
        """
        return map(lambda name: self.zip.extract(name, path=path), frame)

    def synchroniseFrames(self, frameNames):
        """Constructs a list of synchronised depth and RGB frames.

        Returns a list of pairs, where the first is the path of a depth image,
        and the second is the path of a color image.
        """

        # Regular expressions for matching depth and color images
        depthImgProg = re.compile(r'.+/d-.+\.pgm')
        colorImgProg = re.compile(r'.+/r-.+\.ppm')

        # Applies a regex program to the list of names
        def matchNames(prog):
            return map(prog.match, frameNames)

        # Filters out Nones from an iterator
        def filterNone(iter):
            return filter(None.__ne__, iter)

        # Converts regex matches to strings
        def matchToStr(matches):
            return map(lambda match: match.group(0), matches)

        # Retrieves the list of image names matching a certain regex program
        def imageNames(prog):
            return list(matchToStr(filterNone(matchNames(prog))))

        depthImgNames = imageNames(depthImgProg)
        colorImgNames = imageNames(colorImgProg)
        # By sorting the image names we ensure images come in chronological order
        depthImgNames.sort()
        colorImgNames.sort()

        def nameToTimestamp(name):
            """Extracts the timestamp of a RGB / depth image from its name."""
            _, time, _ = name.split('-')
            return float(time)

        frames = []
        colorCount = len(colorImgNames)
        colorIdx = 0
        for depthImgName in depthImgNames:
            depthTime = nameToTimestamp(depthImgName)
            colorTime = nameToTimestamp(colorImgNames[colorIdx])
            diff = abs(depthTime - colorTime)
            # Keep going through the color images until we find
            # the one with the closest timestamp
            while colorIdx + 1 < colorCount:
                colorTime = nameToTimestamp(colorImgNames[colorIdx + 1])
                newDiff = abs(depthTime - colorTime)
                # Moving forward would only result in worse timestamps
                if newDiff > diff:
                    break
                diff = newDiff
                colorIdx = colorIdx + 1
            frames.append((depthImgName, colorImgNames[colorIdx]))
        return frames

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.fnames = []
        self.getFileNames()
        self.bTmp = 300

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        inputs = {}
        for f in range(50):
            fname = self.fnames[f]
            color = self.loadColorImage(fname[1])
            depth = self.loadDepthImage(fname[0])       

            color = torch.from_numpy(color).permute(2,0,1) / 255.
            depth, mask = self.normalizeDepth(depth) 
            depth = torch.from_numpy(depth)[None]
            mask = torch.from_numpy(mask)[None]

            inputs[("color", f)] = color.float()
            inputs[("depth", f)] = depth.float()
            inputs[("mask", f)] = mask.repeat(3,1,1)
        return inputs

    def normalizeDepth(self, relativeDepth):
        maxDepth = 10.0
        nyuConst1 = 351.3
        nyuConst2 = 1092.5
        absoluteDepth = nyuConst1 / (nyuConst2 - relativeDepth)   
        mask1 = absoluteDepth >= 0 
        mask2 = absoluteDepth < maxDepth
        mask = mask1 * mask2
        return np.clip(absoluteDepth, 0, maxDepth), mask

    def getFileNames(self):
        """load file names"""
        if self.opt.loadSampleNYU:
            with open("nyuSample", "rb") as fp:
                self.fnames = pickle.load(fp)
        else:
            dataDir = Path('data')
            rawPath = next(dataDir.glob('office_0017.zip'))
            rawArchive = RawDatasetArchive(rawPath)
            for f in range(50):
                frame = rawArchive[f]
                depthPath = os.path.join(self.opt.dataPath, Path('.') / frame[0])
                colorPath = os.path.join(self.opt.dataPath, Path('.') / frame[1])
                if self.opt.saveSampleNYU:
                    depthPathNew = 'data/nyuSample/'+depthPath.split('/')[-1]
                    colorPathNew = 'data/nyuSample/'+colorPath.split('/')[-1]
                    shutil.copyfile(depthPath, depthPathNew)
                    shutil.copyfile(colorPath, colorPathNew)
                    self.fnames.append((depthPathNew, colorPathNew))

            if self.opt.saveSampleNYU:
                with open("nyuSample", "wb") as fp:
                    pickle.dump(self.fnames, fp)


    def loadDepthImage(self, path):
        """Loads an unprocessed depth map extracted from the raw dataset."""
        with open(path, 'rb') as f:
            return self.readPgm(f)

    def loadColorImage(self, path):
        """Loads an unprocessed color image extracted from the raw dataset."""
        with open(path, 'rb') as f:
            return self.readPpm(f)

    def readPgm(self, pgmFile):
        """Reads a PGM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """
        # First line contains some image meta-info
        p5, width, height, depth = pgmFile.readline().split()

        # Ensure we're actually reading a P5 file
        assert p5 == b'P5'
        assert depth == b'65535', "Only 16-bit PGM files are supported"
        width, height = int(width), int(height)
        data = np.fromfile(pgmFile, dtype='<u2', count=width*height)
        return data.reshape(height, width).astype(float)

    def readPpm(self, ppmFile):
        """Reads a PPM file from a buffer.

        Returns a numpy array of the appropiate size and dtype.
        """
        p6, width, height, depth = ppmFile.readline().split()
        assert p6 == b'P6'
        assert depth == b'255', "Only 8-bit PPM files are supported"
        width, height = int(width), int(height)
        data = np.fromfile(ppmFile, dtype=np.uint8, count=width*height*3)
        return data.reshape(height, width, 3).astype(float)
