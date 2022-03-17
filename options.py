import os
import argparse

class Options():
     def __init__(self):
          self.parser = argparse.ArgumentParser()
          self.parser.add_argument("--saveSampleNYU", 
                                   help="save small segment of NYU Depth V2",
                                   action="store_true",
                                   default=False)
          self.parser.add_argument("--loadSampleNYU", 
                                   help="load small segment of NYU Depth V2",
                                   action="store_true",
                                   default=True)                         
          self.parser.add_argument("--refine", 
                                   help="refine existing calibration or from scratch",
                                   action="store_true",
                                   default=False)
          self.parser.add_argument("--dataPath", 
                                   type=str,
                                   help="path to the training data",
                                   default="data")
          self.parser.add_argument("--numPairs",
                                   type=int,
                                   help="number of rgbd pairs to load",
                                   default=1)
          self.parser.add_argument("--height",
                                   type=int,
                                   help="number of rgbd pairs to load",
                                   default=480)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="number of rgbd pairs to load",
                                   default=640)

     def parse(self):
          self.options = self.parser.parse_args()
          return self.options
