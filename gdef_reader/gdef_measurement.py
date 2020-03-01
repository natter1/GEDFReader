from typing import Optional, List

from gdef_reader.gdef_importer import GDEFHeader, GDEFControlBlock


class GDEFMeasurement:
    def _init_(self):
        self.header: Optional[GDEFHeader] = None

        self.Lines = None
        self.Columns = None
        self.MissingLines = None
        self.LineMean = None
        self.LineMeanOrder = None
        self.InvertLineMean = None
        self.PlaneCorr = None
        self.InvertPlaneCorr = None
        self.MaxWidth = None
        self.MaxHeight = None
        self.OffsetX = None
        self.OffsetY = None
        self.ZUnit = None
        self.Retrace = None
        self.ZLinearized = None
        self.ScanMode = None
        self.ZCalib = None
        self.XCalib = None
        self.YCalib = None
        self.ScanSpeed = None
        self.SetPoint = None
        self.BiasVoltage = None
        self.LoopGain = None
        self.LoopInt = None
        self.PhaseShift = None
        self.ScanDirection = None
        self.DigitalLoop = None
        self.LoopFilter = None
        self.FFTType = None
        self.XYLinearized = None
        self.RetraceType = None
        self.Calculated = None
        self.ScannerRange = None
        self.PixelBlend = None
        self.SourceChannel = None
        self.DirectAC = None
        self.ID = None
        self.QFactor = None
        self.AuxGain = None
        self.FixedPalette = None
        self.FixedMin = None
        self.FixedMax = None
        self.ZeroScan = None
        self.MeasuredAmplitude = None
        self.FrequencyOffset = None
        self.QBoost = None
        self.OffsetPos = None
        self.Data = None

    def read_blocks(self, blocks: List[GDEFControlBlock]):
        pass

        #     read_variable_lists(depth=1)
        #         read block: 3 - block.mark=b'CB')
        #             block variable 0 - Value
        #         read block: 4 - block.mark=b'CB')
        #             block variable 0 - Comment
        #         read block: 5 - block.mark=b'CB')
        #             block variable 0 - Preview
        #     return from read_variable_lists(depth=1)
        #         block variable 48 - TimeInfo
        #     read_variable_lists(depth=1)
        #         read block: 6 - block.mark=b'CB')
        #             block variable 0 - Hour
        #             block variable 1 - Minute
        #             block variable 2 - Second
        #             block variable 3 - Year
        #             block variable 4 - Month
        #             block variable 5 - Day
        #     return from read_variable_lists(depth=1)
        #         block variable 49 - Attributes
        #     read_variable_lists(depth=1)
        #         read block: 7 - block.mark=b'CB')
        #     return from read_variable_lists(depth=1)
