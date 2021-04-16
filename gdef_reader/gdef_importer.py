"""
@author: Nathanael Jöhrmann
"""
import io
import struct
from pathlib import Path
from typing import Optional, BinaryIO, List, Union

import numpy as np

# HEADER_SIZE = 4 + 2 + 2 + 4 + 4
# CONTROL_BLOCK_SIZE = 2 + 2 + 4 + 4 + 1 + 3
# VAR_NAME_SIZE = 50
# VARIABLE_SIZE = 50 + 4
from afm_tools.background_correction import BGCorrectionType
from gdef_reader.gdef_data_strucutres import GDEFHeader, GDEFControlBlock, GDEFVariableType, GDEFVariable, type_sizes
from gdef_reader.gdef_measurement import GDEFMeasurement


class GDEFImporter:
    """
    This class is used to read data from a \*.gdf file (DME AFM) into python. This can be done like:

    .. code:: python

        from gdef_reader.gdef_importer import GDEFImporter
        impported_data = GDEFImporter(gdf_path)  # gdf_path should be a pathlib.Path to a *.gdf file

    :InstanceAttributes:
    basename: Path.stem of the imported \*.gdf file.
    keep_z_offset: If False (default), z-values for each imported measurement are corrected so that mean(z) == 0.
    bg_correction_type: BGCorrectionType for loaded measurements.
    :EndInstanceAttributes:
    """
    def __init__(self, filename: Optional[Path] = None):
        """
        :param filename: Path to \*.gdf file. If it is None (default), a file has to be loaded via GDEFImporter.load().
        """
        self.basename = ""

        self._header: GDEFHeader = GDEFHeader()
        self._buffer: Optional[BinaryIO] = None

        self._blocks: List[GDEFControlBlock] = []
        self._base_blocks: List[GDEFControlBlock] = []

        self.keep_z_offset = False
        self.bg_correction_type = BGCorrectionType.legendre_1
        self._eof = None

        if filename:
            self.load(filename)

    def export_measurements(self, path: Path = None, create_images: bool = False) -> List[GDEFMeasurement]:
        """
        Create a list of GDEFMeasurement-Objects from imported data. The optional parameter create_images
        can be used to show a matplotlib Figure for each GDEFMeasurement (default value is False).
        :param path: Save path for GDEFMeasurement-objects. No saved files, if None.
        :param create_images: Show a matplotlib Figure for each GDEFMeasurement; used for debugging (default: False)
        :return: list of GDEFMeasurement-Objects
        """
        result = []
        for i, block in enumerate(self._blocks):
            if block.n_data != 1 or block.n_variables != 50:
                continue
            measurement = self._get_measurement_from_block(block, create_images)
            measurement.gdf_basename = self.basename
            result.append(measurement)

        for measurement in result:
            if create_images:
                fig = measurement.create_plot()
                if fig:
                    fig.show()
            if path:
                path.mkdir(parents=True, exist_ok=True)
                if create_images:
                    measurement.save_png(f"{path}\\{self.basename}_block_{measurement.gdf_block_id}", dpi=96)
                measurement.save(f"{path}\\{self.basename}_block_{measurement.gdf_block_id:04}.pygdf")  # todo: what happens, when block.id > 9999?

        return result

    def load(self, filename: Union[str, Path]) -> None:
        """
        Import data from a \*.gdf file.
        :param filename: Path to \*.gdf file.
        :return: None
        """
        self.basename = filename.stem
        self._buffer = open(filename, 'rb')
        self._eof = self._buffer.seek(0, 2)
        self._buffer.seek(0)
        self._read_header()
        self._read_variable_lists()
        return None

    def _read_header(self):
        self._buffer.seek(0)  # sets the file's current position at the offset
        self._header.magic = self._buffer.read(4)
        self._header.version = int.from_bytes(self._buffer.read(2), 'little')
        if self._header.version != 0x0200:
            raise Exception(f"File version {self._header.version} is not supported")

        self._buffer.read(2)  # align

        self._header.creation_time = int.from_bytes(self._buffer.read(4), 'little')
        self._header.description_length = int.from_bytes(self._buffer.read(4), 'little')
        self._header.description = self._buffer.read(self._header.description_length).decode("utf-8")

    def _read_control_block(self, block):
        block.mark = self._buffer.read(2)
        if not block.mark == b'CB':
            assert block.mark == b'CB'

        self._buffer.read(2)  # align
        block.n_variables = int.from_bytes(self._buffer.read(4), 'little')
        block.n_data = int.from_bytes(self._buffer.read(4), 'little')

        block.next_byte = self._buffer.read(1)
        self._buffer.read(3)

        return block

    def _read_variable(self, variable):
        variable.name = self._buffer.read(50).decode("utf-8")
        self._buffer.read(2)
        variable.type = int.from_bytes(self._buffer.read(4), 'little')
        assert variable.type < GDEFVariableType.VAR_NVARS.value
        return variable

    def _read_variable_lists(self, depth: int = 0):
        blocks = []
        break_flag = False

        while (not break_flag) and (self._buffer.tell() != self._eof):
            block = GDEFControlBlock()
            block = self._read_control_block(block)

            if block.next_byte == b'\x00':
                break_flag = True

            # read variables
            for i in range(block.n_variables):
                variable = GDEFVariable()
                variable = self._read_variable(variable)
                block.variables.append(variable)

                if variable.type == GDEFVariableType.VAR_DATABLOCK.value:
                    variable.data = self._read_variable_lists(depth + 1)
                    self.flow_offset = ' ' * 4 * depth

            if depth == 0:
                self._read_variable_data(block, depth)

            self._blocks.append(block)
            if depth == 0:
                self._base_blocks.append(block)
            blocks.append(block)
        return blocks  # measurement._blocks

    def _read_variable_data(self, block: GDEFControlBlock, depth: int):
        for variable in block.variables:
            if variable.type == GDEFVariableType.VAR_DATABLOCK.value:
                nestedblocks: GDEFControlBlock = variable.data
                for block in nestedblocks:
                    self._read_variable_data(block, depth + 1)
            else:
                variable.data = self._buffer.read(block.n_data * type_sizes[variable.type])
                if variable.type == GDEFVariableType.VAR_INTEGER.value:
                    variable.data = int.from_bytes(variable.data, 'little')
                elif variable.type == GDEFVariableType.VAR_FLOAT.value:
                    f = io.BytesIO(variable.data)
                    variable.data = []
                    while True:
                        chunk = f.read(4)
                        if chunk == b'':
                            break
                        variable.data.append(struct.unpack('<f', chunk))
                    if len(variable.data) == 1:
                        variable.data = variable.data[0][0]  # [0][0] struct.unpack also returns tuple, not float/double
                elif variable.type == GDEFVariableType.VAR_DOUBLE.value:
                    f = io.BytesIO(variable.data)
                    variable.data = []
                    while True:
                        chunk = f.read(8)
                        if chunk == b'':
                            break
                        variable.data.append(struct.unpack('<d', chunk))
                    if len(variable.data)==1:
                        variable.data = variable.data[0][0]  # [0][0] struct.unpack also returns tuple, not float/double

                elif variable.type == GDEFVariableType.VAR_WORD.value:
                    variable.data = int.from_bytes(variable.data, 'little')
                elif variable.type == GDEFVariableType.VAR_DWORD.value:
                    variable.data = int.from_bytes(variable.data, 'little')
                elif variable.type == GDEFVariableType.VAR_CHAR.value:
                    if len(variable.data) == 1:
                        variable.data = int.from_bytes(variable.data, 'little')
                    else:
                        pass  # variable.data = variable.data.decode("utf-8")
                else:
                    print("should not happen")

    def _get_measurement_from_block(self, block: GDEFControlBlock, create_image) -> GDEFMeasurement:
        result = GDEFMeasurement()
        result.gdf_block_id = block.id

        result.settings.lines = block.variables[0].data
        result.settings.columns = block.variables[1].data
        result.settings.missing_lines = block.variables[2].data
        result.settings.line_mean = block.variables[3].data
        result.settings.line_mean_order = block.variables[4].data
        result.settings.invert_line_mean = block.variables[5].data
        result.settings._plane_corr = block.variables[6].data
        result.settings.invert_plane_corr = block.variables[7].data
        result.settings.max_width = block.variables[8].data
        result.settings.max_height = block.variables[9].data
        result.settings.offset_x = block.variables[10].data
        result.settings.offset_y = block.variables[11].data
        result.settings.z_unit = block.variables[12].data
        result.settings.retrace = block.variables[13].data
        result.settings.z_linearized = block.variables[14].data
        result.settings.scan_mode = block.variables[15].data
        result.settings.z_calib = block.variables[16].data
        result.settings.x_calib = block.variables[17].data
        result.settings.y_calib = block.variables[18].data
        result.settings.scan_speed = block.variables[19].data
        result.settings.set_point = block.variables[20].data
        result.settings.bias_voltage = block.variables[21].data
        result.settings.loop_gain = block.variables[22].data
        result.settings.loop_int = block.variables[23].data
        result.settings.phase_shift = block.variables[24].data
        result.settings.scan_direction = block.variables[25].data
        result.settings.digital_loop = block.variables[26].data
        result.settings.loop_filter = block.variables[27].data
        result.settings.fft_type = block.variables[28].data
        result.settings.xy_linearized = block.variables[29].data
        result.settings.retrace_type = block.variables[30].data
        result.settings.calculated = block.variables[31].data
        result.settings.scanner_range = block.variables[32].data
        result.settings.pixel_blend = block.variables[33].data
        result.settings.source_channel = block.variables[34].data
        result.settings.direct_ac = block.variables[35].data
        result.settings.id = block.variables[36].data
        result.settings.q_factor = block.variables[37].data
        result.settings.aux_gain = block.variables[38].data
        result.settings.fixed_palette = block.variables[39].data
        result.settings.fixed_min = block.variables[40].data
        result.settings.fixed_max = block.variables[41].data
        result.settings.zero_scan = block.variables[42].data
        result.settings.measured_amplitude = block.variables[43].data
        result.settings.frequency_offset = block.variables[44].data
        result.settings.q_boost = block.variables[45].data
        result.settings.offset_pos = block.variables[46].data

        value_data = block.variables[47].data[0].variables[0].data
        result.comment = block.variables[47].data[1].variables[0].data.decode("Latin-1").strip('\x00')
        result.preview = block.variables[47].data[2].variables[0].data

        x, y = result.settings.shape()
        shape = (y, x)
        try:
            result._values_original = np.reshape(value_data, shape)
            result.values = np.reshape(value_data, shape)
            result.correct_background(correction_type=self.bg_correction_type, keep_offset=self.keep_z_offset)
        except:
            result.values = None
        result.settings._pixel_width = result.settings.max_width / result.settings.columns
        result.settings._pixel_height = result.settings.max_height / result.settings.lines

        return result
