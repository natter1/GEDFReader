import io
from enum import Enum
from typing import Optional, BinaryIO, List
import struct
import numpy as np


# HEADER_SIZE = 4 + 2 + 2 + 4 + 4
# CONTROL_BLOCK_SIZE = 2 + 2 + 4 + 4 + 1 + 3
# VAR_NAME_SIZE = 50
# VARIABLE_SIZE = 50 + 4
from gdef_reader.gdef_data_strucutres import GDEFHeader, GDEFControlBlock, GDEFVariableType, GDEFVariable, type_sizes
from gdef_reader.gdef_measurement import GDEFMeasurement

from skimage.io import imsave

class GDEFImporter:
    def __init__(self, filename: str):
        self.header: GDEFHeader = GDEFHeader()
        self.buffer: Optional[BinaryIO] = None

        self.blocks: List[GDEFControlBlock] = []
        self.base_blocks: List[GDEFControlBlock] = []

        self._eof = None
        self.flow_summary = []
        self.flow_offset = ''
        self.load(filename)

    def load(self, filename: str):
        self.buffer = open(filename, 'rb')
        self._eof = self.buffer.seek(0, 2)
        self.buffer.seek(0)
        self.read_header()
        self.read_variable_lists()

    def read_header(self):
        self.flow_summary.append('read_header()')
        self.buffer.seek(0)  # sets the file's current position at the offset
        self.header.magic = self.buffer.read(4)
        self.header.version = int.from_bytes(self.buffer.read(2), 'little')
        if self.header.version != 0x0200:
            raise Exception(f"File version {self.header.version} is not supported")

        self.buffer.read(2)  # align

        self.header.creation_time = int.from_bytes(self.buffer.read(4), 'little')
        self.header.description_length = int.from_bytes(self.buffer.read(4), 'little')
        self.header.description = self.buffer.read(self.header.description_length).decode("utf-8")

    def read_control_block(self, block):
        block.mark = self.buffer.read(2)
        self.flow_summary.append(self.flow_offset + f'    read block: {block.id} - block.mark={block.mark})')
        if not block.mark == b'CB':
            file2 = open("flow_summary.txt", "w")
            file2.write(self.flow_summary)
            assert block.mark == b'CB'

        self.buffer.read(2)  # align
        block.n_variables = int.from_bytes(self.buffer.read(4), 'little')
        block.n_data = int.from_bytes(self.buffer.read(4), 'little')

        block.next_byte = self.buffer.read(1)
        self.buffer.read(3)

        return block

    def read_variable(self, variable):
        variable.name = self.buffer.read(50).decode("utf-8")
        self.buffer.read(2)
        variable.type = int.from_bytes(self.buffer.read(4), 'little')
        assert variable.type < GDEFVariableType.VAR_NVARS.value
        return variable

    def read_variable_lists(self, depth: int = 0):
        blocks = []
        self.flow_offset = ' ' * 4 * depth
        self.flow_summary.append(self.flow_offset + f'read_variable_lists(depth={depth})')
        break_flag = False

        while (not break_flag) and (self.buffer.tell() != self._eof):
            print(f"tell: {self.buffer.tell()} - eof: {self._eof}")
            block = GDEFControlBlock()
            block = self.read_control_block(block)

            if block.next_byte == b'\x00':
                break_flag = True

            # read variables
            for i in range(block.n_variables):
                variable = GDEFVariable()
                variable = self.read_variable(variable)
                self.flow_summary.append(self.flow_offset + f'        block variable {i} - {variable.name}')
                block.variables.append(variable)

                if variable.type == GDEFVariableType.VAR_DATABLOCK.value:
                    variable.data = self.read_variable_lists(depth+1)
                    self.flow_offset = ' ' * 4 * depth

            if depth == 0:
                self.flow_summary.append(self.flow_offset + f'        read variable data for block: {block.id} - (; depth={depth})')
                self.read_variable_data(block, depth)
                self.flow_offset = ' ' * 4 * depth

            self.blocks.append(block)
            if depth==0:
                self.base_blocks.append(block)
            blocks.append(block)
        self.flow_summary.append(self.flow_offset + f'return from read_variable_lists(depth={depth})')
        return blocks  # measurement.blocks

    def read_variable_data(self, block: GDEFControlBlock, depth: int):
        self.flow_offset = '        ' +  ' ' * 4 * depth
        self.flow_summary.append( self.flow_offset + f'read_variable_data(block={block.id}, depth={depth})')

        for variable in block.variables:
            if variable.type == GDEFVariableType.VAR_DATABLOCK.value:
                nestedblocks: GDEFControlBlock = variable.data
                self.flow_summary.append(self.flow_offset + f'    read variable data for nestedblocks: (n_blocks={len(nestedblocks)}; depth={depth+1})')
                for block in nestedblocks:
                    self.read_variable_data(block, depth+1)
            else:
                variable.data = self.buffer.read(block.n_data * type_sizes[variable.type])
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
                    if len(variable.data)==1:
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
                try:
                    self.flow_summary.append(self.flow_offset + f"    variable = {variable.name} - {variable.data[0]}...")
                except:
                    self.flow_summary.append(self.flow_offset + f"    variable = {variable.name} - {variable.data}")

        self.flow_offset = '        ' + ' ' * 4 * depth
        self.flow_summary.append(self.flow_offset + f'return from read_variable_data(block={block.id}, depth={depth})')

    def export_measurements(self) -> List[GDEFMeasurement]:
        """Create a list of GDEFMeasurement-Objects from imported data."""
        result = []
        for i, block in enumerate(self.blocks):
            if block.n_data!=1 or block.n_variables != 50:
                continue
            result.append(self._get_measurement_from_block(block))
        return result

    def _get_measurement_from_block(self, block: GDEFControlBlock)-> GDEFMeasurement:
        result = GDEFMeasurement()

        result.lines = block.variables[0].data
        result.columns = block.variables[1].data
        result.missing_lines = block.variables[2].data
        result.line_mean = block.variables[3].data
        result.line_mean_order = block.variables[4].data
        result.invert_line_mean = block.variables[5].data
        result._plane_corr = block.variables[6].data
        result.invert_plane_corr = block.variables[7].data
        result.max_width = block.variables[8].data
        result.max_height = block.variables[9].data
        result.offset_x = block.variables[10].data
        result.offset_y = block.variables[11].data
        result.z_unit = block.variables[12].data
        result.retrace = block.variables[13].data
        result.z_linearized = block.variables[14].data
        result.scan_mode = block.variables[15].data
        result.z_calib = block.variables[16].data
        result.x_calib = block.variables[17].data
        result.y_calib = block.variables[18].data
        result.scan_speed = block.variables[19].data
        result.set_point = block.variables[20].data
        result.bias_voltage = block.variables[21].data
        result.loop_gain = block.variables[22].data
        result.loop_int = block.variables[23].data
        result.phase_shift = block.variables[24].data
        result.scan_direction = block.variables[25].data
        result.digital_loop = block.variables[26].data
        result.loop_filter = block.variables[27].data
        result.fft_type = block.variables[28].data
        result.xy_linearized = block.variables[29].data
        result.retrace_type = block.variables[30].data
        result.calculated = block.variables[31].data
        result.scanner_range = block.variables[32].data
        result.pixel_blend = block.variables[33].data
        result.source_channel = block.variables[34].data
        result.direct_ac = block.variables[35].data
        result.id = block.variables[36].data
        result.q_factor = block.variables[37].data
        result.aux_gain = block.variables[38].data
        result.fixed_palette = block.variables[39].data
        result.fixed_min = block.variables[40].data
        result.fixed_max = block.variables[41].data
        result.zero_scan = block.variables[42].data
        result.measured_amplitude = block.variables[43].data
        result.frequency_offset = block.variables[44].data
        result.q_boost = block.variables[45].data
        result.offset_pos = block.variables[46].data

        value_data = block.variables[47].data[0].variables[0].data
        shape = (result.columns-result.missing_lines, result.lines)
        try:
            result.value = np.reshape(value_data, shape)
            imsave(f"measurment_blockID{block.id}.png", result.value)
        except:
            pass
        result.comment = block.variables[47].data[1].variables[0].data.decode("utf-8").strip('\x00')
        result.preview = block.variables[47].data[2].variables[0].data

        return result


if __name__ == '__main__':
    dummy = GDEFImporter("AFM.gdf")
    file2 = open("flow_summary.txt", "w")
    file2.write("\n".join(dummy.flow_summary))
    print(dummy)
    measurements = dummy.export_measurements()
    print(measurements)