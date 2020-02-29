import io
from enum import Enum
from typing import Optional, BinaryIO, List
import struct


# HEADER_SIZE = 4 + 2 + 2 + 4 + 4
# CONTROL_BLOCK_SIZE = 2 + 2 + 4 + 4 + 1 + 3
# VAR_NAME_SIZE = 50
# VARIABLE_SIZE = 50 + 4


class GDEFVariableType(Enum):
    VAR_INTEGER = 0
    VAR_FLOAT = 1
    VAR_DOUBLE = 2
    VAR_WORD = 3
    VAR_DWORD = 4
    VAR_CHAR = 5
    VAR_STRING = 6
    VAR_DATABLOCK = 7
    VAR_NVARS = 8  # number of known GDEF types


type_sizes = [4, 4, 8, 2, 4, 1, 0, 0]


class GDEFHeader:
    def __init__(self):
        self.magic = None
        self.version = None
        self.creation_time = None
        self.description_length = None
        self.description = None


class GDEFControlBlock:
    _counter = 0

    def __init__(self):
        GDEFControlBlock._counter += 1
        self.id = GDEFControlBlock._counter
        self.mark = None
        self.n_variables = None
        self.n_data = None

        self.variables: List[GDEFVariable] = []
        self.next_byte = None


class GDEFVariable:
    def __init__(self):
        self.name: str = ''
        self.type: Optional[GDEFVariableType] = None
        self.size = None
        self.data = None


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
        return blocks  # self.blocks

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


if __name__ == '__main__':
    dummy = GDEFImporter("AFM.gdf")
    file2 = open("flow_summary.txt", "w")
    file2.write("\n".join(dummy.flow_summary))
    print(dummy)
