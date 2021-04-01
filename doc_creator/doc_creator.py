"""
This python script is meant to autocreate part of the documentation. This is work in progress right now.
@author: Nathanael Jöhrmann
"""
import inspect
import re
import textwrap
from types import ModuleType
from typing import Tuple

import gdef_reader.gdef_importer as gdef_importer
from gdef_reader import gdef_measurement, gdef_sticher, gdef_indent_analyzer

module_list = [
    gdef_importer,
    gdef_indent_analyzer,
    gdef_measurement,
    gdef_sticher
]


def main():
    readme_api = ""

    for module in module_list:
        readme_api += get_module_doc(module)

    readme = get_readme_header()
    readme += readme_api
    with open('auto_readme.rst', 'w') as f:
        f.writelines(readme)
    return


def get_readme_header():
    with open('readme_header.rst') as f:
        result = f.read()
    return result


def create_header_with_line(header_text: str, line_char='-'):
    result = f"\n{header_text}\n"
    result += len(header_text) * line_char + '\n'
    return result


def create_python_block(code: str, n_indent=0):
    indent = '    '
    result = "\n.. code:: python\n\n"
    result += textwrap.indent(code, indent)
    result = textwrap.indent(result, n_indent * indent) + "\n\n"
    return result


def get_python_module_header(module: ModuleType):
    result = [create_header_with_line(f"Module {module.__name__}", '-')]

    if inspect.getdoc(module):
        return "".join(result)

    doc = inspect.getdoc(module)
    doc = re.sub(r'^@author.*\n?', '', doc, flags=re.MULTILINE)  # remove @author - line
    if doc:
        result.append(f"{doc}\n")
    return "".join(result)


def get_functions_doc(item):
    result = []
    for func in inspect.getmembers(item, inspect.isfunction):
        if func[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        result.append('\n* **' + func[0] + '**\n')  # Get the signature

        result.append(create_python_block(func[0] + str(inspect.signature(func[1])), n_indent=1))
        doc = inspect.getdoc(func[1])
        if doc:
            doc = textwrap.indent(doc, '    ')
            result.append(doc + "\n")
    return "".join(result)


def get_properties_doc(item):
    result = []
    try:
        dummy_obj = item()
    except:
        print(f"Instance of {item.__name__} could not be created. Therefore no class attributes where added to doc")
        return ""

    for attribute in inspect.getmembers(dummy_obj, lambda a: not (inspect.isroutine(a))):
        if attribute[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        result.append(f"* {attribute[0]}\n")
    return "".join(result)


def get_class_doc(cl: Tuple[str, type]) -> str:
    result = [create_header_with_line(f"class {cl[0]}", '~')]
    # print(cl[1].__name__)
    if inspect.getdoc(cl[1]):
        result.append(f"{inspect.getdoc(cl[1])}\n")

    result.append(f"\n{'**Methods:**'}\n")
    result.append(get_functions_doc(cl[1]))
    result.append(f"\n{'**Instance Variables:**'}\n\n")
    result.append(get_properties_doc(cl[1]))
    return "".join(result)


def get_module_doc(module: ModuleType):
    result = [get_python_module_header(module)]

    for cl in inspect.getmembers(module, inspect.isclass):
        if not cl[1].__module__ == module.__name__:  # skip imported classes
            continue
        if cl[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        result.append(get_class_doc(cl))

    return "".join(result)


if __name__ == "__main__":
    main()
