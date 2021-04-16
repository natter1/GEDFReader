"""
This python script is meant to auto-create part of the documentation (mainly the API documentation).
@author: Nathanael Jöhrmann
"""
import inspect
import re
import textwrap
from types import ModuleType, DynamicClassAttribute
from typing import Tuple

import gdef_reader.gdef_importer as gdef_importer
from afm_tools import background_correction
from gdef_reader import gdef_measurement, gdef_sticher, gdef_indent_analyzer

module_list = [
    gdef_importer,
    gdef_indent_analyzer,
    gdef_measurement,
    gdef_sticher,
    background_correction
]


def main():
    readme_api = ""
    from afm_tools.background_correction import BGCorrectionType

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


def get_methods_from_class_doc(class_item):
    result = []
    for func in inspect.getmembers(class_item, inspect.isfunction):
        if func[0].startswith("_") and not func[0] == '__init__':  # Consider anything that starts with _ private and don't document it.
            continue
        result.append('\n* **' + func[0] + '**\n')  # Get the signature

        result.append(create_python_block(func[0] + str(inspect.signature(func[1])), n_indent=1))
        doc = inspect.getdoc(func[1])
        if doc:
            doc = doc.replace(':param ', '\n:').replace(':return:', '\n:return:')
            doc = textwrap.indent(doc, '    ')
            result.append(doc + "\n")
    if len(result) > 0:
        result.insert(0, f"\n{'**Methods:**'}\n")
    return result  # "".join(result)


def get_class_attributes_doc(item):
    result = []
    for attribute in inspect.getmembers(item, lambda a: not (inspect.isroutine(a))):
        if attribute[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue

        if isinstance(attribute[1], property):
            continue

        if type(attribute[1]) == DynamicClassAttribute:  # e.g. skip 'nme' and 'value' for Enum
            print(attribute[0])
            continue

        result.append(f"* {attribute[0]}\n")
    if len(result) > 0:
        result.insert(0, f"\n{'**Class Attributes:**'}\n\n")
    return result  # "".join(result)


def get_class_instance_attributes_doc(class_object):
    result = []
    try:
        class_instance = class_object()
    except:
        print(f"Instance of {class_object.__name__} could not be created. Therefore no instance attributes where added to doc")
        return result
    # todo: write function to read __init__ doc string and extract :param entries -> add it to matching attribute
    attributes_text = class_object.__doc__.partition(":InstanceAttributes:")[2].partition(":EndInstanceAttributes:")[0]
    attributes_text = list(inspect.cleandoc(attributes_text).split("\n"))  # list(attributes_text.split("\n"))

    print(attributes_text)
    for attribute in inspect.getmembers(class_instance, lambda a: not (inspect.isroutine(a))):
        if attribute[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        attribute_description = ""
        for s in attributes_text:
            if s.startswith(attribute[0]):
                attribute_description = s.partition(attribute[0])[2]

        result.append(f"* {attribute[0]}{attribute_description}\n")
    if len(result) > 0:
        result.insert(0, f"\n{'**Instance Attributes:**'}\n\n")
    return result  # "".join(result)


def get_cleaned_class_doc_string(class_type):
    result = ""
    if inspect.getdoc(class_type):
        result = str(f"{inspect.getdoc(class_type)}\n")
        result = result.partition(":InstanceAttributes:")[0] + result.partition(":EndInstanceAttributes:")[2]
        # result.append(f"{inspect.getdoc(class_type)}\n")
    return result


def get_class_doc(cl: Tuple[str, type]) -> str:
    result = [create_header_with_line(f"class {cl[0]}", '~')]

    # # print(cl[1].__name__)
    # if inspect.getdoc(cl[1]):
    #     result.append(f"{inspect.getdoc(cl[1])}\n")

    result.append(get_cleaned_class_doc_string(cl[1]))

    result.extend(get_class_attributes_doc(cl[1]))
    result.extend(get_methods_from_class_doc(cl[1]))
    result.extend(get_class_instance_attributes_doc(cl[1]))
    return "".join(result)

#
#
# def is_mod_function(mod, func):
#     ' checks that func is a function defined in module mod '
#     return inspect.isfunction(func) and inspect.getmodule(func) == mod

def get_module_doc(module: ModuleType):
    result = [get_python_module_header(module)]

    for func in inspect.getmembers(module, inspect.isfunction):
        # print(f"module: {module}")
        # print(f"func: {func}")
        # print(f"get_module: {inspect.getmodule(func)}")
        if func[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        if inspect.getmodule(func[1]) != module:
            continue
        result.append(get_methods_from_class_doc(func))

    for cl in inspect.getmembers(module, inspect.isclass):
        if not cl[1].__module__ == module.__name__:  # skip imported classes
            continue
        if cl[0].startswith("_"):  # Consider anything that starts with _ private and don't document it.
            continue
        result.append(get_class_doc(cl))

    return "".join(filter(None, result))  # filter out empty entries []


if __name__ == "__main__":
    main()
