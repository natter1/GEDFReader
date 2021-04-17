"""
@author: Nathanael Jöhrmann
"""
from pptx.shapes.autoshape import Shape
from pptx_tools.font_style import PPTXFontStyle
from pptx_tools.position import PPTXPosition
from pptx_tools.table_style import PPTXTableStyle


def default_font():
    result = PPTXFontStyle()
    result.size = 14
    return result


def position_2x2_00():
    return PPTXPosition(0.02, 0.115)


def position_2x2_10():
    result = position_2x2_00()
    result.left_rel = 0.355
    return result


def position_2x2_01():
    result = position_2x2_00()
    result.top_rel = 0.62
    return result


def position_2x2_11():
    result = position_2x2_00()
    result.top_rel = position_2x2_01().top_rel
    result.left_rel = position_2x2_10().left_rel
    return result


def summary_table() -> PPTXTableStyle:
    result = PPTXTableStyle()
    result.font_style = default_font()

    result.first_row_header = False
    result.row_banding = True
    result.col_banding = False

    result.set_width_as_fraction(0.33)
    result.col_ratios = [1, 1.15]

    return result


def minimize_table_height(table_shape: Shape):  # todo: make part of PPTXTableStyle
    for row in table_shape.table.rows:
        row.height = 1


# def table_style_summary():
#     result = PPTXTableStyle()
#     result.position = PPTXPosition(0.44, 0.17)
#     result.set_width_as_fraction(0.54)
#     result.col_ratios = [3.95, 1.05, 1, 1]
#     return result
