"""
This code is a set of definitions for math expressions that are
used to be able to give a response to math questions.

Pylint: 10.00 (August 25, 2022)
"""

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"

def add_list(mylist: list):
    """
    Add a list of numbers.

    ### Parameters
        `mylist` (`list`): A list of numerical values.
    ### Return
        The sum of all numbers in given list.
    """

    result = 0
    for var in mylist:
        result += var
    return result

def multiply_list(mylist: list):
    """
    Multiply a list of numbers.

    ### Parameters
        `mylist` (`list`): A list of numerical values.
    ### Return
        The product of all numbers in given list.
    """

    result = 1
    for var in mylist:
        result = result * var
    return result

def subtract_list(mylist: list):
    """
    Subtract a list of numbers.

    ### Parameters
        `mylist` (`list`): A list of numerical values.
    ### Return
        The difference of all numbers in given list.
    """

    result = 0
    for var in mylist:
        result = result - var
    return result

def divide_list(mylist: list):
    """
    Divides a list of numbers.

    ### Parameters
        `mylist` (`list`): A list of numerical values.
    ### Return
        The quotient of all numbers in given list.
    """

    result = 1
    for var in mylist:
        result = result / var
    return result

def circumfrance(radius):
    """
    Gets the circumfrance of a circle using radius.

    ### Parameters
        `radius`: A number representing the radius of a circle.
    ### Return
        The circumfrance of a circle of the inputed radius.
    """

    circumference = 2 * 3.1415 * radius

    return circumference

def area_of_circle(radius):
    """
    Gets the area of a circle using radius.

    ### Parameters
        `radius`: A number representing the radius of a circle.
    ### Return
        The area of a circle of the inputed radius.
    """

    area = 3.1415 * (radius * radius)

    return area

def perimeter_of_square(length, width):
    """
    Gets the perimeter of a rectangle using length and width.

    ### Parameters
        `length`: A numerical value representing the length of a rectangle.
        `width`: A numerical value representing the width of a rectangle.
    ### Return
        The parimeter of a rectangle of inputed size.
    """

    perimeter = (2 * length) + (2 * width)

    return perimeter

def area_of_square(length, width):
    """
    Gets the area of a rectangle using length and width.

    ### Parameters
        `length`: A numerical value representing the length of a rectangle.
        `width`: A numerical value representing the width of a rectangle.
    ### Return
        The area of a rectangle of inputed size.
    """

    area = length * width

    return area

def volume_of_square(length, width, height):
    """
    Gets the volume of a rectangular prism using length, width, and height.

    ### Parameters
        `length`: A numerical value representing the length of a rectangle.
        `width`: A numerical value representing the width of a rectangle.
        `height`: A numerical value representing the height of a prism.
    ### Return
        The volume of a prism of inputed size.
    """

    volume = length * width * height

    return volume

def volume_of_cylinder(radius, height):
    """
    Gets the volume of a cylinder using radius and height

    ### Parameters
        `radius`: A numerical value representing the radius of a circle.
        `height`: A numerical value representing the height of the cylinder.
    ### Return
        The volume of a cylinder of inputed size.
    """

    volume = area_of_circle(radius) * height

    return volume
