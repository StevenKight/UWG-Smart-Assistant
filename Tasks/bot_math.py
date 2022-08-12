"""
This code is a set of definitions for math expressions that are used in the
tasks.py file to be able to give a response to math questions.

Pylint: 10.00 (August 11, 2022)
"""

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"

def addlist(mylist):
    """
    Add a list of numbers.

    :param mylist: A list of numerical values.
    :return: The sum of all numbers in given list.
    """

    result = 0
    for var in mylist:
        result += var
    return result

def multiplylist(mylist):
    """
    Multiply a list of numbers.

    :param mylist: A list of numerical values.
    :return: The product of all numbers in given list.
    """

    result = 1
    for var in mylist:
        result = result * var
    return result

def subtractlist(mylist):
    """
    Subtract a list of numbers.

    :param mylist: A list of numerical values.
    :return: The difference of all numbers in given list.
    """

    result = 0
    for var in mylist:
        result = result - var
    return result

def dividelist(mylist):
    """
    Divides a list of numbers.

    :param mylist: A list of numerical values.
    :return: The quotient of all numbers in given list.
    """

    result = 1
    for var in mylist:
        result = result / var
    return result

def circumfrance(radius):
    """
    Gets the circumfrance of a circle using radius.

    :param radius: A number representing the radius of a circle.
    :return: The circumfrance of a circle of the inputed radius.
    """

    circumference = 2 * 3.1415 * radius

    return circumference

def areaofcircle(radius):
    """
    Gets the area of a circle using radius.

    :param radius: A number representing the radius of a circle.
    :return: The area of a circle of the inputed radius.
    """

    area = 3.1415 * (radius * radius)

    return area

def perimeterofsquare(length, width):
    """
    Gets the perimeter of a rectangle using length and width.

    :param length: A numerical value representing the length of a rectangle.
    :param width: A numerical value representing the width of a rectangle.
    :return: The parimeter of a rectangle of inputed size.
    """

    perimeter = (2 * length) + (2 * width)

    return perimeter

def areaofsquare(length, width):
    """
    Gets the area of a rectangle using length and width.

    :param length: A numerical value representing the length of a rectangle.
    :param width: A numerical value representing the width of a rectangle.
    :return: The area of a rectangle of inputed size.
    """

    area = length * width

    return area

def volumeofsquare(length, width, height):
    """
    Gets the volume of a rectangular prism using length, width, and height.

    :param length: A numerical value representing the length of a rectangle.
    :param width: A numerical value representing the width of a rectangle.
    :param width: A numerical value representing the height of a prism.
    :return: The volume of a prism of inputed size.
    """

    volume = length * width * height

    return volume

def volumeofcylinder(radius, height):
    """
    Gets the volume of a cylinder using radius and height

    :param radius: A numerical value representing the radius of a circle.
    :param height: A numerical value representing the height of the cylinder.
    :return: The volume of a cylinder of inputed size.
    """

    volume = areaofcircle(radius) * height

    return volume
