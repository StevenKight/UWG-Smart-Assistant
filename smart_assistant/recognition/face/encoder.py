"""
Code from the facial_recognition api that utilizes dlib to create encodings for
faces and to find distances between faces for recognition.

Code contains a number of personal changes to alter the speed of recogition.

Pylint: 10.00 (August 25, 2022)
E1101 disabled because pylint cannot find dlib modules
"""

# pylint: disable=E1101

from pkg_resources import resource_filename

import PIL.Image
import dlib
import numpy as np

__author__ = "Adam Geitgey"
__version__ = "1.5"
__pylint__ = "2.14.4"

face_detector = dlib.get_frontal_face_detector()

PREDICTOR_68_POINT_FILE = "Models/Dlib/shape_predictor_68_face_landmarks.dat"
predictor_68_point_model = resource_filename(__name__, PREDICTOR_68_POINT_FILE)
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

PREDICTOR_5_POINT_FILE = "Models/Dlib/shape_predictor_5_face_landmarks.dat"
predictor_5_point_model = resource_filename(__name__, PREDICTOR_5_POINT_FILE)
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

CNN_DETECTION_FILE = "Models/Dlib/mmod_human_face_detector.dat"
cnn_face_detection_model = resource_filename(__name__, CNN_DETECTION_FILE)
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

RECOGNITION_MODEL_FILE = "Models/Dlib/dlib_face_recognition_resnet_model_v1.dat"
face_recognition_model = resource_filename(__name__, RECOGNITION_MODEL_FILE)
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces.
            Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs.
            "cnn" is a more accurate deep-learning model that is GPU/CUDA accelerated (if available)
            The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)

    return face_detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image, face_locations=None, model="small"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels)
                and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    image = PIL.Image.open(file)
    if mode:
        image = image.convert(mode)
    return np.array(image)


def face_locations_frame(img, number_of_times_to_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces.
                                Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs.
                "cnn" is a more accurate deep-learning model which is
                GPU/CUDA accelerated (if available).
                The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        landmarks = _raw_face_locations(img, number_of_times_to_upsample, "cnn")
        trim = [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in landmarks]
        return trim

    landmarks = _raw_face_locations(img, number_of_times_to_upsample, model)
    trim = [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in landmarks]
    return trim


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional- the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding.
                    Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use.
                "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)

    array = []
    for raw_landmark_set in raw_landmarks:
        encoded = face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)
        array.append(np.array(encoded))

    return array


def face_distance(face_encoding_list, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean
    distance for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encoding_list) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encoding_list - face_to_compare, axis=1)
