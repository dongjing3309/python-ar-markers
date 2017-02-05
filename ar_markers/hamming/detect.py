import cv2

from numpy import array, rot90
import numpy as np
from math import fabs

from ar_markers.hamming.coding import decode, extract_hamming_code
from ar_markers.hamming.marker import MARKER_SIZE, HammingMarker

BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]

WARPED_SIZE = 49


def rotate_contour(contour, persp_transf, rot_num):

    # contour_first_warpped must at wrapped origin since getPerspectiveTransform use contour
    rot_num_con = 0
    '''
    # warp contour[0] by persp_transf
    contour_first_warpped = np.dot(persp_transf, np.append(contour[0], np.array([1.0])))
    contour_first_warpped = [contour_first_warpped[0] / contour_first_warpped[2], \
        contour_first_warpped[1] / contour_first_warpped[2]]

    # check warpped point, decide rotation number
    warp_err_thesh = 1e-3

    if fabs(contour_first_warpped[0]) < warp_err_thesh and \
        fabs(contour_first_warpped[1] ) < warp_err_thesh:
        rot_num_con = 0
    elif fabs(contour_first_warpped[0]) < warp_err_thesh and \
        fabs(contour_first_warpped[1] - (WARPED_SIZE-1)) < warp_err_thesh:
        rot_num_con = 1
    elif fabs(contour_first_warpped[0] - (WARPED_SIZE-1)) < warp_err_thesh and \
        fabs(contour_first_warpped[1] - (WARPED_SIZE-1)) < warp_err_thesh:
        rot_num_con = 2
    elif fabs(contour_first_warpped[0] - (WARPED_SIZE-1)) < warp_err_thesh and \
        fabs(contour_first_warpped[1]) < warp_err_thesh:
        rot_num_con = 3
    else:
        raise ValueError('Warp err in rotate_contour.')
    '''

    # plus rot_num, rotate contour
    # note that get_turn_number gives conter-clock, we use clock here
    contour_list = contour.tolist();
    for i in range(rot_num_con + (4-rot_num)):
        contour_list.insert(0, contour_list.pop())

    return np.array(contour_list, dtype='int32')


def validate_and_get_turn_number(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')

    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3

    return rotation


def detect_markers(img):
    width, height, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 10, 100)
    #cv2.imshow("edges", edges)
    #cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    # We only keep the long enough contours
    min_contour_length = min(width, height) / 50
    contours = [contour for contour in contours if len(contour) > min_contour_length]
    WARPED_SIZE = 49
    canonical_marker_coords = array(((0, 0),
                                     (WARPED_SIZE - 1, 0),
                                     (WARPED_SIZE - 1, WARPED_SIZE - 1),
                                     (0, WARPED_SIZE - 1)),
                                     dtype='float32')

    markers_list = []
    for contour in contours:
        approx_curve = cv2.approxPolyDP(contour, len(contour) * 0.01, True)
        if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)):
            continue

        sorted_curve = array(cv2.convexHull(approx_curve, clockwise=False),
                             dtype='float32')
        persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
        warped_img = cv2.warpPerspective(img, persp_transf, (WARPED_SIZE, WARPED_SIZE))
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

        _, warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
        marker = warped_bin.reshape(
            [MARKER_SIZE, WARPED_SIZE / MARKER_SIZE, MARKER_SIZE, WARPED_SIZE / MARKER_SIZE]
        )
        marker = marker.mean(axis=3).mean(axis=1)
        marker[marker < 127] = 0
        marker[marker >= 127] = 1

        try:
            # rotate marker by checking which corner is white
            turn_number = validate_and_get_turn_number(marker)
            marker = rot90(marker, k=turn_number)

            #cv2.imshow("bin", warped_bin)
            #cv2.waitKey(10)
            #cv2.imshow("warped_marker", rot90(warped_bin, k=turn_number))
            #cv2.waitKey(10)
            
            # get id
            hamming_code = extract_hamming_code(marker)
            marker_id = int(decode(hamming_code), 2)

        except ValueError:
            continue

        # rotate corner list
        rotated_contour = rotate_contour(sorted_curve, persp_transf, turn_number)
        markers_list.append(HammingMarker(id=marker_id, contours=rotated_contour))

    return markers_list
