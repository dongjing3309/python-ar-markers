import cv2

from numpy import mean, binary_repr, zeros
from numpy.random import randint

from ar_markers.hamming.coding import encode, HAMMINGCODE_MARKER_POSITIONS

MARKER_SIZE = 7
ZOOM_RATIO = 50


class HammingMarker(object):
    def __init__(self, id, contours=None):
        self.id = id
        self.contours = contours

    def __repr__(self):
        return '<Marker id={} center={}>'.format(self.id, self.center)

    @property
    def center(self):
        if self.contours is None:
            return None
        center_array = mean(self.contours, axis=0).flatten()
        return (int(center_array[0]), int(center_array[1]))

    def generate_image(self):
        img = zeros((MARKER_SIZE, MARKER_SIZE))
        img[1, 1] = 255  # set the orientation marker
        for index, val in enumerate(self.hamming_code):
            coords = HAMMINGCODE_MARKER_POSITIONS[index]
            if val == '1':
                val = 255
            img[coords[0], coords[1]] = int(val)
        # return img
        output_img = zeros((MARKER_SIZE*ZOOM_RATIO, MARKER_SIZE*ZOOM_RATIO))
        cv2.resize(img, dsize=((MARKER_SIZE*ZOOM_RATIO, MARKER_SIZE*ZOOM_RATIO)), dst=output_img, interpolation=cv2.INTER_NEAREST)
        return output_img

    def draw_contour(self, img, color=(0, 255, 0), linewidth=2):
        cv2.drawContours(img, [self.contours], -1, color, linewidth)

    def highlite_marker(self, img, contour_color=(0, 255, 255), text_color=(255, 255, 0), linewidth=2):
        self.draw_contour(img, color=contour_color, linewidth=linewidth)
        cv2.putText(img, str(self.id), self.center, cv2.FONT_HERSHEY_DUPLEX, 1, text_color)

    @classmethod
    def generate(cls):
        return HammingMarker(id=randint(4096))

    @property
    def id_as_binary(self):
        return binary_repr(self.id, width=12)

    @property
    def hamming_code(self):
        return encode(self.id_as_binary)
