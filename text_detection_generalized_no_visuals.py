import easyocr
import cv2
import numpy as np
from math import cos, sin

def load_image(image_path):
    """Load an image from a file path and add padding to make it square."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    if h == w:
        return image, (h, w)
    size = max(h, w)
    padded_image = cv2.copyMakeBorder(image, 
                                      top=(size - h) // 2, 
                                      bottom=(size - h + 1) // 2, 
                                      left=(size - w) // 2, 
                                      right=(size - w + 1) // 2, 
                                      borderType=cv2.BORDER_CONSTANT, 
                                      value=[255, 255, 255])
    return padded_image, (h, w)

def create_ocr_reader(languages):
    """Create an OCR reader object."""
    return easyocr.Reader(languages) # set gpu=False if you want to use CPU

def extract_text_from_image(reader, image, text_threshold=0.7):
    """Extract text from an image using the OCR reader."""
    return reader.readtext(image, text_threshold=text_threshold)

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def rotate_point(point, angle, center):
    """Rotate a point around a center by a given angle."""
    angle_rad = -angle * (3.14159265 / 180.0)
    ox, oy = center
    px, py = point

    qx = ox + (px - ox) * cos(angle_rad) - (py - oy) * sin(angle_rad)
    qy = oy + (px - ox) * sin(angle_rad) + (py - oy) * cos(angle_rad)
    return [int(qx), int(qy)]

def adjust_and_combine_results(original_result, rotated_result, image_shape, angle):
    """Adjust the coordinates of the rotated text and combine results."""
    (h, w) = image_shape[:2]
    center = (w / 2, h / 2)
    for detection in rotated_result:
        rotated_bbox = [rotate_point(point, -angle, center) for point in detection[0]]
        original_result.append([rotated_bbox, detection[1], detection[2]])
    return original_result

def filter_short_texts(detections, min_length=3): # min_length>=3, otherwise false positives
    """Filter out text detections that are shorter than the specified minimum length."""
    return [detection for detection in detections if len(detection[1]) >= min_length]

def filter_forbidden_chars(detections, forbidden_chars):
    """Filter out text detections that are only one letter and are in the forbidden characters list."""
    return [detection for detection in detections if not (len(detection[1]) == 1 and detection[1] in forbidden_chars)]

# Execution
target_image_path = 'images/testcase10.png' # testcase5.png good for vertical text, testcase3.png good for horizontal text
target_img, original_shape = load_image(target_image_path)
easyocr_img = target_img.copy()
forbidden_chars = ['<','.','>',',','/',';',':','\'','"','[',']','{','}','|','\\','`','~','!','^','(',')','_','-','+','=']

reader = create_ocr_reader(['en'])
result = extract_text_from_image(reader, easyocr_img, text_threshold=0.0)

# Rotate the image by 90 degrees counterclockwise and search for text
rotated_img = rotate_image(easyocr_img, 90)
rotated_result = extract_text_from_image(reader, rotated_img, text_threshold=0.0)

# Adjust the coordinates of the rotated text and combine results
result = adjust_and_combine_results(result, rotated_result, easyocr_img.shape, 90)

# Filter out short texts from the results
result = filter_short_texts(result)

# Filter out forbidden characters from the results
result = filter_forbidden_chars(result, forbidden_chars)

import matplotlib.pyplot as plt

def draw_text_on_image(image, detections):
    """Draw the detected text on the image."""
    for detection in detections:
        bbox, text, _ = detection
        bbox = [tuple(point) for point in bbox]
        cv2.polylines(image, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw the text on the image
draw_text_on_image(target_img, result)

# Display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()