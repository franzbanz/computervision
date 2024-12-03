import cv2
import numpy as np

def load_and_prepare_images(target_image_path, device_template_path, subtask_template_path):
    """Loads and resizes the target, device template, and subtask template images."""
    target_img = cv2.imread(target_image_path)
    device_template = cv2.imread(device_template_path)
    subtask_template = cv2.imread(subtask_template_path)
    device_template = cv2.resize(device_template, (180, 200))
    subtask_template = cv2.resize(subtask_template, (80, 80))
    return target_img, device_template, subtask_template

def generate_templates(device_template):
    """Generates various resized templates from the device template."""
    templates = {
        'original': device_template,
        'bottom': cv2.resize(device_template[5:, 5:-5], (180, 200)),
        'top': cv2.resize(device_template[:-5, 5:-5], (180, 200)),
        'left': cv2.resize(device_template[5:-5, :-5], (180, 200)),
        'right': cv2.resize(device_template[5:-5, 5:], (180, 200)),
        'topleftcorner': cv2.resize(device_template[:-5, :-5], (180, 200)),
        'toprightcorner': cv2.resize(device_template[:-5, 5:], (180, 200)),
        'rightbottomcorner': cv2.resize(device_template[5:, 5:], (180, 200)),
        'leftbottomcorner': cv2.resize(device_template[5:, :-5], (180, 200))
    }
    return templates

def find_subtasks(target_img, subtask_template, threshold=0.8):
    """Finds subtasks in the target image using template matching."""
    res = cv2.matchTemplate(target_img, subtask_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    subtasks = [(int(pt[0]), int(pt[1])) for pt in zip(*loc[::-1])]
    return subtasks

def draw_grey_boxes(target_img, subtasks, subtask_template):
    """Draws grey boxes around detected subtasks in the target image."""
    for pt in subtasks:
        top_left = (pt[0] - 10, pt[1] - 10)
        bottom_right = (pt[0] + subtask_template.shape[1] + 10, pt[1] + subtask_template.shape[0] + 10)
        cv2.rectangle(target_img, top_left, bottom_right, (191, 191, 191), -1)
    return target_img

def find_devices(target_img, templates, threshold=0.6):
    """Finds devices in the target image using multiple templates."""
    res = np.zeros(target_img.shape[:2], dtype=np.float32)
    for key, template in templates.items():
        temp_res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        pad_height = target_img.shape[0] - temp_res.shape[0]
        pad_width = target_img.shape[1] - temp_res.shape[1]
        temp_res_padded = np.pad(temp_res, ((0, pad_height), (0, pad_width)), 'constant', constant_values=0)
        res = np.maximum(res, temp_res_padded)
    loc = np.where(res >= threshold)
    return loc, res

def filter_foreground_devices(loc, device_template, target_gray):
    """Filters out background devices based on the grayscale target image."""
    _, target_thresh = cv2.threshold(target_gray, 245, 255, cv2.THRESH_BINARY_INV)
    loc_foreground = [pt for pt in zip(*loc[::-1]) if np.all(target_thresh[pt[1]:pt[1] + device_template.shape[0], pt[0]:pt[0] + device_template.shape[1]] == 255)]
    return loc_foreground

def find_devices_bounding_boxes(loc_foreground, device_template):
    """Finds bounding boxes for detected devices."""
    matched_templates_bbox_filled = np.zeros((max(pt[1] + device_template.shape[0] for pt in loc_foreground), 
                                                max(pt[0] + device_template.shape[1] for pt in loc_foreground)), dtype=np.uint8)
    for pt in loc_foreground:
        cv2.rectangle(matched_templates_bbox_filled, pt, (pt[0] + device_template.shape[1], pt[1] + device_template.shape[0]), 255, -1)
    contours, _ = cv2.findContours(matched_templates_bbox_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    devices = [(x, y, w, h) for cnt in contours for x, y, w, h in [cv2.boundingRect(cnt)]]
    return devices

# Define the paths
target_image_path = 'images/testcase3.png'
device_template_path = 'images/device.png'
subtask_template_path = 'images/subtask.png'

# Load and prepare images
target_img, device_template, subtask_template = load_and_prepare_images(target_image_path, device_template_path, subtask_template_path)
# Create a copy of the target image to draw devices and subtasks
final_image = target_img.copy()

# Generate templates
templates = generate_templates(device_template)

# Find and draw subtasks
subtasks = find_subtasks(target_img, subtask_template)
target_img = draw_grey_boxes(target_img, subtasks, subtask_template)
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# Find devices
loc, res = find_devices(target_img, templates)

# Filter foreground devices
loc_foreground = filter_foreground_devices(loc, templates['original'], target_gray)

# Save bounding boxes
devices = find_devices_bounding_boxes(loc_foreground, templates['original'])