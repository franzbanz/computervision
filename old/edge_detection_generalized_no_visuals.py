import cv2
import numpy as np
from matplotlib import pyplot as plt

# Input
target_image_path = 'images/testcase2.png'
target_img = cv2.imread(target_image_path)
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
print(f'Target image shape: {target_img.shape}')

# Define custom kernels
kernel_vertical = np.tile(np.array([[1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 1, 1]], dtype=np.float32), (7, 1))
kernel_horizontal = np.transpose(kernel_vertical)
kernel_cross = np.ones((12, 12))
kernel_cross[3:9, 0:12] = 0.5
kernel_cross[0:12, 3:9] = 0.5
kernel_cross[4:8, 0:12] = 0
kernel_cross[0:12, 4:8] = 0

# Apply custom filter to find vertical and horizontal edges
result_filter_ver = cv2.filter2D(target_gray, cv2.CV_32F, kernel_vertical)
result_filter_hor = cv2.filter2D(target_gray, cv2.CV_32F, kernel_horizontal)
print(f'Vertical edges: min={result_filter_ver.min()}, max={result_filter_ver.max()}')
print(f'Horizontal edges: min={result_filter_hor.min()}, max={result_filter_hor.max()}')

# Combine the individual results into single arrays
combined_hor = result_filter_hor
combined_ver = result_filter_ver

# Normalize to 0-255 and convert to uint8
result_filter_ver_norm = cv2.normalize(result_filter_ver, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
result_filter_hor_norm = cv2.normalize(result_filter_hor, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Thresholding and visualization
threshold = 230
_, result_filter_ver_thresh = cv2.threshold(result_filter_ver_norm, threshold, 255, cv2.THRESH_BINARY)
_, result_filter_hor_thresh = cv2.threshold(result_filter_hor_norm, threshold, 255, cv2.THRESH_BINARY)
print(f'Vertical edges thresholded at {threshold} to {result_filter_ver_thresh.sum()} pixels')

# Find contours
contours_hor, _ = cv2.findContours(result_filter_hor_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_ver, _ = cv2.findContours(result_filter_ver_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'Found {len(contours_hor)} horizontal contours, and {len(contours_ver)} vertical contours')

# Filter out very short contours
min_contour_length = 20
contours_hor_filtered = [c for c in contours_hor if cv2.arcLength(c, False) > min_contour_length]
contours_ver_filtered = [c for c in contours_ver if cv2.arcLength(c, False) > min_contour_length]
print(f'After filtering, found {len(contours_hor_filtered)} horizontal contours, and {len(contours_ver_filtered)} vertical contours.')

# Convert contours to line segments
def vertical_contours_to_vertical_segments(contours):
    lines = []
    for contour in contours:
        # Simplify contour to ensure it's processed as a line if it's not already
        simplified_contour = cv2.approxPolyDP(contour, epsilon=0.5, closed=False)
        # Extract the vertical extents by finding min and max y coordinates
        x_coords = simplified_contour[:, :, 0]
        y_coords = simplified_contour[:, :, 1]
        y1, y2 = np.min(y_coords), np.max(y_coords)
        x = np.mean(x_coords).astype(int)  # Assuming vertical, x should be constant
        lines.append(((x, y1), (x, y2)))
    return lines
def horizontal_contours_to_horizontal_segments(contours):
    lines = []
    for contour in contours:
        # Simplify contour to ensure it's processed as a line if it's not already
        simplified_contour = cv2.approxPolyDP(contour, epsilon=0.5, closed=False)
        # Extract the horizontal extents by finding min and max x coordinates
        x_coords = simplified_contour[:, :, 0]
        y_coords = simplified_contour[:, :, 1]
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y = np.mean(y_coords).astype(int)  # Assuming horizontal, y should be constant
        lines.append(((x1, y), (x2, y)))
    return lines
line_segments_ver = vertical_contours_to_vertical_segments(contours_ver_filtered)
line_segments_hor = horizontal_contours_to_horizontal_segments(contours_hor_filtered)

# Detect containers
containers_on_target_img = target_img.copy()
container = cv2.imread('images/container.png')
container = cv2.resize(container, (150, 150), interpolation=cv2.INTER_AREA)
threshold = 0.8
# Create a mask for the container with reversed black and white
container_mask = np.ones(container.shape[:2], dtype=np.uint8) * 255
cv2.rectangle(container_mask, (20, 20), (container.shape[1] - 20, container.shape[0] - 20), 0, thickness=-1)
# Perform template matching with the mask
res = cv2.matchTemplate(containers_on_target_img, container, cv2.TM_CCOEFF_NORMED, mask=container_mask)
loc = np.where(res >= threshold)
containers = [(pt[0] + container.shape[1] // 2, pt[1] + container.shape[0] // 2) for pt in zip(*loc[::-1])]
print(f'Found {len(containers)} containers')

# Remove close containers
def remove_close_points(containers, proximity=5):
    containers = sorted(containers, key=lambda pt: (pt[0], pt[1]))
    filtered_containers = []
    while containers:
        current = containers.pop(0)
        filtered_containers.append(current)
        containers = [pt for pt in containers if abs(pt[0] - current[0]) > proximity or abs(pt[1] - current[1]) > proximity]
    return filtered_containers
containers = remove_close_points(containers, proximity=5)
print(f'After removing close containers, {len(containers)} containers remain')

# Connect lines at containers
def connect_lines_at_containers(containers, line_segments_hor, tolerance=5):
    def find_closest_lines(container, line_segments_hor, tolerance=5):
        x, y = container
        left_lines = sorted([seg for seg in line_segments_hor if seg[1][0] < x and abs(seg[1][1] - y) <= tolerance], key=lambda seg: abs(seg[1][0] - x))[:1]
        right_lines = sorted([seg for seg in line_segments_hor if seg[0][0] > x and abs(seg[0][1] - y) <= tolerance], key=lambda seg: abs(seg[0][0] - x))[:1]
        if not left_lines or not right_lines:
            return None, None
        closest_left = left_lines[0]
        closest_right = right_lines[0]
        return closest_left, closest_right
    def merge_lines(closest_left, closest_right):
        merged_hor = [(closest_left[0][0], closest_right[1][0])]
        return merged_hor
    merged_lines = []
    for container in containers:
        closest_left, closest_right = find_closest_lines(container, line_segments_hor, tolerance)
        if closest_left is None or closest_right is None:
            continue  # Skip if there are not enough lines to merge
        merged_hor = merge_lines(closest_left, closest_right)
        line_segments_hor.remove(closest_left)
        line_segments_hor.remove(closest_right)
        new_hor_line = ((merged_hor[0][0], container[1]), (merged_hor[0][1], container[1]))
        line_segments_hor.append(new_hor_line)
        merged_lines.append(new_hor_line)
    return line_segments_hor, merged_lines
line_segments_hor, merged_lines = connect_lines_at_containers(containers, line_segments_hor, tolerance=5)
print(f'containers were connected')

# Detect intersections
cross = cv2.imread('images/cross.png')
res = cv2.matchTemplate(target_img, cross, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
loc = (loc[0] + 6, loc[1] + 6)
intersections = list(zip(*loc[::-1]))
print(f'Found {len(intersections)} intersections')
intersections = remove_close_points(intersections, proximity=5)
print(f'After removing close intersections, {len(intersections)} intersections remain')

# Connect lines at intersections
def connect_lines_at_intersections(intersections, line_segments_ver, line_segments_hor, tolerance=5):
    merged_lines = []
    for intersection in intersections:
        x, y = intersection
        closest_ver = sorted(line_segments_ver, key=lambda seg: min(abs(seg[0][0] - x), abs(seg[1][0] - x)) + tolerance)[:2]
        closest_hor = sorted(line_segments_hor, key=lambda seg: min(abs(seg[0][1] - y), abs(seg[1][1] - y)) + tolerance)[:2]
        if len(closest_ver) < 2 or len(closest_hor) < 2:
            continue
        merged_ver = (min(closest_ver[0][0][1], closest_ver[1][0][1]), max(closest_ver[0][1][1], closest_ver[1][1][1]))
        merged_hor = (min(closest_hor[0][0][0], closest_hor[1][0][0]), max(closest_hor[0][1][0], closest_hor[1][1][0]))
        for line in closest_ver:
            line_segments_ver.remove(line)
        for line in closest_hor:
            line_segments_hor.remove(line)
        new_ver_line = ((x, merged_ver[0]), (x, merged_ver[1]))
        new_hor_line = ((merged_hor[0], y), (merged_hor[1], y))
        line_segments_ver.append(new_ver_line)
        line_segments_hor.append(new_hor_line)
        merged_lines.append((new_ver_line, new_hor_line))
    return line_segments_ver, line_segments_hor, merged_lines
line_segments_ver, line_segments_hor, merged_lines = connect_lines_at_intersections(intersections, line_segments_ver, line_segments_hor, tolerance=5)
print(f'intersections were connected')

# Convert segment chains to polylines
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
line_segments = line_segments_ver + line_segments_hor
max_distance = 20
line_segment_chains = []
used_segments = []
for segment1 in line_segments:
    if segment1 in used_segments:
        continue
    chain = [segment1]
    used_segments.append(segment1)
    chain_growing = True
    while chain_growing:
        chain_growing = False
        for segment2 in line_segments:
            if segment2 in used_segments:
                continue
            for chain_segment in chain:
                if (distance(chain_segment[0], segment2[0]) <= max_distance or
                    distance(chain_segment[0], segment2[1]) <= max_distance or
                    distance(chain_segment[1], segment2[0]) <= max_distance or
                    distance(chain_segment[1], segment2[1]) <= max_distance):
                    chain.append(segment2)
                    used_segments.append(segment2)
                    chain_growing = True
                    break
            if chain_growing:
                break
    line_segment_chains.append(chain)
print(f'Found {len(line_segment_chains)} line segment chains')

# Find endpoints of chains
endpoints_of_chains = []
for chain in line_segment_chains:
    endpoints = []
    for segment in chain:
        for point in segment:
            if all(distance(point, other_point) > max_distance for other_segment in chain for other_point in other_segment if point != other_point):
                endpoints.append(point)
    endpoints_of_chains.append(endpoints)
print(f'Found {sum(len(endpoints) for endpoints in endpoints_of_chains)} endpoints')

# Sort chains
sorted_chains = []
for i, chain in enumerate(line_segment_chains):
    if not endpoints_of_chains[i]:
        sorted_chains.append(chain)
        continue
    start_point = endpoints_of_chains[i][0]
    sorted_chain = [start_point]
    remaining_segments = chain.copy()
    while remaining_segments:
        last_point = sorted_chain[-1]
        next_segment = min(remaining_segments, key=lambda seg: min(distance(last_point, seg[0]), distance(last_point, seg[1])))
        remaining_segments.remove(next_segment)
        if distance(last_point, next_segment[0]) < distance(last_point, next_segment[1]):
            if next_segment[0] != last_point:
                sorted_chain.append(next_segment[0])
            sorted_chain.append(next_segment[1])
        else:
            if next_segment[1] != last_point:
                sorted_chain.append(next_segment[1])
            sorted_chain.append(next_segment[0])
    sorted_chains.append(sorted_chain)
print(f'Sorted {len(sorted_chains)} chains')

# Create polylines
opencv_polylines = []
for chain in sorted_chains:
    polyline = []
    for segment in chain:
        polyline.append(segment[0])
        polyline.append(segment[1])
    opencv_polylines.append(np.array(polyline, dtype=np.int32).reshape((-1, 1, 2)))
filtered_polylines = []
for i, polyline in enumerate(opencv_polylines):
    formatted_points = ', '.join([f"({point[0][0]}, {point[0][1]})" for point in polyline])
    print(f"Polyline {i}: {formatted_points}")