import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# gaussian blur to smooth the image and reduce noise
# applies a convolution function based on the normal distribution,
# so that the values closer to the center of the kernel
# have a greater impact on the surrounding pixels,
# while the values at the edges of the kernel have a lesser impact.
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# canny edge detection to find edges in the image
# an edge represents a sudden change in pixel intensity
# and is often a significant feature of an image,
# indicating the boundaries of objects or structural details.
def detect_edges(image):
    # canny steps:
    #   image filtering with a gaussian filter
    #   gradient calculation
    #   non-maximum suppression
    #   adaptive binarization
    #   edge tracking by hysteresis
    blurred = apply_gaussian_blur(image)
    edges = cv2.Canny(blurred, 100, 200, apertureSize=3, L2gradient=True)
    return edges



# create a binary mask (black and white image) that defines
# a region of interest mask focusing on the road area
def create_roi_mask(image):
    height, width = image.shape
    roi_mask = np.zeros_like(image)

    # Create a trapezoidal ROI focusing on the road area
    roi_points = np.array([
        [width * 0.1, height * 0.5],  # top-left
        [width * 0.9, height * 0.5],  # top-right
        [width, height],  # bottom-right
        [0, height]  # bottom-left
    ], dtype=np.int32)

    # fill the defined ROI in the mask with white
    # only the ROI (the road area) is white
    cv2.fillPoly(roi_mask, [roi_points], 255)
    return roi_mask



# filter lines to keep only those within a specific angle range
def filter_lines_by_angle(lines, min_angle=160, max_angle=200):
    if lines is None:
        return []

    # each line is an array of coordinates
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # skip very short lines less than 20px
        if math.hypot(x2 - x1, y2 - y1) < 20:
            continue

        # calculate angle in degrees (0-360)
        # horizontal line ~180
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360

        # filter for mostly horizontal lines
        if min_angle <= angle <= max_angle or min_angle <= (angle + 180) % 360 <= max_angle:
            filtered_lines.append(line)

    return filtered_lines


# detect lines in the image, restricted to an ROI
def hough_lines(edges, roi_mask=None):
    if roi_mask is not None:
        # apply ROI mask to restrict line detection to road area
        # keep only the edges inside the region of interest (white area of the mask)
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    #  probabilistic hough line transform used to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=20, maxLineGap=10)

    if lines is not None:
        # filter by angle to focus on horizontal-ish lines
        filtered_lines = filter_lines_by_angle(lines)

        # sort by length
        # [0][:2]: starting point [x1, y1]
        # x[0][2:]: ending point [x2, y2]
        filtered_lines = sorted(filtered_lines,
                                key=lambda x: np.linalg.norm(x[0][:2] - x[0][2:]),
                                reverse=True)

        return filtered_lines

    return []


# group lines that likely form a crosswalk pattern
def group_crosswalk_lines(lines, angle_thresh=15, y_dist_thresh=25):
    if lines is None or len(lines) < 3:
        return False, []

    # find groups of lines with similar angles and y-positions
    best_group = []

    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        angle_i = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # use midpoint y for grouping
        mid_y_i = (y1 + y2) / 2

        current_group = [lines[i][0]]

        for j in range(len(lines)):
            if i == j:
                continue

            x3, y3, x4, y4 = lines[j][0]
            angle_j = math.degrees(math.atan2(y4 - y3, x4 - x3))
            mid_y_j = (y3 + y4) / 2

            # check if angles are similar (approximately parallel)
            if abs((angle_i - angle_j) % 180) < angle_thresh:
                # check if y positions are different enough (different stripes)
                if abs(mid_y_i - mid_y_j) > y_dist_thresh:
                    current_group.append(lines[j][0])

        # update best group if current one is better
        if len(current_group) > len(best_group):
            best_group = current_group

    return len(best_group) >= 5, best_group


# draws the lines, roi and writes in case of crosswalk detection
def draw_results(image, edges, lines):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # draw ROI for visualization
    roi_mask = create_roi_mask(image)
    roi_outline = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    roi_outline = np.where(roi_outline > 0, [0, 0, 100], [0, 0, 0]).astype(np.uint8)
    image_color = cv2.addWeighted(image_color, 1, roi_outline, 0.3, 0)

    is_crosswalk, crosswalk_lines = group_crosswalk_lines(lines)

    # draw all detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            match = any(np.array_equal(line[0], cross_line) for cross_line in crosswalk_lines)
            color = (0, 255, 0) if match else (255, 0, 255)
            cv2.line(image_color, (x1, y1), (x2, y2), color, 2)

    if is_crosswalk:
        cv2.putText(image_color, "Crosswalk Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 3, cv2.LINE_AA)

    return image_color


def main():
    # open file picker
    Tk().withdraw()
    path = filedialog.askopenfilename(title="Select image")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return

    # create ROI mask for road area
    roi_mask = create_roi_mask(img)

    edges = detect_edges(img)
    lines = hough_lines(edges, roi_mask)
    result = draw_results(img, edges, lines)

    # create plots
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Edges with ROI")
    roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    plt.imshow(roi_edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Detected Lines")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()