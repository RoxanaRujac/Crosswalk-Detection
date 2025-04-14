#  Crosswalk Detection System

This project is a computer vision-based crosswalk (pedestrian crossing) detection system built using Python and OpenCV. It detects horizontal road stripes (zebra crossings) in images or video frames by applying image processing techniques such as Gaussian blurring, Canny edge detection, and Hough Line Transform.

![crosswalk video example 1](https://github.com/user-attachments/assets/5d04c201-663f-42e9-83a6-e5cbe4226eb8)


##  How It Works

1. **Image Preprocessing**
   - The input image is converted to grayscale.
   - A Gaussian blur is applied to reduce noise and smooth the image.

2. **Edge Detection**
   - Canny edge detection is used to identify strong edges in the image.

3. **Region of Interest (ROI)**
   - A trapezoidal mask is applied to focus on the road area, ignoring irrelevant regions.

4. **Line Detection**
   - Probabilistic Hough Line Transform detects line segments.
   - Lines are filtered based on length and orientation (mostly horizontal).

5. **Crosswalk Grouping**
   - Detected lines are grouped based on angle and position to identify a crosswalk pattern.

6. **Visualization**
   - Results are drawn on the image, highlighting detected crosswalks and overlaying region of interest.

##  Future Improvements

-  Detect traffic signal state (e.g., green pedestrian light).
-  Integrate GPS or location data to confirm crosswalk presence.
-  Provide haptic or audio feedback when a crosswalk is nearby, enhancing accessibility.

##  Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Tkinter (for file dialog GUI)

Install dependencies using:

```bash
pip install -r requirements.txt
