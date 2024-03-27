import numpy as np
import cv2

def read_omr(image_path):
    # Load the image
    image = cv2.imread(image_path)
    cv2.imshow("OMR", image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert it to a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Thresh", thresh)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the detected marks
    marks = []
    objects = image.copy()

    # Iterate through the contours
    for contour in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small contours
        if w * h < 50:
            continue

        # Extract the region of interest (ROI) containing the mark
        mark_roi = thresh[y:y + h, x:x + w]

        # Determine the percentage of white pixels in the ROI
        white_pixels = cv2.countNonZero(mark_roi)
        total_pixels = w * h
        white_percentage = white_pixels / total_pixels

        # If the percentage exceeds a threshold, consider it as a marked region
        if white_percentage > 0.5:
            marks.append((x, y, w, h))
            cv2.circle(objects, (x + 4, y + 4), 4, (255, 0, 0), -1)

    cv2.imshow("Contours", objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return marks

# Test the function
image_path = 'PCF3.png'
marks = read_omr(image_path)
print("Detected marks:", marks)