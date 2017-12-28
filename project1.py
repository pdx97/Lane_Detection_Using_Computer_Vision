import cv2
import io
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import time

time.sleep(15)
camera=PiCamera()
"""
cap=PiCamera()
"""
camera.resolution=(200, 120)
camera.framerate=2
rawCapture=PiRGBArray(camera, size=(200, 120))

#file object=open(values [, a+][, 1])
"""
def opencvtopicam():
    stream=io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.resolution=(640, 480)
        camera.capture(stream, format="jpeg")
    buff=np.fromstring(stream.getvalue(), dtype=np.uint8)
    return buff
"""        
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    if lines is not None:
        for line in lines:
            if line is not None:
                x1,y1,x2,y2 = line
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def separate_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0:
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])

    return right, left

def filter_region(image, vertices):

    """Create the mask using the vertices and apply it to the input image"""

    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):

    """It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black)."""

    imshape=image.shape
    lower_left = [imshape[1]/200,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/200,imshape[0]]
    top_left = [imshape[1]/2-imshape[1],imshape[0]/2]#+imshape[0]/70]
    top_right = [imshape[1]/2+imshape[1],imshape[0]/2]#2+imshape[0]/70]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    mask= np.zeros_like(image)
    mask_color=255;
    cv2.fillPoly(mask,vertices,mask_color)
    masked_img=cv2.bitwise_and(image,mask)


    return filter_region(image, vertices)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                    slope1=slope
                    print("left lane",slope1)
                    
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
                    slope2=slope
                    print("Right lane",slope2)

    # add more weight to longer lines
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    if slope != 0:
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        #print(int(slope))

        return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.55    # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)



    return left_line, right_line

def flatting(line):
    flat_point = tuple()
    if line is not None:
        for i in line:
            flat_point += i
        return flat_point


#cap = cv2.VideoCapture(0)
#cap.start_preview()
#while True:
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame=image.array
    
    #frame=cv2.imdecode(opencvtopicam(), 1)
    #frame = cap.capture(rawCapture, format="bgr", use_video_port=True)
    gray = grayscale(frame)
    blur = cv2.bilateralFilter(gray, 9,75,75)
    edge = canny(blur, 100, 175)
    rio_image = select_region(edge)


    ### Run Hough Lines and separate by +/- slope
    lines = cv2.HoughLinesP(rio_image, rho=4, theta=2*np.pi/180, threshold=20, minLineLength=25, maxLineGap=2)

    l_line, r_line = lane_lines(frame, lines)
    right_line = flatting(r_line)
    left_line = flatting(l_line)
    lines = (left_line, right_line)

    ### Draw lines and return final image
    line_img = np.copy((frame)*0)
    draw_lines(line_img, lines, thickness=3)

    line_img = select_region(line_img)
    final = weighted_img(line_img, frame)

    #cv2.imshow('frame', frame)
    # cv2.imshow('edge', edge)
    # cv2.imshow('edge', rio_image)
    cv2.imshow('final', final)
    cv2.imshow('roi', rio_image)
    rawCapture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
#cap.close()
cv2.destroyAllWindows()
