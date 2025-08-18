import cv2 as cv
import numpy as np


def read_vid_example(vid_file_path):
    cap = cv.VideoCapture(vid_file_path) # using webcam would be 0

    while True:
        isTrue, frame = cap.read() # if the img was read properly, the img frame
        cv.imshow('Video', frame) # Show the frame

        if not isTrue: # to not crash when no more frames or if bad frame
            break

        if cv.waitKey(20) & 0xFF == ord('d'): # stop looping on videos after 20 miliseconds or when "d" is pressed
            break

    cap.release() # closes video file
    cv.destroyAllWindows() # closes all windows


def rescale_frame(frame, scale=.5):
    # works for images, videos, live video
    width = int(frame.shape[1] * scale) # current frame width * rescale factor
    height = int(frame.shape[0] * scale) # current frame height * rescale factor
    dimensions = (width, height) # has to be tuple for cv method

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # resizes frame to particular dimension


def rescale_example(img_file_path):
    img = cv.imread(img_file_path)
    rescaled_img = rescale_frame(img)
    cv.imshow('car', rescaled_img)

    cv.waitKey(0)

def change_res(width, height, capture):
    # Live video only - capture is the cv object containing the frames
    # requesting the camera or video device to provide frames at
    # a specific resolution. 
    capture.set(3,width)
    capture.set(4,height)

def draw_blank_example():
    # 2 ways to draw - drawing on stand alone or dummy image to work with

    # blank image
    blank = np.zeros((500,500, 3), dtype='uint8') # (height, width, color channels), uint8 is img dtype
    blank[:] = 0,0,0 # select all pixels and make them some RGB color
    blank[200:300, 300:400] = 0,0,225

    # Draw rectangle
    cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2) # frame, pt1, pt2, color, thickness (-1 fills it in)

    # Draw Circle
    cv.circle(blank, (250,250), 100, (153,153,255), thickness = 4 ) # frame, center of circle, radius, line color, thickness

    # Draw a line
    cv.line(blank, (0,0), (500,500),(0,255,0), thickness=2 )

    # Write Text
    cv.putText(blank, 'NOICEEE', (0,500), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)

    cv.imshow('draw', blank)
    cv.waitKey(0)

def basic_functions_example(img_file_path):
    img = cv.imread(img_file_path)
    img = rescale_frame(img) # You should rescale before any edge detection

    # 1. Converting to gray scale
    # benefits are it reduces memory usage and simplified feature extraction
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Blur 
    # Removes some of noise in image
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)

    # 3. Edge Cascade (find edges in image)
    canny = cv.Canny(img, 125, 175) # reduce amount of images by applying blur

    # 4. Dilate an image
    # Important for filling gaps/holes to connect nearby objects 
    # that are treats as one unit. Helpful when edge detection creates
    # fragmented results. Also usefulf for noise removal and object expansion
    dilate = cv.dilate(canny, (7,7), iterations=5)

    # 5. Eroding
    eroded = cv.erode(dilate, (7,7), iterations=5)

    # 6. Resize
    resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) # Inter area for making it smaller, inter linear or cubic for making bigger

    # 7. Cropping
    cropped = img[50:200, 200:400]
    
    cv.imshow('car', cropped)
    cv.waitKey(0)

def translate(img_path, x, y):
    # -x -> left
    # +x -> right
    # -y -> up
    # +y -> down
    img = cv.imread(img_path)
    dimensions = (480,480)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow('org', img)
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

def translt_ex(img):
    ex = translate(img, 100, 100)
    cv.imshow('translated', ex)
    cv.waitKey(0)

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width //2, height //2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

def rotate_ex(img_path):
    img = cv.imread(img_path)
    dimensions = (480,480)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow('org', img)
    rot_img = rotate(img, 30)
    cv.imshow('rot', rot_img)
    cv.waitKey(0)

def flip_ex(img_path):
    img = cv.imread(img_path)
    dimensions = (480,480)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow('org', img)
    # 0: Flips the image vertically (around the x-axis).
    # 1: Flips the image horizontally (around the y-axis).
    # -1: Flips the image both vertically and horizontally.
    flip = cv.flip(img, 0)
    cv.imshow('flip', flip)
    cv.waitKey(0)


def countours_ex(img_path):
    # boundaries of the object, not the same as edges
    img = cv.imread(img_path)
    blank = np.zeros(img.shape, dtype='uint8')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)

    # Blur to reduce the amount of noise and sharpness in an image
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    cv.imshow('blur', blur)

    # Contour method 1 - Canny (pref)
    canny = cv.Canny(blur, 100, 175)
    cv.imshow('canny', canny)

    # Contour method 2 - Thresholding
    ret, thresh = cv.threshold(gray, 100, 175, cv.THRESH_BINARY) # tries to "binarize" the image based on thresholds
    cv.imshow('thresh', thresh)

    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # Use canny method first instead of thresholding and then finding contours
    # Canny is simplest and effective

    cv.drawContours(blank, contours, -1, (0,0,255), 2)
    cv.imshow('Blank', blank)
    cv.waitKey(0)


def color_spaces(img_path):
    # more info: https://opencv.org/blog/color-spaces-in-opencv/
    # BGR is the default for openCV
    # RGB is what matplotlib would take

    img = cv.imread(img_path)
    
    # BGR to Grayscale
    # help show pixel intensity clearer, more efficient for storage and processing
    # can't convert gray scale to HSV directly -> have to go back to BGR then hsv
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    cv.imshow('gray', gray)

    # BGR to HSV - hue saturation value
    # simplifies the process of isolating specific colors within an image, better for segmentation
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    cv.imshow('hsv', hsv)

    # BGR to L*a*b
    # Color-based object recognition and segmentation
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cv.imshow('lab', lab)

    # BGR to RGB
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imshow('rgb', rgb)

    cv.waitKey(0)


def color_channels(img_path):
    img = cv.imread(img_path)
    b,g,r = cv.split(img)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    blue = cv.merge([b, blank, blank])
    green = cv.merge([blank, g, blank])
    red = cv.merge([blank, blank, r])

    cv.imshow('b', blue)
    cv.imshow('g', green)
    cv.imshow('r', red)
    cv.imshow('og', img)

    merged_img = cv.merge([b,g,r])
    cv.imshow('mereged', merged_img)

    cv.waitKey(0)


def blurring(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)



    cv.waitKey(0)


def bitwise(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)



    cv.waitKey(0)


def masking(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)



    cv.waitKey(0)


img = 'photos/Prisonmike.jpg'
color_channels(img)