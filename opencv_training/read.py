import cv2 as cv
import numpy as np

# Read Images
# img = cv.imread('Photos/haas.jpg')
# cv.imshow('car', img)
# cv.waitKey(0)

# Read Videos

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



haas = 'photos/haas.jpg'
basic_functions_example(haas)