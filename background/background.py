import cv2
import numpy as np
import sys
from timeit import default_timer as timer
import pyfakewebcam


def resize(dst, img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    return resized


start = timer()
video = cv2.VideoCapture(0)
oceanVideo = cv2.VideoCapture("ocean.mp4")
height, width = 720, 1280
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
video.set(cv2.CAP_PROP_FPS, 60)

# setup the fake camera
fake = pyfakewebcam.FakeWebcam('/dev/video2', width, height)
success, ref_img = video.read()
flag = 0

while 1:
    end = timer()

    success, img = video.read()
    success2, bg = oceanVideo.read()
    bg = resize(bg, ref_img)
    if flag == 0:
        ref_img = img
    # create a mask
    diff1 = cv2.subtract(img, ref_img)
    diff2 = cv2.subtract(ref_img, img)
    diff = diff1 + diff2
    diff[abs(diff) < 8.0] = 0
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray[np.abs(gray) < 10] = 0
    fgmask = gray.astype(np.uint8)
    fgmask[fgmask > 0] = 255
    # invert the mask
    fgmask_inv = cv2.bitwise_not(fgmask)
    # use the masks to extract the relevant parts from FG and BG
    fgimg = cv2.bitwise_and(img, img, mask=fgmask)
    bgimg = cv2.bitwise_and(bg, bg, mask=fgmask_inv)
    # combine both the BG and the FG images
    dst = cv2.add(bgimg, fgimg)
    
    cv2.imshow('Background Removal', dst)
    # cv2.imshow('Original ', img)

    # fake webcam expects RGB
    if end - start > 30:
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        fake.schedule_frame(dst)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fake.schedule_frame(img)
    key = cv2.waitKey(5) & 0xFF
    if ord('q') == key:
        break
    elif ord('d') == key:
        flag = 1
        print("Background Captured")
    elif ord('r') == key:
        flag = 0
        print("Ready to Capture new Background")

cv2.destroyAllWindows()
video.release()
# return jpeg.tobytes()
