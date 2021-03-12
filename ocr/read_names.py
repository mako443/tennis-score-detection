import numpy as np
import cv2
import pytesseract

'''
TODO:
- Tesseract on numbers only!
'''

def disp(img):
    cv2.imshow("",cv2.resize(img, None, fx=3.0, fy=3.0))
    cv2.waitKey()

def show_cc(img, stat):
    show = img.copy()
    if len(img.shape)==2:
        show = np.dstack((show, show, show))
    cv2.rectangle(show, (stat[0], stat[1]), (stat[0]+stat[2]-1, stat[1]+stat[3]-1), (0,0,255), thickness=1)
    return show

def extract_roi(img, invert):
    """Extracts the region of the players name by finding the largest dilated connected component that is not the outer box.

    Args:
        img (np.ndarray): A row-crop of the score box
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray_threshed = np.zeros_like(gray)
    if invert:
        gray_threshed[gray<np.mean(gray)] = 255
    else:
        gray_threshed[gray>np.mean(gray)] = 255

    # First, detect very large regions and remove them
    # print("gray_threshed start")
    # disp(gray_threshed)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_threshed)
    for i in range(len(stats)):
        show = show_cc(gray_threshed, stats[i])
        x0, y0, w, h, _ = stats[i]
        if (w*h) > 0.5*(gray_threshed.shape[0]*gray_threshed.shape[1]):
            gray_threshed[labels == i] = 0
    # print("gray_threshed large removed")
    # disp(gray_threshed)

    ds = 2
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ds*2+1, ds*2+1), (ds, ds))
    gray_threshed_dilated = cv2.dilate(gray_threshed, element)
    gray_threshed_dilated = cv2.erode(gray_threshed_dilated, element)

    # print("gray_threshed dilated")
    # disp(gray_threshed_dilated)
    # return

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_threshed_dilated)
    sorted_indices = np.argsort(-1.0*stats[:,-1]) #Sort by size descending    
    idx = sorted_indices[1] # Take the largest component aside from full the bounding box 

    show = show_cc(img, stats[idx])
    # disp(show)

    x0, y0, x1, y1, _ = stats[idx]
    x1 += x0 + 5
    y1 += y0 + 5
    x0 -= 5
    y0 -= 5
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(x1, img.shape[1]-1)
    y1 = min(y1, img.shape[0]-1)

    return img[y0 : y1, x0 : x1, :]

for i in (0,1,2):
    img = cv2.imread(f'./ocr/testframe{i}.png')

    #Take the left half
    img = img[:, 0:img.shape[1]//2]

    mid = img.shape[0]//2
    top, bot = img[:int(1.15*mid)], img[int(0.85*mid):]

    roi1 = extract_roi(top, True)
    roi2 = extract_roi(top, False)
    if roi1.shape[0]*roi1.shape[1] > roi2.shape[0]*roi2.shape[1]:
        roi = roi1
    else:
        roi = roi2

    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    print(text.lower())

    disp(roi)

quit()
