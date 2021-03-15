import numpy as np
import cv2
# import pytesseract
import time
from dataloading.dataset import ScoreDetectionDataset

'''
TODO:
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

def get_largest_blob(img, invert):
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
    if DEBUG:
        print("gray_threshed start")
        disp(gray_threshed)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_threshed)
    for i in range(len(stats)):
        show = show_cc(gray_threshed, stats[i])
        x0, y0, w, h, _ = stats[i]
        if (w*h) > 0.5*(gray_threshed.shape[0]*gray_threshed.shape[1]):
            gray_threshed[labels == i] = 0
    if DEBUG:
        print("gray_threshed large removed")
        disp(gray_threshed)

    # ds = 2
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (ds*2+1, ds*2+1), (ds, ds))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2), (1, 1))
    gray_threshed_dilated = cv2.dilate(gray_threshed, element)
    gray_threshed_dilated = cv2.erode(gray_threshed_dilated, element)

    if DEBUG:
        print("gray_threshed dilated")
        disp(gray_threshed_dilated)
    # return

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_threshed_dilated)
    sorted_indices = np.argsort(-1.0*stats[:,-1]) #Sort by size descending    
    idx = sorted_indices[1] # Take the largest component aside from full the bounding box 

    show = show_cc(img, stats[idx])
    if DEBUG:
        print('CC')
        disp(show)

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

def extract_roi(img):
    """Crops a row-image to only the player name for reliable OCR extraction by running get_larget_blob() twice and taking the better result.

    Args:
        img (np.ndarray): Pre-cropped row image form extract_rows()

    Returns:
        roi (np.ndarray): Final crop of the row-image to just the player name.
    """
    roi1 = get_largest_blob(img, invert=True)
    roi2 = get_largest_blob(img, invert=False)
    if roi1.shape[0]*roi1.shape[1] > roi2.shape[0]*roi2.shape[1]:
        return roi1
    else:
        return roi2

def extract_rows(img):
    """Splits an image into two rows to read the respective names from.

    Args:
        img (np.ndarray): The image to split into rows.

    Returns:
        top (np.ndarray): The row-image for player 1
        bot (np.ndarray): The row-image for player 2
    """
    img = img[:, 0:img.shape[1]//2]
    mid = img.shape[0]//2
    top, bot = img[:int(1.15*mid)], img[int(0.85*mid):]

    return top, bot

if __name__=='__main__':
    DEBUG = False
    dataset = ScoreDetectionDataset('./data/frames.json', './data/frames')
    if False:
        DEBUG = True
        sample = dataset[12]
        img = cv2.cvtColor(sample['np_images'], cv2.COLOR_RGB2BGR)
        top, bot = extract_rows(img)

        img = top
        
        print("row:")
        disp(img)  

        roi = extract_roi(img)
        print("roi:")
        disp(roi)

        quit()

    t0 = time.time()
    for i_sample, sample in enumerate(dataset):
        img = cv2.cvtColor(sample['np_images'], cv2.COLOR_RGB2BGR)
        data = sample['data']
        
        top, bot = extract_rows(img)
        top = extract_roi(top)
        bot = extract_roi(bot)
        # disp(top)
        # disp(bot)
        cv2.imwrite(f"ocr/cropped_frames/{i_sample:02.0f}_{data['p1_name']}.png", top)
        cv2.imwrite(f"ocr/cropped_frames/{i_sample:02.0f}_{data['p2_name']}.png", bot)

    t1 = time.time()
    print(f'Time: {(t1-t0) / len(dataset)}')