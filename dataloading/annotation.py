import sys
import os
import os.path as osp
import cv2
import json
import numpy as np

WIDTH, HEIGHT = 1280, 720
def annotate_frame(img, frame_pos):
    roi = cv2.selectROI("", img)
    if roi[-1] == 0: #cancelled
        return

    roi = (roi[0]//resize_factor, roi[1]//resize_factor, roi[2]//resize_factor, roi[3]//resize_factor)
    print(roi)
    
    frame_id = f'vid{str(abs(hash(video_name)))[0:4]}_frame{frame_pos:05.0f}' #Hash isn't constant unfortunately

    cv2.imwrite(osp.join(path_frames, frame_id+".png"), cv2.resize(img, (WIDTH, HEIGHT)))

    data = {}
    data['id'] = frame_id
    data['video_name'] = video_name
    data['frame_pos'] = frame_pos
    data['bbox'] = roi
    data['p1_name'] = p1_name
    data['p1_points'] = 0 # a number or "adv" for advantage
    data['p1_sets'] = 0
    data['p1_matches'] = -1 # -1 indicates no match data
    data['p1_serves'] = False
    data['p2_name'] = p2_name
    data['p2_points'] = 0 # a number or "adv" for advantage
    data['p2_sets'] = 0
    data['p2_matches'] = -1 # -1 indicates no match data
    data['p2_serves'] = False    

    try:
        with open(path_data, 'r') as f:
            current_json = json.load(f)
    except:
        current_json = []

    assert isinstance(current_json, list)
    current_json.append(data)    

    with open(path_data, 'w', encoding='utf-8') as f:
        json.dump(current_json, f, ensure_ascii=False, indent=4)

    print(f'Now {len(current_json)} annotations')

def annotate_video(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Video finished')
            break

        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(f'Frame second {frame_pos/fps: 0.2f} / {max_frame/fps: 0.2f}')
        frame = cv2.resize(frame, (WIDTH, HEIGHT)) #Resize to match expected size

        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor) #Resize for better selection

        cv2.imshow("", frame)
        key = cv2.waitKey()

        if key == ord('q'):
            break
        if key == ord('a'):
            annotate_frame(frame, frame_pos)
        if key == ord('k'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 15)
        if key == ord('l'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 60)
        if key == ord('j'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 60)            

    cap.release()

if __name__ == "__main__":
    video_idx, p1_name, p2_name = int(sys.argv[1]), sys.argv[2], sys.argv[3]

    path = './data/video'
    path_data = './data/frames.json'
    path_frames = './data/frames'
    video_names = os.listdir(path)
    video_name = video_names[video_idx]

    resize_factor = 2


    annotate_video(osp.join(path, video_name))