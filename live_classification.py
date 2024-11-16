import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


FRAMES_BACK = 3

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
lms_to_keep = [ 4, 6, 8, 10, 12, 14, 16, 18, 20]

rfc = pickle.load(open('./model_ada.pkl', 'rb'))
# print (model_dict)
# rfc = model_dict['model']

def get_hand_landmarks_sample(hand):

    # TO DO: What if more than one hands were found? 
    # TO DO: What if the handedness is not 'Left' - flip HERE
    handedness = hand.handedness[0][0].display_name
    hand_landmarks = hand.hand_landmarks[0];

    x_lms = [landmark.x for landmark in hand.hand_landmarks[0]]
    y_lms = [landmark.y for landmark in hand.hand_landmarks[0]]
    
    return handedness, x_lms, y_lms

def get_hand_movement(x_lms, y_lms, movement_df, this_video_image_count):
    """
        x_lms, y_lms: arrays with the xs and ys of the hand landmarks 
        movement_df: array with all movement data
        this_video_image_count: which frame of the original video is this
        returns:
            hand_movement: array consisting of x1, y1, x2, y2, iou1, iou2 ..
            (which are the bounding box and the history of ious)
    """
    # Save current bounding box
    curr_hand_bbox = get_hand_bounding_box(x_lms, y_lms,)
    hand_movement = curr_hand_bbox

    # First frame gets None for all ious since there are no prev frames
    if this_video_image_count == 1:
        hand_movement += [None for i in range (FRAMES_BACK)]
        
    # Save ious of 3 frames back (or FRAMES_BACK amount)
    else:
        prev_hand_bbox = movement_df[-1][:4]
        prev_hand_ious = movement_df[-1][4:-1]
        
        # calc IoU for curr and prev frame
        iou1 = get_iou(prev_hand_bbox, curr_hand_bbox)
        hand_movement.append(iou1)

        if len(prev_hand_ious) < FRAMES_BACK - 1:
            conole.log(f'Error: not enough iou elements in movement array for frame before {img_path}')
            return None
        hand_movement += prev_hand_ious

    return hand_movement

def get_box_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def get_bounding_box_ratio(x1, y1, x2, y2):
    return (x2 - x1) / (y2 - y1)

def get_iou(prev_bbox, bbox):
    """
        returns:
            (intersection area) / (union area) of 2 bounding boxes
    """
    prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
    x1, y1, x2, y2 = bbox

    # bounding box of intersection
    x1_inter = max(x1, prev_x1)
    y1_inter = max(y1, prev_y1)
    x2_inter = min(x2, prev_x2)
    y2_inter = min(y2, prev_y2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0

    inter_area = get_box_area(x1_inter, y1_inter, x2_inter, y2_inter)
    bbox_area = get_box_area(x1, y1, x2, y2)
    prev_bbox_area = get_box_area(prev_x1, prev_y1, prev_x2, prev_y2)

    union_area = bbox_area + prev_bbox_area - inter_area

    iou = inter_area / union_area
    return iou

def get_hand_bounding_box(lm_xs, lm_ys):
    """
        takes all xs and ys of the points within dersied bounding box
        returns
        (x1, y1) - upper left corner
        (x2, y2) - lower right corner
    """
    x1, y1 = min(lm_xs), min(lm_ys)
    x2, y2 = max(lm_xs), max(lm_ys)
    return [x1, y1, x2, y2]

def scale(coords):
    return [(coord - min(coords)) / (max(coords) - min(coords)) for coord in coords]


 
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

landmarks_df = []
movement_df = []

n = 0
signs_collected = 0


while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)
    results = detector.detect(mp_image)
    if results.hand_landmarks:
        signs_collected += 1;
        
        for hand_landmarks in results.hand_landmarks:
            for i in range(len(hand_landmarks)):
                x = hand_landmarks[i].x
                y = hand_landmarks[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks)):
                x = hand_landmarks[i].x
                y = hand_landmarks[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                
        handedness, x_lms, y_lms = get_hand_landmarks_sample(results)
        hand_movement = get_hand_movement(x_lms, y_lms, movement_df, signs_collected)
        x1, y1, x2, y2, iou1, iou2, iou3 = hand_movement

        if hand_movement == None:
            continue

        if handedness == 'Left':
            x_lms = [1 - x for x in x_lms]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        scaled_x_lms = scale(x_lms)
        scaled_y_lms = scale(y_lms)
        hand_landmarks_sample = [scaled_x_lms[i] for i in range(len(scaled_x_lms)) if i in lms_to_keep] + \
        [scaled_y_lms[i] for i in range(len(scale(scaled_y_lms))) if i in lms_to_keep]
        landmarks_df.append(hand_landmarks_sample)
        movement_df.append(hand_movement)
        sample = hand_landmarks_sample + [iou1, iou2, iou3, get_bounding_box_ratio(x1, y1, x2, y2)];
        # if None not in sample:
        #     confidence_scores = rfc.predict_proba([sample])
        #     if np.max(confidence_scores, axis=1) > 0.3:
        #         prediction = rfc.predict([sample])
        #         predicted_character = prediction[0]
        #         print (predicted_character)

        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        #         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #                     cv2.LINE_AA)
        if None not in sample:
            # confidence_scores = rfc.predict_proba([sample])
            # if np.max(confidence_scores, axis=1) > 0.3:
            prediction = rfc.predict([sample])
            predicted_character = prediction[0]
            # print (predicted_character)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
    else:
        signs_collected = 0
        movement_df = []

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
