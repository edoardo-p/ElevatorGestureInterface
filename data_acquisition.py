import cv2
import mediapipe as mp
import time
import json
import os
from pynput.keyboard import Key, Listener
import logging
import threading

USER = "gio"

data_path = "./data/values/"
labels_path = "./data/labels/"
labels_dataset = labels_path + "1 label dataset_with_user.json"
values_dataset = data_path + "1 values dataset.json"
stop_event = threading.Event() 

#I save the labels in a new file, then i append the new values to the dataset
files = os.listdir(labels_path)
labelfile_index = max([int(f[:2]) for f in files]) + 1

logging.basicConfig(filename=(labels_path + str(labelfile_index) + " .txt"), level=logging.FATAL, format='%(message)s')

def on_press(key):
    logging.fatal(str(time.time()) +","+ str(key))

def key_logger_thread():
    listener = Listener(on_press=on_press)
    listener.start()
    if stop_event.is_set():
        listener.stop()
        return

def update_labels_files():
    labels = {}
    with open(labels_path + str(labelfile_index) + " .txt", 'r') as f:
        for line in f:
            try:
                float(line[0])
                line = line.split(',')
                key = line[0]
                val = line[1].replace('\n', '')
                if val not in ['Key.left', 'Key.right', "Key.up", "Key.down"]:
                    continue
                labels[line[0]] = {"label": line[1].replace('\n', ''), "user": USER}
            except:
                continue
    dataset = json.load(open(labels_dataset, 'r'))
    dataset.update(labels)
    json.dump(dataset, open(labels_dataset, 'w'))


if __name__ == "__main__":

    logger = threading.Thread(target=key_logger_thread)
    logger.start()
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity= 1)
    mp_draw = mp.solutions.drawing_utils
    landmark_list = {}

    p_time = 0

    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_rgb = cv2.flip(img_rgb, 0)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                point = {}  
                for id, lm in enumerate(hand_landmarks.landmark):
                    point[id] = (lm.x, lm.y, lm.z)
                
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            t = time.time()
            landmark_list[t] = point

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)    
        
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    #firstly i save the single trial data for redundancy
    files = os.listdir(data_path)
    new_index = max([int(f[:1]) for f in files]) + 1
    json.dump(landmark_list, open(data_path + str(new_index) + ' .json', 'w'))

    #then i update the whole dataset of values
    loaded_landmark_list = json.load(open(values_dataset, 'r'))
    loaded_landmark_list.update(landmark_list)
    json.dump(loaded_landmark_list, open(values_dataset, 'w'))

    update_labels_files()

    stop_event.set()
    logger.join() 
    exit(0)