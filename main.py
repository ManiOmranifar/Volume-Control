import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

volume_locked = False
lock_threshold = 10 
lock_timer = None
lock_duration = 2  
target_vol = None 
show_vol = 0  

vol = min_vol

cap = cv2.VideoCapture(0)

def draw_volume_bar(img, vol, vol_percent):
    cv2.putText(img, f'Volume: {int(vol_percent)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(img, (50, 100), (85, 400), (255, 0, 0), 3)
    vol_bar = int(np.interp(vol_percent, [0, 100], [400, 100]))
    cv2.rectangle(img, (50, vol_bar), (85, 400), (255, 0, 0), cv2.FILLED)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_pos = np.array([thumb_tip.x * img.shape[1], thumb_tip.y * img.shape[0], thumb_tip.z * img.shape[1]])
            index_pos = np.array([index_tip.x * img.shape[1], index_tip.y * img.shape[0], index_tip.z * img.shape[1]])

            distance_3d = np.linalg.norm(thumb_pos - index_pos)

            if not volume_locked:
                vol = np.interp(distance_3d, [20, 150], [min_vol, max_vol])
                vol_percent = np.interp(distance_3d, [20, 150], [0, 100])  

                if (target_vol is not None and abs(vol_percent - target_vol) <= lock_threshold) or \
                    (vol_percent <= 1 or vol_percent >= 99):
                    if lock_timer is None:
                        lock_timer = time.time()  
                    elif time.time() - lock_timer >= lock_duration:
                        volume_locked = True  
                else:
                    target_vol = vol_percent 
                    lock_timer = None  

                volume.SetMasterVolumeLevel(vol, None)
                show_vol = vol_percent  

            cv2.circle(img, tuple(thumb_pos[:2].astype(int)), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, tuple(index_pos[:2].astype(int)), 5, (255, 255, 255), cv2.FILLED)
            cv2.line(img, tuple(thumb_pos[:2].astype(int)), tuple(index_pos[:2].astype(int)), (160, 179, 255), 3)

            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pos = np.array([pinky_tip.x * img.shape[1], pinky_tip.y * img.shape[0], pinky_tip.z * img.shape[1]])
            fist_distance = np.linalg.norm(thumb_pos[:2] - pinky_pos[:2])

            if fist_distance < 50: 
                volume_locked = False  
                target_vol = None 
                lock_timer = None

    draw_volume_bar(img, vol, show_vol)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
