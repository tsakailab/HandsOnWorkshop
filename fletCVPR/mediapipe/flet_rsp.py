import flet as ft
import cv2
import mediapipe as mp
import numpy as np
import sys
import requests
import os

sys.path.append("../util")
from flet_util import set_page, ftImShow, cvVideoCapture


resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}
CAMERAS = ["0", "1", "2", "3"]

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': resols['qHD'], 'padding': 10, 
        'cameras': CAMERAS, 'frame_hw': resols['HD'], 
        'images': None}
#args.update({'images': ['../dashcam.jpg', '../park.jpg']}) # works if cap.isOpened() is False


PageOpts = {'TITLE': "Hand Landmarker (mediapipe)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+240, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}


## Please copy and paste line #29 ~ #220 for rock paper scissors game
# defaults
#detector_params = {'model_asset_path': "./object_detector.tflite", 'score_threshold': 0.3}
detector_params = {'model_asset_path': "./hand_detector.task", 'num_hands': 8}
drawer_opts = {'bfps': True, 'margin': 10, 'font_size': 2, 'font_thickness': 2,
               'handedness_text_color': (88, 205, 54)}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}


#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, model_asset_path, num_hands=2):
        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path)
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=num_hands)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    # do not change the function name and args
    def detect(self, bgr):
        return self.detector.detect(self._bgr_to_mp(bgr))

    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp

## "additional parts" refers to the code added from flet_hands.py
## (start) additional parts
import csv
class Hand_classifier():
    def __init__(self):
        self.rps_gesture = {
            0:'fist(rock)', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five(paper)',
            6:'six', 7:'rock', 8:'spiderman', 9:'yeah(scissors)', 10:'ok',
        }
        self.rps_ref = {
            "rock":0, "scissor":9, "paper":5
        }
        ## Download hand landmark samples from github (https://github.com/kairess/Rock-Paper-Scissors-Machine/raw/main/data/gesture_train.csv) to build a gesture classifier
        if os.path.exists("./gesture_train.csv"):
            file = np.genfromtxt('./gesture_train.csv', delimiter=',')
        else:
            file = requests.get('https://github.com/kairess/Rock-Paper-Scissors-Machine/raw/main/data/gesture_train.csv').content.decode('utf-8')
            file = np.array(list(csv.reader(file.splitlines(), delimiter=',')))
        angle = file[:,:-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        self._ReturnLandmarks = lambda landmark: [landmark.x, landmark.y, landmark.z]

    def compute_angle(self, landmarks):
        landmark_points = np.array(list(map(self._ReturnLandmarks, landmarks)))
        # Compute angles between joints
        v1 = landmark_points[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = landmark_points[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3]

        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
        angle = np.degrees([angle]) # Convert radian to degree
        return angle.astype(np.float32)
    
    def detect(self, angle):
        result = self.knn.findNearest(angle, 3)
        return int(result[1][0][0])
    
    def return_class_name(self, index):
        return self.rps_gesture[index]

    def referee(self, classes):
        win_lose = np.zeros(classes.shape)
        rock_nums = (classes==self.rps_ref["rock"]).sum()
        scissor_nums = (classes==self.rps_ref["scissor"]).sum()
        paper_nums = (classes==self.rps_ref["paper"]).sum()
        total = rock_nums+scissor_nums+paper_nums
        if total > 1:
            winners = []
            if rock_nums > 0 and paper_nums > 0 and scissor_nums > 0:
                text = 'Tie'
            elif rock_nums > 0 and paper_nums == 0 and scissor_nums == 0:
                text = 'Tie'
            elif rock_nums == 0 and paper_nums > 0 and scissor_nums == 0:
                text = 'Tie'
            elif rock_nums == 0 and paper_nums == 0 and scissor_nums > 0:
                text = 'Tie'
            elif rock_nums > 0 and scissor_nums > 0 and paper_nums == 0:
                text = 'Rock wins'
                win_lose[classes==self.rps_ref["rock"]] = 1
            elif rock_nums > 0 and paper_nums > 0 and scissor_nums == 0:
                text = 'Paper wins'
                win_lose[classes==self.rps_ref["paper"]] = 1
            elif paper_nums > 0 and scissor_nums > 0 and rock_nums == 0:
                text = 'Scissors wins'
                win_lose[classes==self.rps_ref["scissor"]] = 1
        else:
            win_lose = None
            text = None
        return win_lose, text
## (end) additional parts

#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(bgr)
# will be used in flet_util.ftImshow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
class Drawer():
    def __init__(self, bfps=True,
                 margin=10,  # pixels
                 handedness_text_color=(88,205,54),
                 font_size=1, font_thickness=1):
        self.opts = {'margin': margin, 'handedness_text_color': handedness_text_color,  
                     'font_size': font_size, 'font_thickness': font_thickness}
        self.bfps = bfps
        ## (start) additional parts
        self.HandGestureClassifier = Hand_classifier()
        ## (end) additional parts

    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_result, bgr):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated = self._bgr_to_mp(bgr).numpy_view()

        ## (start) additional parts
        hand_gesture_result = []
        for res in detection_result.hand_landmarks:
            angle = self.HandGestureClassifier.compute_angle(res)
            result = self.HandGestureClassifier.detect(angle)
            hand_gesture_result.append(result)
        hand_gesture_result = np.array(hand_gesture_result)
        win_lose, text = self.HandGestureClassifier.referee(hand_gesture_result)
        ## (end) additional parts

        # Loop through the detected hands to visualize.
        if len(hand_landmarks_list):
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - self.opts['margin']

                # Draw handedness (left or right hand) on the image.
                cv2.putText(annotated, f"{handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            self.opts['font_size'], self.opts['handedness_text_color'], 
                            self.opts['font_thickness'], cv2.LINE_AA)
                
                ## (start) additional parts
                cv2.putText(annotated, text=self.HandGestureClassifier.return_class_name(hand_gesture_result[idx]), 
                            org=(text_x, text_y+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                if win_lose is not None:
                    if win_lose[idx]:
                        cv2.putText(annotated, text="Winner!", 
                                    org=(text_x, text_y-50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,   0,   0), thickness=2)
                ## (end) additional parts
        ## (start) additional parts
        if win_lose is not None:
            cv2.putText(annotated, text=text, 
                        org=(0, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,   0,   0), thickness=2)
        ## (end) additional parts
        return cv2.addWeighted(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), 0.7, bgr, 0.3, 0)

    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp


#### (3/3) Define how to display in a page ####
# you will use this as
# contents = Section(cap, imgproc=imgproc, **section_opts).create()
class Section():
    def __init__(self, cap=None, imgproc=None, **kwargs):
        self.cap = cap
        self.imgproc = imgproc
        self.img_size = (480,640)
        self.cameras = ["0"]
        self.keep_running = False
        self.bottom_margin = 40
        self.elevation = 30
        self.padding = 10
        self.border_radius = 20
        self.__dict__.update(kwargs)
        self.controls = {}

    def set_cap(self, dummy):
        if self.controls['cap_view'].images is None:
            self.controls['cap_view'].SetSource(int(self.controls['dd'].value))
            self.controls['cap_view'].Renew()
        else:
            self.controls['cap_view'].SetSource(self.controls['cap_view'].images.index(self.controls['dd'].value))
            self.controls['cap_view'].Renew()

    def set_mirror(self, dummy):
        self.controls['cap_view'].mirror = self.controls['sw_mirror'].value
        self.controls['cap_view'].Renew()

    def create(self):
        self.controls['cap_view'] = ftImShow(self.cap, imgproc=self.imgproc, keep_running=self.keep_running,
                                             hw=self.img_size, border_radius=self.border_radius)
        ddlist = self.cameras if self.controls['cap_view'].images is None else self.controls['cap_view'].images
        self.controls['dd'] = ft.Dropdown(label="Camera/Image", width=256, 
                        options=[ft.dropdown.Option(c) for c in ddlist],
                        on_change=self.set_cap)
        self.controls['sw_mirror'] = ft.Switch(label="Mirror", value=self.controls['cap_view'].mirror, label_position=ft.LabelPosition.LEFT,
                                               on_change=self.set_mirror)

        section = ft.Container(
            margin=ft.margin.only(bottom=self.bottom_margin),
            content=ft.Column([
                ft.Card(
                    elevation=self.elevation,
                    content=ft.Container(
                        bgcolor=ft.colors.WHITE24,
                        padding=self.padding,
                        border_radius = ft.border_radius.all(self.border_radius),
                        content=ft.Column([
                            self.controls['cap_view'], 
                            ft.Row([self.controls['dd'], self.controls['sw_mirror']])
                        ],
                        tight=True, spacing=0
                        ),
                    )
                ),
            ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
        return section

    def terminate(self):
        self.controls['cap_view'].keep_running = False



imgproc = {'DETECTOR': Detector, 'DETECTOR_PARAMS': detector_params, 
           'DRAWER': Drawer, 'DRAWER_OPTS': drawer_opts,
           'IMAGES': args['images'], 'MIRROR': True}

import sys
def main(page: ft.Page):

    cap = None
    if len(sys.argv) > 1: # force to use the specified camera
        imgproc['IMAGES'] = None
        section_opts['keep_running'] = True
        cap = cvVideoCapture(int(sys.argv[1]), hw=args['frame_hw'])
        args['cameras'] = sys.argv[2:]
    elif imgproc['IMAGES'] is None:
        cap = cvVideoCapture(0, hw=args['frame_hw'])
    else: # use IMAGES
        imgproc['MIRROR'] = False
        section_opts['keep_running'] = False
    #print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), int(cap.get(cv2.CAP_PROP_FPS) + 0.5))

    section = Section(cap, imgproc=imgproc, cameras=args['cameras'], **section_opts)
    contents = section.create()

    # def on_disconnect( _: ft.ControlEvent):
    #         if cap is not None:
    #             cap.release()
    #         print("on_disconnect")
    #         section.terminate()
    # page.on_disconnect = on_disconnect

    set_page(page, PageOpts)
    page.on_window_event = lambda e: (cap.release() if cap is not None else None, cv2.waitKey(1000), page.window_destroy()) if e.data == "close" else None
    page.update()
    page.add(contents)

if __name__ == '__main__':
    ft.app(target=main, **args['app'])