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


PageOpts = {'TITLE': "Pose Landmarker (mediapipe)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+240, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
detector_params = {'model_asset_path': "./pose_detector.task", 
                   'url': "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
                   'output_segmentation_masks': True}
drawer_opts = {'bfps': True}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}


#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, model_asset_path, url=None, output_segmentation_masks=True):
        if url is not None:
            success = os.path.isfile(model_asset_path)
            if not success:
                r = requests.get(url)
                with open(model_asset_path, "wb") as file:
                    file.write(r.content)
                    file.flush()

        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(base_options=base_options,
                                       output_segmentation_masks=output_segmentation_masks)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # do not change the function name and args
    def detect(self, bgr):
        return self.detector.detect(self._bgr_to_mp(bgr))

    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp


#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(bgr)
# will be used in flet_util.ftImShow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
class Drawer():
    def __init__(self, bfps=True, **kwargs):
        self.opts = kwargs
        self.bfps = bfps

    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_result, bgr):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated = self._bgr_to_mp(bgr).numpy_view()

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        
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