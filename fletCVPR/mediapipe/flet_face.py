import flet as ft
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Union
import math
import sys
sys.path.append("../util")
from flet_util import set_page, ftImShow


resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}
CAMERAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': resols['qHD'], 'padding': 10, 'cameras': CAMERAS,
        'images': None}
#args.update({'images': ['../dashcam.jpg', '../park.jpg']}) # works if cap.isOpened() is False


PageOpts = {'TITLE': "Face Detection (mediapipe)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+240, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
#detector_params = {'model_asset_path': "./object_detector.tflite", 'score_threshold': 0.3}
detector_params = {'model_asset_path': {'face_bbox': "./face_detector.tflite", 'face_mesh': "./facemesh_detector.task"},
                   'num_faces': 2, 'mode': 'face_bbox'}
drawer_opts = {'bfps': True, 
               'face_bbox': {'MARGIN': 10, 'ROW_SIZE': 10, 'FONT_SIZE': 1, 'FONT_THICKNESS': 1, 'TEXT_COLOR': (0,0,255)},
               'face_mesh': {}}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}



#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, mode, model_asset_path, num_faces):
        self.mode = mode # 'face_bbox' or 'face_mesh'
        self.detector = {}
        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path['face_bbox'])
        options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options)
        self.detector['face_bbox'] = mp.tasks.vision.FaceDetector.create_from_options(options)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path['face_mesh'])
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, 
                                                        output_face_blendshapes=True,
                                                        output_facial_transformation_matrixes=True,
                                                        num_faces=num_faces)
        self.detector['face_mesh'] = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # do not change the function name and args
    def detect(self, bgr):
        return (self.mode, self.detector[self.mode].detect(self._bgr_to_mp(bgr)))

    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp


#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(bgr)
# will be used in flet_util.ftImshow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
class Drawer():
    def __init__(self, bfps=True, **kwargs):
        self.opts = {}
        self.bfps = bfps
        self.opts.update(kwargs)
        self.drawer = {}
        self.drawer['face_bbox'] = self._draw_bbox
        self.drawer['face_mesh'] = self._draw_mesh

    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_results, bgr):
        mode, detection_result = detection_results
        annotated = self.drawer[mode](bgr, detection_result)
        return cv2.addWeighted(annotated, 0.5, bgr, 0.5, 0)


    def _draw_bbox(self, bgr, detection_result) -> np.ndarray:
        annotated = bgr.copy()
        height, width, _ = bgr.shape

        if len(detection_result.detections):
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(annotated, start_point, end_point, self.opts['face_bbox']['TEXT_COLOR'], 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.opts['face_bbox']['MARGIN'] + bbox.origin_x,
                             self.opts['face_bbox']['MARGIN'] + self.opts['face_bbox']['ROW_SIZE'] + bbox.origin_y)
            cv2.putText(annotated, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.opts['face_bbox']['FONT_SIZE'], self.opts['face_bbox']['TEXT_COLOR'], self.opts['face_bbox']['FONT_THICKNESS'])

        return annotated


    def _draw_mesh(self, bgr, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated = bgr[...,::-1].copy()

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated[...,::-1]


    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp

    def _normalized_to_pixel_coordinates(self,
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


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

    def set_mode(self, e):
        self.controls['cap_view'].RenewDetector({'mode': e.control.value})
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
        self.controls['rg_mode'] = ft.RadioGroup(ft.Row([
                                            ft.Radio(value='face_bbox', label="Box"), 
                                            ft.Radio(value='face_mesh', label="Mesh")]),
                                            on_change=self.set_mode)

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
                            ft.Row([self.controls['dd'], self.controls['sw_mirror'], self.controls['rg_mode']])
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
        cap = cv2.VideoCapture(int(sys.argv[1]))
        args['cameras'] = sys.argv[2:]
    elif imgproc['IMAGES'] is None:
        cap = cv2.VideoCapture(0)
    else: # use IMAGES
        imgproc['MIRROR'] = False
        section_opts['keep_running'] = False

    section = Section(cap, imgproc=imgproc, cameras=args['cameras'], **section_opts)
    contents = section.create()

    # def on_disconnect( _: ft.ControlEvent):
    #         if cap is not None:
    #             cap.release()
    #         print("on_disconnect")
    #         section.terminate()
    # page.on_disconnect = on_disconnect

    set_page(page, PageOpts)
    page.update()
    page.add(contents)

if __name__ == '__main__':
    ft.app(target=main, **args['app'])