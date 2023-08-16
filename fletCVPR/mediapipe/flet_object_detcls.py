import flet as ft
import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.append("../util")
from flet_util import set_page, ftImShow


resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': resols['qHD'], 'padding': 10, 
        'images': None}
args.update({'images': ['../dashcam.jpg', '../park.jpg']}) # works if cap.isOpened() is False


PageOpts = {'TITLE': "Object Detection (YOLOv8)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+240, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
#detector_params = {'model_asset_path': "./object_detector.tflite", 'score_threshold': 0.3}
detector_params = {'model_asset_path': "./efficientdet_lite2_float32.tflite", 'score_threshold': 0.3}
drawer_opts = {'bfps': True, 'margin': 10, 'row_size': 10, 'font_size': 2, 'font_thickness': 2, 'line_thickness': 2}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
                'slider': {'score_threshold': {'width': 400, 'value': int(detector_params['score_threshold']*100), 'min': 1, 'max': 99, 'divisions': 98, 'label': "{value}/100"}}, 
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}


#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, model_asset_path, score_threshold=0.3):
        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path)
        options = mp.tasks.vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=score_threshold)
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

    # do not change the function name and args
    def detect(self, bgr):
        return self.detector.detect(self._bgr_to_mp(bgr))

    def _bgr_to_mp(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return rgb_mp


#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(bgr)
# will be used in flet_util.ftImshow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
class Drawer():
    def __init__(self, bfps=True,
                 margin=10,  # pixels
                 row_size=10,  # pixels
                 line_thickness=3, font_size=1, font_thickness=1):
        self.opts = {'margin': margin, 'row_size': row_size, 'line_thickness': line_thickness, 
                     'font_size': font_size, 'font_thickness': font_thickness}
        self.bfps = bfps
        self.objects = {}
        # 80 objects (https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)
        np.random.seed(0)
        self.colors = [[np.random.randint(0,255) for _ in range(3)] for _ in range(80)]

    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_result, bgr):
        object_names = []
        for detection in detection_result.detections:
            object_names.append(detection.categories[0].category_name)
        self.update_objects_with_unique_ids(object_names)

        annotated = bgr.copy()
        for detection in detection_result.detections:
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            bbox = detection.bounding_box
            obj_color = self.colors[self.objects[category_name]]

            # Draw bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated, start_point, end_point, obj_color, thickness=self.opts['line_thickness'])

            # Draw label and score
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.opts['margin'] + bbox.origin_x,
                            self.opts['margin'] + self.opts['row_size'] + bbox.origin_y)
            cv2.putText(annotated, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.opts['font_size'], obj_color, self.opts['font_thickness'])

        return cv2.addWeighted(annotated, 0.7, bgr, 0.3, 0)

    def update_objects_with_unique_ids(self, object_names):
        new_names = set(object_names) - set(self.objects.keys())
        if len(new_names) > 0:
            unique_ids = list(range(len(self.objects), len(self.objects) + len(new_names)))
            self.objects.update(zip(new_names, unique_ids))


#### (3/3) Define how to display in a page ####
# you will use this as
# contents = Section(cap, imgproc=imgproc, **section_opts).create()
CAMERAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
class Section():
    def __init__(self, cap=None, imgproc=None, **kwargs):
        self.cap = cap
        self.imgproc = imgproc
        self.img_size = (480,640)
        self.keep_running = False
        self.slider = None
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
        ddlist = CAMERAS if self.controls['cap_view'].images is None else self.controls['cap_view'].images
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
                ft.Container(
                    bgcolor=ft.colors.WHITE24,
                    padding=self.padding,
                    border_radius=ft.border_radius.all(self.border_radius),
                    content=ft.Column([
                        ft.Slider(
                            on_change=lambda e: self.controls['cap_view'].RenewDetector({'score_threshold': e.control.value/100}).Renew(),
                            **self.slider['score_threshold']
                        ),
                    ]
                    ),
                )
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
    if imgproc['IMAGES'] is None:
        if len(sys.argv) > 1: # force to use the specified camera
            imgproc['IMAGES'] = None
            section_opts['keep_running'] = True
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(0)
    else: # use IMAGES
        imgproc['MIRROR'] = False
        section_opts['keep_running'] = False

    section = Section(cap, imgproc=imgproc, **section_opts)
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
