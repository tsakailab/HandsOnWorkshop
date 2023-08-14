import flet as ft
import base64
import numpy as np
import cv2
from ultralytics import YOLO

resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}

args = {'resolusion': resols['qHD'], 'padding': 10}

PageOpts = {'TITLE': "Object Detection (YOLOv8)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+400, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

def set_page(page, PageOpts):
    page.title = PageOpts['TITLE']
    page.theme_mode = PageOpts['THEME_MODE']
    page.vertical_alignment = PageOpts['VERTICAL_ALIGNMENT']
    page.horizontal_alignment = PageOpts['HORIZONTAL_ALIGNMENT']
    page.padding = PageOpts['PADDING']
    page.window_height, page.window_width = PageOpts['WINDOW_HW']
    if PageOpts['_WINDOW_TOP_LEFT_INCR']:
        page.window_top += PageOpts['WINDOW_TOP_LEFT'][0]
        page.window_left += PageOpts['WINDOW_TOP_LEFT'][1]
    else:
        page.window_top, page.window_left = PageOpts['WINDOW_TOP_LEFT']


# defaults
#detector_params = {'model_asset_path': "./object_detector.tflite", 'score_threshold': 0.3}
detector_params = {'modelname': "yolov8n", 'imgsz': 960, 'conf': 0.25, 'iou': 0.7}
drawer_opts = {}
section_opts = {'img_size': args['resolution'], 'bottom_margin': 40, 'elevation': 20, 'padding':10, 'text_size': 20, 'border_radius': 20}


#### (1/3) Define a detector ####
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, modelname, imgsz=32, conf=0.25, iou=0.7):
        self.detector = YOLO(modelname)
        self.opts = {'imgsz': imgsz, 'conf': conf, 'iou': iou}

    # do not change the function name and args
    def detect(self, bgr):
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        return self.detector.predict(bgr, verbose=False, **self.opts)


#### (2/3) Define a Rrawer according to the detector result ####
# result = detector.detect(_bgr_to_mp(bgr))
# bgr = Drawer(**draw_opts).draw(result, bgr)
class Drawer():
    def __init__(self):
        self.objects = {}

    # do not change the function name and args
    def draw(self, detection_result, bgr):
        annotated = detection_result[0].plot()
        return cv2.addWeighted(annotated, 0.5, bgr, 0.5, 0)


# macros
imgfmt, base64code = '.jpg', 'ascii'
def _bgr_to_base64(bgr):
    src_base64 = base64.b64encode(
                cv2.imencode(imgfmt, bgr)[1]
                ).decode(base64code)
    return src_base64

class fps_counter():
    def __init__(self, max_count=10):
        self.fps = 0.0
        self.tm = cv2.TickMeter()
        self.count = 0
        self.max_count = max_count
        self.tm.start()

    def count(self):
        self.count += 1

    def count_get(self):
        self.count += 1
        if self.count == self.max_count:
            self.tm.stop()
            self.fps = self.max_count / self.tm.getTimeSec()
            self.tm.reset()
            self.tm.start()
            self.count = 0
        return self.fps


class ftimshow(ft.UserControl):
    def __init__(self, cap, imgproc=None, hw=(480,640)):
        super().__init__()
        self.cap = cap
        self.imgproc = imgproc
        self.hw = hw
        self.capid = 0
        self.mirror = imgproc['MIRROR'] if imgproc is not None and 'MIRROR' in imgproc else None
        self.images = imgproc['IMAGES'] if imgproc is not None and 'IMAGES' in imgproc else None
        self.bgr = np.zeros((hw[0],hw[1],3), np.uint8)
        self.src_base64 = _bgr_to_base64(self.bgr)
        self.img = ft.Image(
            src_base64=self.src_base64,
            width=hw[1], height=hw[0],
            fit=ft.ImageFit.CONTAIN,
            border_radius=ft.border_radius.all(20)
        )
        self.detector=None
        self.drawer=None

    def _imread(self):
        if isinstance(self.images, list) and self.capid < len(self.images):
            success, bgr = True, cv2.imread(self.images[self.capid])
        else:
            success, bgr = False, None
        return success, bgr

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        fps_timer = fps_counter()

        while True:
            if self.cap.isOpened():
                success, frame = self.cap.read()
            else:
                success, frame = self._imread()

            if not success:
                continue

            frame = cv2.resize(frame, (self.hw[1],self.hw[0]))
            if self.mirror:
                frame = cv2.flip(frame, 1)
            if self.imgproc is not None:
                if self.detector is None and 'DETECTOR' in self.imgproc:
                    self.detector = self.imgproc['DETECTOR'](**self.imgproc['DETECTOR_PARAMS'])
            #    self.detector = self.imgproc['detector'](**DETECTOR_PARAMS)
                result = self.detector.detect(frame)
                if self.drawer is None and 'DRAWER' in self.imgproc:
                    self.drawer = self.imgproc['DRAWER'](**self.imgproc['DRAWER_OPTS'])
                #self.bgr = self.imgproc['DRAWER'](result, frame, **self.imgproc['DRAWER_OPTS'])
                self.bgr = self.drawer.draw(result, frame)
            else:
                self.bgr = frame

            fps = fps_timer.count_get()
            cv2.putText(self.bgr, 'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (128,128,0), thickness=2)

            self.src_base64 = _bgr_to_base64(self.bgr)
            self.img.src_base64 = self.src_base64
            self.update()

    def build(self):
        return self.img

    def RenewDetector(self, newparam):
        self.imgproc['DETECTOR_PARAMS'].update(newparam)
        self.detector=None

    def SetSource(self, cid):
        self.capid = cid
        if self.images is None:
            self.cap.release()
            self.cap = cv2.VideoCapture(cid)



#### (3/3) Set controlable values by sliders #### 
# section = CreateSection(cap, imgproc=imgproc, **section_opt)
CAMERAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
def CreateSection(cap, imgproc=None, img_size=(480,640),
                  bottom_margin=40, elevation=30, padding=10, text_size=20, border_radius=20):

    cap_view = ftimshow(cap, imgproc=imgproc, hw=img_size)

    def set_cap(dummy):
        cap_view.SetSource(int(dd.value))
    ddlist = CAMERAS if cap_view.images is None else cap_view.images
    dd = ft.Dropdown(label="Camera/Image", width=256, 
                     options=[ft.dropdown.Option(c) for c in ddlist],
                     on_change=set_cap)

    def set_mirror(dummy):
        cap_view.mirror = sw.value
    sw = ft.Switch(label="Mirror", value=True, label_position=ft.LabelPosition.LEFT,
                   on_change=set_mirror)

    section = ft.Container(
        margin=ft.margin.only(bottom=bottom_margin),
        content=ft.Column([
            ft.Card(
                elevation=elevation,
                content=ft.Container(
                    bgcolor=ft.colors.WHITE24,
                    padding=padding,
                    border_radius = ft.border_radius.all(border_radius),
                    content=ft.Column([
                        cap_view, ft.Row([dd, sw])
                    ],
                    tight=True, spacing=0
                    ),
                )
            ),
            ft.Container(
                bgcolor=ft.colors.WHITE24,
                padding=padding,
                border_radius=ft.border_radius.all(border_radius),
                content=ft.Column([
                    ft.Slider(
                        min=1, max=99, 
                        on_change=lambda e: cap_view.RenewDetector({'conf': e.control.value/100}) #print(e.control.value)
                    ),
                    # ft.Slider(
                    #     min=500, max=900,
                    # )
                ]
                ),
            )
        ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    return section


imgproc = {'DETECTOR': Detector, 'DETECTOR_PARAMS': detector_params, 
           'DRAWER': Drawer, 'DRAWER_OPTS': drawer_opts,
           'IMAGES': None, 'MIRROR': True}
#imgproc.update({'IMAGES': ['image0.jpg', 'image1.png']}) # works if cap.isOpened() is False

cap = cv2.VideoCapture(0) # set None to use IMAGES
caprelease = lambda _: cap.release()
def main(page: ft.Page):

    set_page(page, PageOpts)
    page.update()

    section = CreateSection(cap, imgproc=imgproc, **section_opts)
    page.add(
        section,
    )
    #page.on_disconnect(caprelease)

if __name__ == '__main__':
    ft.app(target=main)
    #ft.app(target=main, view=ft.WEB_BROWSER)
    cap.release()
    cv2.destroyAllWindows()