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

    page.window_prevent_close = True
    #page.add(ft.FilledButton(text="Filled button", on_click=lambda e: page.window_destroy()))


import flet as ft
import base64
import numpy as np
import cv2
import os

_CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
_CAP_PROP_FPS = cv2.CAP_PROP_FPS

class cvVideoCapture():
    def __init__(self, capid=0, width=None, height=None, fps=30, hw=(480,640)):
        self.cid = capid
        self.cap = None
        self.hw = [hw[0], hw[1]]
        if height is not None: self.hw[0] = height
        if width is not None: self.hw[1] = width
        self.fps = fps
        if os.name == "nt": 
            self.cap = cv2.VideoCapture(capid, cv2.CAP_DSHOW)
        elif os.name == "posix":
            self.cap = cv2.VideoCapture(capid)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','2'))
        else:
            self.cap = cv2.VideoCapture(capid)

        self.set(_CAP_PROP_FRAME_WIDTH, self.hw[1])
        self.set(_CAP_PROP_FRAME_HEIGHT, self.hw[0])
        self.set(_CAP_PROP_FPS, self.fps)

    def isOpened(self):
        return self.cap.isOpened()
    def read(self, **kwargs):
        return self.cap.read(**kwargs)
    def release(self):
        return self.cap.release()

    def set(self, src, dst):
        #_decode_forcc = lambda v: "".join([chr((int(v) >> 8 * i) & 0xFF) for i in range(4)])
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*[c for c in fourcc]))
        return self.cap.set(src, dst)


    def get(self, src):
        return self.cap.get(src)


# macros
imgfmt, base64code = '.jpg', 'ascii'
def bgr_to_base64(bgr):
    src_base64 = base64.b64encode(
                cv2.imencode(imgfmt, bgr)[1]
                ).decode(base64code)
    return src_base64

class _fps_counter():
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


class ftImShow(ft.UserControl):
    def __init__(self, cap, imgproc=None, hw=(480,640), VideoCapture=cv2.VideoCapture,
                 border_radius=ft.border_radius.all(20), keep_running=False):
        super().__init__()
        self.cap = cap
        self.imgproc = imgproc
        self.hw = hw
        self.VideoCapture = VideoCapture
        self.capid = 0
        self.mirror = imgproc['MIRROR'] if imgproc is not None and 'MIRROR' in imgproc else None
        self.images = imgproc['IMAGES'] if imgproc is not None and 'IMAGES' in imgproc else None
        self._bgr = np.zeros((hw[0],hw[1],3), np.uint8)
        self._src_base64 = bgr_to_base64(self._bgr)
        self.controls.append(ft.Image(src_base64=self._src_base64, width=hw[1], height=hw[0],
                                  fit=ft.ImageFit.CONTAIN, border_radius=border_radius))
        self.detector=None
        self.drawer=None
        self.keep_running = keep_running
        self.detection_result = None

    def _imread(self):
        success, bgr = False, None
        if isinstance(self.images, list) and self.capid < len(self.images):
            #success, bgr = True, cv2.imread(self.images[self.capid])
            success = os.path.isfile(self.images[self.capid])
            if success:
                bgr = cv2.imread(self.images[self.capid])
            else:
                success, bgr = True, np.zeros((self.hw[0],self.hw[1],3), np.uint8)
        return success, bgr

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        fps_timer = _fps_counter()
        while self.keep_running:
            self._detect_draw()
            if self.drawer is not None and self.drawer.bfps:
                fps = fps_timer.count_get()
                cv2.putText(self._bgr, 'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (128,128,0), thickness=2)
            self._src_base64 = bgr_to_base64(self._bgr)
            self.controls[0].src_base64 = self._src_base64
            self.update()
        if not self.keep_running:
            self.Renew()

    def _detect_draw(self):
        if self.cap is not None and self.cap.isOpened():
            success, frame = self.cap.read()
        else:
            success, frame = self._imread()
        if not success:
            return
            #frame = self._bgr
        frame = cv2.resize(frame, (self.hw[1],self.hw[0]))
        if self.mirror:
            frame = cv2.flip(frame, 1)
        self.detection_result = None
        if self.imgproc is not None:
            if self.detector is None and 'DETECTOR' in self.imgproc:
                self.detector = self.imgproc['DETECTOR'](**self.imgproc['DETECTOR_PARAMS'])
            self.detection_result = self.detector.detect(frame)
            if self.drawer is None and 'DRAWER' in self.imgproc:
                self.drawer = self.imgproc['DRAWER'](**self.imgproc['DRAWER_OPTS'])
            if self.detection_result is not None:
                self._bgr = self.drawer.draw(self.detection_result, frame)
        else:
            self._bgr = frame

    def build(self):
        return self.controls

    def Renew(self):
        self._detect_draw()
        self._src_base64 = bgr_to_base64(self._bgr)
        self.controls[0].src_base64 = self._src_base64
        self.update()

    def RenewDetector(self, newparam):
        self.imgproc['DETECTOR_PARAMS'].update(newparam)
        self.detector=None
        #self.detector = self.imgproc['DETECTOR'](**self.imgproc['DETECTOR_PARAMS'])
        return self

    def RenewDrawer(self, newparam):
        self.imgproc['DRAWER_OPTS'].update(newparam)
        self.drawer=None
        #self.drawer = self.imgproc['DRAWER'](**self.imgproc['DRAWER_OPTS'])
        return self

    def SetSource(self, cid, VideoCapture=cvVideoCapture):
        self.capid = cid
        self.VideoCapture = VideoCapture
        if self.images is None:
        #if self.cap is not None:
            if self.cap is not None:
                w, h = self.cap.get(_CAP_PROP_FRAME_WIDTH), self.cap.get(_CAP_PROP_FRAME_HEIGHT)
                fps = self.cap.get(_CAP_PROP_FPS)
                self.cap.release()
            self.cap = self.VideoCapture(cid, w, h, fps)

        return self




import itertools
def _show_param(cap):
    app_ids = [
        # (cv2.CAP_ANY, 'CAP_ANY'),
        (cv2.CAP_DSHOW, 'CAP_DSHOW'),
        (cv2.CAP_MSMF, 'CAP_MSMF'),
        (cv2.CAP_V4L2, 'CAP_V4L2'),
    ]
    fourcc_list = [
        (cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 'MJPG'),
        (cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'), 'YUYV'),
        (cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'), 'YUY2'),
        (cv2.VideoWriter_fourcc('H', '2', '6', '4'), 'H264'),
        (cv2.VideoWriter_fourcc('B', 'G', 'R', '3'), 'BGR3'),
    ]
    frame_list = [(1920, 1080), (1280, 1024), (1280, 720), (800, 600), (640, 480)]
    fps_list = [60, 30, 24, 20, 15, 10, 5, 2, 1]

    for dev_id, api_id in itertools.product(range(10), app_ids):
        cap = cv2.VideoCapture(dev_id, api_id[0])
        ret = cap.isOpened()
        if ret is False:
            continue

        backend = cap.getBackendName()
        print("Camera #%d (%s : %s) :" % (dev_id, api_id[1], backend))

        for fourcc in fourcc_list:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc[0])
            ret_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc[0] != ret_fourcc:
                continue

            for frame in frame_list:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame[1])
                ret_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ret_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if frame[0] != ret_w or frame[1] != ret_h:
                    continue

                for fps in fps_list:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    ret_fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
                    if fps != ret_fps:
                        continue

                    print('  Frame: %4d x %4d , FPS: %3d , FourCC: %s' % 
                            (ret_w, ret_h, ret_fps, fourcc[1]))




