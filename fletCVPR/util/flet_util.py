
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
        result = None
        if self.imgproc is not None:
            if self.detector is None and 'DETECTOR' in self.imgproc:
                self.detector = self.imgproc['DETECTOR'](**self.imgproc['DETECTOR_PARAMS'])
            result = self.detector.detect(frame)
            if self.drawer is None and 'DRAWER' in self.imgproc:
                self.drawer = self.imgproc['DRAWER'](**self.imgproc['DRAWER_OPTS'])
            if result is not None:
                self._bgr = self.drawer.draw(result, frame)
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

    def SetSource(self, cid, VideoCapture=cv2.VideoCapture):
        self.capid = cid
        self.VideoCapture = VideoCapture
        if self.images is None:
        #if self.cap is not None:
            if self.cap is not None:
                self.cap.release()
            self.cap = self.VideoCapture(cid)#, cv2.CAP_DSHOW)
        return self
