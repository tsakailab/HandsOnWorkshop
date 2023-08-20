import pyrealsense2 as rs
import flet as ft
import base64
import numpy as np
import cv2

# cv2.VideoCapture-like class for realsense
# depth is aligned to color by default.
# Usage:
# cap = rsVideoCapture()
# if cap.isOpened():
#   success, frame = cap.read()
# if success:
#   bgr, depth = frame
class rsVideoCapture():
    def __init__(self, capid=0, hw=(480,640), align=rs.align(rs.stream.color),
                 format={'color': rs.format.bgr8, 'depth': rs.format.z16}, fps=30):
        self.cid = capid
        self.align = align
        config = rs.config()
        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color, hw[1], hw[0], format['color'], fps)
        config.enable_stream(rs.stream.depth, hw[1], hw[0], format['depth'], fps)

        self.pipeline = rs.pipeline()
        self.isopened = False if self.pipeline is None else True
        self.profile = self.pipeline.start(config)
        # Get camera parameters
        self.intr = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.scale = config.resolve(rs.pipeline_wrapper(self.pipeline)).get_device().first_depth_sensor().get_depth_scale()

    def isOpened(self):
        return self.isopened
    def read(self):
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        bgr = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not depth or not bgr:
            self.isopened = False
        bgr = np.asanyarray(bgr.get_data())
        depth = np.asanyarray(depth.get_data())        
        return self.isopened, [bgr, depth]
       
    def release(self):
        self.pipeline.stop()
        return


# macros
imgfmt, base64code = '.jpg', 'ascii'
def _bgr_to_base64(bgr):
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


class ftImShow(ft.UserControl): # for RealSense
    def __init__(self, cap, imgproc=None, hw=(480,640), VideoCapture=rsVideoCapture,
                 border_radius=ft.border_radius.all(20), keep_running=False, cids=(0,1)):
        super().__init__()
        self.cap = cap
        self.imgproc = imgproc
        self.hw = hw
        self.VideoCapture = VideoCapture
        self.capid = 0
        self.mirror = imgproc['MIRROR'] if imgproc is not None and 'MIRROR' in imgproc else None
        #self.images = imgproc['IMAGES'] if imgproc is not None and 'IMAGES' in imgproc else None
        self._frame = np.zeros((hw[0],hw[1],3), np.uint8), np.zeros((hw[0],hw[1]), np.uint16)
        self.cids = cids
        self._src_base64 = _bgr_to_base64(self._frame[0]), _bgr_to_base64(cv2.cvtColor(self._frame[1], cv2.COLOR_GRAY2BGR))
        self.controls.append(ft.Row([ft.Image(src_base64=self._src_base64[cids[0]], width=hw[1], height=hw[0],
                                     fit=ft.ImageFit.CONTAIN, border_radius=border_radius)]))
        for cid in cids[1:]:
            self.controls[0].controls.append(ft.Image(src_base64=self._src_base64[cid], width=hw[1], height=hw[0],
                                    fit=ft.ImageFit.CONTAIN, border_radius=border_radius))
        self.detector=None
        self.drawer=None
        self.keep_running = keep_running
        self.detection_result = None

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        fps_timer = _fps_counter()
        while self.keep_running:
            self._detect_draw()
            if self.drawer is not None and self.drawer.bfps:
                fps = fps_timer.count_get()
                self._frame[0] = cv2.putText(self._frame[0].copy(), 'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (128,128,0), thickness=1)
            # self._src_base64 = _bgr_to_base64(self._frame[0]), _bgr_to_base64(self._frame[1])
            # self.controls[0].controls[0].src_base64 = self._src_base64[0]
            # self.controls[0].controls[1].src_base64 = self._src_base64[1]
            # self.update()
            self.Renew()
        if not self.keep_running:
            self.Renew()

    def _detect_draw(self):
        if self.cap is not None and self.cap.isOpened():
            success, frame = self.cap.read()
        if not success:
            return
            #frame = self._bgr
        #frame = cv2.resize(frame[0], (self.hw[1],self.hw[0])), cv2.resize(frame[1], (self.hw[1],self.hw[0]))
        if self.mirror:
            frame = frame[0][:,::-1,...], frame[1][:,::-1,...] #cv2.flip(frame, 1)
        self.detection_result = None
        if self.imgproc is not None:
            if self.detector is None and 'DETECTOR' in self.imgproc:
                self.detector = self.imgproc['DETECTOR'](**self.imgproc['DETECTOR_PARAMS'])
            self.detection_result = self.detector.detect(frame)    # must accept tuple of a bgr image and depth data
            if self.drawer is None and 'DRAWER' in self.imgproc:
                self.drawer = self.imgproc['DRAWER'](**self.imgproc['DRAWER_OPTS'])
            if self.detection_result is not None:
                self._frame = self.drawer.draw(self.detection_result, frame)   # must return a pair of images in bgr format

    def build(self):
        return self.controls

    def Renew(self):
        self._detect_draw()
        self._src_base64 = _bgr_to_base64(self._frame[0]), _bgr_to_base64(self._frame[1])
        for cid in self.cids:
            self.controls[0].controls[cid].src_base64 = self._src_base64[cid]
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

    def SetSource(self, cid, VideoCapture=rsVideoCapture):
        self.capid = cid
        self.VideoCapture = VideoCapture
        if self.images is None:
        #if self.cap is not None:
            if self.cap is not None:
                self.cap.release()
            self.cap = self.VideoCapture(cid)#, cv2.CAP_DSHOW)
        return self
