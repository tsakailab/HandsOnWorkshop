import flet as ft
import cv2
import sys
sys.path.append("../util")
from flet_util import set_page
from flet_util_rs import ftImShow, rsVideoCapture

import numpy as np
import matplotlib.pyplot as plt


resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}
CAMERAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': (480,640), 'padding': 10, 'cameras': CAMERAS,
        'images': None}
args.update({'images': ['../dashcam.jpg', '../park.jpg']}) # works if cap.isOpened() is False


PageOpts = {'TITLE': "Color and Depth Viewer (RealSense)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]*2+240, args['resolution'][1]*2+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
detector_params = {}
drawer_opts = {'bfps': True, 'min_depth': None, 'max_depth': None}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
#                'slider': {'range': {'width': 400, 'value': int(3000), 'min': 500, 'max': 10000, 'divisions': 95, 'label': "{value}mm"}}, 
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}

r = 30
MarkerPointsCoordinates = ([
    (int(480//2+r*np.cos(np.pi/2)), int(640//2+r*np.sin(np.pi/2))),
    (int(480//2+r*np.cos(7*np.pi/6)), int(640//2+r*np.sin(7*np.pi/6))),
    (int(480//2+r*np.cos(11*np.pi/6)), int(640//2+r*np.sin(11*np.pi/6))),
])

#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.focal_length = [self.intr.fx, self.intr.fy]
        cx, cy = self.intr.ppx, self.intr.ppy # width*0.5, height*0.5
        j_to_u = lambda j: -(j - cx)
        i_to_v = lambda i: -(i - cy)
        self.u, self.v = np.meshgrid(j_to_u(np.arange(self.intr.width)), i_to_v(np.arange(self.intr.height)))

    # Do not change the function name. The first arg is the target bgr image.
    def detect(self, bgr_and_depth):
        Z = bgr_and_depth[1] * self.scale * 1e+3
        X, Y = self.Zuv_to_XY(Z)
        points = np.stack((X, Y, Z), axis=-1)
        p3 = self.split_3_points(points, MarkerPointsCoordinates, average_kernel_size=5)
        th = p3[0, -1] * 0.05

        # compute a unit normal vector
        normal = self.compute_normal_vector(p3)
        
        dist_index = np.abs((points - p3[0]).dot(normal)) < th
        return {"plane_index":dist_index, "points":points}
    
    def Zuv_to_XY(self, Z):
        X = Z*self.u/self.focal_length[0]
        Y = Z*self.v/self.focal_length[1]
        return X, Y
    
    def avelage_filter(self, points, index, kernel_size):
        return points[index[0]-kernel_size:index[0]+kernel_size, index[1]-kernel_size:index[1]+kernel_size].mean((0, 1))

    def split_3_points(self, points, index, average_kernel_size=1):
        return np.stack(
            [self.avelage_filter(points, index[0], average_kernel_size), 
             self.avelage_filter(points, index[1], average_kernel_size), 
             self.avelage_filter(points, index[2], average_kernel_size)]
        )
    def compute_normal_vector(self, points):
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal = normal / np.linalg.norm(normal)
        return normal

#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(bgr)
# will be used in flet_util.ftImshow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
class Drawer():
    def __init__(self, min_depth=None, max_depth=None, colormap=cv2.COLORMAP_JET, **kwargs):
        self.bfps = True
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.colormap = colormap
        self.__dict__.update(kwargs)
        self.depth_to_color = lambda depth: cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), colormap)

        self.point_parameters = {
            "color":(255, 0, 0),
            "scale":5
        }
    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_result, bgr_and_depth):
        dist_index = detection_result["plane_index"]
        bgr = bgr_and_depth[0]
        self.mask = np.zeros(bgr.shape)
        self.mask[..., 0] = 255
        mask = self.mask * dist_index[..., None]
        bgr = cv2.addWeighted(bgr, 1, mask.astype(np.uint8), 0.75, 1)

        cv2.circle(bgr, MarkerPointsCoordinates[0][::-1], self.point_parameters["scale"], self.point_parameters["color"], thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.putText(bgr, '{:4.2f}[mm]'.format(detection_result["points"][MarkerPointsCoordinates[0]][-1]), 
                    MarkerPointsCoordinates[0][::-1], fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=(255, 255, 255), thickness=2)
        cv2.circle(bgr, MarkerPointsCoordinates[1][::-1], self.point_parameters["scale"], self.point_parameters["color"], thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(bgr, MarkerPointsCoordinates[2][::-1], self.point_parameters["scale"], self.point_parameters["color"], thickness=-1, lineType=cv2.LINE_8, shift=0)
        return [bgr, self.depth_to_color(bgr_and_depth[1])]


#### (3/3) Define how to display in a page ####
# you will use this as
# contents = Section(cap, imgproc=imgproc, **section_opts).create()
class Section():
    def __init__(self, cap=None, imgproc=None, **kwargs):
        self.cap = cap
        self.imgproc = imgproc
        self.imgproc["DETECTOR_PARAMS"]["intr"] = self.cap.intr
        self.imgproc["DETECTOR_PARAMS"]["scale"] = self.cap.scale
        self.img_size = (480,640)
        self.cameras = ["0"]
        self.keep_running = False
        self.slider = None
        self.bottom_margin = 40
        self.elevation = 30
        self.padding = 10
        self.border_radius = 20
        self.__dict__.update(kwargs)
        self.controls = {}

    def set_mirror(self, dummy):
        self.controls['cap_view'].mirror = self.controls['sw_mirror'].value
        self.controls['cap_view'].Renew()

    def create(self):
        self.controls['cap_view'] = ftImShow(self.cap, imgproc=self.imgproc, keep_running=self.keep_running,
                                             hw=self.img_size, border_radius=self.border_radius, cids=[0, 1])
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
                            ft.Row([self.controls['sw_mirror']])
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

    cap = rsVideoCapture()
    #section_opts['keep_running'] = True
    #imgproc['IMAGES'] = None

    section = Section(cap, imgproc=imgproc, cameras=args['cameras'], **section_opts)
    contents = section.create()

    # def on_disconnect( _: ft.ControlEvent):
    #         if cap is not None:
    #             cap.release()
    #         print("on_disconnect")
    #         section.terminate()
    # page.on_disconnect = on_disconnect

    set_page(page, PageOpts)
    page.on_window_event = lambda e: (cap.release(), page.window_destroy()) if e.data == "close" else None
    page.update()
    page.add(contents)

if __name__ == '__main__':
    ft.app(target=main, **args['app'])