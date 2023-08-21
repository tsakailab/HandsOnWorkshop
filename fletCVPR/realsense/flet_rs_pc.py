import flet as ft
import cv2
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
CAMERAS = ["0", "1", "2", "3"]

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': (480,640), 'padding': 10, 'cameras': CAMERAS,
        'images': None}

PageOpts = {'TITLE': "Color and Depth Viewer / Plane Detection (RealSense)", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]*2+240, args['resolution'][1]*2+6*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
detector_params = {'rtol': 0.03}
drawer_opts = {'bfps': True, 'min_depth': 0, 'max_depth': 6, 'bblend': False}
section_opts = {'img_size': args['resolution'], 'keep_running': True,
                'slider': {'max_depth': {'width': 400, 'value': drawer_opts['max_depth'], 'min': 1, 'max': 10, 'divisions': 9, 'label': "{value}m"}}, 
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}

#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, **kwargs):
        self.rtol = 0.05
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
        return {"points":points}
    
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
    def __init__(self, min_depth=None, max_depth=None, colormap=cv2.COLORMAP_OCEAN, **kwargs):
        self.bfps = True
        self.bblend = False
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.colormap = colormap
        self.__dict__.update(kwargs)
        self.depth_to_color = lambda depth: cv2.applyColorMap(cv2.normalize(-np.clip(depth, self.min_depth/self.scale, self.max_depth/self.scale), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), colormap)

        self.point_parameters = {
            "color":(255, 0, 0),
            "scale":5
        }
    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, detection_result, bgr_and_depth):
        bgr = bgr_and_depth[0]
        return [bgr, bgr]
    
    def drawfig(self, detection_result, bgr_and_depth, n=5000):
        Z = detection_result["points"][:, :, -1]
        nd = np.count_nonzero(Z)
        p = np.random.choice(nd, min(n,nd), replace=False)
        print("%d out of %d points are displayed." % (n, nd))

        import plotly.graph_objs  as go
        rgb = bgr_and_depth[0][Z>0][p] # * 1.5 # brighter

        trace = go.Scatter3d(
            x=detection_result["points"][:, :, 0][Z>0][p],
            y=detection_result["points"][:, :, 1][Z>0][p],
            z=Z[Z>0][p], mode='markers',
            marker=dict(size=2, 
                        color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(rgb[:,0], rgb[:,1], rgb[:,2])],
                        opacity=0.8)
        )
        layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
        fig = go.Figure(data=[trace], layout=layout)
        return [fig, fig]

#### (3/3) Define how to display in a page ####
# you will use this as
# contents = Section(cap, imgproc=imgproc, **section_opts).create()
class Section():
    def __init__(self, cap=None, imgproc=None, **kwargs):
        self.cap = cap
        self.imgproc = imgproc
        self.imgproc["DETECTOR_PARAMS"]["intr"] = self.cap.intr
        self.imgproc["DETECTOR_PARAMS"]["scale"] = self.cap.scale
        self.imgproc["DRAWER_OPTS"]["scale"] = self.cap.scale
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
    def set_bblend(self, dummy):
        self.controls['cap_view'].drawer.bblend = self.controls['sw_bblend'].value
        self.controls['cap_view'].Renew()
    def set_max_depth(self, dummy):
        self.controls['cap_view'].drawer.max_depth = self.controls['max_depth'].value
        self.controls['cap_view'].Renew()

    def create(self):
        self.controls['cap_view'] = ftImShow(self.cap, imgproc=self.imgproc, keep_running=self.keep_running,
                                             hw=self.img_size, border_radius=self.border_radius, cids=[0, 1])
        self.controls['sw_mirror'] = ft.Switch(label="Mirror", value=self.controls['cap_view'].mirror, label_position=ft.LabelPosition.LEFT,
                                               on_change=self.set_mirror)
        self.controls['sw_bblend'] = ft.Switch(label="Plane", value=False, label_position=ft.LabelPosition.LEFT,
                                               on_change=self.set_bblend)
        self.controls['max_depth'] = ft.Slider(on_change=self.set_max_depth, **self.slider['max_depth'])

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
                            ft.Row([self.controls['sw_mirror'], ft.VerticalDivider(), self.controls['sw_bblend']]),
                            self.controls['max_depth']
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
    page.on_window_event = lambda e: (cap.release() if cap is not None else None, cv2.waitKey(1000), page.window_destroy()) if e.data == "close" else None
    #page.on_window_event = lambda e: (cap.release(), page.window_destroy()) if e.data == "close" else None
    page.update()
    page.add(contents)

if __name__ == '__main__':
    ft.app(target=main, **args['app'])
