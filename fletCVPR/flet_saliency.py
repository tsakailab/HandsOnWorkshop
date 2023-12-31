import flet as ft
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.fftpack import next_fast_len, fft2, ifft2, fftshift, ifftshift
from util.flet_util import set_page, ftImShow, cvVideoCapture


resols = {'nHD': (360,640), 'FWVGA': (480,854), 'qHD': (540,960), 'WSVGA': (576,1024), 
          'HD': (720,1280), 'FWXGA': (768,1366), 'HD+': (900,1600), 'FHD': (1080,1920)}
CAMERAS = ["0", "1", "2", "3"]

args = {'app': {}, # {'view': ft.WEB_BROWSER}, #{'view': ft.FLET_APP},
        'resolution': resols['HD'], 'padding': 10, 
        'cameras': CAMERAS, 'frame_hw': resols['HD'], 
        'images': None}
#args.update({'images': ['dashcam.jpg', 'park.jpg']}) # works if cap.isOpened() is False
args.update({'images': ['dashcam.jpg', 'park.jpg']}) # works if cap.isOpened() is False


PageOpts = {'TITLE': "Visual Saliency", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': args['padding'],
        'WINDOW_HW': (args['resolution'][0]+240, args['resolution'][1]+2*args['padding']), 
        'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}

# defaults
detector_params = {'scale': 0.10, 'sigma': 0.001, 'sigma_smooth': 2.5}
#detector_params = {'modelname': "yolov8n-seg", 'imgsz': 960, 'conf': 0.25, 'iou': 0.7}
drawer_opts = {'bblend': False, 'bcont': False, 'contlevel': 16, 'bbinary': False, 
               'bframew': True, 'bfps': True}
section_opts = {'img_size': args['resolution'], 'keep_running': False, 
                'slider': {'scale': {'width': 400, 'value': 10, 'min': 3, 'max': 30, 'divisions': 27, 'label': "scale={value}"}}, 
                'bottom_margin': 40, 'elevation': 20, 'padding':10, 'border_radius': 20}

def ComputeSaliencyMap(inImage, scale=0.05, sigma=0.001, sigma_smooth = 2.5):
    inImage_size = (inImage.shape[1], inImage.shape[0])  # (width, height)
    # resize
    resize_size = (int(inImage_size[0] * scale), int(inImage_size[1] * scale))
    inImage = np.array(Image.fromarray(inImage).resize(resize_size))  # ndarrayをPIL.Imageに変換，resize

    # give the value from next_fast_len to fft2, then automatically zero paded to the size.
    FT_inImage = fft2(inImage, shape=(next_fast_len(resize_size[1]), next_fast_len(resize_size[0])))
    FT_inImage = fftshift(FT_inImage)

    Amp = np.log(np.abs(FT_inImage)+1e-8)
    Phase = np.angle(FT_inImage)

    # Spectral residual (Laplacian of the Spectrum)
    L_Amp = gaussian_filter(Amp, sigma=sigma)
    R_Amp = L_Amp - Amp
    
    R_complex = np.exp(R_Amp + Phase*1j)
    R_complex = ifftshift(R_complex)  # fftshiftの逆．
    saliencyMap = ifft2(R_complex)
    saliencyMap = np.abs(saliencyMap)
    
    # Gaussian filtering (postprocessing)
    saliencyMap = gaussian_filter(saliencyMap, sigma=sigma_smooth)
    # resize to original size
    saliencyMap = np.array(Image.fromarray(saliencyMap[0:resize_size[1], 0:resize_size[0]]).resize(inImage_size))
    # normalize values from 0 to 255
    saliencyMap = ((saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap)+1e-4) * 255).astype(np.uint8)
    return saliencyMap


#### (1/3) Define a detector ####
# will be used in flet_util.ftImShow as
# detector = Detector(**detector_params)
class Detector():
    def __init__(self, **kwargs):
        self.detector = ComputeSaliencyMap
        self.opts = kwargs

    # Do not change the function name. The first arg is the target bgr image.
    def detect(self, bgr):
        csal = np.zeros(bgr.shape)
        csal[..., 0] = self.detector(bgr[:,:,0], **self.opts)
        csal[..., 1] = self.detector(bgr[:,:,1], **self.opts)
        csal[..., 2] = self.detector(bgr[:,:,2], **self.opts)
        sal = 0.299 * csal[:, :, 2] + 0.587 * csal[:, :, 1] + 0.114 * csal[:, :, 0]
        return np.clip(sal, 0, 255)#.astype(np.uint8)

#### (2/3) Define a Rrawer to show the detector's result as a bgr image ####
# result = detector.detect(_bgr_to_mp(bgr))
# will be used in flet_util.ftImshow as 
# bgr = Drawer(**draw_opts).draw(result, bgr)
class Drawer():
    def __init__(self, **kwargs):
        self.bfps = False
        self.FrameWindow = None
        self.__dict__.update(kwargs)

    # Do not change the function name. The first and secont args are the result and the target bgr image.
    def draw(self, sal, bgr):
        mask = np.ones(bgr.shape[:2], dtype=np.uint8)
        if self.FrameWindow is None:
            self.FrameWindow = np.sqrt(cv2.createHanningWindow((bgr.shape[1], bgr.shape[0]), cv2.CV_32F))
        if self.bframew:
            sal = sal * self.FrameWindow
        if self.bbinary:
            mask = cv2.threshold(sal.astype(np.uint8), 127, 255, cv2.THRESH_OTSU)[1] / 255.0
            #mask = cv2.threshold(((sal.astype(np.float32)**2)/255.0).astype(np.uint8), 127, 255, cv2.THRESH_OTSU)[1] / 255.0
        if self.bcont:
            sal = (sal / self.contlevel).astype(np.uint8) * self.contlevel

        frame = bgr
        if self.bblend:
            #frame = (frame * (sal / 255.0)[..., None]).astype(np.uint8)
            frame = np.clip((frame+64.0) * (sal / 255.0)[..., None], 0, 255).astype(np.uint8)
            if self.bbinary:
                frame = (frame * mask[..., None]).astype(np.uint8)
                #frame = np.clip((frame**self.gamma) * mask[..., None], 0, 255).astype(np.uint8)

        #return cv2.addWeighted(sal, 0.5, bgr, 0.5, 0)
        #return (bgr * (sal / 255.0)[..., None]).astype(np.uint8)
        return cv2.addWeighted(frame, 0.90, bgr, 0.10, 0)
 

#### (3/3) Define how to display in a page ####
# you will use this as
# contents = Section(cap, imgproc=imgproc, **section_opts).create()
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

    def set_bblend(self, dummy):
        self.controls['cap_view'].drawer.bblend = self.controls['sw_bblend'].value
        self.controls['cap_view'].Renew()
    def set_bcont(self, dummy):
        self.controls['cap_view'].drawer.bcont = self.controls['sw_bcont'].value
        self.controls['cap_view'].Renew()
    def set_bbinary(self, dummy):
        self.controls['cap_view'].drawer.bbinary = self.controls['sw_bbinary'].value
        self.controls['cap_view'].Renew()
    def set_bframew(self, dummy):
        self.controls['cap_view'].drawer.bframew = self.controls['sw_bframew'].value
        self.controls['cap_view'].Renew()

    def create(self):
        self.controls['cap_view'] = ftImShow(self.cap, imgproc=self.imgproc, keep_running=self.keep_running,
                                             hw=self.img_size, border_radius=self.border_radius)
        ddlist = self.cameras if self.controls['cap_view'].images is None else self.controls['cap_view'].images
        self.controls['dd'] = ft.Dropdown(label="Camera/Image", width=256, 
                        options=[ft.dropdown.Option(c) for c in ddlist],
                        on_change=self.set_cap)
        self.controls['sw_mirror'] = ft.Switch(label="Mirror", value=False, label_position=ft.LabelPosition.LEFT,
                                               on_change=self.set_mirror)
        self.controls['sw_bblend'] = ft.Switch(label="Saliency", value=False, label_position=ft.LabelPosition.LEFT,
                                               on_change=self.set_bblend)
        self.controls['sw_bcont'] = ft.Switch(label="Contour", value=False, label_position=ft.LabelPosition.LEFT,
                                              on_change=self.set_bcont)
        self.controls['sw_bbinary'] = ft.Switch(label="Mask", value=False, label_position=ft.LabelPosition.LEFT,
                                                on_change=self.set_bbinary)
        self.controls['sw_bframew'] = ft.Switch(label="Border", value=True, label_position=ft.LabelPosition.LEFT,
                                                on_change=self.set_bframew)
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
                            ft.Row([self.controls['dd'], self.controls['sw_bblend'], ft.VerticalDivider(), 
                                    self.controls['sw_bcont'], self.controls['sw_bbinary'], ft.VerticalDivider(), self.controls['sw_mirror']])#, self.controls['sw_bframew']])
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
                            on_change=lambda e: self.controls['cap_view'].RenewDetector({'scale': 1/e.control.value}).Renew(),
                            **self.slider['scale']
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

    def terminate(self):
        self.controls['cap_view'].keep_running = False



imgproc = {'DETECTOR': Detector, 'DETECTOR_PARAMS': detector_params, 
           'DRAWER': Drawer, 'DRAWER_OPTS': drawer_opts,
           'IMAGES': args['images'], 'MIRROR': False}

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
