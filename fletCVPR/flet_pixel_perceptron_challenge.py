import flet as ft
import base64
import numpy as np
import cv2
from sklearn.datasets import load_digits
from matplotlib.colors import TwoSlopeNorm as tsn
import matplotlib.pyplot as plt
from time import sleep
from util.flet_util import set_page

# from importlib import import_module
# flet_utils = import_module("flet_utils.py", "..")
# set_page = flet_utils.set_page

args = {'app': {}, #{'view': ft.WEB_BROWSER},
        'table_size': (8, 8), 'cell_size': 50, 'colormap': plt.cm.bwr, 'font_size': 36, 'pixs': 20,
        'INEQULALITY': ["<", ">"], 'GOOD_BAD': ["Good", " ", "Bad"]}

PageOpts = {'TITLE': "Pixel Perceptron Challenge", 
        'THEME_MODE': ft.ThemeMode.LIGHT, 'WPA': False,
        'VERTICAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 'HORIZONTAL_ALIGNMENT': ft.MainAxisAlignment.CENTER, 
        'PADDING': 10,
        'WINDOW_HW': (900, 1400), 'WINDOW_TOP_LEFT': (50,100), '_WINDOW_TOP_LEFT_INCR': False}


TextFieldOpts = {'height': 2*args['pixs']+8, 'width': 2*args['pixs']-4, 'text_size': 2*args['pixs']-20, 
                 'text_align': ft.TextAlign.RIGHT, 'content_padding': 0, 'dense': False, 'border_width': 0, 'multiline': False,
                 'keyboard_type': ft.KeyboardType.NUMBER, 'read_only': True}
TextButtonOpts = {'height': args['pixs'], 'width': args['pixs']-4}
TextOpts = {'size': args['pixs']-4, 'text_align': ft.alignment.center, 'color': "#000000", 'weight': ft.FontWeight.BOLD} # button

class NumberBox():
    def __init__(self, vlim=[-10, 10], delta=1, colormap=plt.cm.bwr, border=ft.border.all(1, "black"),
                 opts = {'style': ft.ButtonStyle(padding=0, shape=ft.RoundedRectangleBorder(radius=0)), 
                         'TextField': TextFieldOpts, 'TextButton': TextButtonOpts, 'Text': TextOpts}):
        self.delta = delta
        self.colormap = colormap
        self.vlim = vlim
        self.border = border
        self.norm = tsn(vmin=np.minimum(vlim[0],-1e-6), vcenter=0, vmax=np.maximum(vlim[1],1e-6))
        self.txt_up = "^" #f"＋"
        self.txt_dn = "v" #f"－"
        self.opts = opts

    def _get_rgbstr_from_norm(self, normv):
        rgba = self.colormap(normv)
        return '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    def click_up(self, e):
        if self.index is not None: self.obj[self.index] += self.delta
        else: self.obj += self.delta
        value = self.obj[self.index] if self.index is not None else self.obj
        self.TextField.value = self.v2s(value)
        self.content.bgcolor = self._get_rgbstr_from_norm(self.norm(min(value, self.vlim[1])))
        self.TextField.bgcolor = self.content.bgcolor
        if self.on_change is not None: self.on_change({'index': self.index})
        self.content.update()

    def click_dn(self, e):
        if self.index is not None: self.obj[self.index] -= self.delta
        else: self.obj -= self.delta
        value = self.obj[self.index] if self.index is not None else self.obj
        self.TextField.value = self.v2s(value)
        self.content.bgcolor = self._get_rgbstr_from_norm(self.norm(max(value, self.vlim[0])))
        self.TextField.bgcolor = self.content.bgcolor
        if self.on_change is not None: self.on_change({'obj': self.obj, 'index': self.index})
        self.content.update()

    def create(self, obj, index=None, on_change=None):
        self.obj = obj 
        self.index = index
        value = self.obj[self.index] if self.index is not None else self.obj
        self.on_change = on_change
        self.s2v = int if np.issubdtype(value, np.integer) else float
        self.v2s = '{:d}'.format if np.issubdtype(value, np.integer) else '{:2.1f}'.format
        #self.v2s = '{:+d}'.format if np.issubdtype(value, np.integer) else '{:2.1f}'.format
        self.TextField = ft.TextField(value=self.v2s(value), bgcolor=self._get_rgbstr_from_norm(self.norm(value)), **self.opts['TextField'])
        self.btn_up = ft.TextButton(content=ft.Text(self.txt_up, style=self.opts['style'], **self.opts['Text']), 
                               on_click=self.click_up, style=self.opts['style'], **self.opts['TextButton'])
        self.btn_dn = ft.TextButton(content=ft.Text(self.txt_dn, style=self.opts['style'], **self.opts['Text']),
                               on_click=self.click_dn, style=self.opts['style'], **self.opts['TextButton'])
        btns = ft.Column([self.btn_up, self.btn_dn], tight=True, spacing=0, #run_spacing=0, 
                         alignment=ft.MainAxisAlignment.CENTER)#.SPACE_AROUND)
        self.content = ft.Container(
                            ft.Row([self.TextField, btns], tight=True, spacing=0, 
                                    alignment=ft.MainAxisAlignment.END, vertical_alignment=ft.CrossAxisAlignment.CENTER),#ft.MainAxisAlignment.CENTER),
                            margin=0, padding=0, border=self.border, border_radius=0, bgcolor=self._get_rgbstr_from_norm(self.norm(value)))

        return self.content


# Input
class NumberTable():
    def __init__(self, array=None, shape=(3,3), spacing={'row': 0, 'col': 0}, on_change=None, 
                 vlim=[-10, 10], delta=1, colormap=plt.cm.bwr, border=ft.border.all(1, "black")):
        self.shape = shape if array is None else np.array(array).shape
        self.array = array if array is not None else np.zeros(self.shape, dtype=np.int8)
        self.spacing = {'row': 0, 'col': 0}
        self.spacing.update(spacing)
        self.on_change = on_change
        self.vlim = vlim
        self.delta = delta
        self.colormap = colormap
        self.border = border

    def create(self):
        self.content = ft.Column(wrap=False, expand=False, tight=True, spacing=self.spacing['col'], 
                                 alignment=ft.alignment.top_center)
        for i in range(self.shape[0]):
            ftRow = ft.Row(wrap=False, expand=False, tight=True, spacing=self.spacing['row'], 
                           alignment=ft.alignment.center_left)
            for j in range(self.shape[1]):
                ftRow.controls.append(
                    NumberBox(vlim=self.vlim, delta=self.delta, colormap=self.colormap, border=self.border).create(
                                                        self.array, index=(i, j), on_change=self.on_change))
            self.content.controls.append(ftRow)
        return self.content


# Output
class ColoredTable():
    def __init__(self, array=None, shape=(3,3), cell_hw=(45,45), vlim=[-10, 10], colormap=plt.cm.bwr, border=ft.border.all(1, "black")):
        self.shape = shape if array is None else np.array(array).shape
        self.array = array if array is not None else np.zeros(self.shape, dtype=np.int8)
        self.cell_hw = cell_hw
        self.vlim = vlim
        self.colormap = colormap
        self.border = border
        self.norm = tsn(vmin=np.minimum(vlim[0],-1e-6), vcenter=0, vmax=np.maximum(vlim[1],1e-6))

    def get_rgbstr_from_value(self, value):
        rgba = self.colormap(self.norm(value))
        return '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    def create(self):
        self.content = ft.Column(wrap=False, expand=False, tight=True, spacing=0, alignment=ft.alignment.top_center)
        for i in range(self.shape[0]):
            ftRow = ft.Row(wrap=False, expand=False, tight=True, spacing=0, alignment=ft.alignment.center_left)
            for j in range(self.shape[1]):
                bgcolor = self.get_rgbstr_from_value(self.array[i,j])
                ftRow.controls.append(ft.Container(height=self.cell_hw[0], width=self.cell_hw[1], 
                                                   border_radius=0, bgcolor=bgcolor, border=self.border))
            self.content.controls.append(ftRow)
        return self.content


imgfmt, base64code = '.jpg', 'ascii'
def bgr_to_base64(bgr):
    src_base64 = base64.b64encode(
                cv2.imencode(imgfmt, bgr)[1]
                ).decode(base64code)
    return src_base64

def nn_resize(img, shape):
    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)
    return img[per_axis(img.shape[0], shape[0])[:, None],
               per_axis(img.shape[1], shape[1])]

def inequality(l, strings=["<", "=", ">"]):
    s = strings[1]
    if l[0] < l[1]: s = strings[0]
    if l[0] > l[1]: s = strings[2]
    return s

digits = load_digits()
classes = np.unique(digits.target)
#print(classes)
images = []
for c in classes:
    ids = digits.target == c
    images.append(digits.images[ids]*16-1)


def sample_images(classes, images):
    samples = []
    for c in classes:
        id = np.random.choice(len(images[c]))
        samples.append(images[c][id])
    return samples

def create_ftimages(images, hw):
    ftimgs = []
    for img in images:
        ftimgs.append(ft.Image(src_base64=bgr_to_base64(nn_resize(img, hw)),
                            height=hw[0], width=hw[1], fit=ft.ImageFit.FIT_WIDTH, 
                            border_radius=ft.border_radius.all(0)))
    return ftimgs


def main(page: ft.Page):
    set_page(page, PageOpts)
    page.update()

    hw = args['cell_size'] * args['table_size'][0], args['cell_size'] * args['table_size'][1]
    class_a, class_b = 0, 1
    imgs = sample_images([class_a, class_b], images)
    #imga, imgb = imgs[0], imgs[1]
    ftimgs = create_ftimages(imgs, hw)
    #ftimga, ftimgb = ftimgs[0], ftimgs[1]

    w = np.zeros((8,8), dtype=np.int8)

    wimgs = [w * imgs[0] / 255.0, w * imgs[1] / 255.0]
    cell_size = args['cell_size']
    wimg_tables = []
    wimg_tables.append(ColoredTable(wimgs[0], cell_hw=(cell_size,cell_size), vlim=[-10.0, 10.0], colormap=args['colormap'], border=ft.border.all(1, "#a0a0a0")))
    wimg_tables.append(ColoredTable(wimgs[1], cell_hw=(cell_size,cell_size), vlim=[-10.0, 10.0], colormap=args['colormap'], border=ft.border.all(1, "#a0a0a0")))
    wimg_out = [wimg_tables[0].create(), wimg_tables[1].create()]

    logits = np.array([wimgs[0].sum(), wimgs[1].sum()])
    res = ft.Column([
        ft.Row([
        ft.Text(value='{:2.1f}'.format(logits[0]), size=args['font_size'], text_align=ft.alignment.top_right), 
        ft.Text(value=inequality(logits), size=args['font_size'], text_align=ft.alignment.top_center),
        ft.Text(value='{:2.1f}'.format(logits[1]), size=args['font_size'], text_align=ft.alignment.top_right)],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=hw[1]),
        ft.Text(value=" ", size=args['font_size'], text_align=ft.alignment.center),], 
        horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0)

    #ftprtbtn = ft.TextButton("update", on_click=lambda e: page.update())
    #ftprtbtn = ft.TextButton("update", on_click=lambda e: print(logits))

    def update_result():
        res.controls[0].controls[0].value = '{:2.1f}'.format(logits[0])
        res.controls[0].controls[1].value = inequality(logits)
        res.controls[0].controls[2].value = '{:2.1f}'.format(logits[1])
        res.controls[1].value = inequality(logits, args['GOOD_BAD'])
        res.update()

    def update_table_cells(dic):
        i, j = dic['index']
        cell = wimg_tables[0].content.controls[i].controls[j]
        cell.bgcolor = wimg_tables[0].get_rgbstr_from_value(wimgs[0][i,j])
        cell.update()
        cell = wimg_tables[1].content.controls[i].controls[j]
        cell.bgcolor = wimg_tables[1].get_rgbstr_from_value(wimgs[1][i,j])
        cell.update()

    def on_change(dic):
        i, j = dic['index']
        wimgs[0][i,j] = w[i,j] * imgs[0][i,j] / 255
        wimgs[1][i,j] = w[i,j] * imgs[1][i,j] / 255
        logits[0], logits[1] = wimgs[0].sum(), wimgs[1].sum()
        update_result()
        update_table_cells(dic)

    ntable = NumberTable(w, vlim=[-10, 10], colormap=args['colormap'], on_change=on_change)
    #w = ntable.array
    w_in = ntable.create()


    def change_sample():
        imgs = sample_images([class_a, class_b], images)
        ftimgs[0].src_base64 = bgr_to_base64(nn_resize(imgs[0], hw))
        ftimgs[1].src_base64 = bgr_to_base64(nn_resize(imgs[1], hw))
        ftimgs[0].update()
        ftimgs[1].update()

        wimgs[0] = w * imgs[0] / 255
        wimgs[1] = w * imgs[1] / 255
        logits[0], logits[1] = wimgs[0].sum(), wimgs[1].sum()
        update_result()

        cell_size = args['cell_size']
        wimg_tables[0] = ColoredTable(wimgs[0], cell_hw=(cell_size,cell_size), vlim=[-10.0, 10.0], colormap=args['colormap'], border=ft.border.all(1, "#a0a0a0"))
        #wa_out = watable.create()
        wimg_tables[1] = ColoredTable(wimgs[1], cell_hw=(cell_size,cell_size), vlim=[-10.0, 10.0], colormap=args['colormap'], border=ft.border.all(1, "#a0a0a0"))
        #wb_out = wbtable.create()
        page.controls[0].controls[0].controls[1] = wimg_tables[0].create() #wimg_out[0]
        page.controls[0].controls[2].controls[1] = wimg_tables[1].create() #wimg_out[1]

        #ntable = NumberTable(w, vlim=[-10, 10], colormap=args['colormap'], on_change=on_change)
        #w = ntable.array
        #page.controls[0].controls[1].controls[0] = w_in #ntable.create()
        page.update()
        sleep(200)


    ft_tb_change_sample = ft.TextButton("Another Sample", on_click=lambda e: change_sample())
    ft_tb_update_page = ft.TextButton("Update Page", on_click=lambda e: page.update())


    cbar = args['colormap'](np.linspace(0, 1, 400)).reshape(1, -1, 4)[:, ::-1, :3]
    ftcbar = ft.Image(src_base64=bgr_to_base64(cbar*255),
                      height=int(hw[1]/20), width=hw[1], fit=ft.ImageFit.FILL,
                      border_radius=ft.border_radius.all(0))

    page.add(ft.Row([
        ft.Column([ftimgs[0], wimg_out[0]], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Column([w_in, ftcbar, res, ft_tb_change_sample, ft_tb_update_page], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Column([ftimgs[1], wimg_out[1]], horizontal_alignment=ft.CrossAxisAlignment.CENTER)], 
        vertical_alignment=ft.CrossAxisAlignment.START)
    )

ft.app(target=main, **args['app'])
#ft.app(target=main, view=ft.WEB_BROWSER)