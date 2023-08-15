
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
