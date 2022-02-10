import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, colorchooser
from tkinter.ttk import *
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import freeze_support, cpu_count
import argparse
import traceback
import math
import sys
import os
import time

import cv2
import numpy as np
import make_img as mkg
from io_utils import SafeText


def limit_wh(w: int, h: int, max_width: int, max_height: int):
    if h > max_height:
        ratio = max_height / h
        h = max_height
        w = math.floor(w * ratio)
    if w > max_width:
        ratio = max_width / w
        w = max_width
        h = math.floor(h * ratio)
    return w, h


""" tk_ToolTip_class101.py
gives a Tkinter widget a tooltip as the mouse is above the widget
tested with Python27 and Python34  by  vegaseat  09sep2014
www.daniweb.com/programming/software-development/code/484591/a-tooltip-class-for-tkinter

Modified to include a delay time by Victor Zaccardo, 25mar16
Source: https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
"""
class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()


class LabelWithTooltip(Label):
    def __init__(self, *args, **kwargs) -> None:
        _tp = kwargs["tooltip"]
        del kwargs["tooltip"]
        super().__init__(*args, **kwargs)
        self._tp = CreateToolTip(self, _tp)


class CheckbuttonWithTooltip(Checkbutton):
    def __init__(self, *args, **kwargs) -> None:
        _tp = kwargs["tooltip"]
        del kwargs["tooltip"]
        super().__init__(*args, **kwargs)
        self._tp = CreateToolTip(self, _tp)


class RadiobuttonWithTooltip(Radiobutton):
    def __init__(self, *args, **kwargs) -> None:
        _tp = kwargs["tooltip"]
        del kwargs["tooltip"]
        super().__init__(*args, **kwargs)
        self._tp = CreateToolTip(self, _tp)


if __name__ == "__main__":
    freeze_support()
    pool = ThreadPoolExecutor(1)
    root = Tk()
    root.title("Photomosaic maker")

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", action="store_true")
    parser.add_argument("--src", "-s", type=str,
                        default=os.path.join(os.path.dirname(__file__), "img"))
    parser.add_argument("--collage", "-c", type=str,
                        default=os.path.join(os.path.dirname(__file__), "examples", "dest.png"))
    cmd_args = parser.parse_args()
    init_dir = os.path.expanduser("~")

    # ---------------- initialization -----------------
    left_panel = PanedWindow(root)
    left_panel.grid(row=0, column=0, sticky="NSEW")
    # left_panel.rowconfigure(0, weight=1)
    # left_panel.columnconfigure(0, weight=1)
    Separator(root, orient="vertical").grid(
        row=0, column=1, sticky="NSEW", padx=(5, 5))
    right_panel = PanedWindow(root)
    right_panel.grid(row=0, column=2, sticky="N")
    # ---------------- end initialization -----------------

    # ---------------- left panel's children ----------------
    # left panel ROW 0
    canvas = Canvas(left_panel, width=800, height=600)
    canvas.grid(row=0, column=0)

    # left panel ROW 1
    Separator(left_panel, orient="horizontal").grid(row=1, columnspan=2, sticky="NSEW", pady=(5, 5))

    # left panel ROW 2
    log_panel = PanedWindow(left_panel)
    log_panel.grid(row=2, column=0, columnspan=2, sticky="WE")
    log_panel.grid_columnconfigure(0, weight=1)
    log_entry = SafeText(log_panel, height=6, bd=0)
    mkg.pbar_ncols = log_entry.width
    # log_entry.configure(font=("", 10, ""))
    log_entry.grid(row=1, column=0, sticky="NSEW")
    scroll = Scrollbar(log_panel, orient="vertical", command=log_entry.yview)
    log_entry.config(yscrollcommand=scroll.set)
    scroll.grid(row=1, column=1, sticky="NSEW")

    result_img = None
    # ------------------ end left panel ------------------------

    def show_img(img: np.ndarray, printDone: bool = True) -> None:
        global result_img
        if img is None:
            return

        result_img = img
        width, height = canvas.winfo_width(), canvas.winfo_height()
        img_h, img_w, _ = img.shape
        w, h = limit_wh(img_w, img_h, width, height)
        if img_h > height or img_w > width:
            img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        if img.dtype != np.uint8:
            assert img.dtype == np.float32
            img = img * 255
            img = img.astype(np.uint8)
        _, data = cv2.imencode(".ppm", img)
        # prevent the image from being garbage-collected
        root.preview = PhotoImage(data=data.tobytes())
        canvas.delete("all")
        canvas.create_image((width - w) // 2, (height - h) // 2, image=root.preview, anchor=NW)
        save_button.config(state='enabled')
        if printDone:
            print("Done")

    # --------------------- right panel's children ------------------------
    # right panel ROW 0
    file_path = StringVar()
    file_path.set("N/A")
    Label(right_panel, text="Path to tiles:").grid(row=0, columnspan=2, sticky="W", pady=(8, 2))

    # right panel ROW 1
    Label(right_panel, textvariable=file_path, wraplength=150).grid(row=1, columnspan=2, sticky="W")

    # right panel ROW 2
    opt = StringVar()
    opt.set("sort")

    tile_size_panel = PanedWindow(right_panel)
    tile_size_panel.grid(row=2, column=0, columnspan=2, sticky="W")
    
    tile_width = IntVar()
    tile_width.set(50)
    tile_height = IntVar()
    tile_height.set(0)
    LabelWithTooltip(tile_size_panel, text="Tile size: ", tooltip=mkg.PARAMS.size.help).grid(row=0, column=0, sticky="W")
    Entry(tile_size_panel, width=5, textvariable=tile_width).grid(row=0, column=1, sticky="W")
    Label(tile_size_panel, text="x", wraplength=150).grid(row=0, column=2, sticky="W")
    tile_height_entry = Entry(tile_size_panel, width=5, textvariable=tile_height)
    tile_height_entry.grid(row=0, column=3, sticky="W")

    infer_height = BooleanVar()
    infer_height.set(True)
    def action_infer_height():
        if infer_height.get():
            tile_height_entry.config(state='disabled')
        else:
            tile_height_entry.config(state='enabled')
            if tile_height.get() == 0:
                tile_height.set(tile_width.get())
    action_infer_height()
    CheckbuttonWithTooltip(tile_size_panel, text="Infer height", variable=infer_height, command=action_infer_height, 
        tooltip="Infer height from width and the aspect ratios of the images provided").grid(row=1, columnspan=4, sticky="W")

    auto_rotate = IntVar()
    auto_rotate.set(0)
    LabelWithTooltip(tile_size_panel, text="Auto rotate: ", tooltip=mkg.PARAMS.auto_rotate.help).grid(
        row=2, column=0, columnspan=2, sticky="W")
    OptionMenu(tile_size_panel, auto_rotate, "", *mkg.PARAMS.auto_rotate.choices).grid(
        row=2, column=2, columnspan=2, sticky="W")

    # right panel ROW 3
    resize_opt = StringVar()
    resize_opt.set("center")
    LabelWithTooltip(right_panel, text="Resize option: ", tooltip=mkg.PARAMS.resize_opt.help).grid(
        row=3, column=0, sticky="W")
    OptionMenu(right_panel, resize_opt, "", "center", "stretch").grid(
        row=3, column=1, sticky="W")

    # right panel ROW 4
    recursive = BooleanVar()
    recursive.set(True)
    Checkbutton(right_panel, text="Read sub-folders",
                variable=recursive).grid(row=4, columnspan=2, sticky="W")

    imgs = None

    def load_img_action():
        global imgs
        fp = file_path.get()
        try:
            sizes = [tile_width.get()]
            if not infer_height.get():
                sizes.append(tile_height.get())
            imgs = mkg.read_images(fp, sizes, recursive.get(), mp.Pool(cpu_count() // 2), resize_opt.get(), auto_rotate.get())
            shape = imgs[0].shape
            if infer_height.get():
                tile_height.set(shape[0])
            grid = mkg.calc_grid_size(16, 10, len(imgs), shape)
            return mkg.make_collage(grid, imgs.copy(), False)
        except:
            messagebox.showerror("Error", traceback.format_exc())

    
    def reload_images():
        pool.submit(load_img_action).add_done_callback(lambda f: show_img(f.result()))


    def load_images():
        w = tile_width.get()
        h = tile_height.get()
        if w < 2:
            return messagebox.showerror("Illegal Argument", "Tile width must be greater than 2")
        if not infer_height.get() and h < 2:
            return messagebox.showerror("Illegal Argument", "Tile height must be greater than 2")

        fp = filedialog.askdirectory(initialdir=file_path.get() if os.path.isdir(file_path.get()) else init_dir, 
            title="Select folder of tiles")
        if len(fp) <= 0 or not os.path.isdir(fp):
            return

        def callback(f):
            reload_img_button.config(state='enabled')
            sort_button.config(state='enabled')
            collage_button.config(state='enabled')
            show_img(f.result())

        print("Loading tiles from", fp)
        file_path.set(fp)
        pool.submit(load_img_action).add_done_callback(callback)


    # right panel ROW 5
    Button(right_panel, text="Load tiles", command=load_images).grid(row=5, column=0, pady=2)
    reload_img_button = Button(right_panel, text="Reload", command=reload_images)
    reload_img_button.config(state='disabled')
    reload_img_button.grid(row=5, column=1, pady=2, padx=(0, 4))


    def attach_sort():
        right_col_opt_panel.grid_remove()
        right_sort_opt_panel.grid(row=8, columnspan=2, sticky="W")

    def attach_collage():
        right_sort_opt_panel.grid_remove()
        right_col_opt_panel.grid(row=8, columnspan=2, sticky="W")

    # right panel ROW 6
    Radiobutton(right_panel, text="Sort", value="sort", variable=opt,
                state=ACTIVE, command=attach_sort).grid(row=6, column=1, sticky="W")
    Radiobutton(right_panel, text="Photomosaic", value="collage", variable=opt,
                command=attach_collage).grid(row=6, column=0, sticky="W")

    # right panel ROW 7
    Separator(right_panel, orient="horizontal").grid(
        row=7, columnspan=2, pady=(6, 3), sticky="we")

    # right panel ROW 8: Dynamically attached
    # right sort option panel !!OR!! right collage option panel
    right_sort_opt_panel = PanedWindow(right_panel)
    right_sort_opt_panel.grid(row=8, column=0, columnspan=2, sticky="W")

    # ------------------------- right sort option panel --------------------------
    # right sort option panel ROW 0:
    sort_method = StringVar()
    sort_method.set("bgr_sum")
    LabelWithTooltip(right_sort_opt_panel, text="Sort methods:", tooltip=mkg.PARAMS.sort.help).grid(
        row=0, column=0, sticky="W")
    OptionMenu(right_sort_opt_panel, sort_method, "", *mkg.PARAMS.sort.choices).grid(row=0, column=1)

    # right sort option panel ROW 1:
    LabelWithTooltip(right_sort_opt_panel, text="Aspect ratio:", tooltip=mkg.PARAMS.ratio.help).grid(
        row=1, column=0, sticky="W")
    aspect_ratio_panel = PanedWindow(right_sort_opt_panel)
    aspect_ratio_panel.grid(row=1, column=1)
    rw = IntVar()
    rw.set(16)
    rh = IntVar()
    rh.set(10)
    Entry(aspect_ratio_panel, width=3, textvariable=rw).grid(row=0, column=0)
    Label(aspect_ratio_panel, text=":").grid(row=0, column=1)
    Entry(aspect_ratio_panel, width=3, textvariable=rh).grid(row=0, column=2)

    # right sort option panel ROW 2:
    rev_row = BooleanVar()
    rev_row.set(False)
    rev_sort = BooleanVar()
    rev_sort.set(False)
    Checkbutton(right_sort_opt_panel, variable=rev_row,
                text="Reverse consecutive row").grid(row=2, columnspan=2, sticky="W")
    # right sort option panel ROW 3:
    Checkbutton(right_sort_opt_panel, variable=rev_sort,
                text="Reverse sort order").grid(row=3, columnspan=2, sticky="W")

    def generate_sorted_image():
        if imgs is None:
            return messagebox.showerror("Empty set", "Please first load tiles")

        try:
            w, h = rw.get(), rh.get()
            assert w > 0, "Width must be greater than 0"
            assert h > 0, "Height must be greater than 0"
        except AssertionError as e:
            return messagebox.showerror("Illegal Argument", str(e))

        def action():
            try:
                grid, sorted_imgs = mkg.sort_collage(imgs, (w, h), sort_method.get(), rev_sort.get())
                return mkg.make_collage(grid, sorted_imgs, rev_row.get())
            except:
                messagebox.showerror("Error", traceback.format_exc())

        pool.submit(action).add_done_callback(lambda f: show_img(f.result()))

    # right sort option panel ROW 4:
    sort_button = Button(right_sort_opt_panel, text="Generate sorted image", command=generate_sorted_image)
    sort_button.config(state='disabled')
    sort_button.grid(row=4, columnspan=2, pady=5)
    # ------------------------ end right sort option panel -----------------------------

    # ------------------------ right collage option panel ------------------------------
    # right collage option panel ROW 0:
    right_col_opt_panel = PanedWindow(right_panel)
    dest_img_path = StringVar()
    dest_img_path.set("N/A")
    dest_img = None
    Label(right_col_opt_panel, text="Path to the target image: ").grid(
        row=0, columnspan=2, sticky="W", pady=2)

    # right collage option panel ROW 1:
    Label(right_col_opt_panel, textvariable=dest_img_path,
          wraplength=150).grid(row=1, columnspan=2, sticky="W")


    def load_dest_img():
        global dest_img
        if imgs is None:
            return messagebox.showerror("Empty set", "Please first load tiles")

        fp = filedialog.askopenfilename(
            initialdir=os.path.dirname(dest_img_path.get()) if os.path.isdir(os.path.dirname(dest_img_path.get())) else init_dir, 
            title="Select destination image",
            filetypes=(("images", "*.jpg"), ("images", "*.png"), ("images", "*.gif"), ("all files", "*.*")))
        if fp is not None and len(fp) > 0 and os.path.isfile(fp):
            try:
                print("Destination image loaded from", fp)
                dest_img = mkg.imread(fp)
                show_img(dest_img, False)
                dest_img_path.set(fp)
            except:
                messagebox.showerror("Error reading file", traceback.format_exc())


    # right collage option panel ROW 2:
    Button(right_col_opt_panel, text="Load destination image",
           command=load_dest_img).grid(row=2, columnspan=2, pady=(3, 2))

    result_collage = None
    def change_alpha(_=None):
        if result_collage is not None and dest_img is not None:
            if colorization_opt.get() == "brightness":
                show_img(mkg.brightness_blend(result_collage, dest_img, 1 - alpha_scale.get() / 100), False)
            else:
                show_img(mkg.alpha_blend(result_collage, dest_img, 1 - alpha_scale.get() / 100), False)
    
    # right collage option panel ROW 3:
    LabelWithTooltip(right_col_opt_panel, text="Color Blend:", tooltip=mkg.PARAMS.blending.help).grid(
        row=3, column=0, sticky="W", padx=(0, 5))

    # right collage option panel ROW 4:
    colorization_opt = StringVar()
    colorization_opt.set("brightness")
    RadiobuttonWithTooltip(right_col_opt_panel, text="Brightness", variable=colorization_opt, value="brightness", state=ACTIVE, command=change_alpha, 
        tooltip=mkg.PARAMS.blending.help).grid(row=4, column=0, sticky="W")
    RadiobuttonWithTooltip(right_col_opt_panel, text="Alpha", variable=colorization_opt, value="alpha", command=change_alpha,
        tooltip=mkg.PARAMS.blending.help).grid(row=4, column=1, sticky="W")

    # right collage option panel ROW 5:
    alpha_scale = Scale(right_col_opt_panel, from_=0.0, to=100.0, orient=HORIZONTAL, length=150, command=change_alpha)
    alpha_scale.set(0)
    alpha_scale.grid(row=5, columnspan=2, sticky="W")

    # right collage option panel ROW 6:
    colorspace = StringVar()
    colorspace.set("lab")
    LabelWithTooltip(right_col_opt_panel, text="Colorspace:", tooltip=mkg.PARAMS.colorspace.help).grid(row=6, column=0, sticky="W")
    OptionMenu(right_col_opt_panel, colorspace, "", *mkg.PARAMS.colorspace.choices).grid(row=6, column=1, sticky="W")

    # right collage option panel ROW 7:
    dist_metric = StringVar()
    dist_metric.set("euclidean")
    LabelWithTooltip(right_col_opt_panel, text="Metric:", tooltip=mkg.PARAMS.metric.help).grid(row=7, column=0, sticky="W")
    OptionMenu(right_col_opt_panel, dist_metric, "", *mkg.PARAMS.metric.choices).grid(row=7, column=1, sticky="W")

    def attach_even():
        collage_uneven_panel.grid_remove()
        collage_even_panel.grid(row=11, columnspan=2, sticky="W")

    def attach_uneven():
        collage_even_panel.grid_remove()
        collage_uneven_panel.grid(row=11, columnspan=2, sticky="W")

    # right collage option panel ROW 8:
    LabelWithTooltip(right_col_opt_panel, text="Fairness of tiles: ", tooltip=mkg.PARAMS.unfair.help).grid(row=8, columnspan=2, sticky="W")

    # right collage option panel ROW 9:
    even = StringVar()
    even.set("even")
    RadiobuttonWithTooltip(right_col_opt_panel, text="Fair", variable=even, value="even", state=ACTIVE, command=attach_even, 
        tooltip="Require all tiles are used the same amount of times (fair tile usage)").grid(row=9, column=0, sticky="W")
    RadiobuttonWithTooltip(right_col_opt_panel, text="Unfair", variable=even, value="uneven", command=attach_uneven,
        tooltip=mkg.PARAMS.unfair.help).grid(row=9, column=1, sticky="W")

    # right collage option panel ROW 10:
    Separator(right_col_opt_panel, orient="horizontal").grid(row=10, columnspan=2, sticky="we", pady=(5, 5))

    # right collage option panel ROW 11: Dynamically attached
    # could EITHER collage even panel OR collage uneven panel
    # ----------------------- start collage even panel ------------------------
    collage_even_panel = PanedWindow(right_col_opt_panel)
    collage_even_panel.grid(row=11, columnspan=2, sticky="W")

    # collage even panel ROW 1
    LabelWithTooltip(collage_even_panel, text="Duplicates:", tooltip=mkg.PARAMS.dup.help).grid(row=1, column=0, sticky="W")
    dup = DoubleVar()
    dup.set(1)
    Entry(collage_even_panel, textvariable=dup, width=5).grid(row=1, column=1, sticky="W")
    # ----------------------- end collage even panel ------------------------

    # ----------------------- start collage uneven panel --------------------
    collage_uneven_panel = PanedWindow(right_col_opt_panel)

    # collage uneven panel ROW 0
    LabelWithTooltip(collage_uneven_panel, text="Max width:", tooltip=mkg.PARAMS.max_width.help).grid(row=0, column=0, sticky="W")
    max_width = IntVar()
    max_width.set(80)
    Entry(collage_uneven_panel, textvariable=max_width, width=5).grid(row=0, column=1, sticky="W")

    LabelWithTooltip(collage_uneven_panel, text="Freq Mul:", tooltip=mkg.PARAMS.freq_mul.help).grid(row=2, column=0, sticky="W")
    freq_mul = DoubleVar()
    freq_mul.set(1)
    Entry(collage_uneven_panel, textvariable=freq_mul, width=5).grid(row=2, column=1, sticky="W")

    deterministic = BooleanVar()
    deterministic.set(False)
    CheckbuttonWithTooltip(collage_uneven_panel, text="Deterministic", variable=deterministic, 
        tooltip=mkg.PARAMS.deterministic.help).grid(row=3, columnspan=2, sticky="w")
    # ----------------------- end collage uneven panel ----------------------

    def generate_collage():
        if imgs is None:
            return messagebox.showerror("No tiles", "Please first load tiles")
        if dest_img is None:
            return messagebox.showerror("No destination image", "Please first load the image that you're trying to fit")
    
        try:
            if is_salient.get():
                lower_thresh = saliency_thresh_scale.get() / 100
                assert 0.0 < lower_thresh < 1.0, "saliency threshold must be between 0 and 1"
            else:
                lower_thresh = None
            
            alpha = 1 - alpha_scale.get() / 100.0
            assert 0.0 <= alpha <= 1.0
            
            if even.get() == "even":
                _dup = mkg.check_dup_valid(dup.get())
                
                if is_salient.get():
                    def action():
                        return mkg.MosaicFairSalient(
                            dest_img, imgs, _dup, colorspace.get(), dist_metric.get(), 
                            lower_thresh, salient_bg_color, out_wrapper).process_dest_img(dest_img)

                else:               
                    def action():
                        return mkg.MosaicFair(dest_img.shape, imgs, _dup, 
                            colorspace.get(), dist_metric.get()).process_dest_img(dest_img, out_wrapper)
            else:
                assert max_width.get() > 0, "Max width must be a positive number"
                assert freq_mul.get() >= 0, "Max width must be a nonnegative real number"

                def action():
                    return mkg.MosaicUnfair(dest_img.shape, imgs, max_width.get(), colorspace.get(), dist_metric.get(), 
                        lower_thresh, salient_bg_color, freq_mul.get(), not deterministic.get()).process_dest_img(dest_img)

            def wrapper():
                global result_collage
                try:
                    result_collage = action()
                    return mkg.alpha_blend(result_collage, dest_img, alpha) 
                except AssertionError as e:
                    return messagebox.showerror("Error", e)
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            pool.submit(wrapper).add_done_callback(lambda f: show_img(f.result()))

        except AssertionError as e:
            return messagebox.showerror("Error", e)
        except:
            return messagebox.showerror("Error", traceback.format_exc())


    def attach_salient_opt():
        if is_salient.get():
            salient_opt_panel.grid(row=13, columnspan=2, pady=2, sticky="w")
        else:
            salient_opt_panel.grid_remove()

    # right collage option panel ROW 12
    is_salient = BooleanVar()
    is_salient.set(False)
    is_salient_check = Checkbutton(right_col_opt_panel, text="Salient objects only",
                variable=is_salient, command=attach_salient_opt)
    is_salient_check.grid(row=12, columnspan=2, sticky="w")

    # right collage option panel ROW 13
    salient_opt_panel = PanedWindow(right_col_opt_panel)

    change_thresh_queue = []

    def init_change_thresh(_):
        for f in change_thresh_queue:
            f.cancel()
        change_thresh_queue.clear()
        fut = pool.submit(change_thresh)
        change_thresh_queue.append(fut)
    
    def change_thresh():
        if dest_img is not None:
            lower_thresh = saliency_thresh_scale.get() / 100
            assert 0.0 <= lower_thresh <= 1.0
            _, thresh_map = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency((dest_img * 255).astype(np.uint8))
            tmp_dest_img = dest_img.copy()
            tmp_dest_img[thresh_map < lower_thresh] = np.asarray(salient_bg_color[::-1], dtype=np.float32) / 255.0
            show_img(tmp_dest_img, False)
            
    Label(salient_opt_panel, text="Saliency threshold: ").grid(row=0, column=0, sticky="w")
    saliency_thresh_scale = Scale(salient_opt_panel, from_=1.0, to=99.0, orient=HORIZONTAL, length=150, command=init_change_thresh)
    saliency_thresh_scale.set(50.0)
    saliency_thresh_scale.grid(row=1, columnspan=2, sticky="W")
    salient_bg_color = (255, 255, 255)

    def change_bg_color():
        global salient_bg_color, last_resize_time
        rbg_color, hex_color = colorchooser.askcolor(color=salient_bg_color)
        if hex_color:
            last_resize_time = time.time()
            salient_bg_color = rbg_color
            salient_bg_chooser["bg"] = hex_color
            salient_bg_chooser.update()

    salient_bg_chooser = tk.Button(salient_opt_panel, text="Select Background Color",
                                   command=change_bg_color, bg="#FFFFFF")
    salient_bg_chooser.grid(row=2, columnspan=2, pady=(3, 1))

    # right collage option panel ROW 14
    collage_button = Button(right_col_opt_panel, text=" Generate Collage ", command=generate_collage)
    collage_button.config(state='disabled')
    collage_button.grid(row=14, columnspan=2, pady=(3, 5))
    # ------------------------ end right collage option panel --------------------

    # right panel ROW 9:
    Separator(right_panel, orient="horizontal").grid(
        row=9, columnspan=2, sticky="we", pady=(4, 10))

    save_img_init_dir = init_dir

    def save_img():
        global save_img_init_dir
        if result_img is None:
            messagebox.showerror("Error", "You don't have any image to save yet!")
            return
        
        fp = filedialog.asksaveasfilename(initialdir=save_img_init_dir, title="Save your collage",
                                          filetypes=(("images", "*.jpg"), ("images", "*.png")),
                                          defaultextension=".png", initialfile="result.png")
        dir_name = os.path.dirname(fp)
        if fp is not None and len(fp) > 0 and os.path.isdir(dir_name):
            save_img_init_dir = dir_name
            print("Saving image to", fp)
            try:
                mkg.imwrite(fp, result_img)
                print("Saved!")
            except:
                messagebox.showerror("Error", traceback.format_exc())

    # right panel ROW 10:
    save_button = Button(right_panel, text=" Save image ", command=save_img)
    save_button.config(state='disabled')
    save_button.grid(row=10, columnspan=2)
    # -------------------------- end right panel -----------------------------------

    # make the window appear at the center
    # https://www.reddit.com/r/Python/comments/6m03sh/make_tkinter_window_in_center_of_screen_newbie/
    root.update_idletasks()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    size = tuple(int(pos) for pos in root.geometry().split('+')[0].split('x'))
    x = w / 2 - size[0] / 2
    y = h / 2 - size[1] / 2 - 10
    root.geometry("%dx%d+%d+%d" % (size + (x, y)))
    root.update()

    right_panel_width = right_panel.winfo_width()
    log_entry_height = left_panel.winfo_height() - canvas.winfo_height()
    last_resize_time = time.time()

    def canvas_resize(event):
        global last_resize_time
        if time.time() - last_resize_time > 0.1 and event.width >= 850 and event.height >= 550:
            last_resize_time = time.time()
            log_entry.configure(height=6 + math.floor((event.height - 500) / 80))
            log_entry.width = log_entry.initial_width + math.floor((event.width - 800) / 10)
            mkg.pbar_ncols = log_entry.width
            log_entry.update()
            canvas.configure(
                width=event.width - right_panel_width - 20, 
                height=event.height - log_entry.winfo_height() - 15)
            canvas.update()
            if result_img is not None:
                show_img(result_img, False)

    root.bind("<Configure>", canvas_resize)
    out_wrapper = log_entry
    sys.stdout = out_wrapper
    sys.stderr = out_wrapper
    mkg.enable_gpu()

    if cmd_args.D:
        file_path.set(cmd_args.src)
        print("Loading tiles from", cmd_args.src)
        pool.submit(load_img_action).add_done_callback(lambda f: show_img(f.result()))

        print("Destination image loaded from", cmd_args.collage)
        dest_img = mkg.imread(cmd_args.collage)
        show_img(dest_img, False)
        dest_img_path.set(cmd_args.collage)

        reload_img_button.config(state='enabled')
        sort_button.config(state='enabled')
        collage_button.config(state='enabled')

    root.mainloop()
