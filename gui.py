import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, colorchooser
from tkinter.ttk import *
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, freeze_support
from typing import Tuple, List
import argparse
import traceback
import math
import sys
import os
import time

import cv2
import numpy as np
import make_img as mkg
from PIL import Image, ImageTk


def limit_wh(w: int, h: int, max_width: int, max_height: int) -> Tuple[int, int]:
    if h > max_height:
        ratio = max_height / h
        h = max_height
        w = math.floor(w * ratio)
    if w > max_width:
        ratio = max_width / w
        w = max_width
        h = math.floor(h * ratio)
    return w, h


# adapted from
# https://stackoverflow.com/questions/16745507/tkinter-how-to-use-threads-to-preventing-main-event-loop-from-freezing
class SafeText(Text):
    def __init__(self, master, **options):
        Text.__init__(self, master, **options)
        self.queue = Queue()
        self.encoding = "utf-8"
        self.gui = True
        self.initial_width = 85
        self.width = self.initial_width
        self.update_me()

    def write(self, line: str):
        self.queue.put(line)

    def flush(self):
        pass

    # this one run in the main thread
    def update_me(self):
        try:
            while True:
                line = self.queue.get_nowait()

                # a naive way to process the \r control char
                if line.find("\r") > -1:
                    line = line.replace("\r", "")
                    row = int(self.index(END).split(".")[0])
                    self.delete("{}.0".format(row - 1),
                                "{}.{}".format(row - 1, len(line)))
                    self.insert("end-1c linestart", line)
                else:
                    self.insert(END, line)
                self.see("end-1c")
        except:
            pass
        self.update_idletasks()
        self.after(50, self.update_me)


if __name__ == "__main__":
    freeze_support()
    pool = ThreadPoolExecutor(1)
    root = Tk()
    root.title("Collage Maker")

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", action="store_true")
    parser.add_argument("--init_dir", "-d", type=str,
                        default=os.path.dirname(__file__))
    parser.add_argument("--src", "-s", type=str,
                        default=os.path.join(os.path.dirname(__file__), "img"))
    parser.add_argument("--collage", "-c", type=str,
                        default=os.path.join(os.path.dirname(__file__), "examples", "dest.png"))
    cmd_args = parser.parse_args()
    init_dir = cmd_args.init_dir
    if not os.path.isdir(init_dir):
        init_dir = os.path.dirname(__file__)

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
    canvas = Canvas(left_panel, width=800, height=500)
    canvas.grid(row=0, column=0)

    # left panel ROW 1
    Separator(left_panel, orient="horizontal").grid(
        row=1, columnspan=2, sticky="NSEW", pady=(5, 5))

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

    out_wrapper = log_entry
    sys.stdout = out_wrapper
    sys.stderr = out_wrapper
    result_img = None
    # ------------------ end left panel ------------------------

    def show_img(img: np.ndarray, printDone: bool = True) -> None:
        global result_img
        result_img = img
        width, height = canvas.winfo_width(), canvas.winfo_height()
        img_h, img_w, _ = img.shape
        w, h = limit_wh(img_w, img_h, width, height)
        if img_h > height or img_w > width:
            img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        preview = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # prevent the image from being garbage-collected
        root.preview = ImageTk.PhotoImage(image=Image.fromarray(preview))
        canvas.delete("all")
        canvas.create_image((width - w) // 2, (height - h) // 2,
                            image=root.preview, anchor=NW)
        if printDone:
            print("Done")

    # --------------------- right panel's children ------------------------
    # right panel ROW 0
    file_path = StringVar()
    file_path.set("N/A")
    Label(right_panel, text="Path of source images:").grid(
        row=0, columnspan=2, sticky="W", pady=(10, 2))

    # right panel ROW 1
    Label(right_panel, textvariable=file_path, wraplength=150).grid(
        row=1, columnspan=2, sticky="W")

    # right panel ROW 2
    opt = StringVar()
    opt.set("sort")
    img_size = IntVar()
    img_size.set(50)
    Label(right_panel, text="Image size: ").grid(
        row=2, column=0, sticky="W", pady=(3, 2))
    Entry(right_panel, width=5, textvariable=img_size).grid(
        row=2, column=1, sticky="W", pady=(3, 2))

    # right panel ROW 3
    resize_opt = StringVar()
    resize_opt.set("center")
    Label(right_panel, text="Resize flag: ").grid(
        row=3, column=0, sticky="W", pady=(2, 3))
    OptionMenu(right_panel, resize_opt, "", "center", "stretch").grid(
        row=3, column=1, sticky="W", pady=(2, 3))

    # right panel ROW 4
    recursive = BooleanVar()
    recursive.set(True)
    Checkbutton(right_panel, text="Read sub-folders",
                variable=recursive).grid(row=4, columnspan=2, sticky="W")

    imgs = None
    current_image = None

    def load_images():
        global imgs, current_image
        fp = filedialog.askdirectory(
            initialdir=init_dir, title="Select folder of source images")
        if len(fp) > 0 and os.path.isdir(fp):
            file_path.set(fp)
        else:
            return
        try:
            size = img_size.get()
            if size < 1:
                return messagebox.showerror("Illegal Argument", "Img size must be greater than 1")

            def action():
                global imgs
                try:
                    imgs = mkg.read_images(
                        fp, (size, size), recursive.get(), 4, resize_opt.get())
                    grid = mkg.calc_grid_size(
                        16, 10, len(imgs))
                    return mkg.make_collage(grid, imgs, False)
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            print("Loading source images from", fp)
            pool.submit(action).add_done_callback(
                lambda f: show_img(f.result()))

        except:
            messagebox.showerror("Error", traceback.format_exc())

    # right panel ROW 5
    Button(right_panel, text=" Load source images ", command=load_images).grid(
        row=5, columnspan=2, pady=(3, 4))

    def attach_sort():
        right_col_opt_panel.grid_remove()
        right_sort_opt_panel.grid(row=8, columnspan=2, sticky="W")

    def attach_collage():
        right_sort_opt_panel.grid_remove()
        right_col_opt_panel.grid(row=8, columnspan=2, sticky="W")

    # right panel ROW 6
    Radiobutton(right_panel, text="Sort", value="sort", variable=opt,
                state=ACTIVE, command=attach_sort).grid(row=6, column=0, sticky="W")
    Radiobutton(right_panel, text="Collage", value="collage", variable=opt,
                command=attach_collage).grid(row=6, column=1, sticky="W")

    # right panel ROW 7
    Separator(right_panel, orient="horizontal").grid(
        row=7, columnspan=2, pady=(5, 3), sticky="we")

    # right panel ROW 8: Dynamically attached
    # right sort option panel !!OR!! right collage option panel
    right_sort_opt_panel = PanedWindow(right_panel)
    right_sort_opt_panel.grid(
        row=8, column=0, columnspan=2, pady=2, sticky="W")

    # ------------------------- right sort option panel --------------------------
    # right sort option panel ROW 0:
    sort_method = StringVar()
    sort_method.set("bgr_sum")
    Label(right_sort_opt_panel, text="Sort methods:").grid(
        row=0, column=0, pady=5, sticky="W")
    OptionMenu(right_sort_opt_panel, sort_method, "", *
               mkg.all_sort_methods).grid(row=0, column=1)

    # right sort option panel ROW 1:
    Label(right_sort_opt_panel, text="Aspect ratio:").grid(
        row=1, column=0, sticky="W", pady=(2, 2))
    aspect_ratio_panel = PanedWindow(right_sort_opt_panel)
    aspect_ratio_panel.grid(row=1, column=1, pady=(2, 2))
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
            messagebox.showerror(
                "Empty set", "Please first load source images")
        else:
            try:
                w, h = rw.get(), rh.get()
                assert w > 0, "Width must be greater than 0"
                assert h > 0, "Height must be greater than 0"
            except:
                return messagebox.showerror("Illegal Argument", traceback.format_exc())

            def action():
                try:
                    grid, sorted_imgs = mkg.sort_collage(imgs, (w, h), sort_method.get(),
                                                         rev_sort.get())
                    return mkg.make_collage(grid, sorted_imgs, rev_row.get())
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            pool.submit(action).add_done_callback(
                lambda f: show_img(f.result()))

    # right sort option panel ROW 4:
    Button(right_sort_opt_panel, text="Generate sorted image",
           command=generate_sorted_image).grid(row=4, columnspan=2, pady=5)
    # ------------------------ end right sort option panel -----------------------------

    # ------------------------ right collage option panel ------------------------------
    # right collage option panel ROW 0:
    right_col_opt_panel = PanedWindow(right_panel)
    dest_img_path = StringVar()
    dest_img_path.set("N/A")
    dest_img = None
    Label(right_col_opt_panel, text="Path of destination image: ").grid(
        row=0, columnspan=2, sticky="W", pady=2)

    # right collage option panel ROW 1:
    Label(right_col_opt_panel, textvariable=dest_img_path,
          wraplength=150).grid(row=1, columnspan=2, sticky="W")

    def load_dest_img():
        global dest_img
        if imgs is None:
            messagebox.showerror(
                "Empty set", "Please first load source images")
        else:
            fp = filedialog.askopenfilename(initialdir=init_dir, title="Select destination image",
                                            filetypes=(("images", "*.jpg"), ("images", "*.png"), ("images", "*.gif"),
                                                       ("all files", "*.*")))
            if fp is not None and len(fp) > 0 and os.path.isfile(fp):
                try:
                    print("Destination image loaded from", fp)
                    dest_img = mkg.imread(fp)
                    show_img(dest_img, False)
                    dest_img_path.set(fp)
                except:
                    messagebox.showerror("Error reading file",
                                         traceback.format_exc())
            else:
                return

    # right collage option panel ROW 2:
    Button(right_col_opt_panel, text="Load destination image",
           command=load_dest_img).grid(row=2, columnspan=2, pady=(3, 2))

    # right collage option panel ROW 3:
    sigma = StringVar()
    sigma.set("1.0")
    Label(right_col_opt_panel, text="Sigma: ").grid(
        row=3, column=0, sticky="W", pady=(5, 2))
    Entry(right_col_opt_panel, textvariable=sigma, width=8).grid(
        row=3, column=1, sticky="W", pady=(5, 2))

    # right collage option panel ROW 4:
    colorspace = StringVar()
    colorspace.set("lab")
    Label(right_col_opt_panel, text="Colorspace: ").grid(
        row=4, column=0, sticky="W")
    OptionMenu(right_col_opt_panel, colorspace, "", *
               mkg.all_colorspaces).grid(row=4, column=1, sticky="W")

    # right collage option panel ROW 5:
    dist_metric = StringVar()
    dist_metric.set("euclidean")
    Label(right_col_opt_panel, text="Metric: ").grid(
        row=5, column=0, sticky="W")
    OptionMenu(right_col_opt_panel, dist_metric, "", *
               mkg.all_metrics).grid(row=5, column=1, sticky="W")

    def attach_even():
        collage_uneven_panel.grid_remove()
        collage_even_panel.grid(row=8, columnspan=2, sticky="W")

    def attach_uneven():
        collage_even_panel.grid_remove()
        collage_uneven_panel.grid(row=8, columnspan=2, sticky="W")

    # right collage option panel ROW 6:
    even = StringVar()
    even.set("even")
    Radiobutton(right_col_opt_panel, text="Even", variable=even, value="even",
                state=ACTIVE, command=attach_even).grid(row=6, column=0, sticky="W")
    Radiobutton(right_col_opt_panel, text="Uneven", variable=even, value="uneven",
                command=attach_uneven).grid(row=6, column=1, sticky="W")

    # right collage option panel ROW 7:
    Separator(right_col_opt_panel, orient="horizontal").grid(
        row=7, columnspan=2, sticky="we", pady=(5, 5))

    # right collage option panel ROW 8: Dynamically attached
    # could EITHER collage even panel OR collage uneven panel
    # ----------------------- start collage even panel ------------------------
    collage_even_panel = PanedWindow(right_col_opt_panel)
    collage_even_panel.grid(row=8, columnspan=2, sticky="W")

    # collage even panel ROW 0
    Label(collage_even_panel, text="C Types: ").grid(
        row=0, column=0, sticky="W")
    ctype = StringVar()
    ctype.set("float16")
    OptionMenu(collage_even_panel, ctype, "", *
               mkg.all_ctypes).grid(row=0, column=1, sticky="W")

    # collage even panel ROW 1
    Label(collage_even_panel, text="Duplicates: ").grid(
        row=1, column=0, sticky="W", pady=2)
    dup = IntVar()
    dup.set(1)
    Entry(collage_even_panel, textvariable=dup,
          width=5).grid(row=1, column=1, sticky="W", pady=2)
    # ----------------------- end collage even panel ------------------------

    # ----------------------- start collage uneven panel --------------------
    collage_uneven_panel = PanedWindow(right_col_opt_panel)

    # collage uneven panel ROW 0
    Label(collage_uneven_panel, text="Max width: ").grid(
        row=0, column=0, sticky="W")
    max_width = IntVar()
    max_width.set(80)
    Entry(collage_uneven_panel, textvariable=max_width,
          width=5).grid(row=0, column=1, sticky="W")
    # ----------------------- end collage uneven panel ----------------------

    def generate_collage():
        if imgs is None:
            return messagebox.showerror("Empty set", "Please first load source images")
        if not os.path.isfile(dest_img_path.get()):
            return messagebox.showerror("No destination image", "Please first load the image that you're trying to fit")
        else:
            try:
                if is_salient.get():
                    lower_thresh = salient_lower_thresh.get()
                    assert 0 <= lower_thresh < 255 or lower_thresh == -1, \
                        "Lower salient threshold must be -1 (auto) or between 0 and 255"
                    if even.get() == "even":
                        assert dup.get() > 0, "Duplication must be a positive number"
                        assert float(
                            sigma.get()) != 0, "Sigma must be non-zero"

                        def action():
                            try:
                                grid, sorted_imgs, _ = mkg.calc_salient_col_even_fast(dest_img_path.get(), imgs,
                                                                                      dup.get(), colorspace.get(),
                                                                                      ctype.get(), float(sigma.get()),
                                                                                      dist_metric.get(), lower_thresh,
                                                                                      salient_bg_color, out_wrapper)
                                return mkg.make_collage(grid, sorted_imgs, False)
                            except:
                                messagebox.showerror(
                                    "Error", traceback.format_exc())
                    else:
                        assert max_width.get() > 0, "Max width must be a positive number"

                        def action():
                            try:
                                grid, sorted_imgs, _ = mkg.calc_salient_col_dup(dest_img_path.get(), imgs,
                                                                                max_width.get(), colorspace.get(),
                                                                                float(
                                                                                    sigma.get()), dist_metric.get(),
                                                                                lower_thresh, salient_bg_color)
                                return mkg.make_collage(grid, sorted_imgs, False)
                            except:
                                messagebox.showerror(
                                    "Error", traceback.format_exc())
                else:
                    if even.get() == "even":
                        assert dup.get() > 0, "Duplication must be a positive number"
                        assert float(
                            sigma.get()) != 0, "Sigma must be non-zero"

                        def action():
                            try:
                                grid, sorted_imgs, _ = mkg.calc_col_even(dest_img_path.get(), imgs,
                                                                         dup.get(), colorspace.get(),
                                                                         ctype.get(), float(sigma.get()),
                                                                         dist_metric.get(), out_wrapper)
                                return mkg.make_collage(grid, sorted_imgs, False)
                            except:
                                messagebox.showerror(
                                    "Error", traceback.format_exc())
                    else:
                        assert max_width.get() > 0, "Max width must be a positive number"

                        def action():
                            try:
                                grid, sorted_imgs, _ = mkg.calc_col_dup(dest_img_path.get(), imgs,
                                                                        max_width.get(), colorspace.get(),
                                                                        float(sigma.get()), dist_metric.get())
                                return mkg.make_collage(grid, sorted_imgs, False)
                            except:
                                messagebox.showerror(
                                    "Error", traceback.format_exc())

                pool.submit(action).add_done_callback(
                    lambda f: show_img(f.result()))

            except AssertionError as e:
                return messagebox.showerror("Error", e)
            except:
                return messagebox.showerror("Error", traceback.format_exc())

    def attach_salient_opt():
        if is_salient.get():
            salient_opt_panel.grid(row=10, columnspan=2, pady=2, sticky="w")
        else:
            salient_opt_panel.grid_remove()

    # right collage option panel ROW 9
    is_salient = BooleanVar()
    is_salient.set(False)
    Checkbutton(right_col_opt_panel, text="Salient objects only",
                variable=is_salient, command=attach_salient_opt).grid(row=9, columnspan=2, sticky="w")

    # right collage option panel ROW 10
    salient_opt_panel = PanedWindow(right_col_opt_panel)
    salient_lower_thresh = IntVar()
    salient_lower_thresh.set(127)
    salient_opt_label = Label(salient_opt_panel, text="Lower threshold: ")
    salient_opt_entry = Entry(
        salient_opt_panel, textvariable=salient_lower_thresh, width=8)
    salient_opt_label.grid(row=0, column=0, sticky="w")
    salient_opt_entry.grid(row=0, column=1, sticky="w")
    salient_bg_color = np.array((255, 255, 255), np.uint8)

    def change_bg_color():
        global salient_bg_color, last_resize_time
        rbg_color, hex_color = colorchooser.askcolor(
            color=tuple(salient_bg_color))
        if hex_color:
            last_resize_time = time.time()
            salient_bg_color = np.array(rbg_color, np.uint8)
            salient_bg_chooser["bg"] = hex_color
            salient_bg_chooser.update()

    salient_bg_chooser = tk.Button(salient_opt_panel, text="Select Background Color",
                                   command=change_bg_color, bg="#FFFFFF")
    salient_bg_chooser.grid(row=1, columnspan=2, pady=(3, 1))

    # right collage option panel ROW 11
    Button(right_col_opt_panel, text=" Generate Collage ",
           command=generate_collage).grid(row=11, columnspan=2, pady=(3, 5))
    # ------------------------ end right collage option panel --------------------

    # right panel ROW 9:
    Separator(right_panel, orient="horizontal").grid(
        row=9, columnspan=2, sticky="we", pady=(4, 10))

    def save_img():
        if result_img is None:
            messagebox.showerror(
                "Error", "You don't have any image to save yet!")
        else:

            def imwrite(filename, img):
                try:
                    ext = os.path.splitext(filename)[1]
                    result, n = cv2.imencode(ext, img)

                    if result:
                        with open(filename, mode='wb') as f:
                            n.tofile(f)
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            fp = filedialog.asksaveasfilename(initialdir=init_dir, title="Save your collage",
                                              filetypes=(
                                                  ("images", "*.jpg"), ("images", "*.png")),
                                              defaultextension=".png", initialfile="result.png")
            if fp is not None and len(fp) > 0 and os.path.isdir(os.path.dirname(fp)):
                print("Image saved to", fp)
                imwrite(fp, result_img)

    # right panel ROW 10:
    Button(right_panel, text=" Save image ",
           command=save_img).grid(row=10, columnspan=2)
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
        if time.time() - last_resize_time > 0.25 and event.width >= 850 and event.height >= 550:
            last_resize_time = time.time()
            log_entry.configure(
                height=6 + math.floor((event.height - 500) / 80))
            log_entry.width = log_entry.initial_width + \
                math.floor((event.width - 800) / 10)
            mkg.pbar_ncols = log_entry.width
            log_entry.update()
            w, h = event.width - right_panel_width - \
                20, event.height - log_entry.winfo_height() - 15
            pw, ph = canvas.winfo_width(), canvas.winfo_height()
            w_scale, h_scale = w / pw, h / ph
            canvas.configure(width=w, height=h)
            canvas.update()
            if result_img is not None:
                show_img(result_img, False)

    root.bind("<Configure>", canvas_resize)

    if cmd_args.D:
        def debug_act(fp):
            global imgs
            try:
                imgs = mkg.read_images(
                    fp, (40, 40), True, 4, "center")
                grid = mkg.calc_grid_size(
                    16, 10, len(imgs))
                return mkg.make_collage(grid, imgs, False)
            except:
                messagebox.showerror("Error", traceback.format_exc())

        file_path.set(cmd_args.src)
        print("Loading source images from", cmd_args.src)
        pool.submit(debug_act, cmd_args.src).add_done_callback(
            lambda f: show_img(f.result()))

        print("Destination image loaded from", cmd_args.collage)
        dest_img = mkg.imread(cmd_args.collage)
        show_img(dest_img, False)
        dest_img_path.set(cmd_args.collage)

    root.mainloop()
