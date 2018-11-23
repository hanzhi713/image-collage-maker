from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import *
import traceback
import os
import cv2
from PIL import Image, ImageTk
from multiprocessing.pool import ThreadPool
from multiprocessing import Queue, freeze_support
import math
import sys
import make_img

make_img.pbar_ncols = 110


def limit_wh(w: int, h: int, max_width: int, max_height: int) -> [int, int]:
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
        self.width = 110
        self.gui = True
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
    root = Tk()
    left_panel = PanedWindow(root)
    left_panel.grid(row=0, column=0)
    Separator(root, orient="vertical").grid(
        row=0, column=1, sticky="nsew", padx=(5, 5))
    right_panel = PanedWindow(root)
    right_panel.grid(row=0, column=2)

    cmd_log = StringVar()
    cmd_log.set("")

    Separator(root, orient="horizontal").grid(
        row=1, columnspan=2, sticky="nsew", pady=(5, 5))

    log_panel = PanedWindow(root)
    log_panel.grid(row=2, column=0, columnspan=3, sticky="WE")
    log_panel.grid_columnconfigure(0, weight=1)
    log_entry = SafeText(log_panel, height=6, bd=0)
    log_entry.grid(row=1, column=0, sticky="nsew")
    scroll = Scrollbar(log_panel, orient="vertical", command=log_entry.yview)
    log_entry.config(yscrollcommand=scroll.set)
    scroll.grid(row=1, column=1, sticky="nsew")

    out_wrapper = log_entry
    sys.stdout = out_wrapper
    sys.stderr = out_wrapper

    canvas = Canvas(root, width=800, height=500)
    result_img = None


    def show_img(img):
        global result_img
        result_img = img
        w, h = limit_wh(*img.shape[:2][::-1], 800, 500)
        preview = cv2.cvtColor(cv2.resize(img, (w, h),
                                          cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        root.preview = ImageTk.PhotoImage(image=Image.fromarray(preview))
        canvas.delete("all")
        canvas.create_image((800 - w) // 2, (500 - h) // 2,
                            image=root.preview, anchor=NW)
        print("Done")


    left_panel.add(canvas)

    right_top_panel = PanedWindow(right_panel)

    file_path = StringVar()
    file_path.set("N/A")
    Label(right_top_panel, text="Path of source images:").grid(
        columnspan=2, sticky="W")
    Label(right_top_panel, textvariable=file_path, wraplength=150).grid(
        row=1, columnspan=2, sticky="W")

    opt = StringVar()
    opt.set("sort")
    img_size = IntVar()
    img_size.set(50)
    Label(right_top_panel, text="Image size: ").grid(
        row=2, column=0, pady=(5, 2))
    Entry(right_top_panel, width=5, textvariable=img_size).grid(
        row=2, column=1, sticky="W", pady=(5, 2))
    recursive = BooleanVar()
    recursive.set(True)
    Checkbutton(right_top_panel, text="Read sub-folders",
                variable=recursive).grid(row=3, columnspan=2, sticky="W")

    imgs = None
    current_image = None


    def load_images():
        global imgs, current_image
        fp = filedialog.askdirectory(
            initialdir=os.path.dirname(__file__), title="Select folder of source images")
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
                    imgs = make_img.read_images(
                        fp, (size, size), recursive.get(), 4)
                    grid = make_img.calculate_grid_size(
                        16, 10, len(imgs))
                    return make_img.make_collage(grid, imgs, False)
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            pool = ThreadPool(1)
            print("Loading source images from", fp)
            pool.apply_async(action, args=(), callback=show_img)
            pool.close()

        except:
            t = traceback.format_exc()
            messagebox.showerror("Error", t)


    Button(right_top_panel, text=" Load source images ", command=load_images).grid(
        row=4, columnspan=2, pady=(2, 5))

    right_top_panel.grid(row=0, column=0, pady=10, sticky="W")

    Separator(right_panel, orient="horizontal").grid(
        row=1, columnspan=2, sticky="we")

    right_sort_opt_panel = PanedWindow(right_panel)
    right_sort_opt_panel.grid(row=2, column=0, pady=10, sticky="W")

    sort_method = StringVar()
    sort_method.set("bgr_sum")
    Label(right_sort_opt_panel, text="Sort methods:").grid(
        row=0, column=0, pady=5, sticky="W")
    OptionMenu(right_sort_opt_panel, sort_method, "", *
    make_img.all_sort_methods).grid(row=0, column=1)

    Label(right_sort_opt_panel, text="Aspect ratio:").grid(
        row=1, column=0, sticky="W", pady=(2, 2))
    aspect_ratio_panel = PanedWindow(right_sort_opt_panel)
    rw = IntVar()
    rw.set(16)
    rh = IntVar()
    rh.set(10)
    aspect_ratio_panel.grid(row=1, column=1, pady=(2, 2))
    Entry(aspect_ratio_panel, width=3, textvariable=rw).grid(row=0, column=0)
    Label(aspect_ratio_panel, text=":").grid(row=0, column=1)
    Entry(aspect_ratio_panel, width=3, textvariable=rh).grid(row=0, column=2)

    rev_row = BooleanVar()
    rev_row.set(False)
    rev_sort = BooleanVar()
    rev_sort.set(False)
    Checkbutton(right_sort_opt_panel, variable=rev_row,
                text="Reverse consecutive row").grid(row=2, columnspan=2, sticky="W")
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
                    grid, sorted_imgs = make_img.sort_collage(imgs, (w, h), sort_method.get(),
                                                            rev_sort.get())
                    return make_img.make_collage(grid, sorted_imgs, rev_row.get())
                except:
                    messagebox.showerror("Error", traceback.format_exc())

            pool = ThreadPool(1)
            pool.apply_async(action, args=(), callback=show_img)
            pool.close()


    Button(right_sort_opt_panel, text="Generate sorted image",
           command=generate_sorted_image).grid(row=4, columnspan=2, pady=5)

    right_collage_opt_panel = PanedWindow(right_panel)
    sigma = StringVar()
    sigma.set("1.0")
    color_space = StringVar()
    color_space.set("lab")
    dest_img_path = StringVar()
    dest_img_path.set("N/A")
    dest_img = None
    Label(right_collage_opt_panel, text="Path of destination image").grid(
        row=0, columnspan=2, sticky="W", pady=(2, 3))
    Label(right_collage_opt_panel, textvariable=dest_img_path,
          wraplength=150).grid(row=1, columnspan=2, sticky="W")


    def load_dest_img():
        global dest_img
        if imgs is None:
            messagebox.showerror(
                "Empty set", "Please first load source images")
        else:
            fp = filedialog.askopenfilename(initialdir=os.path.dirname(__file__), title="Select destination image",
                                            filetypes=(("images", "*.jpg"), ("images", "*.png"), ("images", "*.gif"),
                                                       ("all files", "*.*")))
            if fp is not None and len(fp) > 0 and os.path.isfile(fp):
                try:
                    print("Destination image loaded from", fp)
                    dest_img = cv2.imread(fp)
                    show_img(dest_img)
                    dest_img_path.set(fp)
                except:
                    messagebox.showerror("Error reading file",
                                         traceback.format_exc())
            else:
                return


    Button(right_collage_opt_panel, text="Load destination image",
           command=load_dest_img).grid(row=2, columnspan=2)
    Label(right_collage_opt_panel, text="Sigma: ").grid(
        row=3, column=0, sticky="W", pady=(5, 2))
    Entry(right_collage_opt_panel, textvariable=sigma, width=8).grid(
        row=3, column=1, sticky="W", pady=(5, 2))
    Label(right_collage_opt_panel, text="Colorspace: ").grid(
        row=4, column=0, sticky="W")
    OptionMenu(right_collage_opt_panel, color_space, "", *
    make_img.all_color_spaces).grid(row=4, column=1, sticky="W")

    ctype = StringVar()
    ctype.set("float16")
    dup = IntVar()
    dup.set(1)

    collage_even_panel = PanedWindow(right_collage_opt_panel)
    Label(collage_even_panel, text="C Types: ").grid(
        row=0, column=0, sticky="W")
    OptionMenu(collage_even_panel, ctype, "", *
    make_img.all_ctypes).grid(row=0, column=1, sticky="W")
    Label(collage_even_panel, text="Duplicates: ").grid(
        row=1, column=0, sticky="W")
    Entry(collage_even_panel, textvariable=dup,
          width=5).grid(row=1, column=1, sticky="W")
    collage_even_panel.grid(row=8, columnspan=2, sticky="W")

    max_width = IntVar()
    max_width.set(50)
    collage_uneven_panel = PanedWindow(right_collage_opt_panel)
    Label(collage_uneven_panel, text="Max width: ").grid(
        row=0, column=0, sticky="W")
    Entry(collage_uneven_panel, textvariable=max_width,
          width=5).grid(row=0, column=1, sticky="W")


    def attach_even():
        collage_uneven_panel.grid_remove()
        collage_even_panel.grid(row=8, columnspan=2, sticky="W")


    def attach_uneven():
        collage_even_panel.grid_remove()
        collage_uneven_panel.grid(row=8, columnspan=2, sticky="W")


    distance_metric = StringVar()
    distance_metric.set("euclidena")
    Label(right_collage_opt_panel, text="Metric: ").grid(row=5, column=0, sticky="W")
    OptionMenu(right_collage_opt_panel, distance_metric, "", *make_img.all_metrics).grid(row=5, column=1, sticky="W")

    even = StringVar()
    even.set("even")
    Radiobutton(right_collage_opt_panel, text="Even", variable=even, value="even",
                state=ACTIVE, command=attach_even).grid(row=6, column=0, sticky="W")
    Radiobutton(right_collage_opt_panel, text="Uneven", variable=even, value="uneven",
                command=attach_uneven).grid(row=6, column=1, sticky="W")
    
    Separator(right_collage_opt_panel, orient="horizontal").grid(
        row=7, columnspan=2, sticky="we", pady=(5, 5))

    def generate_collage():
        if imgs is None:
            return messagebox.showerror("Empty set", "Please first load source images")
        if not os.path.isfile(dest_img_path.get()):
            return messagebox.showerror("No destination image", "Please first load the image that you're trying to fit")
        else:
            try:
                if even.get() == "even":
                    assert dup.get() > 0, "Duplication must be a positive number"
                    assert float(sigma.get()) != 0, "Sigma must be non-zero"

                    def action():
                        try:
                            grid, sorted_imgs, _ = make_img.calculate_collage_bipartite(dest_img_path.get(), imgs,
                                                                                        dup.get(), color_space.get(),
                                                                                        ctype.get(), float(sigma.get()), 
                                                                                        distance_metric.get(), out_wrapper)
                            return make_img.make_collage(grid, sorted_imgs, False)
                        except:
                            messagebox.showerror(
                                "Error", traceback.format_exc())
                else:
                    assert max_width.get() > 0, "Max width must be a positive number"

                    def action():
                        try:
                            grid, sorted_imgs, _ = make_img.calculate_collage_dup(dest_img_path.get(), imgs,
                                                                                  max_width.get(), color_space.get(),
                                                                                  float(sigma.get()), distance_metric.get())
                            return make_img.make_collage(grid, sorted_imgs, False)
                        except:
                            messagebox.showerror(
                                "Error", traceback.format_exc())

                pool = ThreadPool(1)
                pool.apply_async(action, callback=show_img)
                pool.close()

            except AssertionError as e:
                return messagebox.showerror("Error", e)
            except:
                return messagebox.showerror("Error", traceback.format_exc())


    Button(right_collage_opt_panel, text=" Generate Collage ",
           command=generate_collage).grid(row=9, columnspan=2, pady=(5, 3))


    def attach_sort():
        right_collage_opt_panel.grid_remove()
        right_sort_opt_panel.grid(row=2, columnspan=2, sticky="W")


    def attach_collage():
        right_sort_opt_panel.grid_remove()
        right_collage_opt_panel.grid(row=2, columnspan=2, sticky="W")


    Radiobutton(right_top_panel, text="Sort", value="sort", variable=opt,
                state=ACTIVE, command=attach_sort).grid(row=5, column=0, sticky="W")
    Radiobutton(right_top_panel, text="Collage", value="collage", variable=opt,
                command=attach_collage).grid(row=5, column=1, sticky="W")

    Separator(right_panel, orient="horizontal").grid(
        row=3, columnspan=2, sticky="we", pady=(0, 10))


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

            fp = filedialog.asksaveasfilename(initialdir=os.path.dirname(__file__), title="Save your collage",
                                              filetypes=(
                                                  ("images", "*.jpg"), ("images", "*.png")),
                                              defaultextension=".png", initialfile="result.png")
            if fp is not None and len(fp) > 0 and os.path.isdir(os.path.dirname(fp)):
                print("Image saved to", fp)
                imwrite(fp, result_img)


    Button(right_panel, text=" Save image ",
           command=save_img).grid(row=4, columnspan=2)

    # make the window appear at the center
    # https://www.reddit.com/r/Python/comments/6m03sh/make_tkinter_window_in_center_of_screen_newbie/
    root.update_idletasks()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    size = tuple(int(pos) for pos in root.geometry().split('+')[0].split('x'))
    x = w / 2 - size[0] / 2
    y = h / 2 - size[1] / 2 - 10
    root.geometry("%dx%d+%d+%d" % (size + (x, y)))
    root.mainloop()
