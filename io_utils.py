import os
import io
import sys
import time
import queue
import ctypes
import platform
import tempfile
import threading
from tkinter import Text, END
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def stdout_redirector(stream: io.TextIOBase):
    """
    write stuff from stdout to stream that has a .write method 

    part of the following function is borrowed from
    1. https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    2. https://gist.github.com/natedileas/8eb31dc03b76183c0211cdde57791005
    """
    if platform.system() == "Windows":
        if hasattr(sys, 'gettotalrefcount'): # debug build
            libc = ctypes.CDLL('ucrtbased')
        else:
            libc = ctypes.CDLL('api-ms-win-crt-stdio-l1-1-0')
    else:
        libc = ctypes.CDLL(None)
        
    stdout = sys.__stdout__
    stdout_fd = stdout.fileno()

    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied, tempfile.TemporaryFile(mode='w+b') as tfile: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        os.dup2(tfile.fileno(), stdout_fd)  # $ exec >&to

        try:
            flag = True

            def poll_once():
                libc.fflush(None)
                stdout.flush()
                tfile.flush()
                tfile.seek(0)
                stream.write(tfile.read().decode("utf-8"))
                tfile.truncate(0)

            def poll():
                while flag:
                    poll_once()
                    time.sleep(0.1)
                poll_once()

            poll_thread = threading.Thread(target=poll)
            poll_thread.start()

            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            flag = False
            poll_thread.join()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            stream.close()


class JVOutWrapper:
    """
    The output wrapper for displaying the progress of J-V algorithm
    """

    def __init__(self, io_wrapper, ncols):
        self.io_wrapper = io_wrapper
        self.tqdm = None
        self.ncols = ncols

    def write(self, lines: str):
        if self.io_wrapper is None:
            return
        for line in lines.split("\n"):
            if not line.startswith("lapjv: "):
                # self.io_wrapper.write(line + "\n")
                continue
            if line.startswith("lapjv: AUGMENT SOLUTION row "):
                line = line.replace(" ", "")
                slash_idx = line.find("/")
                s_idx = line.find("[")
                e_idx = line.find("]")
                if s_idx > -1 and slash_idx > -1 and e_idx > -1:
                    if self.tqdm:
                        self.tqdm.n = int(line[s_idx + 1:slash_idx])
                        self.tqdm.update(0)
                    else:
                        self.tqdm = tqdm(file=self.io_wrapper, ncols=self.ncols, total=int(line[slash_idx + 1:e_idx]), desc="lapjv: ")
                continue
            if not self.tqdm:
                self.io_wrapper.write(line + "\n")

    def flush(self):
        pass

    def close(self):
        if self.tqdm:
            self.tqdm.n = self.tqdm.total
            self.tqdm.refresh()
            self.tqdm.close()


# adapted from
# https://stackoverflow.com/questions/16745507/tkinter-how-to-use-threads-to-preventing-main-event-loop-from-freezing
class SafeText(Text):
    def __init__(self, master, **options):
        Text.__init__(self, master, **options)
        self.queue = queue.Queue()
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
        while not self.queue.empty():
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
        self.update_idletasks()
        self.after(50, self.update_me)
