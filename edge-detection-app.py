import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math
# -----------------------------------------------------------------------------------------------------------
# HARROUCHE BASMA 
# -----------------------------------------------------------------------------------------------------------


# -----------------------
# Fonctions utilitaires
# -----------------------
def np_to_pil(img_np):
    if img_np is None:
        return None
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode='L')

def load_image_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def show_image_in_label(pil_img, label, maxsize=(480, 360)):
    if pil_img is None:
        label.configure(image=None)
        label.image = None
        return
    w, h = pil_img.size
    mw, mh = maxsize
    scale = min(1.0, mw / w, mh / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    img_r = pil_img.resize(new_size)
    tkimg = ImageTk.PhotoImage(img_r)
    label.configure(image=tkimg)
    label.image = tkimg

def convolution_2d(img, kernel):
    H, W = img.shape
    k = kernel.shape[0]
    pad = k // 2
    img_pad = np.pad(img, pad, mode="edge")
    res = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            res[i, j] = np.sum(img_pad[i:i+k, j:j+k] * kernel)

    return np.clip(res, 0, 255).astype(np.uint8)

#masque des filtres

K_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

K_SOBEL_Y = np.array([[-1, -2, -1],
                      [0,  0,  0],
                      [1,  2,  1]], dtype=np.float32)

K_PREWITT_X = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)

K_PREWITT_Y = np.array([[-1, -1, -1],
                        [0,  0,  0],
                        [1,  1,  1]], dtype=np.float32)

K_ROBERTS_X = np.array([[1, 0],
                        [0, -1]], dtype=np.float32)

K_ROBERTS_Y = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

K_LAPLACE = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=np.float32)





# les filtres---------------------------------------------------------------------------------------

def filtre_gaussien_cv(img, sigma=1.0):
    if sigma <= 0:
        return img.copy()

    size = int(round(6 * sigma))
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1

    c = (size - 1) / 2
    ax = np.arange(size) - c

    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx*xx + yy*yy) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)

    return convolution_2d(img, kernel)

def gradient_from_filters(Gx, Gy):
    G = np.sqrt(Gx.astype(np.float32)**2 + Gy.astype(np.float32)**2)
    G = np.clip(G, 0, 255).astype(np.uint8)
    angle = np.arctan2(Gy, Gx)
    return G, angle

# seuillage --------------------------------------------------------------------
class ThresholdDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("choisir seuil")
        self.res = None

        tk.Label(self, text="Type de suillage:").pack()

        self.mode = tk.StringVar(value="simple")

        tk.Radiobutton(self, text="Simple", value="simple",
                       variable=self.mode, command=self.update_ui).pack(anchor="w")

        tk.Radiobutton(self, text="Hysteresis", value="hyst",
                       variable=self.mode, command=self.update_ui).pack(anchor="w")

        frame = tk.Frame(self)
        frame.pack(pady=8)

        # seuillage simple ------------------------------------------
        self.label_simple = tk.Label(frame, text="T :")
        self.label_simple.grid(row=0, column=0)
        self.val_simple = tk.IntVar(value=100)
        self.entry_simple = tk.Entry(frame, textvariable=self.val_simple, width=6)
        self.entry_simple.grid(row=0, column=1)

        # hysteresis---------------------------------------------
        self.label_low = tk.Label(frame, text="seuill_low :")
        self.val_low = tk.IntVar(value=50)
        self.entry_low = tk.Entry(frame, textvariable=self.val_low, width=6)

        self.label_high = tk.Label(frame, text="seuill_high :")
        self.val_high = tk.IntVar(value=120)
        self.entry_high = tk.Entry(frame, textvariable=self.val_high, width=6)

        tk.Button(self, text="OK", command=self.validate).pack(pady=6)

        self.update_ui()
        self.grab_set()
        self.wait_window()

    def update_ui(self):
        if self.mode.get() == "simple":
            self.label_low.grid_remove()
            self.entry_low.grid_remove()
            self.label_high.grid_remove()
            self.entry_high.grid_remove()

            self.label_simple.grid(row=0, column=0)
            self.entry_simple.grid(row=0, column=1)

        else:
            self.label_simple.grid_remove()
            self.entry_simple.grid_remove()

            self.label_low.grid(row=0, column=0)
            self.entry_low.grid(row=0, column=1)
            self.label_high.grid(row=1, column=0)
            self.entry_high.grid(row=1, column=1)

    def validate(self):
        if self.mode.get() == "simple":
            self.res = ("simple", self.val_simple.get())
        else:
            self.res = ("hyst", self.val_low.get(), self.val_high.get())
        self.destroy()

#________________________ seuillage _________________________________________________________________________________
def seuil_simple(img, T):
    return np.where(img >= T, 255, 0).astype(np.uint8)

def seuil_hysteresis(img, t_low, t_high):
    H, W = img.shape
    res = np.zeros((H, W), dtype=np.uint8)
    strong = 255
    weak = 80

    strong_pixels = img >= t_high
    weak_pixels = (img >= t_low) & (img < t_high)

    res[strong_pixels] = strong
    res[weak_pixels] = weak

    changed = True
    while changed:
        changed = False
        for i in range(1, H-1):
            for j in range(1, W-1):
                if res[i, j] == weak and np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i, j] = strong
                    changed = True

    res[res != strong] = 0
    return res

# laplacien --------------------------------------------------------
def laplacien(img):
    H, W = img.shape
    k = K_LAPLACE
    pad = 1
    img_pad = np.pad(img, pad, mode="edge").astype(np.int32)
    res = np.zeros((H, W), dtype=np.int32)

    for i in range(H):
        for j in range(W):
            res[i, j] = np.sum(img_pad[i:i+3, j:j+3] * k)

    return res

def zero_crossing(lap_img, seuil):
    H, W = lap_img.shape
    zc = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            center = lap_img[i, j]
            patch = lap_img[i - 1:i + 2, j - 1:j + 2]

            has_pos = np.any(patch > 0)
            has_neg = np.any(patch < 0)
            if not (has_pos and has_neg):
                continue

            amplitude = np.max(np.abs(patch - center))
            if amplitude >= seuil:
                zc[i, j] = 255

    return zc

def LoG(img, sigma, seuil):
    gauss = filtre_gaussien_cv(img, sigma)
    lap = laplacien(gauss)
    zc = zero_crossing(lap, seuil)
    return gauss, lap, zc

# windoow_________________________________________________________________________
class OperationWindow(tk.Toplevel):
    def __init__(self, master, title):
        super().__init__(master)
        self.title(title)

        self.img_orig = None
        self.img_result = None

        tk.Label(self, text="Image originale").pack()
        self.canvas_orig = tk.Label(self)
        self.canvas_orig.pack()

        tk.Label(self, text="Resultat").pack()
        self.canvas_res = tk.Label(self)
        self.canvas_res.pack()

        frame = tk.Frame(self)
        frame.pack(pady=6)

        tk.Button(frame, text="choisir img", command=self.load).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="appliquer", command=self.apply).pack(side=tk.LEFT, padx=4)

        tk.Button(frame, text="afficher Gx/Gy/Mag/Phase",
                  command=self.show_gradient).pack(side=tk.LEFT, padx=4)

        tk.Button(frame, text="sauver", command=self.save).pack(side=tk.LEFT, padx=4)
        tk.Button(frame, text="fermer", command=self.destroy).pack(side=tk.LEFT, padx=4)

    # gradient____________________________________________________________________________________
    def show_gradient(self):
        if self.img_orig is None:
            messagebox.showwarning("Avertissement", "Charger une image d'abord.")
            return

        # detminer le masque de puis le filtre
        if isinstance(self, SobelWindow):
            kx, ky = K_SOBEL_X, K_SOBEL_Y
        elif isinstance(self, PrewittWindow):
            kx, ky = K_PREWITT_X, K_PREWITT_Y
        elif isinstance(self, RobertsWindow):
            kx, ky = K_ROBERTS_X, K_ROBERTS_Y
        else:
            messagebox.showwarning("Avertissement",
                                   "Gx/Gy non disponibles pour ce filtre.")
            return

        Gx = convolution_2d(self.img_orig, kx)
        Gy = convolution_2d(self.img_orig, ky)
        Mag, Phase = gradient_from_filters(Gx, Gy)

        # Normalize phase
        phase_img = ((Phase - Phase.min()) /
                     (Phase.max() - Phase.min() + 1e-6) * 255).astype(np.uint8)

        win = tk.Toplevel(self)
        win.title("Gx / Gy / Magnitude / Phase")

        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --------------------------------

        def add_img(img, title):
            tk.Label(scroll_frame, text=title).pack()
            lbl = tk.Label(scroll_frame)

            lbl.pack()
            show_image_in_label(np_to_pil(img), lbl)

        add_img(Gx, "Gx")
        add_img(Gy, "Gy")
        add_img(Mag, "|G| Magnitude")
        add_img(phase_img, "Phase")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.bmp")])
        if path:
            self.img_orig = load_image_gray(path)
            show_image_in_label(np_to_pil(self.img_orig), self.canvas_orig)

    def save(self):
        if self.img_result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            np_to_pil(self.img_result).save(path)

    def apply(self):
        raise NotImplementedError
# --------------------------sobel----------------------------------
class SobelWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Sobel")

    def apply(self):
        if self.img_orig is None:
            return

        dlg = ThresholdDialog(self)
        if dlg.res is None:
            return

        Gx = convolution_2d(self.img_orig, K_SOBEL_X)
        Gy = convolution_2d(self.img_orig, K_SOBEL_Y)
        G, _ = gradient_from_filters(Gx, Gy)

        if dlg.res[0] == "simple":
            self.img_result = seuil_simple(G, dlg.res[1])
        else:
            self.img_result = seuil_hysteresis(G, dlg.res[1], dlg.res[2])

        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)


#---------------------------prewittwindow-----------------------------------------

class PrewittWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Prewitt")

    def apply(self):
        if self.img_orig is None:
            return

        dlg = ThresholdDialog(self)
        if dlg.res is None:
            return

        Gx = convolution_2d(self.img_orig, K_PREWITT_X)
        Gy = convolution_2d(self.img_orig, K_PREWITT_Y)
        G, _ = gradient_from_filters(Gx, Gy)

        if dlg.res[0] == "simple":
            self.img_result = seuil_simple(G, dlg.res[1])
        else:
            self.img_result = seuil_hysteresis(G, dlg.res[1], dlg.res[2])

        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)

# -----------------------------------robert-------------------------------------------
class RobertsWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Roberts")

    def apply(self):
        if self.img_orig is None:
            return

        dlg = ThresholdDialog(self)
        if dlg.res is None:
            return

        Gx = convolution_2d(self.img_orig, K_ROBERTS_X)
        Gy = convolution_2d(self.img_orig, K_ROBERTS_Y)
        G, _ = gradient_from_filters(Gx, Gy)

        if dlg.res[0] == "simple":
            self.img_result = seuil_simple(G, dlg.res[1])
        else:
            self.img_result = seuil_hysteresis(G, dlg.res[1], dlg.res[2])

        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)

# ---------------------------laplacien-------------------------------------------
class LaplacienWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "laplacien")

    def apply(self):
        if self.img_orig is None:
            return

        dlg = ThresholdDialog(self)
        if dlg.res is None:
            return

        seuil = dlg.res[1]

        L = laplacien(self.img_orig)
        zc = zero_crossing(L, seuil)

        self.img_result = zc
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)

# ----------------------laplacien de gausien ---------------------------------------------
class LoGWindow(OperationWindow):
    def __init__(self, master):
        super().__init__(master, "Laplacian de Gaussian")

        params = tk.Frame(self)
        params.pack()

        tk.Label(params, text="sigma:").grid(row=0, column=0)
        self.sigma = tk.DoubleVar(value=1.0)
        tk.Entry(params, textvariable=self.sigma, width=6).grid(row=0, column=1)

    def apply(self):
        if self.img_orig is None:
            return

        dlg = ThresholdDialog(self)
        if dlg.res is None:
            return

        seuil = dlg.res[1]
        sigma = self.sigma.get()

        gauss, lap, zc = LoG(self.img_orig, sigma, seuil)
        self.img_result = zc
        show_image_in_label(np_to_pil(self.img_result), self.canvas_res)



class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("detection des contours – TP3")
        self.geometry("560x360")

        tk.Label(self, text="TP3 : detection des Contours", font=("Arial", 14)).pack(pady=10)

        frame = tk.Frame(self)
        frame.pack()

        tk.Button(frame, text="filtre Sobel", width=24, command=lambda: SobelWindow(self)).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(frame, text="filtre Prewitt", width=24, command=lambda: PrewittWindow(self)).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(frame, text="filtre Roberts", width=24, command=lambda: RobertsWindow(self)).grid(row=1, column=0, padx=6, pady=6)
        tk.Button(frame, text="laplacien", width=24, command=lambda: LaplacienWindow(self)).grid(row=1, column=1, padx=6, pady=6)
        tk.Button(frame, text="lapdeGauss", width=24, command=lambda: LoGWindow(self)).grid(row=2, column=0, padx=6, pady=6)

        tk.Label(self, text="choisissez un filtre pour detecter les contours").pack(pady=10)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
