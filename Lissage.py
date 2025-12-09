import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optionnel : spline de lissage (SciPy)
try:
    from scipy.interpolate import UnivariateSpline
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


###########################################################
# Frame scrollable pour la colonne de gauche
###########################################################
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self.canvas)
        self.interior_id = self.canvas.create_window(
            (0, 0), window=self.interior, anchor="nw"
        )

        def _configure_interior(event):
            size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
            self.canvas.config(scrollregion=(0, 0, size[0], size[1]))
            if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
                self.canvas.config(width=self.interior.winfo_reqwidth())

        self.interior.bind("<Configure>", _configure_interior)

        def _configure_canvas(event):
            if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
                self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())

        self.canvas.bind("<Configure>", _configure_canvas)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)


###########################################################
# Utilitaires : parsing des points, Hermite, lissage
###########################################################
def parse_points_string(text):
    """
    Parse un texte de lignes 'x y pente' ou 'x y'.
    Retourne trois arrays numpy (x, y, m).
    Si pente absente, elle est estimée (différences finies).
    """
    xs, ys, ms = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for c in ",;":
            line = line.replace(c, " ")
        parts = line.split()
        if len(parts) not in (2, 3):
            raise ValueError(f"Ligne invalide (attendu 2 ou 3 nombres) : '{line}'")
        x = float(parts[0])
        y = float(parts[1])
        xs.append(x)
        ys.append(y)
        if len(parts) == 3:
            ms.append(float(parts[2]))

    if len(xs) < 2:
        raise ValueError("Il faut au moins 2 points pour tracer une courbe.")

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Tri par abscisse
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # Pentes : soit fournies, soit estimées
    if len(ms) == 0:
        ms = estimate_slopes(xs, ys)
    else:
        ms = np.array(ms, dtype=float)[order]

    return xs, ys, ms


def estimate_slopes(xs, ys):
    """
    Estime les pentes par différences finies centrées.
    """
    n = len(xs)
    ms = np.zeros(n, dtype=float)
    if n == 1:
        ms[0] = 0.0
        return ms

    ms[0] = (ys[1] - ys[0]) / (xs[1] - xs[0])
    ms[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])

    for i in range(1, n - 1):
        dx1 = xs[i] - xs[i - 1]
        dx2 = xs[i + 1] - xs[i]
        ms[i] = (((ys[i] - ys[i - 1]) / dx1) * dx2 +
                 ((ys[i + 1] - ys[i]) / dx2) * dx1) / (dx1 + dx2)
    return ms


def hermite_eval(xgrid, xs, ys, ms):
    """
    Évalue la spline de Hermite pièce par pièce sur xgrid.
    xs, ys, ms : arrays 1D triés.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ms = np.asarray(ms, dtype=float)
    n = len(xs)

    ygrid = np.full_like(xgrid, np.nan, dtype=float)
    mask = (xgrid >= xs[0]) & (xgrid <= xs[-1])
    xv = xgrid[mask]
    if xv.size == 0:
        return ygrid

    idx = np.searchsorted(xs, xv) - 1
    idx[idx < 0] = 0
    idx[idx > n - 2] = n - 2

    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    m0 = ms[idx]
    m1 = ms[idx + 1]
    h = x1 - x0
    t = (xv - x0) / h

    H00 = 2 * t**3 - 3 * t**2 + 1
    H10 = t**3 - 2 * t**2 + t
    H01 = -2 * t**3 + 3 * t**2
    H11 = t**3 - t**2

    yv = H00 * y0 + H10 * h * m0 + H01 * y1 + H11 * h * m1
    ygrid[mask] = yv
    return ygrid


def build_curve_samples(xs, ys, ms, method, xmin, xmax, ns, smooth_param=0.0):
    """
    Construit un échantillonnage (xvals, yvals) pour une courbe
    à partir des points (xs, ys, ms) et de la méthode :
      - 'hermite' : spline d'Hermite
      - 'lissage' : spline de lissage (UnivariateSpline)
    """
    xvals = np.linspace(xmin, xmax, ns)

    if method == "hermite":
        yvals = hermite_eval(xvals, xs, ys, ms)
    elif method == "lissage":
        if not HAVE_SCIPY:
            raise RuntimeError(
                "SciPy (scipy.interpolate.UnivariateSpline) est requis pour la spline de lissage."
            )
        s = float(smooth_param)
        spline = UnivariateSpline(xs, ys, s=s)
        yvals = spline(xvals)
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    return xvals, yvals


###########################################################
# Génération du TikZ à partir des échantillons
###########################################################
def generate_tikz_code_from_samples(
    curves_samples,
    curve_labels,
    styles, colors, line_widths,
    xmin, xmax, ymin, ymax,
    x_step, y_step,
    show_grid=True, show_ticks=True, show_tick_labels=True,
    hide_extremes=False,
    axis_label_x="x", axis_label_y="y",
    scale_ratio_x=1.0, scale_ratio_y=1.0,
    max_abs_y=100.0,
    label_positions=None
):
    if label_positions is None:
        label_positions = {}

    style_map = {
        "solid":   "solid",
        "dashed":  "dash pattern=on 5pt off 5pt",
        "dotted":  "dash pattern=on 1pt off 3pt",
        "dashdot": "dash pattern=on 4pt off 2pt on 1pt off 2pt",
    }

    pgf_options = [
        f"xmin={xmin}",
        f"xmax={xmax}",
        f"ymin={ymin}",
        f"ymax={ymax}",
        "axis lines=middle",
        "trig format=rad"
    ]
    pgf_options.append("grid=major" if show_grid else "grid=none")

    if show_ticks:
        if not hide_extremes:
            pgf_options.append(f"xtick distance={x_step}")
            pgf_options.append(f"ytick distance={y_step}")
        else:
            xticks = []
            c = xmin + x_step
            while c < (xmax - 1e-9):
                xticks.append(round(c, 5))
                c += x_step
            if len(xticks) == 0:
                pgf_options.append("xtick=\\empty")
            else:
                xs_str = ",".join(str(v) for v in xticks)
                pgf_options.append(f"xtick={{ {xs_str} }}")

            yticks = []
            c = ymin + y_step
            while c < (ymax - 1e-9):
                yticks.append(round(c, 5))
                c += y_step
            if len(yticks) == 0:
                pgf_options.append("ytick=\\empty")
            else:
                ys_str = ",".join(str(v) for v in yticks)
                pgf_options.append(f"ytick={{ {ys_str} }}")
    else:
        pgf_options.append("xtick=\\empty")
        pgf_options.append("ytick=\\empty")

    if not show_tick_labels:
        pgf_options.append("xticklabel=\\empty")
        pgf_options.append("yticklabel=\\empty")

    def ensure_math_mode(s):
        return s if (s.startswith('$') and s.endswith('$')) else f'${s}$'

    xlabel_proc = ensure_math_mode(axis_label_x)
    ylabel_proc = ensure_math_mode(axis_label_y)
    pgf_options.append(f"xlabel={xlabel_proc}")
    pgf_options.append(f"ylabel={ylabel_proc}")

    pgf_options.append("scale only axis")
    pgf_options.append(f"x={scale_ratio_x}cm")
    pgf_options.append(f"y={scale_ratio_y}cm")

    restrict_str = f"restrict y to domain=-{max_abs_y}:{max_abs_y}"

    tikz = []
    tikz.append(r"\begin{tikzpicture}")
    tikz.append("  \\begin{axis}[%")
    opts_str = ",\n    ".join(pgf_options)
    tikz.append(f"    {opts_str}")
    tikz.append("  ]")

    for i, (xvals, yvals) in enumerate(curves_samples):
        lbl = curve_labels[i] if i < len(curve_labels) else f"$C_{{{i+1}}}$"
        stp = styles[i] if i < len(styles) else "solid"
        col = colors[i] if i < len(colors) else "black"
        lw = line_widths[i] if i < len(line_widths) else 1.0
        style_pgf = style_map.get(stp, "solid")

        cstyle = (
            f"line width={lw}pt, color={col}, {style_pgf}, "
            f"unbounded coords=jump, {restrict_str}"
        )

        coord_lines = []
        for xv, yv in zip(xvals, yvals):
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            coord_lines.append(f"      ({xv:.6f},{yv:.6f})")

        if coord_lines:
            tikz.append(f"    \\addplot[{cstyle}]")
            tikz.append("    coordinates {")
            tikz.extend(coord_lines)
            tikz.append("    };")

        if i in label_positions:
            (xlbl, ylbl) = label_positions[i]
            anchor = "west" if i % 2 == 0 else "east"
            tikz.append(
                f"    \\draw[color={col}] (axis cs:{xlbl},{ylbl}) "
                f"node[anchor={anchor}]{{\\color{{{col}}}{lbl}}};"
            )

    tikz.append("  \\end{axis}")
    tikz.append(r"\end{tikzpicture}")

    return "\n".join(tikz)


###########################################################
# Application Tkinter
###########################################################
class PlotTikzApp:
    def __init__(self, master):
        self.master = master
        master.title("Générateur TikZ (2 courbes à partir de points + Hermite / lissage)")

        # Layout général : gauche = contrôles (scrollable), droite = figure
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=0)
        master.columnconfigure(1, weight=1)

        # Colonne gauche : frame scrollable
        self.scroll = ScrollableFrame(master)
        self.scroll.grid(row=0, column=0, sticky="nsw")
        self.mainframe = self.scroll.interior

        # Colonne droite : frame pour Matplotlib
        plot_frame = ttk.Frame(master)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Variables
        self.label1_var = tk.StringVar(value="$f$")
        self.label2_var = tk.StringVar(value="$g$")

        self.style1_var = tk.StringVar(value="solid")
        self.color1_var = tk.StringVar(value="black")
        self.linewidth1_var = tk.DoubleVar(value=1.5)

        self.style2_var = tk.StringVar(value="dashed")
        self.color2_var = tk.StringVar(value="gray")
        self.linewidth2_var = tk.DoubleVar(value=1.5)

        self.xmin_var = tk.StringVar(value="-5")
        self.xmax_var = tk.StringVar(value="5")
        self.ymin_var = tk.StringVar(value="-5")
        self.ymax_var = tk.StringVar(value="5")

        self.xstep_var = tk.StringVar(value="1.0")
        self.ystep_var = tk.StringVar(value="1.0")
        self.xlabel_var = tk.StringVar(value="x")
        self.ylabel_var = tk.StringVar(value="y")

        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_ticks_var = tk.BooleanVar(value=True)
        self.show_tick_labels_var = tk.BooleanVar(value=True)
        self.hide_extremes_var = tk.BooleanVar(value=False)

        self.scale_ratio_x_var = tk.DoubleVar(value=1.0)
        self.scale_ratio_y_var = tk.DoubleVar(value=1.0)
        self.max_abs_y_var = tk.DoubleVar(value=100.0)
        self.num_samples_var = tk.IntVar(value=200)

        self.method_var = tk.StringVar(value="hermite")
        self.smooth_param_var = tk.DoubleVar(value=0.0)

        self.label_positions = {}
        self.dragging_text = None
        self.label_texts = []

        ####################################################
        # Widgets dans la colonne gauche (scrollable)
        ####################################################
        row = 0

        # Courbe 1
        ttk.Label(
            self.mainframe,
            text="Courbe 1 : points (x y pente) :"
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.points1_text = tk.Text(self.mainframe, width=35, height=5)
        self.points1_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.points1_text.insert("1.0", "-2  1  0\n-1  0  1\n0  0  0\n1  1  0\n2  0 -1")
        row += 1

        ttk.Label(self.mainframe, text="Label 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=10, textvariable=self.label1_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        style_opts = ["solid", "dashed", "dotted", "dashdot"]
        color_opts = ["black", "red", "blue", "gray", "green", "orange"]

        ttk.Label(self.mainframe, text="Style 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=style_opts, textvariable=self.style1_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Couleur 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=color_opts, textvariable=self.color1_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Linewidth 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=5, textvariable=self.linewidth1_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # Courbe 2
        ttk.Label(
            self.mainframe,
            text="Courbe 2 : points (x y pente) :"
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        self.points2_text = tk.Text(self.mainframe, width=35, height=5)
        self.points2_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.points2_text.insert("1.0", "-2 -1  0\n-1  0 -1\n0  0  0\n1 -1  0\n2  0  1")
        row += 1

        ttk.Label(self.mainframe, text="Label 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=10, textvariable=self.label2_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Style 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=style_opts, textvariable=self.style2_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Couleur 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=color_opts, textvariable=self.color2_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Linewidth 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=5, textvariable=self.linewidth2_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # Domaine
        ttk.Label(self.mainframe, text="xmin :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xmin_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="xmax :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xmax_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="ymin :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ymin_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="ymax :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ymax_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="x step :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xstep_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="y step :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ystep_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Label axe X :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=8, textvariable=self.xlabel_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="Label axe Y :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=8, textvariable=self.ylabel_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Checkbutton(
            self.mainframe,
            text="Ne pas afficher extremes axes",
            variable=self.hide_extremes_var
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Échelle X (cm) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.scale_ratio_x_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        ttk.Label(self.mainframe, text="Échelle Y (cm) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.scale_ratio_y_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="|y| max (TikZ) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.max_abs_y_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Nb. échantillons :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.num_samples_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Méthode :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(
            self.mainframe,
            values=["hermite", "lissage"],
            textvariable=self.method_var,
            width=10,
            state="readonly"
        ).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.mainframe, text="Paramètre de lissage s (0=interp) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=8, textvariable=self.smooth_param_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        self.grid_check = ttk.Checkbutton(self.mainframe, text="Afficher la grille", variable=self.show_grid_var)
        self.grid_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        self.ticks_check = ttk.Checkbutton(self.mainframe, text="Afficher les graduations", variable=self.show_ticks_var)
        self.ticks_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        self.ticklabels_check = ttk.Checkbutton(
            self.mainframe,
            text="Afficher les labels de ticks",
            variable=self.show_tick_labels_var
        )
        self.ticklabels_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        self.plot_button = ttk.Button(self.mainframe, text="Tracer / Mettre à jour", command=self.update_plot)
        self.plot_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        self.tikz_button = ttk.Button(self.mainframe, text="Générer le code TikZ", command=self.generate_tikz)
        self.tikz_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Connexions événements Matplotlib
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    ####################################################
    # Logique
    ####################################################
    def update_plot(self):
        """Aperçu Matplotlib à partir des points + spline Hermite ou lissage."""
        try:
            method = self.method_var.get()
            smooth_param = self.smooth_param_var.get()

            l1 = self.label1_var.get().strip() or "Courbe1"
            l2 = self.label2_var.get().strip() or "Courbe2"

            sty1 = self.style1_var.get()
            col1 = self.color1_var.get()
            lw1 = self.linewidth1_var.get()

            sty2 = self.style2_var.get()
            col2 = self.color2_var.get()
            lw2 = self.linewidth2_var.get()

            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            ymin = float(self.ymin_var.get())
            ymax = float(self.ymax_var.get())
            ns = self.num_samples_var.get()

            curves_data = []

            txt1 = self.points1_text.get("1.0", "end").strip()
            if txt1:
                xs1, ys1, ms1 = parse_points_string(txt1)
                curves_data.append((xs1, ys1, ms1, l1, sty1, col1, lw1))

            txt2 = self.points2_text.get("1.0", "end").strip()
            if txt2:
                xs2, ys2, ms2 = parse_points_string(txt2)
                curves_data.append((xs2, ys2, ms2, l2, sty2, col2, lw2))

            self.ax.clear()
            self.label_texts = []

            style_mpl_map = {
                "solid": "solid",
                "dashed": "dashed",
                "dotted": "dotted",
                "dashdot": "dashdot"
            }

            if not hasattr(self, 'label_positions'):
                self.label_positions = {}

            for i, (xs, ys, ms, lab, sty, col, lw) in enumerate(curves_data):
                xvals, yvals = build_curve_samples(
                    xs, ys, ms,
                    method=method,
                    xmin=xmin, xmax=xmax,
                    ns=ns,
                    smooth_param=smooth_param
                )

                st_ = style_mpl_map.get(sty, "solid")
                c_ = col if col else "black"
                l_ = lw

                self.ax.plot(xvals, yvals, linestyle=st_, color=c_, linewidth=l_)

                if i in self.label_positions:
                    xlab, ylab = self.label_positions[i]
                    vanchor = "center"
                else:
                    finite_mask = np.isfinite(yvals)
                    inside_mask = finite_mask & (yvals >= ymin) & (yvals <= ymax)

                    if np.any(inside_mask):
                        valid_idx = np.where(inside_mask)[0]
                    elif np.any(finite_mask):
                        valid_idx = np.where(finite_mask)[0]
                    else:
                        xlab = (xmin + xmax) / 2
                        ylab = (ymin + ymax) / 2
                        vanchor = "center"
                        valid_idx = None

                    if valid_idx is not None and len(valid_idx) > 0:
                        frac = 0.8 if i == 0 else 0.2
                        idx_in_valid = int(frac * (len(valid_idx) - 1))
                        idx = valid_idx[idx_in_valid]

                        xlab = xvals[idx]
                        ylab = yvals[idx]

                        dx = 0.02 * (xmax - xmin)
                        dy = 0.05 * (ymax - ymin)

                        if i == 0:
                            ylab += dy
                            vanchor = "bottom"
                        else:
                            ylab -= dy
                            vanchor = "top"

                        xlab += dx

                        xmarg = 0.02 * (xmax - xmin)
                        ymarg = 0.02 * (ymax - ymin)
                        xlab = min(max(xlab, xmin + xmarg), xmax - xmarg)
                        ylab = min(max(ylab, ymin + ymarg), ymax - ymarg)

                    self.label_positions[i] = (xlab, ylab)

                txt_obj = self.ax.text(
                    xlab, ylab, lab,
                    color=c_,
                    fontsize=9,
                    picker=True,
                    ha='center',
                    va=vanchor
                )
                txt_obj._curve_index = i
                self.label_texts.append(txt_obj)

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.grid(self.show_grid_var.get())

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def on_pick(self, event):
        if isinstance(event.artist, matplotlib.text.Text):
            self.dragging_text = event.artist

    def on_motion(self, event):
        if self.dragging_text and event.inaxes == self.ax:
            self.dragging_text.set_position((event.xdata, event.ydata))
            self.canvas.draw()

    def on_release(self, event):
        if self.dragging_text and event.inaxes == self.ax:
            i = getattr(self.dragging_text, '_curve_index', None)
            if i is not None:
                self.label_positions[i] = (event.xdata, event.ydata)
            self.dragging_text = None
            self.canvas.draw()

    def generate_tikz(self):
        """Génère le code TikZ à partir des points et des splines."""
        try:
            method = self.method_var.get()
            smooth_param = self.smooth_param_var.get()

            l1 = self.label1_var.get().strip() or "$C_1$"
            l2 = self.label2_var.get().strip() or "$C_2$"

            sty1 = self.style1_var.get()
            col1 = self.color1_var.get()
            lw1 = float(self.linewidth1_var.get())

            sty2 = self.style2_var.get()
            col2 = self.color2_var.get()
            lw2 = float(self.linewidth2_var.get())

            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            ymin = float(self.ymin_var.get())
            ymax = float(self.ymax_var.get())
            xstep = float(self.xstep_var.get())
            ystep = float(self.ystep_var.get())
            xlabel = self.xlabel_var.get()
            ylabel = self.ylabel_var.get()

            hide_ext = self.hide_extremes_var.get()
            sg = self.show_grid_var.get()
            st = self.show_ticks_var.get()
            stl = self.show_tick_labels_var.get()

            sx = float(self.scale_ratio_x_var.get())
            sy = float(self.scale_ratio_y_var.get())
            maxy = float(self.max_abs_y_var.get())
            ns = self.num_samples_var.get()

            curves_samples = []
            labs = []
            stys_ = []
            cols_ = []
            wids = []

            txt1 = self.points1_text.get("1.0", "end").strip()
            if txt1:
                xs1, ys1, ms1 = parse_points_string(txt1)
                xvals1, yvals1 = build_curve_samples(
                    xs1, ys1, ms1,
                    method=method,
                    xmin=xmin, xmax=xmax,
                    ns=ns,
                    smooth_param=smooth_param
                )
                curves_samples.append((xvals1, yvals1))
                labs.append(l1)
                stys_.append(sty1)
                cols_.append(col1)
                wids.append(lw1)

            txt2 = self.points2_text.get("1.0", "end").strip()
            if txt2:
                xs2, ys2, ms2 = parse_points_string(txt2)
                xvals2, yvals2 = build_curve_samples(
                    xs2, ys2, ms2,
                    method=method,
                    xmin=xmin, xmax=xmax,
                    ns=ns,
                    smooth_param=smooth_param
                )
                curves_samples.append((xvals2, yvals2))
                labs.append(l2)
                stys_.append(sty2)
                cols_.append(col2)
                wids.append(lw2)

            if not curves_samples:
                raise ValueError("Aucune courbe définie (pas de points).")

            tikz_code = generate_tikz_code_from_samples(
                curves_samples=curves_samples,
                curve_labels=labs,
                styles=stys_,
                colors=cols_,
                line_widths=wids,
                xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax,
                x_step=xstep, y_step=ystep,
                show_grid=sg,
                show_ticks=st,
                show_tick_labels=stl,
                hide_extremes=hide_ext,
                axis_label_x=xlabel,
                axis_label_y=ylabel,
                scale_ratio_x=sx,
                scale_ratio_y=sy,
                max_abs_y=maxy,
                label_positions=self.label_positions
            )

            cwin = tk.Toplevel(self.master)
            cwin.title("Code TikZ")

            txt_widget = tk.Text(cwin, wrap="none", width=80, height=25)
            txt_widget.insert("1.0", tikz_code)
            txt_widget.pack(fill="both", expand=True)

            def cb_copy():
                self.master.clipboard_clear()
                self.master.clipboard_append(tikz_code)
                messagebox.showinfo("Copié", "Code TikZ copié dans le presse-papiers.")

            ttk.Button(cwin, text="Copier", command=cb_copy).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Erreur", str(e))


def main():
    root = tk.Tk()
    app = PlotTikzApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()