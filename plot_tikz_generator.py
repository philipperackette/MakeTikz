import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

##############################################
# Détection pôles rationnels + scission
##############################################
def find_rational_singularities(expr, x_sym, xmin, xmax):
    if expr.is_rational_function(x_sym):
        num, den = sp.fraction(expr)
        sols = sp.solve(sp.Eq(den, 0), x_sym, dict=True)
        poles = []
        for d in sols:
            val = d.get(x_sym, None)
            if val is not None and val.is_real:
                vf = float(val)
                if xmin < vf < xmax:
                    poles.append(vf)
        poles.sort()
        return poles
    return []

def make_subintervals(xmin, xmax, singularities, margin=1e-2):
    points = [xmin] + singularities + [xmax]
    intervals = []
    for i in range(len(points)-1):
        a, b = points[i], points[i+1]
        if (b-a) <= 2*margin:
            continue
        left  = a if i==0 else a+margin
        right = b if i==(len(points)-2) else b-margin
        if left<right:
            intervals.append((left,right))
    return intervals

##################################################
# Découpe piecewise "dans l'ordre" (le premier True prime)
##################################################
def piecewise_subfunctions_in_order(expr, x_sym, global_xmin, global_xmax):
    """
    Retourne une liste (sub_expr, a,b) pour un Piecewise(...) en ordre.
    Le premier morceau recouvre la zone où sa condition est vraie;
    la zone "couverte" est enlevée du domaine. 
    Si un morceau a cond_i=True, on prend tout le reste et on arrête.
    """
    uncovered = [(global_xmin, global_xmax)]
    sub_list = []

    for (expr_i, cond_i) in expr.args:
        if not uncovered:
            break
        cond_str = str(cond_i).strip()
        if cond_i is True or cond_str=="True":
            # prend tout le reste
            for (ua, ub) in uncovered:
                if ua<ub:
                    sub_list.append((expr_i, ua, ub))
            uncovered=[]
            break
        else:
            new_uncovered=[]
            for (ua, ub) in uncovered:
                if ua>=ub:
                    continue
                sub_dom= sp.Interval(ua, ub)
                sol_set= sp.solveset(cond_i, x_sym, domain=sub_dom)
                if sol_set.is_EmptySet:
                    new_uncovered.append((ua,ub))
                else:
                    covered_intervals=[]
                    if isinstance(sol_set, sp.Interval):
                        covered_intervals=[sol_set]
                    elif isinstance(sol_set, sp.Union):
                        for part in sol_set.args:
                            if isinstance(part, sp.Interval):
                                covered_intervals.append(part)
                    remain= sub_dom
                    for ci in covered_intervals:
                        remain= remain - ci
                    if remain.is_EmptySet:
                        pass
                    elif isinstance(remain, sp.Interval):
                        new_uncovered.append((float(remain.start), float(remain.end)))
                    elif isinstance(remain, sp.Union):
                        for part in remain.args:
                            if isinstance(part, sp.Interval):
                                new_uncovered.append((float(part.start), float(part.end)))
                    for ci in covered_intervals:
                        a_= float(ci.start)
                        b_= float(ci.end)
                        if a_< b_:
                            sub_list.append((expr_i, a_, b_))
            uncovered= new_uncovered

    return sub_list


##############################################
# generate_tikz_code : scinde piecewise + pôles
# Remplace "log(" par "ln(" en latex
##############################################
def generate_tikz_code(
    functions, curve_labels,
    styles, colors, line_widths,
    xmin, xmax, ymin, ymax,
    x_step, y_step,
    show_grid=True, show_ticks=True, show_tick_labels=True,
    hide_extremes=False,
    axis_label_x="x", axis_label_y="y",
    scale_ratio_x=1.0, scale_ratio_y=1.0,
    max_abs_y=100.0,
    num_samples=200,
    label_positions=None
):
    if label_positions is None:
        label_positions = {}

    # Conversion style python -> pgf
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

    # Ticks
    if show_ticks:
        if not hide_extremes:
            pgf_options.append(f"xtick distance={x_step}")
            pgf_options.append(f"ytick distance={y_step}")
        else:
            # on fabrique la liste de ticks en évitant xmin,xmax,ymin,ymax
            xticks=[]
            c= xmin + x_step
            while c < (xmax - 1e-9):
                xticks.append(round(c,5))
                c+= x_step
            if len(xticks)==0:
                pgf_options.append("xtick=\\empty")
            else:
                xs= ",".join(str(v) for v in xticks)
                pgf_options.append(f"xtick={{ {xs} }}")

            yticks=[]
            c= ymin + y_step
            while c < (ymax - 1e-9):
                yticks.append(round(c,5))
                c+= y_step
            if len(yticks)==0:
                pgf_options.append("ytick=\\empty")
            else:
                ys= ",".join(str(v) for v in yticks)
                pgf_options.append(f"ytick={{ {ys} }}")
    else:
        pgf_options.append("xtick=\\empty")
        pgf_options.append("ytick=\\empty")

    if not show_tick_labels:
        pgf_options.append("xticklabel=\\empty")
        pgf_options.append("yticklabel=\\empty")

    # Axes labels
    def ensure_math_mode(s):
        return s if (s.startswith('$') and s.endswith('$')) else f'${s}$'
    xlabel_proc= ensure_math_mode(axis_label_x)
    ylabel_proc= ensure_math_mode(axis_label_y)
    pgf_options.append(f"xlabel={xlabel_proc}")
    pgf_options.append(f"ylabel={ylabel_proc}")

    pgf_options.append("scale only axis")
    pgf_options.append(f"x={scale_ratio_x}cm")
    pgf_options.append(f"y={scale_ratio_y}cm")

    restrict_str= f"restrict y to domain=-{max_abs_y}:{max_abs_y}"

    tikz=[]
    tikz.append(r"\begin{tikzpicture}")
    tikz.append("  \\begin{axis}[%")
    opts_str= ",\n    ".join(pgf_options)
    tikz.append(f"    {opts_str}")
    tikz.append("  ]")

    x_sym= sp.Symbol('x', real=True)
    nb_fun= min(len(functions),2)
    for i in range(nb_fun):
        fstr= functions[i]
        lbl= curve_labels[i] if i<len(curve_labels) else f"$C_{{{i+1}}}$"
        stp= styles[i] if i<len(styles) else "solid"
        col= colors[i] if i<len(colors) else "black"
        lw= line_widths[i] if i<len(line_widths) else 1.0
        style_pgf= style_map.get(stp,"solid")

        # parse
        local_dict= {
            "x": x_sym,
            "sin":sp.sin,"cos":sp.cos,"tan":sp.tan,
            "csc":sp.csc,"sec":sp.sec,"cot":sp.cot,
            "sinh":sp.sinh,"cosh":sp.cosh,"tanh":sp.tanh,
            "asinh":sp.asinh,"acosh":sp.acosh,"atanh":sp.atanh,
            "exp":sp.exp,"log":sp.log
        }
        try:
            expr= sp.parse_expr(fstr, local_dict)
        except:
            expr= None
        if expr is None:
            continue

        sub_exprs=[]
        if isinstance(expr, sp.Piecewise):
            sub_list= piecewise_subfunctions_in_order(expr, x_sym, xmin, xmax)
            sub_exprs.extend(sub_list) # => (ex, a,b)
        else:
            sub_exprs.append((expr, xmin, xmax))

        for (subE, subA, subB) in sub_exprs:
            if subA>=subB:
                continue
            # scinde sur pôles
            if subE.is_rational_function(x_sym):
                poles= find_rational_singularities(subE, x_sym, subA, subB)
            else:
                poles=[]
            intervals= make_subintervals(subA, subB, poles, margin=1e-2)

            subE_tex= str(subE).replace("**","^")
            # => On remplace "log(" par "ln(" pour pgf
            subE_tex= subE_tex.replace("log(","ln(")

            for (L,R) in intervals:
                cstyle= (
                    f"samples={num_samples}, unbounded coords=jump, "
                    f"line width={lw}pt, color={col}, {style_pgf}, "
                    f"{restrict_str}, domain={L}:{R}"
                )
                tikz.append(f"    \\addplot[{cstyle}]{{{subE_tex}}};")

        if i in label_positions:
            (xlbl, ylbl)= label_positions[i]
            anchor= "west" if i%2==0 else "east"
            tikz.append(
                f"    \\draw[color={col}] (axis cs:{xlbl},{ylbl}) "
                f"node[anchor={anchor}]{{\\color{{{col}}}{lbl}}};"
            )

    tikz.append("  \\end{axis}")
    tikz.append(r"\end{tikzpicture}")

    return "\n".join(tikz)

######################################################
class PlotTikzApp:
    def __init__(self, master):
        self.master= master
        master.title("Générateur TikZ (2 courbes, log->ln, piecewise)")

        self.mainframe= ttk.Frame(master, padding="10 10 10 10")
        self.mainframe.grid()

        # 2 fonctions
        self.func1_var= tk.StringVar(value="Piecewise((x+1, x<0),(x-1, True))")
        self.func2_var= tk.StringVar(value="log(x)")
        self.label1_var= tk.StringVar(value="$f$")
        self.label2_var= tk.StringVar(value="$g$")

        # style/couleur
        self.style1_var= tk.StringVar(value="solid")
        self.color1_var= tk.StringVar(value="black")
        self.linewidth1_var= tk.DoubleVar(value=1.5)

        self.style2_var= tk.StringVar(value="dashed")
        self.color2_var= tk.StringVar(value="gray")
        self.linewidth2_var= tk.DoubleVar(value=1.5)

        self.xmin_var= tk.StringVar(value="-5")
        self.xmax_var= tk.StringVar(value="5")
        self.ymin_var= tk.StringVar(value="-5")
        self.ymax_var= tk.StringVar(value="5")

        self.xstep_var= tk.StringVar(value="1.0")
        self.ystep_var= tk.StringVar(value="1.0")
        self.xlabel_var= tk.StringVar(value="x")
        self.ylabel_var= tk.StringVar(value="y")

        self.show_grid_var= tk.BooleanVar(value=True)
        self.show_ticks_var= tk.BooleanVar(value=True)
        self.show_tick_labels_var= tk.BooleanVar(value=True)

        # On utilise la même variable pour le check:
        self.hide_extremes_var= tk.BooleanVar(value=False)

        self.scale_ratio_x_var= tk.DoubleVar(value=1.0)
        self.scale_ratio_y_var= tk.DoubleVar(value=1.0)
        self.max_abs_y_var= tk.DoubleVar(value=100.0)
        self.num_samples_var= tk.IntVar(value=200)

        self.label_positions={}
        self.dragging_text=None
        self.label_texts=[]

        row=0
        # Fct1
        ttk.Label(self.mainframe, text="Fonction 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=35, textvariable=self.func1_var).grid(row=row, column=1, sticky=(tk.W, tk.E))
        row+=1
        ttk.Label(self.mainframe, text="Label 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=10, textvariable=self.label1_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Style 1 :").grid(row=row, column=0, sticky=tk.W)
        style_opts=["solid","dashed","dotted","dashdot"]
        ttk.Combobox(self.mainframe, values=style_opts, textvariable=self.style1_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Couleur 1 :").grid(row=row, column=0, sticky=tk.W)
        color_opts=["black","red","blue","gray","green","orange"]
        ttk.Combobox(self.mainframe, values=color_opts, textvariable=self.color1_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Linewidth 1 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=5, textvariable=self.linewidth1_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        # fct2
        ttk.Label(self.mainframe, text="Fonction 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=35, textvariable=self.func2_var).grid(row=row, column=1, sticky=(tk.W, tk.E))
        row+=1
        ttk.Label(self.mainframe, text="Label 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=10, textvariable=self.label2_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Style 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=style_opts, textvariable=self.style2_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Couleur 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(self.mainframe, values=color_opts, textvariable=self.color2_var, width=8).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Linewidth 2 :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=5, textvariable=self.linewidth2_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        # domain
        ttk.Label(self.mainframe, text="xmin :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xmin_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="xmax :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xmax_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="ymin :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ymin_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="ymax :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ymax_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="x step :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.xstep_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="y step :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.ystep_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Label axe X :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=8, textvariable=self.xlabel_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="Label axe Y :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=8, textvariable=self.ylabel_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        # Checkbutton => même variable hide_extremes_var
        ttk.Checkbutton(
            self.mainframe,
            text="Ne pas afficher extremes axes",
            variable=self.hide_extremes_var
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Échelle X (cm) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.scale_ratio_x_var).grid(row=row, column=1, sticky=tk.W)
        row+=1
        ttk.Label(self.mainframe, text="Échelle Y (cm) :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.scale_ratio_y_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="|y| max :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.max_abs_y_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        ttk.Label(self.mainframe, text="Nb. échantillons :").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(self.mainframe, width=6, textvariable=self.num_samples_var).grid(row=row, column=1, sticky=tk.W)
        row+=1

        self.grid_check= ttk.Checkbutton(self.mainframe, text="Afficher la grille", variable=self.show_grid_var)
        self.grid_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row+=1

        self.ticks_check= ttk.Checkbutton(self.mainframe, text="Afficher les graduations", variable=self.show_ticks_var)
        self.ticks_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row+=1

        self.ticklabels_check= ttk.Checkbutton(self.mainframe, text="Afficher les labels de ticks",
                                               variable=self.show_tick_labels_var)
        self.ticklabels_check.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row+=1

        self.plot_button= ttk.Button(self.mainframe, text="Tracer / Mettre à jour", command=self.update_plot)
        self.plot_button.grid(row=row, column=0, columnspan=2, pady=5)
        row+=1

        self.tikz_button= ttk.Button(self.mainframe, text="Générer le code TikZ", command=self.generate_tikz)
        self.tikz_button.grid(row=row, column=0, columnspan=2, pady=5)
        row+=1

        # figure
        self.fig, self.ax= plt.subplots(figsize=(5,4))
        self.canvas= FigureCanvasTkAgg(self.fig, master=self.mainframe)
        self.canvas_widget= self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=2, rowspan=90, padx=10, pady=5, sticky=(tk.N, tk.S))

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def update_plot(self):
        """Aperçu Matplotlib => simple lambdify."""
        try:
            f1= self.func1_var.get().strip()
            f2= self.func2_var.get().strip()
            l1= self.label1_var.get().strip() or "Courbe1"
            l2= self.label2_var.get().strip() or "Courbe2"

            sty1= self.style1_var.get()
            col1= self.color1_var.get()
            lw1= self.linewidth1_var.get()

            sty2= self.style2_var.get()
            col2= self.color2_var.get()
            lw2= self.linewidth2_var.get()

            xmin= float(self.xmin_var.get())
            xmax= float(self.xmax_var.get())
            ymin= float(self.ymin_var.get())
            ymax= float(self.ymax_var.get())
            ns= self.num_samples_var.get()

            self.ax.clear()
            self.label_texts=[]

            x_sym= sp.Symbol('x', real=True)
            local_dict= {
                "x": x_sym,
                "sin":sp.sin,"cos":sp.cos,"tan":sp.tan,
                "csc":sp.csc,"sec":sp.sec,"cot":sp.cot,
                "sinh":sp.sinh,"cosh":sp.cosh,"tanh":sp.tanh,
                "asinh":sp.asinh,"acosh":sp.acosh,"atanh":sp.atanh,
                "exp":sp.exp,"log":sp.log
            }

            style_mpl_map={
                "solid": "solid",
                "dashed":"dashed",
                "dotted":"dotted",
                "dashdot":"dashdot"
            }

            # 2 courbes
            fstrs=[f1,f2]
            labs=[l1,l2]
            stys=[sty1, sty2]
            cols=[col1, col2]
            lwlist=[lw1, lw2]

            xvals= np.linspace(xmin, xmax, ns)

            for i in range(2):
                if not fstrs[i]:
                    continue
                expr= sp.parse_expr(fstrs[i], local_dict)
                f= sp.lambdify(x_sym, expr,"numpy")
                yvals= f(xvals)

                st_= style_mpl_map.get(stys[i],"solid")
                c_ = cols[i] if cols[i] else "black"
                l_ = lwlist[i]

                self.ax.plot(xvals, yvals, linestyle=st_, color=c_, linewidth=l_)

                pos_frac= 0.8 if i==0 else 0.2
                idx= int(pos_frac*len(xvals))
                idx= max(0, min(idx, len(xvals)-1))
                xlab= xvals[idx]
                ylab= yvals[idx]
                offset= 0.2 if i==0 else -0.2

                txt_obj= self.ax.text(xlab, ylab+offset, labs[i],
                                      color=c_, fontsize=9, picker=True)
                txt_obj._curve_index= i
                self.label_texts.append(txt_obj)

            self.ax.set_xlim(xmin,xmax)
            self.ax.set_ylim(ymin,ymax)
            self.ax.grid(self.show_grid_var.get())

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def on_pick(self, event):
        if isinstance(event.artist, matplotlib.text.Text):
            self.dragging_text= event.artist

    def on_motion(self, event):
        if self.dragging_text and event.inaxes==self.ax:
            self.dragging_text.set_position((event.xdata, event.ydata))
            self.canvas.draw()

    def on_release(self, event):
        if self.dragging_text and event.inaxes==self.ax:
            i= getattr(self.dragging_text, '_curve_index', None)
            if i is not None:
                self.label_positions[i]= (event.xdata, event.ydata)
            self.dragging_text= None
            self.canvas.draw()

    def generate_tikz(self):
        """
        Génére code TikZ => scinde piecewise/pôles, 
        remplace "log(" par "ln(",
        lit hide_extremes_var pour omettre xmin,xmax etc.
        """
        try:
            f1= self.func1_var.get().strip()
            f2= self.func2_var.get().strip()
            l1= self.label1_var.get().strip() or "$C_1$"
            l2= self.label2_var.get().strip() or "$C_2$"

            sty1= self.style1_var.get()
            col1= self.color1_var.get()
            lw1= float(self.linewidth1_var.get())

            sty2= self.style2_var.get()
            col2= self.color2_var.get()
            lw2= float(self.linewidth2_var.get())

            xmin= float(self.xmin_var.get())
            xmax= float(self.xmax_var.get())
            ymin= float(self.ymin_var.get())
            ymax= float(self.ymax_var.get())
            xstep= float(self.xstep_var.get())
            ystep= float(self.ystep_var.get())
            xlabel= self.xlabel_var.get()
            ylabel= self.ylabel_var.get()

            # On récupère la variable "ne pas afficher extremes axes"
            hide_ext= self.hide_extremes_var.get()

            sg= self.show_grid_var.get()
            st= self.show_ticks_var.get()
            stl= self.show_tick_labels_var.get()

            sx= float(self.scale_ratio_x_var.get())
            sy= float(self.scale_ratio_y_var.get())
            maxy= float(self.max_abs_y_var.get())
            ns=  self.num_samples_var.get()

            funs=[]
            labs=[]
            stys_=[]
            cols_=[]
            wids=[]

            if f1:
                funs.append(f1)
                labs.append(l1)
                stys_.append(sty1)
                cols_.append(col1)
                wids.append(lw1)
            if f2:
                funs.append(f2)
                labs.append(l2)
                stys_.append(sty2)
                cols_.append(col2)
                wids.append(lw2)

            tikz_code= generate_tikz_code(
                functions=funs,
                curve_labels=labs,
                styles=stys_,
                colors=cols_,
                line_widths=wids,
                xmin=xmin,xmax=xmax,
                ymin=ymin,ymax=ymax,
                x_step=xstep,y_step=ystep,
                show_grid=sg,
                show_ticks=st,
                show_tick_labels=stl,
                hide_extremes=hide_ext,
                axis_label_x=xlabel,
                axis_label_y=ylabel,
                scale_ratio_x=sx,
                scale_ratio_y=sy,
                max_abs_y=maxy,
                num_samples=ns,
                label_positions=self.label_positions
            )

            cwin= tk.Toplevel(self.master)
            cwin.title("Code TikZ")

            txt= tk.Text(cwin, wrap="none", width=80, height=25)
            txt.insert("1.0", tikz_code)
            txt.pack(fill="both", expand=True)

            def cb_copy():
                self.master.clipboard_clear()
                self.master.clipboard_append(tikz_code)
                messagebox.showinfo("Copié", "Code TikZ copié dans le presse-papiers.")

            ttk.Button(cwin, text="Copier", command=cb_copy).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

def main():
    root= tk.Tk()
    app= PlotTikzApp(root)
    root.mainloop()

if __name__=="__main__":
    main()