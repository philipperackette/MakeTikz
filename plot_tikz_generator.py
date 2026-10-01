import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Le moteur (lecture des fonctions, traduction pgfplots, aperçu) est partagé
# avec la version web : hf_space/make_tikz_engine.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))
from make_tikz_engine import check_pgf, generate_tikz_code, parse_function, sample_function  # noqa: E402

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
        """Aperçu Matplotlib => simple lambdify avec gestion optimale des labels."""
        try:
            # Récupération des paramètres
            f1 = self.func1_var.get().strip()
            f2 = self.func2_var.get().strip()
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
            
            # Réinitialisation
            self.ax.clear()
            self.label_texts = []
            
            # Initialisation du dictionnaire de positions seulement s'il n'existe pas
            if not hasattr(self, 'label_positions'):
                self.label_positions = {}
            
            style_mpl_map = {
                "solid": "solid",
                "dashed": "dashed",
                "dotted": "dotted",
                "dashdot": "dashdot"
            }
            
            # Préparation des données pour 2 courbes
            fstrs = [f1, f2]
            labs = [l1, l2]
            stys = [sty1, sty2]
            cols = [col1, col2]
            lwlist = [lw1, lw2]
            
            for i in range(2):
                if not fstrs[i]:
                    continue
                
                # Même lecture que le code TikZ (^, 2x, abs, ln...) ; courbe coupée aux pôles
                try:
                    expr = parse_function(fstrs[i])
                    check_pgf(expr)
                except ValueError as exc:
                    raise ValueError(f"Fonction {i+1} : {exc}") from None
                xvals, yvals = sample_function(expr, xmin, xmax, max(int(ns), 400),
                                               float(self.max_abs_y_var.get()))
                
                # Style de la courbe
                st_ = style_mpl_map.get(stys[i], "solid")
                c_ = cols[i] if cols[i] else "black"
                l_ = lwlist[i]
                self.ax.plot(xvals, yvals, linestyle=st_, color=c_, linewidth=l_)
                
                # --- Positionnement intelligent du label ---
                
                # Si l'utilisateur a déjà défini une position manuellement, l'utiliser
                if i in self.label_positions:
                    xlab, ylab = self.label_positions[i]
                    anchor = "center"
                else:
                    # Sinon, calculer une position optimale
                    
                    # 1. Filtrer les valeurs exploitables
                    finite_mask = np.isfinite(yvals) & (np.abs(yvals) < 1e10)
                    inside_mask = finite_mask & (yvals >= ymin) & (yvals <= ymax)
                    
                    if np.any(inside_mask):
                        # Cas idéal : valeurs dans la fenêtre visible
                        valid_idx = np.where(inside_mask)[0]
                    elif np.any(finite_mask):
                        # Fallback : valeurs finies mais hors fenêtre
                        valid_idx = np.where(finite_mask)[0]
                    else:
                        # Dernier recours : centre du graphique
                        xlab = (xmin + xmax) / 2
                        ylab = (ymin + ymax) / 2
                        anchor = "center"
                        valid_idx = None
                    
                    if valid_idx is not None and len(valid_idx) > 0:
                        # Positionnement stratégique : 80% pour courbe 1, 20% pour courbe 2
                        frac = 0.8 if i == 0 else 0.2
                        idx_in_valid = int(frac * (len(valid_idx) - 1))
                        idx = valid_idx[idx_in_valid]
                        
                        xlab = xvals[idx]
                        ylab = yvals[idx]
                        
                        # Décalage intelligent
                        dx = 0.02 * (xmax - xmin)
                        dy = 0.05 * (ymax - ymin)
                        
                        # Décalage alterné pour éviter chevauchements
                        if i == 0:
                            ylab += dy
                            anchor = "bottom"
                        else:
                            ylab -= dy
                            anchor = "top"
                        
                        xlab += dx
                        
                        # S'assurer de rester dans la fenêtre avec marge
                        xmarg = 0.02 * (xmax - xmin)
                        ymarg = 0.02 * (ymax - ymin)
                        xlab = min(max(xlab, xmin + xmarg), xmax - xmarg)
                        ylab = min(max(ylab, ymin + ymarg), ymax - ymarg)
                    
                    # Mémoriser la position calculée
                    self.label_positions[i] = (xlab, ylab)
                
                # Création du label
                txt_obj = self.ax.text(
                    xlab, ylab, labs[i],
                    color=c_,
                    fontsize=9,
                    picker=True,
                    ha='center',
                    va=anchor
                )
                txt_obj._curve_index = i
                self.label_texts.append(txt_obj)
            
            # Configuration finale du graphique
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
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