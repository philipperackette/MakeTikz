import gradio as gr
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from make_tikz_engine import generate_tikz_code


# Petites fonctions utilitaires pour parser proprement
def to_float(s, default):
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return default


def to_int(s, default):
    try:
        return int(str(s))
    except Exception:
        return default


def make_tikz(
    func1,
    func2,
    label1,
    label2,
    style1,
    color1,
    lw1,
    style2,
    color2,
    lw2,
    xmin, xmax,
    ymin, ymax,
    xstep, ystep,
    xlabel, ylabel,
    show_grid,
    show_ticks,
    show_tick_labels,
    hide_extremes,
    scale_x, scale_y,
    max_abs_y,
    num_samples,
):
    # Valeurs numériques "propres"
    xmin_f = to_float(xmin, -5.0)
    xmax_f = to_float(xmax, 5.0)
    ymin_f = to_float(ymin, -5.0)
    ymax_f = to_float(ymax, 5.0)
    xstep_f = to_float(xstep, 1.0)
    ystep_f = to_float(ystep, 1.0)
    scale_x_f = to_float(scale_x, 1.0)
    scale_y_f = to_float(scale_y, 1.0)
    max_abs_y_f = to_float(max_abs_y, 100.0)
    num_samples_i = to_int(num_samples, 200)

    # Préparation des listes pour generate_tikz_code
    functions = []
    labels = []
    styles = []
    colors = []
    widths = []

    if func1.strip():
        functions.append(func1.strip())
        labels.append(label1.strip() or "$C_1$")
        styles.append(style1 or "solid")
        colors.append(color1 or "black")
        widths.append(to_float(lw1, 1.5))

    if func2.strip():
        functions.append(func2.strip())
        labels.append(label2.strip() or "$C_2$")
        styles.append(style2 or "dashed")
        colors.append(color2 or "gray")
        widths.append(to_float(lw2, 1.5))

    # -----------------------------
    # 1) Génération du code TikZ
    # -----------------------------
    try:
        tikz = generate_tikz_code(
            functions=functions,
            curve_labels=labels,
            styles=styles,
            colors=colors,
            line_widths=widths,
            xmin=xmin_f,
            xmax=xmax_f,
            ymin=ymin_f,
            ymax=ymax_f,
            x_step=xstep_f,
            y_step=ystep_f,
            show_grid=show_grid,
            show_ticks=show_ticks,
            show_tick_labels=show_tick_labels,
            hide_extremes=hide_extremes,
            axis_label_x=xlabel or "x",
            axis_label_y=ylabel or "y",
            scale_ratio_x=scale_x_f,
            scale_ratio_y=scale_y_f,
            max_abs_y=max_abs_y_f,
            num_samples=num_samples_i,
            label_positions={},  # version web : pas de drag & drop
        )
    except Exception as e:
        tikz = f"% Erreur lors de la génération TikZ : {e}"

    # -----------------------------
    # 2) Aperçu Matplotlib
    # -----------------------------
    fig, ax = plt.subplots(figsize=(5, 4))

    x_sym = sp.Symbol("x", real=True)
    local_dict = {
        "x": x_sym,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "csc": sp.csc,
        "sec": sp.sec,
        "cot": sp.cot,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "exp": sp.exp,
        "log": sp.log,
        "pi": sp.pi,
        "e": sp.E,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "sign": sp.sign,
    }

    style_mpl_map = {
        "solid": "solid",
        "dashed": "dashed",
        "dotted": "dotted",
        "dashdot": "dashdot",
    }

    xvals = np.linspace(xmin_f, xmax_f, max(num_samples_i, 200))
    curves_plotted = False

    for i in range(len(functions)):
        fstr = functions[i]
        lab = labels[i]
        stp = styles[i]
        col = colors[i]
        lw = widths[i]
        try:
            expr = sp.parse_expr(fstr, local_dict)
            f = sp.lambdify(x_sym, expr, "numpy")
            yvals = np.array(f(xvals), dtype=float)

            linestyle = style_mpl_map.get(stp, "solid")
            color = col if col else "black"

            ax.plot(
                xvals,
                yvals,
                linestyle=linestyle,
                color=color,
                linewidth=lw,
                label=lab,
            )
            curves_plotted = True
        except Exception as e:
            # On affiche l'erreur dans le graphique
            ax.text(0.5, 0.5, f"Erreur dans {lab}:\n{str(e)}", 
                   transform=ax.transAxes, ha='center', color='red')
            continue

    ax.set_xlim(xmin_f, xmax_f)
    ax.set_ylim(ymin_f, ymax_f)

    # grille
    ax.grid(show_grid)

    # ticks + labels
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # labels d'axes
    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")

    if curves_plotted and any(labels):
        ax.legend(loc="best")

    fig.tight_layout()

    return tikz, fig


# ==============================
# Interface Gradio
# ==============================

style_choices = ["solid", "dashed", "dotted", "dashdot"]
color_choices = ["black", "red", "blue", "gray", "green", "orange", "purple", "brown"]

with gr.Blocks() as demo:
    gr.Markdown("# MakeTikZ - Générateur de graphiques LaTeX")
    
    with gr.Accordion("📚 Aide sur la syntaxe SymPy", open=False):
        gr.Markdown("""
        ### Opérateurs mathématiques
        - **Addition/ soustraction** : `+`, `-`
        - **Multiplication** : `*` ou espace implicite
        - **Division** : `/`
        - **Puissance** : `**` ou `^`
        - **Racine carrée** : `sqrt(x)`
        
        ### Fonctions disponibles
        - **Trigonométrie** : `sin(x)`, `cos(x)`, `tan(x)`
        - **Trigonométrie hyperbolique** : `sinh(x)`, `cosh(x)`, `tanh(x)`
        - **Exponentielle/logarithme** : `exp(x)`, `log(x)` (ln), `log10(x)`
        - **Valeur absolue** : `abs(x)` ou `Abs(x)`
        - **Signe** : `sign(x)`
        
        ### Constantes
        - **π (pi)** : `pi`
        - **e (exponentielle)** : `E` ou `exp(1)`
        
        ### Exemples de fonctions
        1. **Polynôme** : `x**2 + 3*x - 2`
        2. **Fonction rationnelle** : `(x+1)/(x-1)`
        3. **Fonction trigonométrique** : `2*sin(pi*x/2)`
        4. **Fonction exponentielle** : `exp(-x**2)`
        5. **Fonction définie par morceaux** :
           ```
           Piecewise((x+1, x<0), (x-1, True))
           ```
           (pour x<0: x+1, sinon: x-1)
        
        ### Conseils
        - Utilisez `*` pour la multiplication explicite
        - Les parenthèses sont importantes pour la priorité des opérations
        - Pour les fonctions par morceaux, importez `Piecewise` si nécessaire
        """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Courbes")

            func1 = gr.Textbox(
                label="Fonction 1",
                value="Piecewise((x+1, x<0),(x-1, True))",
                placeholder="Ex: x**2 + sin(x), log(x), etc.",
            )
            label1 = gr.Textbox(label="Label 1", value="$f$")
            style1 = gr.Dropdown(
                choices=style_choices, value="solid", label="Style 1"
            )
            color1 = gr.Dropdown(
                choices=color_choices, value="black", label="Couleur 1"
            )
            lw1 = gr.Textbox(label="Épaisseur 1 (pt)", value="1.5")

            gr.HTML("<hr>")

            func2 = gr.Textbox(
                label="Fonction 2 (optionnelle)", 
                value="log(x)",
                placeholder="Ex: exp(-x**2), cos(pi*x), etc.",
            )
            label2 = gr.Textbox(label="Label 2", value="$g$")
            style2 = gr.Dropdown(
                choices=style_choices, value="dashed", label="Style 2"
            )
            color2 = gr.Dropdown(
                choices=color_choices, value="gray", label="Couleur 2"
            )
            lw2 = gr.Textbox(label="Épaisseur 2 (pt)", value="1.5")

            gr.Markdown("### Domaine & axes")

            xmin = gr.Textbox(label="xmin", value="-5")
            xmax = gr.Textbox(label="xmax", value="5")
            ymin = gr.Textbox(label="ymin", value="-5")
            ymax = gr.Textbox(label="ymax", value="5")

            xstep = gr.Textbox(label="Pas en x", value="1.0")
            ystep = gr.Textbox(label="Pas en y", value="1.0")

            xlabel = gr.Textbox(label="Label axe X", value="x")
            ylabel = gr.Textbox(label="Label axe Y", value="y")

            gr.Markdown("### Options d'affichage")

            show_grid = gr.Checkbox(
                label="Afficher la grille", value=True
            )
            show_ticks = gr.Checkbox(
                label="Afficher les graduations", value=True
            )
            show_tick_labels = gr.Checkbox(
                label="Afficher les labels de graduations", value=True
            )
            hide_extremes = gr.Checkbox(
                label="Ne pas afficher les graduations aux extrémités",
                value=False,
            )

            gr.Markdown("### Paramètres avancés (TikZ)")

            scale_x = gr.Textbox(label="Échelle X (cm)", value="1.0")
            scale_y = gr.Textbox(label="Échelle Y (cm)", value="1.0")
            max_abs_y = gr.Textbox(label="|y| max", value="100.0")
            num_samples = gr.Textbox(label="Nb. échantillons", value="200")

            bouton = gr.Button("Générer / Mettre à jour", variant="primary")

        with gr.Column(scale=2):
            preview = gr.Plot(label="Aperçu")
            output = gr.Textbox(
                label="Code TikZ / pgfplots",
                lines=30,
            )
            with gr.Row():
                copy_btn = gr.Button("📋 Copier le code TikZ")
                clear_btn = gr.Button("🗑️ Effacer")

    # clic sur "Générer / Mettre à jour"
    bouton.click(
        make_tikz,
        inputs=[
            func1,
            func2,
            label1,
            label2,
            style1,
            color1,
            lw1,
            style2,
            color2,
            lw2,
            xmin,
            xmax,
            ymin,
            ymax,
            xstep,
            ystep,
            xlabel,
            ylabel,
            show_grid,
            show_ticks,
            show_tick_labels,
            hide_extremes,
            scale_x,
            scale_y,
            max_abs_y,
            num_samples,
        ],
        outputs=[output, preview],
    )

    # bouton "Copier le code TikZ" (JS côté client)
    copy_btn.click(
        fn=None,
        inputs=output,
        outputs=None,
        js="""
        (tikz) => {
            navigator.clipboard.writeText(tikz);
            alert('Code TikZ copié dans le presse-papiers.');
        }
        """,
    )
    
    # bouton "Effacer"
    def clear_fields():
        return ["", "", "$f$", "$g$", "solid", "black", "1.5", 
                "dashed", "gray", "1.5", "-5", "5", "-5", "5",
                "1.0", "1.0", "x", "y", True, True, True, False,
                "1.0", "1.0", "100.0", "200"]
    
    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[
            func1, func2, label1, label2, style1, color1, lw1,
            style2, color2, lw2, xmin, xmax, ymin, ymax,
            xstep, ystep, xlabel, ylabel, show_grid, show_ticks,
            show_tick_labels, hide_extremes, scale_x, scale_y,
            max_abs_y, num_samples
        ]
    )

if __name__ == "__main__":
    demo.launch()