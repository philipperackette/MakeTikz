import gradio as gr
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from make_tikz_engine import generate_tikz_code


# Small helpers to parse numeric fields safely
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
    # Clean numeric values
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

    # Prepare lists for generate_tikz_code
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
    # 1) Generate TikZ code
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
            label_positions={},  # web version: no drag & drop
        )
    except Exception as e:
        tikz = f"% Error while generating TikZ: {e}"

    # -----------------------------
    # 2) Matplotlib preview
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
            # Show error message inside the plot for this curve
            ax.text(
                0.5,
                0.5,
                f"Error in {lab}:\n{str(e)}",
                transform=ax.transAxes,
                ha="center",
                color="red",
            )
            continue

    ax.set_xlim(xmin_f, xmax_f)
    ax.set_ylim(ymin_f, ymax_f)

    # grid
    ax.grid(show_grid)

    # ticks + labels
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # axis labels
    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")

    if curves_plotted and any(labels):
        ax.legend(loc="best")

    fig.tight_layout()

    return tikz, fig


# ==============================
# Gradio interface
# ==============================

style_choices = ["solid", "dashed", "dotted", "dashdot"]
color_choices = ["black", "red", "blue", "gray", "green", "orange", "purple", "brown"]

with gr.Blocks() as demo:
    gr.Markdown("# MakeTikZ – LaTeX graph generator")

    with gr.Accordion("📚 SymPy syntax help", open=False):
        gr.Markdown(
            """
        ### Basic operators
        - **Addition / subtraction**: `+`, `-`
        - **Multiplication**: `*` or implicit
        - **Division**: `/`
        - **Power**: `**` or `^`
        - **Square root**: `sqrt(x)`

        ### Available functions
        - **Trigonometric**: `sin(x)`, `cos(x)`, `tan(x)`
        - **Hyperbolic**: `sinh(x)`, `cosh(x)`, `tanh(x)`
        - **Exponential / logarithm**: `exp(x)`, `log(x)` (natural log), `log10(x)`
        - **Absolute value**: `abs(x)` or `Abs(x)`
        - **Sign**: `sign(x)`

        ### Constants
        - **π (pi)**: `pi`
        - **e (Euler)**: `E` or `exp(1)`

        ### Example functions
        1. **Polynomial**: `x**2 + 3*x - 2`
        2. **Rational function**: `(x+1)/(x-1)`
        3. **Trigonometric**: `2*sin(pi*x/2)`
        4. **Exponential**: `exp(-x**2)`
        5. **Piecewise function**:
           ```python
           Piecewise((x+1, x < 0), (x-1, True))
           ```
           (for x < 0: x+1, otherwise: x-1)

        ### Tips
        - Use `*` explicitly for multiplication
        - Parentheses matter for precedence
        - For piecewise functions, use `Piecewise((expr1, cond1), (expr2, cond2), ...)`
        """
        )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Curves")

            func1 = gr.Textbox(
                label="Function 1",
                value="Piecewise((x+1, x<0),(x-1, True))",
                placeholder="e.g. x**2 + sin(x), log(x), etc.",
            )
            label1 = gr.Textbox(label="Label 1", value="$f$")
            style1 = gr.Dropdown(
                choices=style_choices, value="solid", label="Style 1"
            )
            color1 = gr.Dropdown(
                choices=color_choices, value="black", label="Color 1"
            )
            lw1 = gr.Textbox(label="Line width 1 (pt)", value="1.5")

            gr.HTML("<hr>")

            func2 = gr.Textbox(
                label="Function 2 (optional)",
                value="log(x)",
                placeholder="e.g. exp(-x**2), cos(pi*x), etc.",
            )
            label2 = gr.Textbox(label="Label 2", value="$g$")
            style2 = gr.Dropdown(
                choices=style_choices, value="dashed", label="Style 2"
            )
            color2 = gr.Dropdown(
                choices=color_choices, value="gray", label="Color 2"
            )
            lw2 = gr.Textbox(label="Line width 2 (pt)", value="1.5")

            gr.Markdown("### Domain & axes")

            xmin = gr.Textbox(label="xmin", value="-5")
            xmax = gr.Textbox(label="xmax", value="5")
            ymin = gr.Textbox(label="ymin", value="-5")
            ymax = gr.Textbox(label="ymax", value="5")

            xstep = gr.Textbox(label="x step", value="1.0")
            ystep = gr.Textbox(label="y step", value="1.0")

            xlabel = gr.Textbox(label="x-axis label", value="x")
            ylabel = gr.Textbox(label="y-axis label", value="y")

            gr.Markdown("### Display options")

            show_grid = gr.Checkbox(
                label="Show grid", value=True
            )
            show_ticks = gr.Checkbox(
                label="Show ticks", value=True
            )
            show_tick_labels = gr.Checkbox(
                label="Show tick labels", value=True
            )
            hide_extremes = gr.Checkbox(
                label="Hide ticks at extremes",
                value=False,
            )

            gr.Markdown("### Advanced TikZ settings")

            scale_x = gr.Textbox(label="X scale (cm)", value="1.0")
            scale_y = gr.Textbox(label="Y scale (cm)", value="1.0")
            max_abs_y = gr.Textbox(label="|y| max", value="100.0")
            num_samples = gr.Textbox(label="Number of samples", value="200")

            bouton = gr.Button("Generate / Update", variant="primary")

        with gr.Column(scale=2):
            preview = gr.Plot(label="Preview")
            output = gr.Textbox(
                label="TikZ / pgfplots code",
                lines=30,
            )
            with gr.Row():
                copy_btn = gr.Button("📋 Copy TikZ code")
                clear_btn = gr.Button("🗑️ Clear inputs")

    # "Generate / Update" button
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

    # "Copy TikZ code" button (client-side JS)
    copy_btn.click(
        fn=None,
        inputs=output,
        outputs=None,
        js="""
        (tikz) => {
            navigator.clipboard.writeText(tikz);
            alert('TikZ code copied to clipboard.');
        }
        """,
    )

    # "Clear inputs" button
    def clear_fields():
        return [
            "", "log(x)", "$f$", "$g$",
            "solid", "black", "1.5",
            "dashed", "gray", "1.5",
            "-5", "5", "-5", "5",
            "1.0", "1.0", "x", "y",
            True, True, True, False,
            "1.0", "1.0", "100.0", "200",
        ]

    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[
            func1, func2, label1, label2, style1, color1, lw1,
            style2, color2, lw2, xmin, xmax, ymin, ymax,
            xstep, ystep, xlabel, ylabel, show_grid, show_ticks,
            show_tick_labels, hide_extremes, scale_x, scale_y,
            max_abs_y, num_samples,
        ],
    )

if __name__ == "__main__":
    demo.launch()