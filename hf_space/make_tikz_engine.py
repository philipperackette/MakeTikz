import sympy as sp

##############################################
# Détection pôles rationnels + scission
##############################################
def find_rational_singularities(expr, x_sym, xmin, xmax):
    """
    Renvoie la liste triée des pôles réels de expr dans ]xmin, xmax[,
    si expr est une fonction rationnelle en x_sym. Sinon [].
    """
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
    """
    Découpe [xmin, xmax] en intervalles ouverts séparés par les pôles,
    en retirant une marge 'margin' autour de chaque pôle pour éviter
    de tracer pile dessus.
    """
    points = [xmin] + singularities + [xmax]
    intervals = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if (b - a) <= 2 * margin:
            continue
        left = a if i == 0 else a + margin
        right = b if i == (len(points) - 2) else b - margin
        if left < right:
            intervals.append((left, right))
    return intervals


##################################################
# Découpe piecewise "dans l'ordre" (le premier True prime)
##################################################
def piecewise_subfunctions_in_order(expr, x_sym, global_xmin, global_xmax):
    """
    Retourne une liste (sub_expr, a, b) pour un Piecewise(...) en ordre.
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
        if cond_i is True or cond_str == "True":
            # prend tout le reste
            for (ua, ub) in uncovered:
                if ua < ub:
                    sub_list.append((expr_i, ua, ub))
            uncovered = []
            break
        else:
            new_uncovered = []
            for (ua, ub) in uncovered:
                if ua >= ub:
                    continue
                sub_dom = sp.Interval(ua, ub)
                sol_set = sp.solveset(cond_i, x_sym, domain=sub_dom)

                if sol_set.is_EmptySet:
                    # rien de couvert sur [ua,ub]
                    new_uncovered.append((ua, ub))
                else:
                    covered_intervals = []
                    if isinstance(sol_set, sp.Interval):
                        covered_intervals = [sol_set]
                    elif isinstance(sol_set, sp.Union):
                        for part in sol_set.args:
                            if isinstance(part, sp.Interval):
                                covered_intervals.append(part)

                    # on enlève les morceaux couverts de sub_dom
                    remain = sub_dom
                    for ci in covered_intervals:
                        remain = remain - ci

                    if remain.is_EmptySet:
                        pass
                    elif isinstance(remain, sp.Interval):
                        new_uncovered.append((float(remain.start), float(remain.end)))
                    elif isinstance(remain, sp.Union):
                        for part in remain.args:
                            if isinstance(part, sp.Interval):
                                new_uncovered.append((float(part.start), float(part.end)))

                    # ajoute les sous-domaines où cond_i est vraie
                    for ci in covered_intervals:
                        a_ = float(ci.start)
                        b_ = float(ci.end)
                        if a_ < b_:
                            sub_list.append((expr_i, a_, b_))

            uncovered = new_uncovered

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
    """
    Génère le code TikZ/pgfplots pour les fonctions données (max 2 courbes),
    en gérant :
      - Piecewise sympy (découpage du domaine),
      - pôles rationnels (découpage + unbounded coords=jump),
      - remplacement log(...) -> ln(...),
      - options d'axes (grille, graduations, labels, échelles).
    """
    if label_positions is None:
        label_positions = {}

    # Conversion style python -> pgf
    style_map = {
        "solid":   "solid",
        "dashed":  "dash pattern=on 5pt off 5pt",
        "dotted":  "dash pattern=on 1pt off 3pt",
        "dashdot": "dash pattern=on 4pt off 2pt on 1pt off 2pt",
    }

    # Options de base pgfplots
    pgf_options = [
        f"xmin={xmin}",
        f"xmax={xmax}",
        f"ymin={ymin}",
        f"ymax={ymax}",
        "axis lines=middle",
        "trig format=rad"
    ]
    pgf_options.append("grid=major" if show_grid else "grid=none")

    # Gestion des ticks
    if show_ticks:
        if not hide_extremes:
            pgf_options.append(f"xtick distance={x_step}")
            pgf_options.append(f"ytick distance={y_step}")
        else:
            # fabrique la liste de ticks en évitant xmin,xmax,ymin,ymax
            xticks = []
            c = xmin + x_step
            while c < (xmax - 1e-9):
                xticks.append(round(c, 5))
                c += x_step
            if len(xticks) == 0:
                pgf_options.append("xtick=\\empty")
            else:
                xs = ",".join(str(v) for v in xticks)
                pgf_options.append(f"xtick={{ {xs} }}")

            yticks = []
            c = ymin + y_step
            while c < (ymax - 1e-9):
                yticks.append(round(c, 5))
                c += y_step
            if len(yticks) == 0:
                pgf_options.append("ytick=\\empty")
            else:
                ys = ",".join(str(v) for v in yticks)
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

    xlabel_proc = ensure_math_mode(axis_label_x)
    ylabel_proc = ensure_math_mode(axis_label_y)
    pgf_options.append(f"xlabel={xlabel_proc}")
    pgf_options.append(f"ylabel={ylabel_proc}")

    # Échelles
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

    x_sym = sp.Symbol('x', real=True)
    nb_fun = min(len(functions), 2)

    for i in range(nb_fun):
        fstr = functions[i]
        lbl = curve_labels[i] if i < len(curve_labels) else f"$C_{{{i+1}}}$"
        stp = styles[i] if i < len(styles) else "solid"
        col = colors[i] if i < len(colors) else "black"
        lw = line_widths[i] if i < len(line_widths) else 1.0
        style_pgf = style_map.get(stp, "solid")

        # Dictionnaire de parsing
        local_dict = {
            "x": x_sym,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "csc": sp.csc, "sec": sp.sec, "cot": sp.cot,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
            "exp": sp.exp, "log": sp.log
        }
        try:
            expr = sp.parse_expr(fstr, local_dict)
        except Exception:
            expr = None
        if expr is None:
            continue

        sub_exprs = []
        if isinstance(expr, sp.Piecewise):
            sub_list = piecewise_subfunctions_in_order(expr, x_sym, xmin, xmax)
            sub_exprs.extend(sub_list)  # => (ex, a, b)
        else:
            sub_exprs.append((expr, xmin, xmax))

        for (subE, subA, subB) in sub_exprs:
            if subA >= subB:
                continue

            # scinde sur les pôles rationnels
            if subE.is_rational_function(x_sym):
                poles = find_rational_singularities(subE, x_sym, subA, subB)
            else:
                poles = []

            intervals = make_subintervals(subA, subB, poles, margin=1e-2)

            subE_tex = str(subE).replace("**", "^")
            # => On remplace "log(" par "ln(" pour pgf
            subE_tex = subE_tex.replace("log(", "ln(")

            for (L, R) in intervals:
                cstyle = (
                    f"samples={num_samples}, unbounded coords=jump, "
                    f"line width={lw}pt, color={col}, {style_pgf}, "
                    f"{restrict_str}, domain={L}:{R}"
                )
                tikz.append(f"    \\addplot[{cstyle}]{{{subE_tex}}};")

        # gestion des labels de courbes (texte) si positions fournies
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