import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.printing.str import StrPrinter


X = sp.Symbol('x', real=True)

# Noms acceptés dans la saisie (aperçu et code TikZ utilisent la même lecture).
LOCAL_DICT = {
    "x": X,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "csc": sp.csc, "sec": sp.sec, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "log10": lambda u: sp.log(u, 10), "log2": lambda u: sp.log(u, 2),
    "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling,
    "pi": sp.pi, "e": sp.E, "E": sp.E,
    "Piecewise": sp.Piecewise,
}

# ^ pour la puissance et multiplication implicite (2x, 3sin(x)), comme indiqué dans l'aide.
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_function(text):
    """Lit une fonction de x saisie par l'utilisateur ; ValueError si illisible."""
    try:
        expr = parse_expr(text, local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)
    except Exception as exc:
        raise ValueError(f"expression illisible « {text} » ({exc.__class__.__name__})") from None
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"« {text} » n'est pas une fonction de x")
    others = expr.free_symbols - {X}
    if others:
        names = ", ".join(sorted(str(s) for s in others))
        raise ValueError(f"« {text} » contient une variable inconnue : {names} (seule x est permise)")
    return expr


##############################################
# Traduction sympy -> syntaxe pgfmath
##############################################
class PgfPrinter(StrPrinter):
    """
    Écrit une expression sympy dans la syntaxe de pgfplots (pgfmath).

    Points délicats de pgfmath :
      - noms de fonctions : ln, abs, cosec ; pas de fonctions hyperboliques
        réciproques (réécrites avec ln) ni de log en base quelconque ;
      - parenthèses explicites autour d'une puissance précédée d'un moins
        (-(x^2)) : sans effet avec pgf 3.1, mais sûr avec les versions
        anciennes qui lisaient -x^2 comme (-x)^2.
    """

    FUNCTIONS = {
        "sin": "sin", "cos": "cos", "tan": "tan", "sec": "sec", "csc": "cosec", "cot": "cot",
        "asin": "asin", "acos": "acos", "atan": "atan",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
        "exp": "exp", "Abs": "abs", "sign": "sign", "floor": "floor", "ceiling": "ceil",
    }

    def _print_Function(self, expr):
        name = expr.func.__name__
        if name not in self.FUNCTIONS:
            raise ValueError(f"la fonction {name} n'existe pas dans pgfplots")
        return f"{self.FUNCTIONS[name]}({', '.join(self._print(a) for a in expr.args)})"

    def _print_log(self, expr):
        if len(expr.args) == 2:
            return f"ln({self._print(expr.args[0])})/ln({self._print(expr.args[1])})"
        return f"ln({self._print(expr.args[0])})"

    def _print_asinh(self, expr):
        u = self._print(expr.args[0])
        return f"ln(({u}) + sqrt(({u})^2 + 1))"

    def _print_acosh(self, expr):
        u = self._print(expr.args[0])
        return f"ln(({u}) + sqrt(({u})^2 - 1))"

    def _print_atanh(self, expr):
        u = self._print(expr.args[0])
        return f"0.5*ln((1 + ({u}))/(1 - ({u})))"

    def _print_Exp1(self, expr):
        return "e"

    def _print_Pi(self, expr):
        return "pi"

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Rational(self, expr):
        return f"({expr.p}/{expr.q})"

    def _print_Pow(self, expr, rational=False):
        base, exp = expr.as_base_exp()
        if exp == sp.S.Half:
            return f"sqrt({self._print(base)})"
        if exp == -1:
            return f"1/({self._print(base)})"
        if exp.is_number and exp.is_negative:
            return f"1/({self._print(sp.Pow(base, -exp, evaluate=False))})"
        if base == sp.E:
            return f"exp({self._print(exp)})"
        b = self._print(base)
        if not (base.is_Symbol or (base.is_Integer and base >= 0) or base.is_Function or base == sp.pi):
            b = f"({b})"
        if exp == 1:
            return b
        e = self._print(exp)
        if not (exp.is_Integer and exp >= 0) and not exp.is_Rational:
            e = f"({e})"
        return f"{b}^{e}"

    def _print_Mul(self, expr):
        coeff, rest = expr.as_coeff_Mul()
        if coeff.is_negative:
            inner = self._print(-expr)
            # parenthèses explicites : -(x^2), voir la docstring
            return f"-({inner})" if ("^" in inner or "/" in inner or "*" in inner) else f"-{inner}"
        return super()._print_Mul(expr)


def sample_function(expr, xmin, xmax, num_samples=400, max_abs_y=100.0):
    """
    Points (x, y) pour l'aperçu matplotlib. Les valeurs non définies ou
    au-delà de max_abs_y deviennent NaN, et la courbe est coupée aux pôles
    (pas de trait vertical entre deux branches, comme dans le code TikZ).
    """
    xs = np.linspace(xmin, xmax, max(int(num_samples), 2))
    f = sp.lambdify(X, expr, "numpy")
    with np.errstate(all="ignore"):
        ys = f(xs)
        ys = np.broadcast_to(np.asarray(ys, dtype=complex), xs.shape)
        ys = np.where(np.abs(ys.imag) > 1e-12, np.nan, ys.real).astype(float)
    ys[~np.isfinite(ys) | (np.abs(ys) > max_abs_y)] = np.nan
    if expr.is_rational_function(X):
        for pole in find_rational_singularities(expr, X, xmin, xmax):
            ys[np.argmin(np.abs(xs - pole))] = np.nan
            ys[np.nonzero((xs[:-1] < pole) & (xs[1:] > pole))[0]] = np.nan
    return xs, ys


def to_pgf(expr):
    """Expression sympy -> chaîne pgfmath (ValueError si non traduisible)."""
    return PgfPrinter().doprint(expr)


def check_pgf(expr):
    """ValueError si une partie de l'expression ne peut pas être tracée par pgfplots."""
    parts = [e for e, _ in expr.args] if isinstance(expr, sp.Piecewise) else [expr]
    for part in parts:
        to_pgf(part)

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

                if sol_set is sp.S.EmptySet:
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

                    if remain is sp.S.EmptySet:
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
      - traduction exacte vers pgfmath (ln, abs, parenthèses autour de -x^2...),
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

    x_sym = X
    nb_fun = min(len(functions), 2)

    for i in range(nb_fun):
        fstr = functions[i]
        lbl = curve_labels[i] if i < len(curve_labels) else f"$C_{{{i+1}}}$"
        stp = styles[i] if i < len(styles) else "solid"
        col = colors[i] if i < len(colors) else "black"
        lw = line_widths[i] if i < len(line_widths) else 1.0
        style_pgf = style_map.get(stp, "solid")

        try:
            expr = parse_function(fstr)
        except ValueError as exc:
            raise ValueError(f"Fonction {i+1} : {exc}") from None

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

            try:
                subE_tex = to_pgf(subE)
            except ValueError as exc:
                raise ValueError(f"Fonction {i+1} : {exc}") from None

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