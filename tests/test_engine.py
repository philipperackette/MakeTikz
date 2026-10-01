"""
Tests du moteur TikZ (hf_space/make_tikz_engine.py).

    pip install sympy numpy pytest
    python -m pytest tests

Si pdflatex est installé, le code produit est aussi compilé avec pgfplots.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hf_space"))
from make_tikz_engine import (  # noqa: E402
    check_pgf,
    generate_tikz_code,
    parse_function,
    sample_function,
    to_pgf,
)


def pgf(text):
    return to_pgf(parse_function(text))


def plots(code):
    return re.findall(r"\\addplot\[[^\]]*domain=([^\]]*)\]\{(.*)\};", code)


def tikz(*functions, xmin=-3, xmax=3):
    return generate_tikz_code(list(functions), [], [], [], [], xmin, xmax, -3, 3, 1, 1)


@pytest.mark.parametrize("text, expected", [
    ("x^2", "x^2"),                      # ^ accepté comme puissance
    ("2x+1", "2*x + 1"),                 # multiplication implicite
    ("exp(-x**2)", "exp(-(x^2))"),       # moins devant une puissance : parenthèses
    ("-x**2+1", "1 - (x^2)"),
    ("abs(x)", "abs(x)"),                # pgfmath : abs, pas Abs
    ("ln(x)", "ln(x)"),
    ("log(x)", "ln(x)"),
    ("log10(x)", "ln(x)/ln(10)"),
    ("csc(x)", "cosec(x)"),
    ("e^x", "exp(x)"),
    ("pi*x", "pi*x"),
    ("1/(x^2-1)", "1/(x^2 - 1)"),
    ("x^(1/3)", "x^(1/3)"),
    ("x^(-1/2)", "1/(sqrt(x))"),
    ("asinh(x)", "ln((x) + sqrt((x)^2 + 1))"),
])
def test_traduction_pgf(text, expected):
    assert pgf(text) == expected


@pytest.mark.parametrize("text, message", [
    ("gamma(x)", "n'existe pas dans pgfplots"),
    ("y+1", "variable inconnue"),
    ("x+(", "illisible"),
])
def test_erreurs_explicites(text, message):
    with pytest.raises(ValueError, match=message):
        check_pgf(parse_function(text))


def test_erreur_dans_generate_tikz():
    with pytest.raises(ValueError, match="Fonction 1"):
        tikz("gamma(x)")


def test_poles_rationnels_decoupes():
    domains = [d for d, _ in plots(tikz("1/(x^2-1)"))]
    assert domains == ["-3:-1.01", "-0.99:0.99", "1.01:3"]


def test_piecewise():
    assert plots(tikz("Piecewise((x+1, x<0), (x-1, True))")) == [
        ("-3.0:0.0", "x + 1"), ("0.0:3.0", "x - 1")]


def test_apercu_coupe_aux_poles():
    xs, ys = sample_function(parse_function("1/x"), -1, 1, 401)
    assert np.isnan(ys[np.argmin(np.abs(xs))])
    assert np.isnan(ys).sum() >= 1
    xs, ys = sample_function(parse_function("log(x)"), -1, 1, 401)
    assert np.isnan(ys[xs < 0]).all()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex absent")
def test_compilation_latex(tmp_path):
    functions = ["exp(-x^2)", "2x+1", "abs(x)", "asinh(x)", "csc(x)", "1/(x^2-1)",
                 "Piecewise((x+1, x<0), (x-1, True))", "log10(x)", "-2x^3+x"]
    body = "\n".join(tikz(f) for f in functions)
    doc = ("\\documentclass{article}\\usepackage{pgfplots}\\pgfplotsset{compat=1.16}"
           "\\begin{document}\n" + body + "\n\\end{document}\n")
    (tmp_path / "t.tex").write_text(doc)
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                         cwd=tmp_path, capture_output=True, text=True)
    errors = [line for line in run.stdout.splitlines() if line.startswith("!")]
    assert run.returncode == 0, errors
