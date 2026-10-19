"""
Tests des calculs de Lissage.py (interpolation par points), sans ouvrir de fenêtre.

    pip install numpy matplotlib scipy pytest
    python -m pytest tests

Nécessite tkinter (importé par Lissage.py) ; si pdflatex est installé, le code
produit est aussi compilé avec pgfplots.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tkinter")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Lissage import (  # noqa: E402
    build_curve_samples,
    generate_tikz_code_from_samples,
    hermite_eval,
    parse_points_string,
)


def cubique(x):
    return x**3 - 2 * x


def test_lecture_des_points_tri_et_pentes_estimees():
    xs, ys, ms = parse_points_string("# commentaire\n2, 4\n0 0\n\n1;1\n")
    assert xs.tolist() == [0, 1, 2]
    assert ys.tolist() == [0, 1, 4]
    assert ms.tolist() == pytest.approx([1, 2, 3])


def test_lecture_des_pentes_fournies_suivent_le_tri():
    xs, ys, ms = parse_points_string("1 1 5\n0 0 -1\n")
    assert xs.tolist() == [0, 1]
    assert ms.tolist() == [-1, 5]


@pytest.mark.parametrize("texte", ["1 2 3 4", "1", "a b", "3 4"])
def test_lecture_refuse_les_saisies_invalides(texte):
    with pytest.raises(ValueError):
        parse_points_string(texte)


def test_hermite_exacte_pour_une_cubique_et_vide_hors_des_points():
    xs = np.array([-2.0, -0.5, 1.0, 2.0])
    grille = np.linspace(-3, 3, 241)
    ys = hermite_eval(grille, xs, cubique(xs), 3 * xs**2 - 2)
    dedans = (grille >= -2) & (grille <= 2)
    assert ys[dedans] == pytest.approx(cubique(grille[dedans]))
    assert np.isnan(ys[~dedans]).all()


def test_echantillons_hermite():
    xs, ys, ms = parse_points_string("0 0\n1 1\n2 0")
    xv, yv = build_curve_samples(xs, ys, ms, "hermite", -1, 3, 81)
    assert len(xv) == len(yv) == 81
    assert yv[np.isclose(xv, 1)] == pytest.approx([1])


def test_echantillons_spline_de_lissage_passe_par_les_points():
    pytest.importorskip("scipy")
    xs = np.array([0.0, 1, 2, 3, 4])
    ys = np.array([0.0, 1, 0, 1, 0])
    xv, yv = build_curve_samples(xs, ys, None, "lissage", 0, 4, 9, smooth_param=0)
    assert yv[::2] == pytest.approx(ys, abs=1e-9)


def test_methode_inconnue():
    with pytest.raises(ValueError):
        build_curve_samples(np.array([0.0, 1]), np.array([0.0, 1]), np.array([1.0, 1]), "autre", 0, 1, 5)


def code_exemple():
    xs, ys, ms = parse_points_string("-2 -4 0\n0 0 2\n2 4 0")
    courbe = build_curve_samples(xs, ys, ms, "hermite", -3, 3, 61)
    droite = (np.array([-1.0, 0.0, 1.0]), np.array([1.0, np.nan, -1.0]))
    return generate_tikz_code_from_samples(
        [courbe, droite], ["$f$", "$g$"], ["solid", "dashed"], ["blue", "red"], [1.2, 0.8],
        -3, 3, -5, 5, 1, 1, label_positions={0: (2, 4)},
    )


def test_code_tikz_produit():
    code = code_exemple()
    assert code.startswith("\\begin{tikzpicture}") and code.endswith("\\end{tikzpicture}")
    assert code.count("\\addplot[") == 2
    assert "dash pattern=on 5pt off 5pt" in code
    assert "nan" not in code.lower()
    # Seuls les points définis (dans [-2, 2] pour l'interpolation) sont écrits.
    assert code.count("      (") == 41 + 2
    assert "node[anchor=west]{\\color{blue}$f$}" in code


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex absent")
def test_compilation_latex(tmp_path):
    doc = ("\\documentclass{article}\\usepackage{pgfplots}\\pgfplotsset{compat=1.16}"
           "\\begin{document}\n" + code_exemple() + "\n\\end{document}\n")
    (tmp_path / "t.tex").write_text(doc)
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                         cwd=tmp_path, capture_output=True, text=True)
    errors = [line for line in run.stdout.splitlines() if line.startswith("!")]
    assert run.returncode == 0, errors
