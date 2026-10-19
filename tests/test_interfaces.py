"""
Les interfaces se chargent sans erreur (sans ouvrir de fenêtre ni de serveur).

- plot_tikz_generator.py et Lissage.py : nécessitent tkinter ;
- hf_space/app.py (Space Hugging Face) : nécessite gradio
  (pip install -r hf_space/requirements.txt).
"""
import importlib
import os
import sys

import pytest

RACINE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "hf_space"))


@pytest.mark.parametrize("module", ["plot_tikz_generator", "Lissage"])
def test_applications_bureau(module):
    pytest.importorskip("tkinter")
    application = importlib.import_module(module)
    assert callable(application.main)
    assert hasattr(application, "PlotTikzApp")


def test_space_hugging_face():
    gr = pytest.importorskip("gradio")
    app = importlib.import_module("app")
    assert isinstance(app.demo, gr.Blocks)
