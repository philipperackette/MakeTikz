# MakeTikz

> **Vous ne voulez pas installer Python ou vous ne savez pas l’utiliser ?**  
> Utilisez directement la **version web** (interface très proche, prévisualisation + bouton « Copier le code TikZ ») :  
> 👉 https://huggingface.co/spaces/rackette/MakeTikz

MakeTikz est un petit utilitaire Python/Tkinter qui permet de **générer automatiquement du code TikZ/pgfplots** à partir d’expressions symboliques (SymPy), avec **aperçu interactif** de la courbe avant export.  
L’objectif : préparer rapidement de beaux tracés pour des cours, feuilles d’exercices ou rapports LaTeX, sans écrire à la main les commandes `\addplot`.

---

## Fonctionnalités

- **Interface graphique (Tkinter)**
  - Saisie de deux fonctions (ex. `Piecewise((x+1, x<0),(x-1, True))`, `log(x)`),
  - Choix du label LaTeX pour chaque courbe (ex. `$C_f$`, `$C_g$`),
  - Réglage du style (plein, tirets, pointillés…) et de la couleur,
  - Réglage du domaine : `xmin`, `xmax`, `ymin`, `ymax`, pas de graduation, labels des axes, etc.

- **Support d’expressions SymPy courantes**
  - Variable : `x`
  - Fonctions : `sin`, `cos`, `tan`, `exp`, `log`, fonctions hyperboliques, puissances (`x**2`), valeurs absolues, `Piecewise`, etc.

- **Gestion des singularités**
  - Détection des pôles sur le domaine (dénominateur nul),
  - Découpage du domaine en sous-intervalles,
  - Génération d’un code pgfplots propre (sauts aux singularités, restrictions sur `y`…).

- **Code TikZ/pgfplots prêt à coller**
  - Génération d’un environnement `tikzpicture` + `axis`,
  - Conversion de `log(...)` en `\ln(...)` dans les labels,
  - Paramètres configurables : grille, graduations, échelles, nombre d’échantillons…

- **Placement interactif des labels**
  - Affichage des labels sur la figure Matplotlib,
  - Déplacement des labels à la souris,
  - Export des positions dans le code TikZ via `\draw ... node{...}`.

---

## Dépendances

- Python 3.x  
- `tkinter`  
- `numpy`  
- `sympy`  
- `matplotlib`

Installation typique (version bureau) :

```bash
git clone https://github.com/philipperackette/MakeTikz.git
cd MakeTikz
pip install numpy sympy matplotlib
python plot_tikz_generator.py