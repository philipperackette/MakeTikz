# MakeTikz

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Vous ne voulez pas installer Python ou vous ne savez pas l'utiliser ?**  
> Utilisez directement la **version web** (interface très proche, prévisualisation + bouton « Copier le code TikZ ») :  
> 👉 https://huggingface.co/spaces/rackette/MakeTikz

MakeTikz est un ensemble d'utilitaires Python/Tkinter permettant de **générer automatiquement du code TikZ/pgfplots** pour différents types de tracés mathématiques, avec **aperçu interactif** avant export.  
L’objectif : préparer rapidement de beaux tracés pour des cours, feuilles d’exercices ou rapports LaTeX, sans écrire à la main les commandes `\addplot`.


## Aperçu

| Tracés symboliques (`plot_tikz_generator.py`) | Interpolation par points (`Lissage.py`) |
|:---:|:---:|
| ![Interface 1](Interface1.png) | ![Interface 2](Interface2.png) |

---

## Deux outils disponibles

### 1. `plot_tikz_generator.py` – Tracés à partir d’expressions symboliques (SymPy)

- **Pour** : fonctions mathématiques explicites (ex. `sin(x)`, `exp(x)`, `Piecewise`, `log(x)`, etc.)
- **Entrée** : deux expressions symboliques en `x`
- **Support d’expressions SymPy courantes**
  - Variable : `x`
  - Fonctions : `sin`, `cos`, `tan`, `exp`, `log`, fonctions hyperboliques, puissances (`x**2`), valeurs absolues, `Piecewise`, etc.
- **Gestion des singularités**
  - Détection des pôles sur le domaine (dénominateur nul)
  - Découpage automatique du domaine en sous-intervalles
  - Génération d’un code pgfplots propre (sauts aux singularités, restrictions sur `y`, etc.)
- **Exécution** :
  ```bash
  python plot_tikz_generator.py
  ```

### 2. `Lissage.py` – Tracés à partir de points (interpolation / spline)

- **Pour** : courbes définies par des points discrétisés (jusqu’à deux courbes simultanées)
- **Entrée** : listes de points  
  - soit `(x y pente)` si vous précisez la pente au point,
  - soit `(x y)` si vous laissez le programme estimer les pentes.
- **Méthodes disponibles** :
  - **Hermite** : interpolation par splines cubiques de Hermite (ne nécessite que NumPy)
  - **Lissage** : spline de lissage (`UnivariateSpline` – nécessite SciPy)
- **Placement interactif** :
  - Affichage des labels de courbe sur la figure Matplotlib
  - Déplacement des labels à la souris
  - Les positions sont intégrées dans le code TikZ via `\draw ... node{...}`.
- **Exécution** :
  ```bash
  python Lissage.py
  ```

---

## Fonctionnalités communes

- **Interface graphique (Tkinter)**
  - Saisie des données :
    - expressions symboliques pour `plot_tikz_generator.py`
    - points `(x y pente)` ou `(x y)` pour `Lissage.py`
  - Choix du label LaTeX pour chaque courbe (ex. `$C_f$`, `$C_g$`, `$f$`, `$g$`)
  - Réglage du style (plein, tirets, pointillés, tiret-point, etc.) et de la couleur
  - Réglage du domaine : `xmin`, `xmax`, `ymin`, `ymax`
  - Réglage des graduations, labels d’axes, échelles, etc.

- **Code TikZ/pgfplots prêt à coller**
  - Génération d’un environnement complet :
    ```tex
    \begin{tikzpicture}
      \begin{axis}[...]
        \addplot[...] coordinates { ... };
      \end{axis}
    \end{tikzpicture}
    ```
  - Paramètres configurables : présence de la grille, distance entre graduations, échelles d’axes (`x`, `y`), nombre d’échantillons, etc.
  - Conversion de `log(...)` en `\ln(...)` dans les labels pour la partie SymPy.

- **Placement interactif des labels**
  - Labels affichés sur la figure Matplotlib
  - Déplacement des labels à la souris
  - Export des positions des labels via des commandes `\draw` dans le code TikZ (optionnel).

---

## Dépendances

- Python 3.x  
- `tkinter`  
- `numpy`  
- `sympy` (pour `plot_tikz_generator.py`)  
- `matplotlib`

**Pour `Lissage.py` (optionnel mais recommandé pour le lissage) :**

- `scipy` (uniquement pour la méthode de lissage via `UnivariateSpline`)

### Installation typique (version « bureau »)

```bash
git clone https://github.com/philipperackette/MakeTikz.git
cd MakeTikz

# Dépendances de base
pip install numpy sympy matplotlib

# Pour utiliser la méthode de lissage dans Lissage.py :
pip install scipy
```

---

## Utilisation rapide

### 1. Tracés à partir d’expressions symboliques (`plot_tikz_generator.py`)

```bash
python plot_tikz_generator.py
```

1. Saisissez vos deux expressions en `x` (ex. `x**2`, `Piecewise((x+1, x<0),(x-1, True))`, `log(x)`).
2. Ajustez le domaine (`xmin`, `xmax`, `ymin`, `ymax`) et les styles de courbe.
3. Cliquez sur **« Tracer / Mettre à jour »** pour visualiser.
4. Cliquez sur **« Générer le code TikZ »** pour obtenir le code pgfplots à coller dans votre document LaTeX.

---

### 2. Tracés à partir de points (`Lissage.py`)

```bash
python Lissage.py
```

1. Saisissez les points de vos courbes dans les zones de texte prévues :
   - **Courbe 1 :** zones `x y pente` (ou `x y`)
   - **Courbe 2 :** idem, si vous souhaitez tracer une deuxième courbe.
2. Choisissez la **méthode** :
   - `hermite` (spline de Hermite, pas besoin de SciPy)
   - `lissage` (spline de lissage, nécessite SciPy)
3. Réglez le domaine (`xmin`, `xmax`, `ymin`, `ymax`), le nombre d’échantillons, les styles de courbes, etc.
4. Cliquez sur **« Tracer / Mettre à jour »** pour l’aperçu.
5. Ajustez éventuellement la position des labels en les déplaçant à la souris.
6. Cliquez sur **« Générer le code TikZ »** pour obtenir le code à coller dans votre document LaTeX.

---

## Formats d’entrée pour `Lissage.py`

Vous pouvez saisir les points de deux façons :

### 1. Points avec pentes explicites

Chaque ligne contient `x y pente` :

```text
x1 y1 m1
x2 y2 m2
...
```

Exemple :

```text
-2  1  0
-1  0  1
0   0  0
1   1  0
2   0 -1
```

### 2. Points sans pentes (pentes estimées automatiquement)

Chaque ligne contient `x y` :

```text
x1 y1
x2 y2
...
```

Les pentes sont alors **estimées automatiquement** par différences finies (centrées lorsque c’est possible).

Vous pouvez également utiliser des séparateurs `,` ou `;` : ils seront interprétés comme des espaces.


---

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).
