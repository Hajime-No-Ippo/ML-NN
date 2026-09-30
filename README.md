# ML-NN

Coursework for **CS401 Machine Learning & Neural Networks**.

Two notebooks, built around the Andrew Ng *Machine Learning* Programming
Exercise 1 (linear regression), plus an exploratory pass over a real-estate
price dataset.

## Contents

```
Linear_Regression/
  Week2 Ex 1 - Linear Regression-Start.ipynb   Exercise 1, start to finish
  Linear_Regression.ipynb                     EDA on Real estate.csv
input/
  ex1data1.txt         97 x 2    population -> profit
  ex1data2.txt         47 x 3    house size, bedrooms -> price
  Real estate.csv     414 x 8    UCI real-estate valuation
```

### `Week2 Ex 1 - Linear Regression-Start.ipynb`

Implements linear regression by hand, then checks it against scikit-learn.

**One variable** — population vs profit.

- `compute_cost_one_variable` — the cost $J(\theta)$
- `gradient_descent` — converges to $\theta = [-3.6303,\ 1.1664]$
  (Ng's expected values are `[-3.6303, 1.1664]`)
- A contour plot of $J(\theta)$ over a $\theta_0$/\ $\theta_1$ grid
- `sklearn.linear_model.LinearRegression` for comparison — agrees, as it must

**Multiple variables** — house size, bedrooms vs price.

- `feature_normalize` — house sizes are ~1000x the bedroom counts, so the
  features are scaled before gradient descent. Without this the cost surface
  is a narrow valley and the learning rate has to be tiny.
- A sweep over `alpha = [0.3, 0.1, 0.03, 0.01]` to pick a learning rate
- $\theta = [340412.66,\ 110630.27,\ -6648.69]$ after 250 iterations
- `normal_eqn` — the closed-form $\theta = (X^TX)^{-1}X^Ty$, which lands on
  `[89597.91, 139.21, -8738.02]`
- The two methods are compared directly: a 1650 sq-ft, 3-bedroom house is
  ~$293,081 either way (293081.6351 vs 293081.4643)

The recurring point of the exercise is that the hand-rolled gradient descent
and both closed-form solutions agree. The normal-equation $\theta$ differs in
scale from the gradient-descent one because it is fitted on the *unnormalised*
features — same model, different parameterisation.

### `Linear_Regression.ipynb`

Exploratory analysis of `input/Real estate.csv` (414 rows, the UCI real-estate
dataset): `df.info`, a `df.corr()` annotated heatmap, and a `seaborn.pairplot`
over all eight columns.

Useful on its own: transaction date and house price are almost uncorrelated
(-0.049) while distance to the nearest MRT station is one of the stronger
predictors — which is why this dataset is a common exercise.

## Setup

Python 3.10+. The notebooks expect a kernel named `ml_env`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn hvplot jupyter
```

Dependencies actually imported: `numpy`, `pandas`, `matplotlib`, `seaborn`,
`sklearn`, `hvplot`.

`.venv` is gitignored. There is no `requirements.txt` yet.

## Notes

- **Notebook 1, cell 28 is broken.** It reads `data/ex1data2.txt`, but the
  file lives at `input/ex1data2.txt`, so the path should be
  `../input/ex1data2.txt` like every other read in these notebooks. The saved
  output is from a run when the layout was different; re-running from a clean
  checkout will fail there.
- **Notebook 1, cell 0** runs `!source .venv/bin/activate`, which is bash
  syntax in what is otherwise a Python kernel. It works in some setups and
  not others. `uv`'s "Checked 8 packages" output in the saved result suggests
  it was last run under a different setup than this one describes.
- **The two notebooks disagree on Python** — 3.13.13 and 3.10.18 in their
  saved metadata. Nothing here needs 3.13, so 3.10 is the safer pin.
- `.ipynb_checkpoints/` is committed. It is editor scratch and should be
  ignored; the two checkpoint copies of the notebooks are currently tracked.
- Saved outputs are committed alongside the code, so the numbers above can be
  read without running anything.

## License

GPL-3.0 — see [LICENSE](LICENSE).
