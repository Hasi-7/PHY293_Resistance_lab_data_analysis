# Linear fits with uncertainty in x and y

Fit one or more CSV datasets to `y = m*x + b`. For each file, the tool prints the
slope and intercept with their 1-standard-deviation uncertainties, chi-squared,
degrees of freedom, and reduced chi-squared. It saves a PNG containing the best-fit
line, x and y error bars, and a vertical-residual plot, plus a TXT file with the
same results printed in the terminal.

## Setup and use

```bash
python -m pip install -r requirements.txt
python linear_fit.py measurements.csv --x-label Current --x-unit A --y-label Voltage --y-unit V
```

The resulting files are `measurements_fit.png` and `measurements_fit.txt` beside
the CSV. To process several datasets, pass several filenames; add
`--output-dir plots` to place all outputs in one directory. Run
`python linear_fit.py --help` for the full list of options.

The CSV must contain a header and **one measurement per row**:

```csv
x,y,sigma_x,sigma_y
0.5,1.1,0.02,0.05
1.0,2.0,0.02,0.05
1.5,3.2,0.02,0.05
2.0,4.0,0.02,0.05
```

Use the *same units* within each column. `sigma_x` and `sigma_y` are positive,
absolute **1-sigma** uncertainties (not percentages). Supply your chosen final
uncertainty for each measurement; the *Brief Notes on Linear Fits* recommend
using the greater of the reading and random uncertainties for y error bars and
the reading uncertainty for x. The tool does not infer these from the readings.
At least three rows and two distinct x values are required. Each input CSV is
fitted independently; the CLI axis names/units apply to every file in a run.

## How the calculations work

The three PDFs in this repo motivate a least-squares line, parameter
uncertainties, error bars on both axes, reduced chi-squared, and residuals.
The closed-form slope and intercept and residual-scatter uncertainty formulas
in *Uncertainties in linear fit.pdf* assume x is exact and equal y weighting.
Since these CSVs supply uncertainties in **both** axes, this program instead
uses SciPy orthogonal distance regression (ODR). It minimizes

`chi^2 = sum_i [(delta_x_i/sigma_x_i)^2 + (delta_y_i/sigma_y_i)^2]`

subject to each adjusted point `(x_i + delta_x_i, y_i + delta_y_i)` lying on
`y = m*x + b`. For a straight line with independent errors this is equivalent
to minimizing `sum_i [y_i - (m*x_i + b)]^2 / (sigma_y_i^2 + m^2*sigma_x_i^2)`
over m and b. Errors on x and y are treated as independent across points.
The two reported parameter uncertainties are the square roots of the diagonal
of ODR's **unscaled covariance matrix**, so the input sigmas determine their
absolute scale. No post-fit inflation by the residual scatter is applied.

With `N` rows and two fitted parameters, `reduced chi^2 = chi^2/(N-2)`;
values near one indicate scatter consistent with the supplied uncertainties.
The residual panel shows `y_i - (m*x_i + b)` against the *measured* x_i,
including the original x and y error bars. These vertical residuals are useful
for spotting patterns; chi-squared uses the adjusted two-axis distances above,
not the vertical residuals divided by y uncertainties alone. The plot and
terminal output give the line's parameter values; use the optional unit flags
to label slope in `y_unit/x_unit` and intercept in `y_unit`.
