"""Fit a straight line to CSV measurements with uncertainties in both axes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures even on machines without a graphical display.
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr


REQUIRED_COLUMNS = ("x", "y", "sigma_x", "sigma_y")


def read_measurements(path):
    """Return four arrays, rejecting incomplete or invalid measurement rows."""
    columns = {name: [] for name in REQUIRED_COLUMNS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing CSV column(s): {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {row_number}: too many fields")
            try:
                values = {name: float(row[name]) for name in REQUIRED_COLUMNS}
            except (ValueError, TypeError) as exc:
                raise ValueError(f"row {row_number}: expected four numeric values") from exc
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"row {row_number}: values must be finite")
            if values["sigma_x"] <= 0 or values["sigma_y"] <= 0:
                raise ValueError(f"row {row_number}: uncertainties must be positive")
            for name in REQUIRED_COLUMNS:
                columns[name].append(values[name])

    if len(columns["x"]) < 3:
        raise ValueError("at least three measurements are needed (degrees of freedom = N - 2)")
    if len(set(columns["x"])) < 2:
        raise ValueError("at least two distinct x values are needed")
    return tuple(np.asarray(columns[name]) for name in REQUIRED_COLUMNS)


def fit_line(x, y, sigma_x, sigma_y):
    """Return the ODR output, parameter standard uncertainties, and diagnostics."""
    # The PDFs' ordinary least-squares formulas minimize sum (y_i - m*x_i - b)^2,
    # assuming exact x. Orthogonal distance regression instead finds corrections
    # delta_i to x_i and epsilon_i to y_i that minimize
    # chi^2 = sum [(delta_i/sigma_x_i)^2 + (epsilon_i/sigma_y_i)^2]
    # subject to y_i + epsilon_i = m*(x_i + delta_i) + b (signs are immaterial
    # after squaring). Thus BOTH supplied measurement uncertainties affect the fit.
    model = odr.Model(lambda beta, values: beta[0] * values + beta[1])
    data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
    initial_slope, initial_intercept = np.polyfit(x, y, 1)
    result = odr.ODR(data, model, beta0=[initial_slope, initial_intercept]).run()
    if result.info not in (1, 2, 3):
        raise ValueError(f"fit did not converge: {'; '.join(result.stopreason)}")

    # The diagonal of cov_beta gives the parameter VARIANCES for the supplied
    # absolute sigma_x and sigma_y. Square roots are the 1-sigma uncertainties
    # on m and b. Unlike sd_beta, cov_beta is NOT multiplied by chi^2_reduced;
    # scaling would replace the supplied error-bar sizes with residual scatter.
    variances = np.diag(result.cov_beta)
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("fit could not estimate finite parameter uncertainties")
    sigma_m, sigma_b = np.sqrt(variances)

    # Two fitted parameters (m and b) use two degrees of freedom: nu = N - 2.
    # ODR's sum_square is the minimized chi^2 above; chi^2_reduced = chi^2/nu.
    chi_squared = result.sum_square
    reduced_chi_squared = chi_squared / (len(x) - 2)

    # The lower panel uses VERTICAL residuals r_i = y_i - (m*x_i + b),
    # evaluated at the measured x. These differ from ODR's adjusted distances
    # used in chi^2, which account for uncertainties in BOTH coordinates.
    residuals = y - (result.beta[0] * x + result.beta[1])
    return result.beta, (sigma_m, sigma_b), chi_squared, reduced_chi_squared, residuals


def plot_fit(path, x, y, sigma_x, sigma_y, beta, errors, chi_squared, reduced_chi_squared,
             residuals, x_label, y_label, x_unit, y_unit, title):
    slope, intercept = beta
    sigma_m, sigma_b = errors
    intercept_sign = "+" if intercept >= 0 else "-"
    slope_unit = f" {y_unit}/{x_unit}" if x_unit and y_unit else ""
    intercept_unit = f" {y_unit}" if y_unit else ""
    x_axis = f"{x_label} ({x_unit})" if x_unit else x_label
    y_axis = f"{y_label} ({y_unit})" if y_unit else y_label
    x_note_unit = f" ({x_unit})" if x_unit else ""
    y_note_unit = f" ({y_unit})" if y_unit else ""

    fig = plt.figure(figsize=(11, 7), layout="constrained")
    grid_spec = fig.add_gridspec(3, 2, height_ratios=[1, 3, 1])
    fit_info_ax = fig.add_subplot(grid_spec[0, 0])
    legend_ax = fig.add_subplot(grid_spec[0, 1])
    ax = fig.add_subplot(grid_spec[1, :])
    residual_ax = fig.add_subplot(grid_spec[2, :], sharex=ax)
    fit_info_ax.axis("off")
    legend_ax.axis("off")
    fig.suptitle(title)
    # Error bars show each CSV row's supplied uncertainties on both coordinates.
    ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt="o", capsize=3,
                 label="Measurements with uncertainties")
    grid = np.linspace(np.min(x - sigma_x), np.max(x + sigma_x), 200)
    ax.plot(grid, slope * grid + intercept, color="tab:orange", label="Best-fit line")
    ax.set_ylabel(y_axis)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.25)

    # The same measurement bars appear on the vertical-residual plot. A
    # horizontal zero line makes trends or curvature easy to spot.
    residual_ax.errorbar(x, residuals, xerr=sigma_x, yerr=sigma_y, fmt="o", capsize=3)
    residual_ax.axhline(0, color="tab:orange", linestyle="--")
    residual_ax.set_xlabel(x_axis)
    residual_ax.set_ylabel(f"{y_label} residual" + (f" ({y_unit})" if y_unit else ""))
    residual_ax.grid(alpha=0.25)

    summary = (f"{y_label} = {slope:.5g} {x_label} "
               f"{intercept_sign} {abs(intercept):.5g}\n"
               f"m = {slope:.5g} ± {sigma_m:.2g}{slope_unit}\n"
               f"b = {intercept:.5g} ± {sigma_b:.2g}{intercept_unit}\n"
               f"χ² = {chi_squared:.4g}; χ²/(N-2) = {reduced_chi_squared:.4g}")
    fit_info_ax.text(0, 0.98, summary, transform=fit_info_ax.transAxes,
                     fontsize=10, va="top")
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", borderaxespad=0)
    legend_ax.text(0, 0.48,
                   f"Horizontal: {x_label} uncertainty{x_note_unit}\n"
                   "(propagated systematic and random)\n"
                   f"Vertical: Calculated {y_label.lower()} uncertainty{y_note_unit}",
                   transform=legend_ax.transAxes, fontsize=10, va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="CSV files with x,y,sigma_x,sigma_y columns")
    parser.add_argument("--output-dir", type=Path, help="save plots and text results here (default: beside each CSV)")
    parser.add_argument("--x-label", default="x")
    parser.add_argument("--y-label", default="y")
    parser.add_argument("--x-unit", default="", help="unit of x; used for axis and slope")
    parser.add_argument("--y-unit", default="", help="unit of y; used for axis and parameters")
    args = parser.parse_args()

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.csv:
        try:
            x, y, sx, sy = read_measurements(csv_path)
            beta, errors, chi2, reduced, residuals = fit_line(x, y, sx, sy)
            image_path = (args.output_dir or csv_path.parent) / f"{csv_path.stem}_fit.png"
            plot_fit(image_path, x, y, sx, sy, beta, errors, chi2, reduced,
                     residuals, args.x_label, args.y_label, args.x_unit, args.y_unit,
                     csv_path.stem)
            slope_unit = f" {args.y_unit}/{args.x_unit}" if args.x_unit and args.y_unit else ""
            intercept_unit = f" {args.y_unit}" if args.y_unit else ""
            intercept_sign = "+" if beta[1] >= 0 else "-"
            report = "\n".join([
                f"{csv_path}: {args.y_label} = m {args.x_label} + b",
                f"  {args.y_label} = ({beta[0]:.6g}{slope_unit}) {args.x_label} "
                f"{intercept_sign} {abs(beta[1]):.6g}{intercept_unit}",
                f"  m = {beta[0]:.6g} +/- {errors[0]:.2g}{slope_unit}",
                f"  b = {beta[1]:.6g} +/- {errors[1]:.2g}{intercept_unit}",
                f"  chi^2 = {chi2:.6g}; N - 2 = {len(x) - 2}; reduced chi^2 = {reduced:.6g}",
                f"  plot: {image_path}",
            ]) + "\n"
            image_path.with_suffix(".txt").write_text(report, encoding="utf-8")
        except (OSError, ValueError) as exc:
            parser.error(f"{csv_path}: {exc}")

        print(report, end="")


if __name__ == "__main__":
    main()
