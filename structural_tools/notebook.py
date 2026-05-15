import math
import os

import handcalcs
import pandas as pd
from IPython.display import Latex, display


def set_params_columns(num_cols: int):
    handcalcs.set_option("param_columns", num_cols)


def check_value(value: float, check_value: float = 1, inequality: str = "leq"):
    pass_latex = r"\textbf{\color{OK}OK}\ {\color{OK}\checkmark}"
    fail_latex = r"\textbf{{\color{NG}NG !}}"
    outcome = fail_latex
    operator = "ERROR"
    match inequality:
        case "leq" | "<=":
            if value <= check_value:
                outcome = pass_latex
                operator = "\\leq"
            else:
                operator = "\\gt"
        case "lt" | "<":
            if value < check_value:
                outcome = pass_latex
                operator = "\\lt"
            else:
                operator = "\\geq"
        case "geq" | ">=":
            if value >= check_value:
                outcome = pass_latex
                operator = "\\geq"
            else:
                operator = "\\lt"
        case "gt" | ">":
            if value > check_value:
                outcome = pass_latex
                operator = "\\gt"
            else:
                operator = "\\leq"
        case "eq" | "=":
            if value == check_value:
                outcome = pass_latex
                operator = "="
            else:
                operator = "\\neq"
    return f"{value:.3g} {operator} {check_value:.3g} \\quad {outcome}"


def display_table(
    styler: pd.io.formats.style.Styler,
    hrules=True,
    position="H",
    **kwargs,
) -> Latex | None:
    """Displays a table if run in a Jupyter ipynb, or returns LaTeX code if run by nbconvert when exporting to PDF

    Args:
        styler (pd.io.formats.style.Styler): Formatted table styler created by the `create_formatted_table` function
        position_float (str, optional): See pandas documentation for Styler.to_latex(). Defaults to "centering".
        hrules (bool, optional): See pandas documentation for Styler.to_latex(). Defaults to True.
        position (str, optional): See pandas documentation for Styler.to_latex(). Defaults to "H".

    Returns:
        _type_: _description_
    """
    # Need to use the "export_to_*" scripts to set the NBCONVERT environment variable for this to work,
    # otherwise it will just display the dataframe as normal without LaTeX formatting
    if "NBCONVERT" in os.environ:
        return Latex(
            styler.to_latex(
                hrules=hrules,
                position=position,
                **kwargs,
            )
        )
    else:
        display(styler.data)


def sig_figs(x: float, sig_figs: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    if pd.isna(x):
        return ""

    if x == 0:
        return float(0)

    x = float(x)
    sig_figs = int(sig_figs)

    if 1e-3 <= abs(x) < 1e6:
        decimals = sig_figs - int(math.floor(math.log10(abs(x)))) - 1
        decimals = max(decimals, 0)
        return f"{x:.{decimals}f}"

    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig_figs - 1))
