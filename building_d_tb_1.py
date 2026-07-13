# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: beaverton-creek
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: Building D D-7.3 + D-F.5 Transfer Beam, and Posts
# project: Beaverton Creak Apartments
# location: Beaverton, OR
# client: GBD Architects
# job_number: 10022500446
# author:
#   name: Andy Sazima
#   initials: AGS
#   email: andy.sazima@kpff.com
# ---


# %%
# %reset -f
from copy import deepcopy
from math import sqrt

import forallpeople as units
import handcalcs
import handcalcs.render
import matplotlib.pyplot as plt
import numpy as np

# import polars as pl
import pandas as pd
import sympy as sp
from Pynite import FEModel3D
from steelpy import aisc
from structural_tools.notebook.calculation import check_value, set_params_columns
from structural_tools.notebook.calculation import feet_inches as fi
from structural_tools.notebook.display import display_math, display_table, display_text, sig_figs
from structural_tools.notebook.plotting import plot_internal_diagram, set_plot_style
from structural_tools.steel import calculate_critical_stress

# %%
units.environment(env_name="structural", top_level=True)
handcalcs.set_option("custom_symbols", {"__": ",", "plus": "+", "minus": "-"})
handcalcs.set_option("greek_exclusions", ["psi"])
handcalcs.set_option("display_precision", 4)
handcalcs.set_option("param_columns", 1)
handcalcs.set_option("latex_block_start", "\\begin{equation*}")
handcalcs.set_option("latex_block_end", "\\end{equation*}")
set_plot_style()

# %%
IDENTIFIER = "e1and2"
GENERATE_NEW_IMAGES = False

# %% [markdown]
"""
# Flush beam and post analysis

## Loading

We have the following loading conditions:
"""
# %%
set_params_columns(6)

# %%
# %%render params
DL = 28.5 * psf
LL = 40 * psf

# %% [markdown]
"""
Since we are designing interior beams, live load reduction according to ASCE 7 Section 4.7 will apply. Each beam will have a different reduction factor, but the following applies:

1. Interior beam $\rightarrow K_{LL} = 2$ (ASCE 7 Table 4.7-1)
2. The transfer beams all support at least 2 floors $\rightarrow L_{red} \geq 0.4 L_o = L_{red,limit}$ (ASCE 7 Sec. 4.7.2)
"""

# %%
K_LL = 2
L_red__limit = 0.4 * LL

# %% [markdown]
"""
We will assume that the top flanges are fully braced by the floor (no LTB).

We will first try to design this beam with a single span. If that does not work, we will use 2 spans with a post in the center wall. Dimensions of the problem are shown in \autoref{fig:a_tb_1}.

![Tributary diagram of the beam in question](images/e_tb_1.png){width=50% #fig:a_tb_1}
"""

# %%
set_params_columns(6)

# %%
# %%render params
DL = 28.5 * psf
LL = 40 * psf

# %% [markdown]
"""
Since we are designing interior beams, live load reduction according to ASCE 7 Section 4.7 will apply. Each beam will have a different reduction factor, but the following applies:

1. Interior beam $\rightarrow K_{LL} = 2$ (ASCE 7 Table 4.7-1)
2. The transfer beams all support at least 2 floors $\rightarrow L_{red} \geq 0.4 L_o = L_{red,limit}$ (ASCE 7 Sec. 4.7.2)
"""
# %%
K_LL = 2
L_red__limit = 0.4 * LL

# %% [markdown]
"""
We will assume that the top flanges are fully braced by the floor (no LTB).
"""

# %% [markdown]
"""
## Beam

The beam loading and geometry are as follows
"""

# %%
# %%render
w_trib_2 = (fi(8, 6) + fi(9, 10)) / 2 * ft
w_trib_35 = fi(8, 6) / 2 * ft
l_trib = fi(8, 11) * ft
n_levels_2 = 1
n_levels_35 = 3

# %%
# %%render
A_trib_2 = w_trib_2 * l_trib * n_levels_2
A_trib_35 = w_trib_35 * l_trib * n_levels_35
A_trib = A_trib_2 + A_trib_35

# %% [markdown]
"""
# We can check if the live load can be reduced.
"""

# %%
# %%render
# fmt: off
if A_trib * K_LL >= 400 * ft * ft: L_red = LL * (0.25 + 15 / sqrt(K_LL * A_trib))
else: L_red = LL
# fmt: on

# %%
# %%render
# fmt: off
if L_red >= L_red__limit: L_red = L_red
else: L_red = L_red__limit
# fmt: on

# %% [markdown]
"""
The governing load combinations will be:

Strength

1. $1.4 D$
2. $1.2 D + 1.6 L$

Service

1. $L$
2. $D+L$ where $D=0.5DL$ according to OSSC 2025 for wood beams

This means the load cases are:
"""

# %%
# %%render params
D = DL
L = L_red

# %%
# %%render
D_line_2 = D * w_trib_2
D_line_35 = D * w_trib_35
L_line_2 = L * w_trib_2
L_line_35 = L * w_trib_35

# %% [markdown]
"""
There is also dead load from the interior walls directly on top of the beam.
"""

# %%
# %%render
W_wall = 10 * psf
h_wall = 10.5 * ft
D_wall = W_wall * h_wall

# %% [markdown]
"""
Along with the self weight of the beam, which will be automatically incorporated in the analysis.
"""

# %%
# Initialize FE Model
base_model = FEModel3D()

# Materials
E = 29000  # Modulus of elasticity (ksi)
Fy = 50  # Yield stress (ksi)
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus of elasticity (ksi)
rho = 489 / (12**3) / 1000  # Density (kci)
steel = base_model.add_material(name="steel", E=E, G=G, nu=nu, rho=rho, fy=Fy)

# Load combinations
base_model.add_load_combo("1.4D", {"D": 1.4}, ["strength"])
base_model.add_load_combo("1.2D+1.6L", {"D": 1.2, "L": 1.6}, ["strength"])
base_model.add_load_combo("L", {"L": 1}, ["service"])


# %%
tb_1_steel = deepcopy(base_model)

# Nodes
tb_1_steel.add_node("n1", 0, 0, 0)
tb_1_steel.add_node("n2", float(l_trib.to("inch")), 0, 0)

# Wide flange shape
wf = aisc.W_shapes.W14X26
display_text(f"Try a {wf.name}")

# Add a section with the following properties:
tb_1_steel.add_section(
    "wf",
    A=wf.area,
    Iy=wf.Iy,
    Iz=wf.Ix,
    J=wf.J,
)

# Add a member
tb_1_steel.add_member("m1", "n1", "n2", "steel", "wf")

# Provide simple supports
tb_1_steel.def_support("n1", True, True, True, False, False, False)
tb_1_steel.def_support("n2", False, True, True, True, False, False)

# Uniform loads
total_dead_load = (n_levels_2 * float(D_line_2) + n_levels_35 * float(D_line_35)) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_dead_load, -total_dead_load, case="D")

wall_dead_load = n_levels * float(D_wall) / 1000 / 12  # kip/in
l_wall_1 = fi(6, 6, return_unit="in")
l_wall_2 = fi(3, 9, return_unit="in")
l_wall_3 = fi(3, 4, return_unit="in")
l_gap_1_2 = fi(3, 4, return_unit="in")
l_gap_2_3 = fi(2, 8, return_unit="in")
x0 = 0
x1 = l_wall_1
x2 = l_wall_1 + l_gap_1_2
x3 = l_wall_1 + l_gap_1_2 + l_wall_2
x4 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3
x5 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3 + l_wall_3
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x0,
    x2=x1,
    case="D",
)
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x2,
    x2=x3,
    case="D",
)
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x4,
    x2=x5,
    case="D",
)

tb_1_steel.add_member_self_weight("FY", -1, case="D")

total_live_load = n_levels * float(L_line) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_live_load, -total_live_load, case="L")

# Analyze the beam
tb_1_steel.analyze()

# %%
if GENERATE_NEW_IMAGES:
    # from Pynite.Rendering import Renderer  # or Pynite.Visualization
    import vtk
    from Pynite.Visualization import Renderer  # or Pynite.Visualization

    # Create renderer
    renderer = Renderer(tb_1_steel)

    for combo_name in base_model.load_combos.keys():
        renderer.combo_name = combo_name
        renderer.window.SetSize(1920, 540)

        renderer.theme = "print"
        renderer.annotation_size = 3
        renderer.update()
        renderer.renderer.ResetCamera()
        camera = renderer.renderer.GetActiveCamera()
        camera.Zoom(3.25)
        actors = renderer.renderer.GetActors()
        actors.InitTraversal()
        for i in range(0, actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if i >= 4:
                if actor.GetClassName() == "vtkFollower":
                    actor.GetProperty().SetColor(0, 0.75, 0)  # red labels
        renderer.window.Render()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderer.window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.SetFileName(f"images/{IDENTIFIER}_tb_1_steel_loading_1_span_{combo_name}.png")
        writer.Write()

        # Now that we're done with the render window, finalize it
        renderer.window.Finalize()

    # Save result
    # renderer.screenshot("images/tb_1_steel_loading.png", interact=False, reset_camera=False)

# %% [markdown]
"""
See the beam loading in \autoref{fig:tb_a_1_loading_1_span_14D} through \autoref{fig:tb_a_1_loading_1_span_L}.

- Note that all units are in kips and inches.

![Loading for $1.4D$](images/a_tb_1_steel_loading_1_span_1.4D.png){width=100% #fig:tb_a_1_loading_1_span_14D}

![Loading for $1.2D+1.6L$](images/a_tb_1_steel_loading_1_span_1.2D+1.6L.png){width=100% #fig:tb_a_1_loading_1_span_12D16L}

![Loading for $L$](images/a_tb_1_steel_loading_1_span_L.png){width=100% #fig:tb_a_1_loading_1_span_L}
"""

# %%
member = tb_1_steel.members["m1"]
x, M_14D = member.moment_array("Mz", 100, combo_name="1.4D")
_, M_12D16L = member.moment_array("Mz", 100, combo_name="1.2D+1.6L")
fig, ax = plot_internal_diagram(
    x,
    [M_14D, M_12D16L],
    labels=["1.4D", "1.2D+1.6L"],
    envelope=True,
    xlabel="Location [in]",
    ylabel="Moment [kip-in]",
    title="Moment Diagram",
    save_png=f"images/{IDENTIFIER}_tb_1_steel_moments_1_span",
    show_plot=False,
)

M_dem, _ = member.min_moment("Mz", ["strength"])
M_dem = M_dem * -1 * kip * inch
M_dem__kipft = M_dem * 1 * ft / 12 / inch * kip

# %% [markdown]
"""
![Moment Diagram](images/a_tb_1_steel_moments_1_span.png){#fig:moment_diagram_steel_1_span}

Moment demands are:
"""

# %%
display_math(
    [
        "M_{dem} &= " + str(sig_figs(M_dem, 3)) + "\\textrm{ kip-in}",
        "M_{dem,kipft} &= " + str(sig_figs(M_dem__kipft, 3)) + "\\textrm{ kip-ft}",
    ]
)

# %% [markdown]
"""
We will check the maximum bending moment from strength-based load combinations against the strength-based moment capacity.
"""
# %%
# %%render params
phi_b = 0.9
F_y = base_model.materials["steel"].fy * ksi
Z_x = wf.Zx * inch**3

# %%
# %%render
M_cap = phi_b * F_y * Z_x
M_cap__kipft = M_cap * 1 * ft / (12 * inch)

# %%
# %%render
DCR = M_dem / M_cap
check_DCR = check_value(DCR, 1, "<=")

# %% [markdown]
"""
We will also check deflection against a limit of $l/360$ as it is the governing case for this beam that supports both the floor and ceiling on level 2.
"""

# %%
delta_L_val = tb_1_steel.members["m1"].min_deflection("dy", "L")

# %%
# %%render long
"""

# %% [markdown]
"""
# fmt: off
delta_L = (delta_L_val * inch * -1)
delta_max__L = l_trib.to("inch") / 360
check_delta__L = check_value(delta_L, delta_max__L)
# fmt: on

# %% [markdown]
"""
## Post Design

### 6x6 Timber Post

We will now want to check that the 6x6 wood posts are adequate.

The design assumptions we will use are:

1. non-eccentric loading
2. Douglas Fir No. 1
3. Dry

The loading stress on the post, $f_c$, is
"""

# %%
max_reaction = max(
    [
        abs(tb_1_steel.nodes[node].RxnFY[combo])
        for node in tb_1_steel.nodes.keys()
        for combo in tb_1_steel.load_combos.keys()
    ]
)

# %%
# %%render
b = 5.5 * inch
d = 5.5 * inch
A = b * d
P = max_reaction * kip
f_c = P / A


# %% [markdown]
"""
We must adjust the design compressive value, $F_c$
"""

# %%
set_params_columns(3)

# %%
# %%render params
C_D = 1.0
C_M = 1.0
C_t = 1.0
C_F = 1.0
C_i = 1.0
C_fu = 1.0
C_T = 1.0

F_c = 1000 * psi
E_min = 580_000 * psi

# %% [markdown]
"""
The column stability factor, $C_P$, is
"""

# %%
# %%render
F_star_C = F_c * C_D * C_M * C_t * C_F * C_i
E_prime_min = E_min * C_M * C_t * C_fu * C_i * C_T

# %%
# %%render
# fmt: off
c = 0.8  # sawn lumber
h_level_1 = 11.5 * ft
K_e = 1  # pinned ends
l_e = (K_e * h_level_1.to("inch"))
check_le__d = check_value(l_e / d, 50, "<=")
F_cE = (0.822 * E_prime_min) / (l_e / d) ** 2
# fmt: on

# %%
C_P = (1 + (F_cE / F_star_C)) / (2 * c) - sqrt(((1 + (F_cE / F_star_C)) / (2 * c)) ** 2 - (F_cE / F_star_C) / c)

# %% [markdown]
"""
The fully adjusted compressive design value is
"""

# %%
# %%render
F_prime_C = F_star_C * C_P

# %%
# %%render
check_compression = check_value(f_c, F_prime_C, "<=")

# %%
if "NG" in check_compression:
    display_text("A 6x6 post will not work--we need to use HSS")

# %% [markdown]
"""
### HSS Post
"""

# %% [markdown]
"""
The following \autoref{tab:hss_capacities} values are the reference design compression capacities of various HSS posts that could be used in this project.
"""

# %%
hss_shape_list = [
    "HSS3_1_2X3_1_2X1_8",
    "HSS4X4X1_8",
    "HSS4X4X3_16",
    "HSS4X4X1_4",
    "HSS4_1_2X4_1_2X1_4",
    "HSS5X5X1_4",
    "HSS5X5X5_16",
    "HSS5X5X3_8",
]
capacities = pd.DataFrame({}, index=hss_shape_list)
capacities.index.name = "Shape"
K = 1.0
L_c = float(h_level_1.to("inch") * K)
L_c = 10.75 * 12
E = 29000
F_y = 50
phi_c = 0.9
for shape in hss_shape_list:
    hss = aisc.HSS_shapes.sections[shape]
    r = hss.rx
    A = hss.area
    F_cr = calculate_critical_stress(yield_stress=F_y, youngs_modulus=E, effective_length=L_c, radius_of_gyration=r)
    capacities.loc[shape, "adjusted capacity"] = phi_c * F_cr * A
    capacities.loc[shape, "radius of gyration"] = r
    capacities.loc[shape, "gross area"] = A
    capacities.loc[shape, "slenderness ratio"] = L_c / r
    capacities.loc[shape, "critical buckling stress"] = F_cr

column_name_map = {"adjusted capacity": r"$\phi_c P_n$ [kip]"}
display_table(
    dataframe=capacities,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Common HSS shape capacities",
    label="tab:hss_capacities",
)

# %%
minimum_viable_shape = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].idxmin()
minimum_viable_capacity = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].min()
display_text(rf"The minimum viable HSS shape is {minimum_viable_shape}")
display_math(
    [
        rf"P_u = {sig_figs(max_reaction, 3)} \text{{ kip}}",
        rf"\phi_c P_n = {sig_figs(minimum_viable_capacity, 3)} \text{{ kip}}",
    ]
)

# %% [markdown]
"""
# Two Spans
"""

# %%
set_params_columns(6)

# %%
# %%render params
DL = 28.5 * psf
LL = 40 * psf

# %% [markdown]
"""
Since we are designing interior beams, live load reduction according to ASCE 7 Section 4.7 will apply. Each beam will have a different reduction factor, but the following applies:

1. Interior beam $\rightarrow K_{LL} = 2$ (ASCE 7 Table 4.7-1)
2. The transfer beams all support at least 2 floors $\rightarrow L_{red} \geq 0.4 L_o = L_{red,limit}$ (ASCE 7 Sec. 4.7.2)
"""

# %%
K_LL = 2
L_red__limit = 0.4 * LL

# %% [markdown]
"""
We will assume that the top flanges are fully braced by the floor (no LTB).
"""

# %% [markdown]
"""
## Beam design
"""

# %% [markdown]
"""
The beam loading and geometry are as follows
"""

# %%
# %%render
w_trib = fi(25, 7) / 2 * ft
l_trib_1 = fi(6 + 3, 6 + 4) * ft
l_trib_2 = fi(3 + 2 + 3, 9 + 8 + 4) * ft
l_trib = max(l_trib_1, l_trib_2)
n_levels = 3

# %%
# %%render
A_trib = w_trib * l_trib * n_levels

# %% [markdown]
"""
We can check if the live load can be reduced.
"""

# %%
# %%render
# fmt: off
if A_trib * K_LL >= 400 * ft * ft: L_red = LL * (0.25 + 15 / sqrt(K_LL * A_trib))
else: L_red = LL
# fmt: on

# %%
# %%render
# fmt: off
if L_red >= L_red__limit: L_red = L_red
else: L_red = L_red__limit
# fmt: on

# %% [markdown]
"""
The beam at A-7.5 and A-A resolves the load from levels 2, 3, and 4 (not the roof); therefore, the load combinations are as follows:

Strength

1. $1.4 D$
2. $1.2 D + 1.6 L$

Service

1. $L$
"""

# %% [markdown]
"""
This means the load cases are:
"""

# %%
# %%render params
D = DL
L = L_red

# %%
# %%render
D_line = D * w_trib
L_line = L * w_trib

# %% [markdown]
"""
There is also dead load from the interior walls directly on top of the beam.
"""

# %%
# %%render
W_wall = 10 * psf
h_wall = 10.5 * ft
D_wall = W_wall * h_wall

# %% [markdown]
"""
Along with the self weight of the beam, which will be automatically incorporated in the analysis.
"""

# %%
# Initialize FE Model
base_model = FEModel3D()

# Materials
E = 29000  # Modulus of elasticity (ksi)
Fy = 50  # Yield stress (ksi)
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus of elasticity (ksi)
rho = 489 / (12**3) / 1000  # Density (kci)
steel = base_model.add_material(name="steel", E=E, G=G, nu=nu, rho=rho, fy=Fy)

E = 1700  # Modulus of elasticity (ksi)
Fy = 2.400  # Yield stress (ksi)
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus of elasticity (ksi)
rho = 33 / (12**3) / 1000  # Density (kci)
glulam = base_model.add_material(name="glulam", E=E, G=G, nu=nu, rho=rho, fy=Fy)

# Load combinations
base_model.add_load_combo("1.4D", {"D": 1.4}, ["strength"])
base_model.add_load_combo("1.2D+1.6L", {"D": 1.2, "L": 1.6}, ["strength"])
base_model.add_load_combo("L", {"L": 1}, ["service"])
base_model.add_load_combo("0.5D+L", {"D": 0.5, "L": 1}, ["service wood"])


# %% [markdown]
"""
### Steel
"""

# %%
tb_1_steel = deepcopy(base_model)

# Nodes
tb_1_steel.add_node("n1", 0, 0, 0)
tb_1_steel.add_node("n3", float(l_trib_1.to("inch")), 0, 0)
tb_1_steel.add_node("n2", float(l_trib_1.to("inch") + l_trib_2.to("inch")), 0, 0)

# Provide simple supports
tb_1_steel.def_support("n1", True, True, True, False, False, False)
tb_1_steel.def_support("n2", False, True, True, True, False, False)
tb_1_steel.def_support("n3", False, True, False, False, False, False)

# Wide flange shape
wf = aisc.W_shapes.W14X26
display_text(f"Try a {wf.name}")

# Add a section with the following properties:
tb_1_steel.add_section(
    "wf",
    A=wf.area,
    Iy=wf.Iy,
    Iz=wf.Ix,
    J=wf.J,
)

# Add a member
tb_1_steel.add_member("m1", "n1", "n2", "steel", "wf")

# Uniform loads
total_dead_load = n_levels * float(D_line) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_dead_load, -total_dead_load, case="D")

wall_dead_load = n_levels * float(D_wall) / 1000 / 12  # kip/in
l_wall_1 = fi(6, 6, return_unit="in")
l_wall_2 = fi(3, 9, return_unit="in")
l_wall_3 = fi(3, 4, return_unit="in")
l_gap_1_2 = fi(3, 4, return_unit="in")
l_gap_2_3 = fi(2, 8, return_unit="in")
x0 = 0
x1 = l_wall_1
x2 = l_wall_1 + l_gap_1_2
x3 = l_wall_1 + l_gap_1_2 + l_wall_2
x4 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3
x5 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3 + l_wall_3
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x0,
    x2=x1,
    case="D",
)
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x2,
    x2=x3,
    case="D",
)
tb_1_steel.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x4,
    x2=x5,
    case="D",
)

tb_1_steel.add_member_self_weight("FY", -1, case="D")

total_live_load = n_levels * float(L_line) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_live_load, -total_live_load, case="L")

# Analyze the beam
tb_1_steel.analyze()

# %%
if GENERATE_NEW_IMAGES:
    # from Pynite.Rendering import Renderer  # or Pynite.Visualization
    import vtk
    from Pynite.Visualization import Renderer  # or Pynite.Visualization

    # Create renderer
    renderer = Renderer(tb_1_steel)

    for combo_name in base_model.load_combos.keys():
        renderer.combo_name = combo_name
        renderer.window.SetSize(1920, 540)

        renderer.theme = "print"
        renderer.annotation_size = 3
        renderer.update()
        renderer.renderer.ResetCamera()
        camera = renderer.renderer.GetActiveCamera()
        camera.Zoom(3.25)
        actors = renderer.renderer.GetActors()
        actors.InitTraversal()
        for i in range(0, actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if i >= 4:
                if actor.GetClassName() == "vtkFollower":
                    actor.GetProperty().SetColor(0, 0.75, 0)  # red labels
        renderer.window.Render()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renderer.window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.SetFileName(f"images/{IDENTIFIER}_tb_1_steel_loading_2_span_{combo_name}.png")
        writer.Write()

        # Now that we're done with the render window, finalize it
        renderer.window.Finalize()

    # Save result
    # renderer.screenshot("images/tb_1_steel_loading.png", interact=False, reset_camera=False)

# %% [markdown]
"""
See the beam loading in \autoref{fig:tb_a_1_loading_2_span_14D} through \autoref{fig:tb_a_1_loading_2_span_L}.

- Note that all units are in kips and inches.

![Loading for $1.4D$](images/a_tb_1_steel_loading_2_span_1.4D.png){width=100% #fig:tb_a_1_loading_2_span_14D}

![Loading for $1.2D+1.6L$](images/a_tb_1_steel_loading_2_span_1.2D+1.6L.png){width=100% #fig:tb_a_1_loading_2_span_12D16L}

![Loading for $L$](images/a_tb_1_steel_loading_2_span_L.png){width=100% #fig:tb_a_1_loading_2_span_L}
"""

# %%
member = tb_1_steel.members["m1"]
x, M_14D = member.moment_array("Mz", 500, combo_name="1.4D")
_, M_12D16L = member.moment_array("Mz", 500, combo_name="1.2D+1.6L")
fig, ax = plot_internal_diagram(
    x,
    [M_14D, M_12D16L],
    labels=["1.4D", "1.2D+1.6L"],
    envelope=True,
    xlabel="Location [in]",
    ylabel="Moment [kip-in]",
    title="Moment Diagram",
    save_png="images/a_tb_1_steel_moments_2_span",
    show_plot=False,
)
M_dem_min, _ = member.min_moment("Mz", ["strength"])
M_dem_max, _ = member.max_moment("Mz", ["strength"])
M_dem = max(abs(M_dem_min), abs(M_dem_max)) * kip * inch
M_dem__kipft = M_dem * 1 * ft / 12 / inch * kip

# %% [markdown]
"""
![Moment Diagram](images/a_tb_1_steel_moments_2_span.png){#fig:moment_diagram_steel_2_span}
"""

# %% [markdown]
"""
Moment demands are:
"""

# %%
display_math(
    [
        "M_{dem} &= " + str(sig_figs(M_dem, 3)) + "\\textrm{ kip-in}",
        "M_{dem,kipft} &= " + str(sig_figs(M_dem__kipft, 3)) + "\\textrm{ kip-ft}",
    ]
)

# %% [markdown]
"""
We will check the maximum bending moment from strength-based load combinations against the strength-based moment capacity.
"""

# %%
# %%render params
phi_b = 0.9
F_y = base_model.materials["steel"].fy * ksi
Z_x = wf.Zx * inch**3

# %%
# %%render
M_cap = phi_b * F_y * Z_x
M_cap__kipft = M_cap * 1 * ft / (12 * inch)

# %%
# %%render
DCR = M_dem / M_cap
check_DCR = check_value(DCR, 1, "<=")

# %% [markdown]
"""
We will also check deflection against a limit of $l/360$ as it is the governing case for this beam that supports both the floor and ceiling on level 2.
"""

# %%
m1_d_max = abs(tb_1_steel.members["m1"].max_deflection("dy", "L"))
m1_d_min = abs(tb_1_steel.members["m1"].min_deflection("dy", "L"))
delta_L_val = max([m1_d_max, m1_d_min])

# %%
# %%render long
# fmt: off
delta_L = (delta_L_val * inch)
delta_max__L = l_trib.to("inch") / 360
check_delta__L = check_value(delta_L, delta_max__L)
# fmt: on

# %% [markdown]
"""
### Glulam
"""

# %%
tb_1_glulam = deepcopy(base_model)

# Nodes
tb_1_glulam.add_node("n1", 0, 0, 0)
tb_1_glulam.add_node("n3", float(l_trib_1.to("inch")), 0, 0)
tb_1_glulam.add_node("n2", float(l_trib_1.to("inch") + l_trib_2.to("inch")), 0, 0)

# Provide simple supports
tb_1_glulam.def_support("n1", True, True, True, False, False, False)
tb_1_glulam.def_support("n2", False, True, True, True, False, False)
tb_1_glulam.def_support("n3", False, True, False, False, False, False)

# Wide flange shape
display_text("Try a 5.5x11-7/8 24F-V4 glulam")
b_glulam = 5.5
d_glulam = 11.875  # minimum that works by trial and error (11.25 is close)

# Add a section with the following properties:
tb_1_glulam.add_section(
    "glulam_5.5_11.875",
    A=b_glulam * d_glulam,
    Iz=1 / 12 * b_glulam * d_glulam**3,
    Iy=1 / 12 * d_glulam * b_glulam**3,
    J=1,
)

# Add a member
tb_1_glulam.add_member("m1", "n1", "n2", "glulam", "glulam_5.5_11.875")

# Uniform loads
total_dead_load = n_levels * float(D_line) / 1000 / 12  # kip/in
tb_1_glulam.add_member_dist_load("m1", "Fy", -total_dead_load, -total_dead_load, case="D")

wall_dead_load = n_levels * float(D_wall) / 1000 / 12  # kip/in
l_wall_1 = fi(6, 6, return_unit="in")
l_wall_2 = fi(3, 9, return_unit="in")
l_wall_3 = fi(3, 4, return_unit="in")
l_gap_1_2 = fi(3, 4, return_unit="in")
l_gap_2_3 = fi(2, 8, return_unit="in")
x0 = 0
x1 = l_wall_1
x2 = l_wall_1 + l_gap_1_2
x3 = l_wall_1 + l_gap_1_2 + l_wall_2
x4 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3
x5 = l_wall_1 + l_gap_1_2 + l_wall_2 + l_gap_2_3 + l_wall_3
tb_1_glulam.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x0,
    x2=x1,
    case="D",
)
tb_1_glulam.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x2,
    x2=x3,
    case="D",
)
tb_1_glulam.add_member_dist_load(
    "m1",
    "Fy",
    -wall_dead_load,
    -wall_dead_load,
    x1=x4,
    x2=x5,
    case="D",
)

tb_1_glulam.add_member_self_weight("FY", -1, case="D")

total_live_load = n_levels * float(L_line) / 1000 / 12  # kip/in
tb_1_glulam.add_member_dist_load("m1", "Fy", -total_live_load, -total_live_load, case="L")

# Analyze the beam
tb_1_glulam.analyze()

# %%
member = tb_1_glulam.members["m1"]
x, M_14D = member.moment_array("Mz", 500, combo_name="1.4D")
_, M_12D16L = member.moment_array("Mz", 500, combo_name="1.2D+1.6L")
fig, ax = plot_internal_diagram(
    x,
    [M_14D, M_12D16L],
    labels=["1.4D", "1.2D+1.6L"],
    envelope=True,
    xlabel="Location [in]",
    ylabel="Moment [kip-in]",
    title="Moment Diagram",
    save_png=f"images/{IDENTIFIER}_tb_1_glulam_moments_2_span",
    show_plot=False,
)
M_dem_min, _ = member.min_moment("Mz", ["strength"])
M_dem_max, _ = member.max_moment("Mz", ["strength"])
M_dem = max(abs(M_dem_min), abs(M_dem_max)) * kip * inch
M_dem__kipft = M_dem * 1 * ft / 12 / inch * kip

# %% [markdown]
"""
![Moment Diagram](images/a_tb_1_steel_moments_1_span.png){#fig:moment_diagram_glulam_2_span}

Moment demands are:
"""

# %%
display_math(
    [
        "M_{dem} &= " + str(sig_figs(M_dem, 3)) + "\\textrm{ kip-in}",
        "M_{dem,kipft} &= " + str(sig_figs(M_dem__kipft, 3)) + "\\textrm{ kip-ft}",
    ]
)

# %%
min_results = [
    (abs(val), val, combo)
    for val, combo in (tb_1_glulam.members[m].min_moment("Mz", ["strength"]) for m in tb_1_glulam.members)
]

max_results = [
    (abs(val), val, combo)
    for val, combo in (tb_1_glulam.members[m].max_moment("Mz", ["strength"]) for m in tb_1_glulam.members)
]

_, M_min, min_combo = max(min_results)
_, M_max, max_combo = max(max_results)

# overall governing (by absolute value)
if abs(M_min) > abs(M_max):
    M_dem = M_min
    governing_combo = min_combo
else:
    M_dem = M_max
    governing_combo = max_combo

M_dem = M_dem * kip * inch

display_text(f"From Load Combination: {governing_combo}")

# %% [markdown]
"""
We will check the maximum bending moment from strength-based load combinations against the strength-based moment capacity.
"""

# %%
set_params_columns(2)

# %%
# %%render params
F_b = 2.4 * ksi
C_M = 1.0
C_t = 1.0
C_L = 1.0
C_fu = 1.0
C_c = 1.0
C_I = 1.0
K_f = 2.54
phi = 0.85
lamb = 0.8  # NDS 2024 Supplement N.3.3


# %% [markdown]
"""
The volume factor (NDS 2024 5.3.6) is calculated as:
"""


# %%
def moment_zero(member, combo, n=400):
    x, M = member.moment_array("Mz", n, combo)

    x = np.array(x)
    M = np.array(M)

    idx = np.where(M[:-1] * M[1:] < 0)[0]

    return [x[i] - M[i] * (x[i + 1] - x[i]) / (M[i + 1] - M[i]) for i in idx]


x_zero__moment = moment_zero(tb_1_glulam.members["m1"], "1.2D+1.6L")

# %%
# %%render
L = x_zero__moment[0] / 12
d = d_glulam
b = b_glulam
x = 10  # for non-southern pine species
C_V__init = (21 / L) ** (1 / x) * (12 / d) ** (1 / x) * (5.125 / b) ** (1 / x)
C_V = min(1, C_V__init)

# %%
# %%render
F_prime_b = F_b * C_M * C_t * min(C_L, C_V) * C_fu * C_c * C_I * K_f * phi * lamb

# %%
# %%render
I = 1 / 12 * b_glulam * d_glulam**3
S = I / (d_glulam / 2) * inch**3
f_b = M_dem / S
check_DCR = check_value(f_b, F_prime_b, "<=")

# %% [markdown]
"""
We will also check deflection against a limit of $l/360$ as it is the governing case for this beam that supports both the floor and ceiling on level 2.
"""

# %%
m1_d_max = abs(tb_1_glulam.members["m1"].max_deflection("dy", "L"))
m1_d_min = abs(tb_1_glulam.members["m1"].min_deflection("dy", "L"))
delta_L_val = max([m1_d_max, m1_d_min])

# %%
# %%render long
# fmt: off
delta_L = (delta_L_val * inch)
delta_max__L = l_trib.to("inch") / 360
check_delta__L = check_value(delta_L, delta_max__L)
# fmt: on

# %% [markdown]
"""
## Post Design
"""

# %% [markdown]
"""
### 6x6 Timber Post
"""

# %% [markdown]
"""
We will now want to check that the 6x6 wood posts are adequate.

The design assumptions we will use are:

1. non-eccentric loading
2. Douglas Fir No. 1
3. Dry

The loading stress on the post, $f_c$, is
"""

# %%
max_reaction = max(
    [
        abs(tb_1_steel.nodes[node].RxnFY[combo])
        for node in tb_1_steel.nodes.keys()
        for combo in tb_1_steel.load_combos.keys()
    ]
)
end_reaction = max(
    [abs(tb_1_steel.nodes[node].RxnFY[combo]) for node in ["n1", "n2"] for combo in tb_1_steel.load_combos.keys()]
)

# %%
# %%render
b = 5.5 * inch
d = 5.5 * inch
A = b * d
P_midspan = max_reaction * kip
P_endpost = end_reaction * kip
f_c__midspan = P_midspan / A
f_c__endpost = P_endpost / A


# %%
set_params_columns(3)

# %% [markdown]
"""
We must adjust the design compressive value, $F_c$
"""

# %%
# %%render params
C_D = 1.0
C_M = 1.0
C_t = 1.0
C_F = 1.0
C_i = 1.0
C_fu = 1.0
C_T = 1.0

F_c = 1000 * psi
E_min = 580_000 * psi

# %% [markdown]
"""
The column stability factor, $C_P$, is
"""

# %%
# %%render
F_star_C = F_c * C_D * C_M * C_t * C_F * C_i
E_prime_min = E_min * C_M * C_t * C_fu * C_i * C_T

# %%
# %%render
# fmt: off
c = 0.8  # sawn lumber
h_level_1 = 11.5 * ft
K_e = 1  # pinned ends
l_e = (K_e * h_level_1.to("inch"))
check_le__d = check_value(l_e / d, 50, "<=")
F_cE = (0.822 * E_prime_min) / (l_e / d) ** 2
# fmt: on

# %%
C_P = (1 + (F_cE / F_star_C)) / (2 * c) - sqrt(((1 + (F_cE / F_star_C)) / (2 * c)) ** 2 - (F_cE / F_star_C) / c)

# %% [markdown]
"""
The fully adjusted compressive design value is
"""

# %%
# %%render
F_prime_C = F_star_C * C_P

# %%
# %%render
check_c__midspan = check_value(f_c__midspan, F_prime_C, "<=")
check_c__endpost = check_value(f_c__endpost, F_prime_C, "<=")

# %%
if "NG" in check_c__midspan:
    display_text("A 6x6 post will not work at the midspan--we need to use HSS")
else:
    display_text("A 6x6 post will work at the midspan")
if "NG" in check_c__endpost:
    display_text("A 6x6 post will not work at the endpost--we need to use HSS")
else:
    display_text("A 6x6 post will work at the endpost")

# %% [markdown]
"""
### HSS Post
"""

# %% [markdown]
"""
The following \autoref{tab:hss_capacities} values are the reference design compression capacities of various HSS posts that could be used in this project.
"""

# %%
hss_shape_list = [
    "HSS3_1_2X3_1_2X1_8",
    "HSS4X4X1_8",
    "HSS4X4X3_16",
    "HSS4X4X1_4",
    "HSS4_1_2X4_1_2X1_4",
    "HSS5X5X1_4",
    "HSS5X5X5_16",
    "HSS5X5X3_8",
]
capacities = pd.DataFrame({}, index=hss_shape_list)
capacities.index.name = "Shape"
K = 1.0
L_c = float(h_level_1.to("inch") * K)
L_c = 10.75 * 12
E = 29000
F_y = 50
phi_c = 0.9
for shape in hss_shape_list:
    hss = aisc.HSS_shapes.sections[shape]
    r = hss.rx
    A = hss.area
    F_cr = calculate_critical_stress(yield_stress=F_y, youngs_modulus=E, effective_length=L_c, radius_of_gyration=r)
    capacities.loc[shape, "adjusted capacity"] = phi_c * F_cr * A
    capacities.loc[shape, "radius of gyration"] = r
    capacities.loc[shape, "gross area"] = A
    capacities.loc[shape, "slenderness ratio"] = L_c / r
    capacities.loc[shape, "critical buckling stress"] = F_cr

column_name_map = {"adjusted capacity": r"$\phi_c P_n$ [kip]"}
display_table(
    dataframe=capacities,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Common HSS shape capacities",
    label="tab:hss_capacities",
)

# %%
minimum_viable_shape = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].idxmin()
minimum_viable_capacity = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].min()
display_text(rf"The minimum viable HSS shape is {minimum_viable_shape}")
display_math(
    [
        rf"P_u = {sig_figs(max_reaction, 3)} \text{{ kip}}",
        rf"\phi_c P_n = {sig_figs(minimum_viable_capacity, 3)} \text{{ kip}}",
    ]
)
