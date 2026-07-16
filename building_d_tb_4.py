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

try:
    import handcalcs.render
except AttributeError:
    pass

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
IDENTIFIER = "d6_6"
GENERATE_NEW_IMAGES = True

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

![Tributary diagram of the beam in question](images/d_tb_6_6.png){width=50% #fig:a_tb_1}

We will assume that the top flanges are fully braced by the floor (no LTB).
"""

# %% [markdown]
"""
## Beam

The beam loading and geometry are as follows
"""

# %%
# %%render
w_trib = (fi(11, 6) + fi(21, 4)) / 2 * ft
l_trib = fi(18, 11) * ft
n_levels_2 = 1
n_levels_34 = 2
n_levels = n_levels_2 + n_levels_34

# %%
# %%render
A_trib_2 = w_trib * l_trib * n_levels_2
A_trib_34 = w_trib * l_trib * n_levels_34
A_trib = w_trib * l_trib * n_levels

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

# %% [markdown]
"""
We must also consider the point loads of the joists that support the building overhang. These overhangs are resolved by (a) joists that cantilever the 1'-0" overhang and frame into (b) deeper LVLs that frame into the transfer beam. 

The worst-case downwards point loads given in calculations by others for J-5 are as follows:
"""

# %%
# %%render
D_point = 651 / 1000
L_point = 1353 / 1000
S_point = 320 / 1000


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
base_model.add_load_combo("1a", {"D": 1.4}, ["strength"])
base_model.add_load_combo("2a-1", {"D": 1.2, "L": 1.6, "L_r": 0.5}, ["strength"])
base_model.add_load_combo("2a-2", {"D": 1.2, "L": 1.6, "S": 0.3}, ["strength"])
base_model.add_load_combo("3a-1", {"D": 1.2, "L_r": 1.6, "L": 1.0}, ["strength"])
base_model.add_load_combo("3a-2", {"D": 1.2, "S": 1.0, "L": 1.0}, ["strength"])
base_model.add_load_combo("3a-3", {"D": 1.2, "L_r": 1.6, "W": 0.5}, ["strength"])
base_model.add_load_combo("3a-4", {"D": 1.2, "S": 1.0, "W": 0.5}, ["strength"])
# base_model.add_load_combo("4a-1", {"D": 1.2, "W": 1.0, "L": 1.0, "L_r": 0.5}, ["strength"])
# base_model.add_load_combo("4a-2", {"D": 1.2, "W": 1.0, "L": 1.0, "S": 0.3}, ["strength"])
# base_model.add_load_combo("5a", {"D": 0.9, "W": -1.0}, ["strength"])
# base_model.add_load_combo("6", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0, "S": 0.15}, ["strength"])
# base_model.add_load_combo("7", {"D": 0.9, "E_v": -1.0, "E_h": 1.0}, ["strength"])
base_model.add_load_combo("L", {"L": 1}, ["service"])
base_model.add_load_combo("L_r", {"L_r": 1}, ["service"])
base_model.add_load_combo("S", {"S": 1}, ["service"])
# base_model.add_load_combo("W", {"W": 1}, ["service"])
# base_model.add_load_combo("0.5D+L", {"D": 0.5, "L": 1}, ["service OSSC25 wood"])
# base_model.add_load_combo("CC.2-1a", {"D": 1, "L": 1}, ["service ASCE7-22"])
# base_model.add_load_combo("CC.2-1b", {"D": 1, "S_ser": 1}, ["service ASCE7-22"])
# base_model.add_load_combo("CC.2-2", {"D": 1, "L": 0.5}, ["creep ASCE7-22"])
base_model.add_load_combo("D", {"D": 1}, ["self weight"])
base_model.add_load_combo("all_nominal", {"D": 1, "L": 1, "L_r": 1, "S": 1, "W": 1}, ["all_nominal"])


# %%
tb_1_steel = deepcopy(base_model)

joist_overhang_length = fi(6, 3, return_unit="in")

# Nodes
tb_1_steel.add_node("n1", 0, 0, 0)
tb_1_steel.add_node("n2", float(l_trib.to("inch")), 0, 0)
tb_1_steel.add_node("n3", float(l_trib.to("inch")) - joist_overhang_length, 0, 0)

# Wide flange shape
wf = aisc.W_shapes.W16X31
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
x0 = 0
x_j5 = float(l_trib.to("inch")) - joist_overhang_length

total_dead_load_2 = (n_levels_2 * float(D_line)) / 1000 / 12  # kip/in
total_dead_load_34 = (n_levels_34 * float(D_line)) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_dead_load_2, -total_dead_load_2, x1=x0, x2=x_j5, case="D")
tb_1_steel.add_member_dist_load("m1", "Fy", -total_dead_load_34, -total_dead_load_34, case="D")

wall_dead_load = n_levels * float(D_wall) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -wall_dead_load, -wall_dead_load, case="D")

tb_1_steel.add_member_self_weight("FY", -1, case="D")

total_live_load_2 = n_levels_2 * float(L_line) / 1000 / 12  # kip/in
total_live_load_34 = n_levels_34 * float(L_line) / 1000 / 12  # kip/in
tb_1_steel.add_member_dist_load("m1", "Fy", -total_live_load_2, -total_live_load_2, x1=x0, x2=x_j5, case="L")
tb_1_steel.add_member_dist_load("m1", "Fy", -total_live_load_34, -total_live_load_34, case="L")

# Point loads
tb_1_steel.add_node_load("n3", "FY", -2 * D_point, "D")
tb_1_steel.add_node_load("n3", "FY", -2 * L_point, "L")
tb_1_steel.add_node_load("n3", "FY", -2 * S_point, "S")

# Analyze the beam
tb_1_steel.analyze()

# %%
if GENERATE_NEW_IMAGES:
    # from Pynite.Rendering import Renderer  # or Pynite.Visualization
    import vtk
    from Pynite.Visualization import Renderer  # or Pynite.Visualization

    # Create renderer
    renderer = Renderer(tb_1_steel)

    combo_name = "all_nominal"
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

![All nominal loads](images/d6_6_tb_1_steel_loading_1_span_all_nominal.png){width=100% #fig:tb_a_1_loading_1_span_14D}
"""

# %%
member = tb_1_steel.members["m1"]
moments_list = []
labels_list = []
strength_combos = [combo for combo in tb_1_steel.load_combos.values() if "strength" in combo.combo_tags]
for combo in strength_combos:
    x, M = member.moment_array("Mz", 100, combo_name=combo.name)
    moments_list += [M]
    combo_str = "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
    combo_str = combo_str.replace("+ -", "-")
    labels_list.append(combo_str)

fig, ax = plot_internal_diagram(
    x,
    moments_list,
    labels=labels_list,
    envelope=True,
    figsize=(12, 6),
    xlabel="Location [in]",
    ylabel="Moment [kip-in]",
    title="Moment Diagram",
    save_png=f"images/{IDENTIFIER}_tb_1_moments",
    show_plot=False,
)

M_dem_tuple_max = member.max_moment("Mz", ["strength"])
M_dem_tuple_min = member.min_moment("Mz", ["strength"])

critical_tuple = max(
    [M_dem_tuple_max, M_dem_tuple_min],
    key=lambda t: abs(t[0]),
)

M_dem = abs(critical_tuple[0]) * kip * inch
governing_combo = critical_tuple[1]
M_dem__kipft = M_dem * 1 * ft / 12 / inch

# %% [markdown]
"""
![Moment Diagram](images/d6_6_tb_1_moments.png){#fig:moment_diagram_steel_1_span}

Moment demands are:
"""

# %%
# %%render
M_dem
M_dem__kipft

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
defl_list = []
labels_list = []
strength_combos = [combo for combo in tb_1_steel.load_combos.values() if "service" in combo.combo_tags]
for combo in strength_combos:
    x, defl = member.deflection_array("dy", 100, combo_name=combo.name)
    defl_list += [defl]
    combo_str = "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
    combo_str = combo_str.replace("+ -", "-")
    labels_list.append(combo_str)

fig, ax = plot_internal_diagram(
    x,
    defl_list,
    labels=labels_list,
    envelope=True,
    figsize=(12, 6),
    xlabel="Location [in]",
    ylabel="Deflection [in]",
    title="Deflection Diagram",
    save_png=f"images/{IDENTIFIER}_deflections",
    show_plot=False,
)

delta_tuple_max = member.max_deflection("dy", ["service"])
delta_tuple_min = member.min_deflection("dy", ["service"])

self_weight_max = member.max_deflection("dy", "D")
self_weight_min = member.min_deflection("dy", "D")
self_weight__defl = max(abs(self_weight_max), abs(self_weight_min))

critical_tuple = max(
    [delta_tuple_max, delta_tuple_min],
    key=lambda t: abs(t[0]),
)

governing_combo = critical_tuple[1]

# %% [markdown]
"""
![End beam deflection diagram](images/d6_6_deflections.png){#fig:end_beam_deflection_diagram}
"""

# %%
delta_L_val = tb_1_steel.members["m1"].min_deflection("dy", "L")

# %%
# %%render long
# fmt: off
delta_L = (delta_L_val * inch * -1)
delta_max__L = l_trib.to("inch") / 360
check_delta__L = check_value(delta_L, delta_max__L)
# fmt: on

# %% [markdown]
"""
## Post Design

### HSS Post
"""

# %%
max_reaction = max(
    [
        abs(tb_1_steel.nodes[node].RxnFY[combo])
        for node in tb_1_steel.nodes.keys()
        for combo in tb_1_steel.load_combos.keys()
    ]
)
print(max_reaction)

# %% [markdown]
"""
The following \autoref{tab:hss_capacities} values are the reference design compression capacities of various HSS posts that could be used in this project.
"""

# %%
# %%render
h_level_1 = 13.5 * ft

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
