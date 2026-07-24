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
import os
from copy import deepcopy
from math import sqrt

import forallpeople as units
import handcalcs
from IPython.display import Markdown, display

try:
    import handcalcs.render
except AttributeError:
    pass

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# import polars as pl
import pandas as pd
import structural_tools.model as model
import sympy as sp
from Pynite import FEModel3D
from steelpy import aisc
from structural_tools.model import (
    LoadCase,
    add_named_member_dist_load,
    add_named_member_pt_load,
    add_named_node_load,
    add_named_self_weight,
    create_base_model,
    get_load_combos_from_tags,
)
from structural_tools.notebook.calculation import check_value, set_params_columns
from structural_tools.notebook.calculation import feet_inches as fi
from structural_tools.notebook.display import (
    display_figure,
    display_markdown,
    display_math,
    display_table,
    display_text,
    sig_figs,
)
from structural_tools.notebook.plotting import plot_internal_diagram, plot_member_loads, set_plot_style
from structural_tools.notebook.runtime import image_path, runtime
from structural_tools.steel import Section, calculate_critical_stress

# %%
units.environment(env_name="structural", top_level=True)
handcalcs.set_option("custom_symbols", {"__": ",", "plus": "+", "minus": "-"})
handcalcs.set_option("greek_exclusions", ["psi"])
handcalcs.set_option("display_precision", 4)
handcalcs.set_option("param_columns", 1)
handcalcs.set_option("latex_block_start", "\\begin{equation*}")
handcalcs.set_option("latex_block_end", "\\end{equation*}")
set_plot_style()

#######################################################################################################################
############################################## USER INPUT #############################################################
BEAM_SELECTION = "W10x19"
#######################################################################################################################
#######################################################################################################################

IDENTIFIER = runtime.identifier
PATH = runtime.path
MEMBER_NAME = "m_1"


# %% [markdown]
"""
# Beam and post analysis

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

Dimensions of the problem are shown in \autoref{fig:tributary_diagram}. Note that the framing plan for levels 3 and up are different. It is shown in \autoref{fig:level_3_framing}
"""

# %%
display_figure(
    runtime.image_path("tributary_diagram"), "Tributary diagram of the beam", "fig:tributary_diagram", width="70%"
)
display_figure(
    runtime.image_path("level_3_framing"), "Framing of levels 3 and above", "fig:level_3_framing", width="70%"
)


# %% [markdown]
"""
## Beam

The beam loading and geometry are as follows
"""

# %% [markdown]
"""
Tributary widths:
"""

# %%
# %%render
w_trib__1 = fi(24, 9) / 2 * ft
w_trib__2 = fi(8, 11) / 2 * ft
w_trib__3 = fi(18, 0) / 2 * ft
w_trib__4 = fi(20, 5) / 2 * ft

# %% [markdown]
"""
Tributary lengths:
"""

# %%
# %%render
l_trib__1 = fi(29, 5) * ft
l_trib__2 = fi(8, 11) * ft
l_trib__3 = fi(11, 10) * ft
l_trib__4 = fi(8, 8) * ft
l_trib = l_trib__2 + l_trib__3 + l_trib__4

# %% [markdown]
"""
Level numbers:
"""

# %%
# %%render
n_levels__2 = 1
n_levels__35 = 3
n_levels = n_levels__2 + n_levels__35

# %%
assert l_trib == l_trib__1, "The total beam length must match the addition of all trib segments."


# %% [markdown]
"""
Tributary areas:
"""

# %%
# %%render
A_trib__1 = l_trib__1 * w_trib__1
A_trib__2 = l_trib__2 * w_trib__2
A_trib__3 = l_trib__3 * w_trib__3
A_trib__4 = l_trib__4 * w_trib__4
A_trib = (A_trib__1 + A_trib__2 + A_trib__3 + A_trib__4) * n_levels

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
### Load Cases

The basic uniform load cases are:
"""

# %%
# %%render params
D = DL
L = L_red

# %%
# %%render
D_line__seg1 = D * w_trib__1
D_line__seg2 = D * w_trib__2
D_line__seg3 = D * w_trib__3
D_line__seg4 = D * w_trib__4
L_line__seg1 = L * w_trib__1
L_line__seg2 = L * w_trib__2
L_line__seg3 = L * w_trib__3
L_line__seg4 = L * w_trib__4

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
D_point = 2 * 651 / 1000
L_point = 2 * 1353 / 1000
S_point = 2 * 320 / 1000

# %%
beam = create_base_model(
    include_ossc_creep_combos=False,
    load_cases=[LoadCase.DEAD, LoadCase.LIVE, LoadCase.SNOW],
)

joist_overhang_length = fi(6, 4, return_unit="in")
x_start = 0
x_joist = joist_overhang_length
x_post_1 = fi(9, 10, return_unit="inch")
x_post_2 = x_post_1 + fi(10, 11, return_unit="inch")
x_seg2 = float(l_trib__2.to("inch"))
x_seg3 = x_seg2 + float(l_trib__3.to("inch"))
x_end = float(l_trib.to("inch"))

# Nodes
beam.add_node("n_start", 0, 0, 0)
beam.add_node("n_end", float(l_trib.to("inch")), 0, 0)
beam.add_node("n_joist", joist_overhang_length, 0, 0)
beam.add_node("n_post_1", x_post_1, 0, 0)
beam.add_node("n_post_2", x_post_2, 0, 0)

# Wide flange shape
section = Section.from_shape(BEAM_SELECTION)
display_text(f"Try a {section.name}")

# Add a section with the following properties:
beam.add_section(
    "wf",
    A=section.area,
    Iy=section.second_moment_of_area_y_axis,
    Iz=section.second_moment_of_area_x_axis,
    J=section.torsional_constant,
)

# Add a member
beam.add_member(MEMBER_NAME, "n_start", "n_end", "steel 50ksi", "wf")

# Provide simple supports
beam.def_support("n_start", True, True, True, False, False, False)
beam.def_support("n_post_1", False, True, True, True, False, False)
beam.def_support("n_post_2", False, True, True, True, False, False)
beam.def_support("n_end", False, True, True, True, False, False)

# Uniform loads

# Dead
dead_load_seg1_level2 = (n_levels__2 * float(D_line__seg1)) / 1000 / 12  # kip/in
dead_load_seg2_level2 = (n_levels__2 * float(D_line__seg2)) / 1000 / 12  # kip/in
dead_load_seg3_level2 = (n_levels__2 * float(D_line__seg3)) / 1000 / 12  # kip/in
dead_load_seg4_level2 = (n_levels__2 * float(D_line__seg4)) / 1000 / 12  # kip/in
# note that segment 2 doesn't exist on levels 3 and up
dead_load_seg1_level35 = (n_levels__35 * float(D_line__seg1)) / 1000 / 12  # kip/in
dead_load_seg3_level35 = (n_levels__35 * float(D_line__seg3)) / 1000 / 12  # kip/in
dead_load_seg4_level35 = (n_levels__35 * float(D_line__seg4)) / 1000 / 12  # kip/in
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg1_level2,
    -dead_load_seg1_level2,
    x1=x_joist,
    x2=x_end,
    case="D",
    name="Dead seg 1 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg2_level2,
    -dead_load_seg2_level2,
    x1=x_joist,
    x2=x_seg2,
    case="D",
    name="Dead seg 2 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg3_level2,
    -dead_load_seg3_level2,
    x1=x_seg2,
    x2=x_seg3,
    case="D",
    name="Dead seg 3 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg4_level2,
    -dead_load_seg4_level2,
    x1=x_seg3,
    x2=x_end,
    case="D",
    name="Dead seg 4 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg1_level35,
    -dead_load_seg1_level35,
    x1=x_start,
    x2=x_end,
    case="D",
    name="Dead seg 1 level 3+",
)
# seg2 doesn't exist on levels 3 and up
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg3_level35,
    -dead_load_seg3_level35,
    x1=x_start,
    x2=x_seg3,
    case="D",
    name="Dead seg 3 level 3+",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -dead_load_seg4_level35,
    -dead_load_seg4_level35,
    x1=x_seg3,
    x2=x_end,
    case="D",
    name="Dead seg 4 level 3+",
)
# Wall weight
wall_dead_load = n_levels * float(D_wall) / 1000 / 12  # kip/in
add_named_member_dist_load(beam, MEMBER_NAME, "Fy", -wall_dead_load, -wall_dead_load, case="D", name="Interior walls")
# Live
live_load_seg1_level2 = (n_levels__2 * float(L_line__seg1)) / 1000 / 12  # kip/in
live_load_seg2_level2 = (n_levels__2 * float(L_line__seg2)) / 1000 / 12  # kip/in
live_load_seg3_level2 = (n_levels__2 * float(L_line__seg3)) / 1000 / 12  # kip/in
live_load_seg4_level2 = (n_levels__2 * float(L_line__seg4)) / 1000 / 12  # kip/in
# note that segment 2 doesn't exist on levels 3 and up
live_load_seg1_level35 = (n_levels__35 * float(L_line__seg1)) / 1000 / 12  # kip/in
live_load_seg3_level35 = (n_levels__35 * float(L_line__seg3)) / 1000 / 12  # kip/in
live_load_seg4_level35 = (n_levels__35 * float(L_line__seg4)) / 1000 / 12  # kip/in
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg1_level2,
    -live_load_seg1_level2,
    x1=x_joist,
    x2=x_end,
    case="L",
    name="Live seg 1 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg2_level2,
    -live_load_seg2_level2,
    x1=x_joist,
    x2=x_seg2,
    case="L",
    name="Live seg 2 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg3_level2,
    -live_load_seg3_level2,
    x1=x_seg2,
    x2=x_seg3,
    case="L",
    name="Live seg 3 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg4_level2,
    -live_load_seg4_level2,
    x1=x_seg3,
    x2=x_end,
    case="L",
    name="Live seg 4 level 2",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg1_level35,
    -live_load_seg1_level35,
    x1=x_start,
    x2=x_end,
    case="L",
    name="Live seg 1 level 3+",
)
# seg2 doesn't exist on levels 3 and up
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg3_level35,
    -live_load_seg3_level35,
    x1=x_start,
    x2=x_seg3,
    case="L",
    name="Live seg 3 level 3+",
)
add_named_member_dist_load(
    beam,
    MEMBER_NAME,
    "Fy",
    -live_load_seg4_level35,
    -live_load_seg4_level35,
    x1=x_seg3,
    x2=x_end,
    case="L",
    name="Live seg 4 level 3+",
)
# Self weight
add_named_self_weight(beam, "FY", -1, case="D")

# Point loads
add_named_node_load(beam, "n_joist", "FY", -2 * D_point, case="D", name="2x J5 Dead")
add_named_node_load(beam, "n_joist", "FY", -2 * L_point, case="L", name="2x J5 Live")
add_named_node_load(beam, "n_joist", "FY", -2 * S_point, case="S", name="2x J5 Snow")

# Analyze the beam
beam.analyze()

# %% [markdown]
"""
The beam loading is shown in \autoref{fig:nominal_loads}.
"""

# %%
fig, ax = plot_member_loads(beam.members[MEMBER_NAME], save_png=image_path("nominal_loads"), figsize=(12, 14))

display_figure(
    image_path=image_path("nominal_loads"), caption="All nominal loads", label="fig:nominal_loads", width="100%"
)

# %% [markdown]
"""
We will check the moment capacity for each load combination. This could be plastic deformation, inelastic lateral torsional buckling, or elastic lateral torsional buckling.

First, we need to find the moments in the beam (see \autoref{fig:moments}).
"""

# %%
x, moments, labels, _ = model.get_moments_from_tags(model=beam)
fig, ax = plot_internal_diagram(
    x,
    moments,
    labels=labels,
    envelope=True,
    figsize=(12, 6),
    xlabel="Location [in]",
    ylabel="Moment [kip-in]",
    title="Moment Diagram",
    save_png=image_path("moments"),
    show_plot=False,
)

display_figure(image_path=image_path("moments"), caption="Moment diagram", label="fig:moments", width="100%")

# %% [markdown]
"""
Then, we follow the procedure in AISC 360 Section F2 for wide flange shapes. The summary of this analysis is shown in \autoref{tab:flexure_results}
"""

# %%
flexure_results = model.analyze_member_flexure(beam, section, bottom_flange_brace_points=[x_post_1, x_post_2])

column_name_map = {
    "flange": "Flange",
    "segment start": r"$x_{start}$",
    "segment end": r"$x_{end}$",
    "unbraced length": r"$L_b$",
    "ltb modification factor": r"$C_b$",
    "moment demand": r"$M_u$",
    "factored moment capacity": r"$\phi M_n$",
    "limit region": "Regime",
    "DCR": "DCR",
}
display_table(
    flexure_results,
    column_names_filter_and_map=column_name_map,
    caption="Beam flexure analysis",
    label="tab:flexure_results",
)

# %% [markdown]
"""
We will also check deflection against a limit of $l/360$ as it is the governing case for this beam that supports both the floor and ceiling on level 2.

The deflection diagram is shown in \autoref{fig:deflections}
"""

# %%
x, defl_list, labels_list, _ = model.get_deflections_from_tags(model=beam)
fig, ax = plot_internal_diagram(
    x,
    defl_list,
    labels=labels_list,
    envelope=True,
    figsize=(12, 6),
    xlabel="Location [in]",
    ylabel="Deflection [in]",
    title="Deflection Diagram",
    save_png=image_path("deflections"),
    show_plot=False,
)

display_figure(
    image_path=image_path("deflections"), caption="Deflection diagram", label="fig:deflections", width="100%"
)


member = beam.members[MEMBER_NAME]
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

# %%
delta_L_val = beam.members[MEMBER_NAME].min_deflection("dy", "L OSSC25")
delta_S_val = beam.members[MEMBER_NAME].min_deflection("dy", "S OSSC25")

# %% [markdown]
"""
The deflection checks are shown below.
"""

# %%
# %%render long
# fmt: off
delta_L = (delta_L_val * inch * -1)
delta_S = (delta_S_val * inch * -1)
delta_max__L = l_trib.to("inch") / 360
delta_max__S = l_trib.to("inch") / 360
check_delta__L = check_value(delta_L, delta_max__L)
check_delta__S = check_value(delta_S, delta_max__S)
# fmt: on

# %% [markdown]
"""
## Post Design

### HSS Post

We will look at the worst-case post reaction for the beam.
"""

# %%
max_reaction = max(
    [abs(beam.nodes[node].RxnFY[combo]) for node in beam.nodes.keys() for combo in beam.load_combos.keys()]
)
display_math(
    [
        rf"P_u = {sig_figs(max_reaction, 3)} \text{{ kip}}",
    ]
)

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
minimum_viable__shape = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].idxmin()
minimum_viable__capacity = capacities.loc[capacities["adjusted capacity"] >= max_reaction, "adjusted capacity"].min()
display_text(rf"The minimum viable HSS shape is {minimum_viable__shape}")
display_math(
    [
        rf"\phi_c P_n = {sig_figs(minimum_viable__capacity, 3)} \text{{ kip}}",
    ]
)

# %%
# %%render long
check_post = check_value(minimum_viable__capacity, max_reaction, inequality="geq")
