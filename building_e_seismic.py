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
# title: Building E Seismic Design
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
try:
    import handcalcs.render  # type: ignore[import-not-found]  # noqa: F401
except AttributeError:
    pass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from structural_tools import DesignMethod, LoadCase
from structural_tools.asce import RiskCategory, SeismicParameters, SiteClass
from structural_tools.asce.elf import SeismicLoads
from structural_tools.conversions import *
from structural_tools.notebook import (
    check_value,
    display_figure,
    display_table,
    display_text,
    initialize_notebook,
    set_params_columns,
    sig_figs,
)
from structural_tools.notebook.calculation import feet_inches as fi
from structural_tools.notebook.runtime import image_path
from structural_tools.structure import LateralSystem, Level, Structure
from structural_tools.wood.shear_walls import (
    assign_force_schedule,
    assign_values_for_all_levels_per_wall,
    assign_values_for_all_walls_per_level,
    calculate_end_post_forces,
    check_shear_wall_tributary_area,
    create_shear_walls_dataframe_from_dict,
    design_shear_walls_envelope,
)
from structural_tools.wood.sheathing import (
    Nail,
    PanelType,
    Sheathing,
    SheathingApplication,
    SheathingMaterial,
    get_sheathing_properties,
)

if TYPE_CHECKING:
    from structural_tools.units import ft, inch, ksf, lb, psi

initialize_notebook()

# %% [markdown]
# # Effective Seismic Weight Calculation

# %% [markdown]
# We calculate the effective seismic weight with unit weights and square footages.

# %% [markdown]
# ## Unit Weights

# %% [markdown]
# Weights will be assigned with area unit weights.

# %%
set_params_columns(3)

# %%
# %%render params
U_floor = 35 / 1000 * ksf
U_roof = 25 / 1000 * ksf
U_wall = 12 / 1000 * ksf

# %% [markdown]
# ## Building Geometry

# %% [markdown]
# The lengths and widths of the weight areas are consistent across levels 2, 3, 4, and 5. Lengths, $L$, are the E-W dimensions, while widths, $w$, are the N-S dimensions.
#
# Due to the building geometry, the floor levels are split into different zones shown in ![Building area labels](images/building_e_areas.png){width=60% #fig:diaphragm_trib_area}

# %%
set_params_columns(4)

# %%
# %%render params
L_1 = 131 + (10 / 12) * ft
L_2 = 170 + (2 / 12) * ft
L_3 = 36 + (3 / 12) * ft
L_4 = 30 + (9 / 12) * ft
w_1 = 36 + (3 / 12) * ft
w_2 = 30 + (9 / 12) * ft
w_3 = 115 + (1 / 12) * ft
w_4 = 120 + (1 / 12) * ft
L_jog = 5 * ft

# %% [markdown]
# The wall heights vary across the levels.

# %%
set_params_columns(5)

# %%
# %%render params
h_wall__1 = 11.5 * ft
h_wall__2 = 10.5 * ft
h_wall__3 = 10.5 * ft
h_wall__4 = 10.5 * ft
h_wall__5 = 10 * ft

# %% [markdown]
# The perimeter of the building is:
#
# > Note: there is a small overlap of zones 2 and 4 between lines E-9 and E-10

# %%
Overlap = (6 + 7 / 12) * ft

# %%
# %%render
P_wall = L_1 + w_1 + L_jog + w_2 + L_2 + w_2 - Overlap + L_4 + w_4 + L_4 + L_jog + L_3 + w_3 - w_1

# %% [markdown]
# Areas can now be calculated. Note that the areas of the walls are the actual wall areas; they have not been distributed to each floor based on tributary area, yet.

# %%
# %%render long
A_level = L_1 * w_1 + L_2 * w_2 + L_3 * w_3 + L_4 * w_4

# %%
# %%render
A_wall__1 = h_wall__1 * P_wall
A_wall__2 = h_wall__2 * P_wall
A_wall__3 = h_wall__3 * P_wall
A_wall__4 = h_wall__4 * P_wall
A_wall__5 = h_wall__5 * P_wall

# %% [markdown]
# ## Component Weights

# %% [markdown]
# The typical floor and corridor weights are

# %%
# %%render
W_level = U_floor * A_level

# %% [markdown]
# This applies to levels 2, 3, and 4.

# %%
set_params_columns(4)

# %%
# %%render params
W_level__2 = W_level
W_level__3 = W_level
W_level__4 = W_level
W_level__5 = W_level

# %% [markdown]
# The roof weight is

# %%
# %%render
W_roof = U_roof * A_level

# %% [markdown]
# The exterior walls are divided in half to the floor above and below.

# %%
# %%render
W_wall__2 = U_wall * (A_wall__2 / 2 + A_wall__1 / 2)
W_wall__3 = U_wall * (A_wall__3 / 2 + A_wall__2 / 2)
W_wall__4 = U_wall * (A_wall__4 / 2 + A_wall__3 / 2)
W_wall__5 = U_wall * (A_wall__5 / 2 + A_wall__4 / 2)
W_wall__roof = U_wall * (A_wall__5 / 2)

# %% [markdown]
# ## Level Weights

# %% [markdown]
# Each level's weight is

# %%
# %%render
W_level__2 = W_level__2 + W_wall__2
W_level__3 = W_level__3 + W_wall__3
W_level__4 = W_level__4 + W_wall__4
W_level__5 = W_level__5 + W_wall__5
# be sure to add the weight of the truss later
W_level__roof = W_roof + W_wall__roof

# %% [markdown]
# ## Total Weight

# %% [markdown]
# The total weight of the structure is

# %%
# %%render long
W_total = W_level__2 + W_level__3 + W_level__4 + W_level__5 + W_level__roof

# %% [markdown]
# # Seismic Equivalent Lateral Force (ELF) Analysis

# %% [markdown]
# We will be using the equivalent lateral force method prescribed by ASCE 7-22 Section 12.8. We will envelope the forces with a fully flexible diaphragm assumption and a fully rigid diaphragm assumption.

# %% [markdown]
# ## Seismic Parameters

# %% [markdown]
r"""
Seismic design parameters are obtained from the ASCE Hazard Tool and ASCE 7-22 Sec. 11 and 12.
"""

# %%
latitude = 45.501
longitude = -122.834
seismic_params = SeismicParameters(latitude=latitude, longitude=longitude, risk_category=RiskCategory.II, site_class=SiteClass.D)

# %%
display_text(f"For latitude = {latitude} and longitude = {longitude}:")

# %%
set_params_columns(9)

# %%
# %%render params
I_e = seismic_params.i_e
S_S = seismic_params.s_s
S_1 = seismic_params.s_1
S_MS = seismic_params.s_ms
S_M1 = seismic_params.s_m1
S_DS = seismic_params.s_ds
S_D1 = seismic_params.s_d1
SDC = seismic_params.sdc
T_L = seismic_params.t_l

# %% [markdown]
# ## Base Shear
#
# We must calculate the period

# %%
set_params_columns(1)

# %%
# %%render params
h_top__roof = (68 + 3 / 12) * ft
h_bottom__roof = (55 + 10 / 12) * ft
h_n = np.mean([h_top__roof, h_bottom__roof])  # ASCE 7-22 Sec. 11.2

# %%
# %%render
h_n__check = check_value(h_n, 65 * ft, "<=")

# %% [markdown]
# We will approximate the building period.

# %%
h_n = float(h_n)  # to prevent weirdness with units

# %%
structure = Structure(
    lateral_system_x=LateralSystem(system_index="A16"),
    lateral_system_y=LateralSystem(system_index="A16"),
    levels_input={
        1: Level(height=h_wall__1, weight=0),  # ground level weight not considered
        2: Level(height=h_wall__2, weight=W_level__2),
        3: Level(height=h_wall__3, weight=W_level__3),
        4: Level(height=h_wall__4, weight=W_level__4),
        5: Level(height=h_wall__5, weight=W_level__5),
        6: Level(height=0, weight=W_level__roof),  # roof level height not considered
    },
    structural_height=h_n,
)

# %%
display_text(
    f"The lateral system in the E-W/x direction is: {structure.lateral_system_x.system_index} = {structure.lateral_system_x.system}"
)

# %%
set_params_columns(4)

# %%
# %%render params
R_x = structure.lateral_system_x.r
Omega_0__x = structure.lateral_system_x.omega_0
C_d__x = structure.lateral_system_x.c_d
rho_x = 1.0

# %%
display_text(
    f"The lateral system in the N-S/y direction is: {structure.lateral_system_y.system_index} = {structure.lateral_system_y.system}"
)

# %%
# %%render params
R_y = structure.lateral_system_y.r
Omega_0__y = structure.lateral_system_y.omega_0
C_d__y = structure.lateral_system_y.c_d
rho_y = 1.0


# %% [markdown]
r"""
We will approximate the building period with ASCE 7-22 Sec. 12.8.2.

\begin{equation*}
    T_a = C_t h_n^x
\end{equation*}
"""

# %%
set_params_columns(2)

# %%
# %%render params
c_t__x = structure.lateral_system_x.building_period_coefficient
x_x = structure.lateral_system_x.building_period_exponent
c_t__y = structure.lateral_system_y.building_period_coefficient
x_y = structure.lateral_system_y.building_period_exponent

# %%
# %%render params
T_x = structure.t_x
T_y = structure.t_y

# %% [markdown]
r"""
And now, we can calculate the base shear, $V$
"""

# %%
seismic_loads = SeismicLoads(structure, seismic_params)

# %%
set_params_columns(2)

# %%
# %%render params
C_s__x = seismic_loads.c_s_x
C_s__y = seismic_loads.c_s_y
V_x = seismic_loads.v_x
V_y = seismic_loads.v_y

print(C_s__x, C_s__y, V_x, V_y)
# %% [markdown]
r"""
## Distribute the seismic forces

With the base shear calculated, we must now distribute the base shear on each level.

Calculating for each floor leads to the following \autoref{tab:seismic_forces_calculation_table_ew} and \autoref{tab:seismic_forces_calculation_table_ns}.
"""

# %%
# Format the dataframe
# Reverse the order of the dataframe with '.loc[::-1]' so roof is the top row and ground is the bottom row, then apply formatting with the 'format_dataframe' function
column_name_map = seismic_loads.column_name_map
del column_name_map["unbounded diaphragm design force"]
del column_name_map["minimum diaphragm design force"]
del column_name_map["maximum diaphragm design force"]
del column_name_map["diaphragm design force"]
display_table(
    dataframe=seismic_loads.seismic_loads_x.iloc[::-1],
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Seismic Forces Calculation Table - EW",
    label="tab:seismic_forces_calculation_table_ew",
    position="H",
)
display_table(
    dataframe=seismic_loads.seismic_loads_y.iloc[::-1],
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Seismic Forces Calculation Table - NS",
    label="tab:seismic_forces_calculation_table_ns",
    position="H",
)

# %% [markdown]
# # Diaphragm and Shear Wall Design

# %% [markdown]
# ## Redundancy Factor
#
# The redundancy factor, $\rho$, is either 1.3 or 1.0 according to ASCE 7-22 12.3.4.
#
# No single story resists more than 35% of the shear force (see $C_{vx}$ in \autoref{tab:seismic_forces_calculation_table}), so all values of $\rho$ are $1.0$.

# %% [markdown]
# ## Shear Wall Design
#
# We will envelope the forces with a fully flexible diaphragm assumption and a fully rigid diaphragm assumption.

# %% [markdown]
"""
#### north-south direction

most of the shear walls are not in line which presents an issue: if we assume a diaphragm is the full depth, there will be an extremely long collector element without a shear wall on both sides. we will avoid this by just assuming the flexible diaphragm does not extend the full depth and instead is just tributary to the nearest wall (\autoref{fig:diaphragm_trib_area}). this means that there are essentially two diaphragms through the depth of the building. this is a rough approximation to ensure all the shear load is resolved in the shear walls.

the walls that are close to inline (a-5 and a-5.1, a-6 and a-6.1, and a-9 and a-9.1) in reality will be treated as inline and have collectors to transfer the shear load from the diaphragm to those walls.

proper detailing will be done to ensure continuity through the depth of the diaphragm.

shear wall shear forces are tabulated in \autoref{tab:shear_wall_forces_roof_ns} through \autoref{tab:shear_wall_forces_level2_ns}.

shear wall sheathing will be assumed to have the following properties:

1. wsp sheathing
2. 15/32" thickness
3. 10d common nails
4. plywood panels
5. framing uses douglas-fir-larch (no specific gravity adjustment)

shear capacity and stiffness values come from **awc sdpws 2021 table 4.3a**.

notes:

- cumulative shear force, $f_{x,cum}$, is cumulative from the roof downwards. for example, level 3 includes the shear forces from the roof, level 4, and level 3.
"""

# %%
# set baseline sheathing properties. we set the number of sheathed sides later
sheathing = Sheathing(
    sheathing_material=SheathingMaterial.WSP_SHEATHING,
    minimum_nominal_panel_thickness=15 / 32,
    nail=Nail.COMMON_10D,
    panel_type=PanelType.PLY,
)

# %%
# Plan dimensions
l_E1__E2 = 26 + 1 / 12
l_E2__E3 = 24 + 8 / 12
l_E3__E4 = 26 + 4 / 12
l_E4__E5 = 21 + 4 / 12
l_E5__E6 = 24 + 8 / 12
l_E6__E7 = 8 + 9 / 12
l_E7__E8 = 30 + 9 / 12
l_E8__E9 = 5 + 6 / 12
l_E9__E10 = 30 + 9 / 12

l_E1__E24 = 35 + 11 / 12
l_E24__E34 = 24 + 8 / 12
l_E34__E48 = 34 + 4 / 12
l_E48__E59 = 24 + 8 / 12
l_E59__E74 = 24 + 8 / 12
l_E74__E9 = 23 + 10 / 12
l_E74__E92 = l_E74__E9 + float(Overlap)

l_E2__E24 = l_E1__E24 - l_E1__E2
l_E24__E3 = l_E2__E3 - l_E2__E24
l_E3__E33 = 8 + 10 / 12
l_E33__E4 = l_E3__E4 - l_E3__E33
l_E59__E6 = l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 - (l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59)
l_E59__E7 = (
    l_E1__E24
    + l_E24__E34
    + l_E34__E48
    + l_E48__E59
    + l_E59__E74
    - (l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7)
)
l_E7__E74 = l_E7__E8 + l_E8__E9 - l_E74__E9

l_EA__EB = 35 + 11 / 12
l_EB__EC = 21 + 4 / 12
l_EC__ED = 17 + 11 / 12
l_ED__EF = 8 + 8 / 12
l_EF__EG = 30 + 9 / 12
l_EG__EH = 5 + 6 / 12
l_EH__EI = 30 + 9 / 12

l_EA__EB1 = 38 + 10.25 / 12
l_EB1__EC2 = 24 + 8.25 / 12
l_EC2__EF = 20 + 3.5 / 12
l_EF__EF2 = 6

l_EC2__EC5 = 8 + 10 / 12
l_EC5__EF = l_EC2__EF - l_EC2__EC5
l_EC2__EF2 = l_EC2__EF + l_EF__EF2
l_EC5__EF2 = l_EC5__EF + l_EF__EF2
l_EF2__EG = l_EF__EG - l_EF__EF2
l_EF2__EH = l_EF2__EG + l_EG__EH

d_trib_ns_1 = 67 / 2
d_trib_ew_1 = 67 / 2

# %%
# Wall tributary widths and lengths

# Zone 1 walls
t_w__E2 = l_E1__E2 + (l_E2__E3 / 2)
t_w__E3 = (l_E2__E3 / 2) + (l_E3__E33 / 2)
t_w__E33 = (l_E3__E33 / 2) + (l_E33__E4 / 2)
t_w__E4 = (l_E33__E4 / 2) + (l_E4__E5 / 2)
t_w__E5 = (l_E4__E5 / 2) + (l_E5__E6 / 2)
t_w__E6 = (l_E5__E6 / 2) + l_E6__E7

# Zone 2 walls
t_w__E24 = l_E1__E24 + (l_E24__E34 / 2) - float(L_jog)
t_w__E34 = (l_E24__E34 / 2) + (l_E34__E48 / 2)
t_w__E48 = (l_E34__E48 / 2) + (l_E48__E59 / 2)
t_w__E59 = (l_E48__E59 / 2) + l_E59__E7  # to reentrant corner
t_w__E74 = l_E7__E74 + l_E74__E92

# Zones 3 & 4 walls
t_w__E8 = l_E7__E8 + l_E8__E9 / 2
t_w__E9 = l_E8__E9 / 2 + l_E9__E10

# Zone 1 walls
t_l__E2 = d_trib_ns_1
t_l__E3 = d_trib_ns_1
t_l__E33 = d_trib_ns_1
t_l__E4 = d_trib_ns_1
t_l__E5 = d_trib_ns_1
t_l__E6 = d_trib_ns_1

# Zone 2 walls
t_l__E24 = d_trib_ns_1
t_l__E34 = d_trib_ns_1
t_l__E48 = d_trib_ns_1
t_l__E59 = d_trib_ns_1
t_l__E74 = l_EH__EI  # specific case for this bottom wall

# Zones 3 & 4 walls
t_l__E8__1 = 28
t_l__E8__2 = 18.33
t_l__E8__3 = 12.25 + 1 / 12  # make trib areas match
t_l__E8__4 = 17.75
t_l__E8__5 = 20
t_l__E8__6 = 18.75

t_l__E9__1 = 47.67
t_l__E9__2 = 20.67
t_l__E9__3 = 20.25 + 2 / 12  # make trib areas match
t_l__E9__4 = 31.5 + 0.5 / 12  # make trib areas match
# t_l__E8__1 + t_l__E8__2 + t_l__E8__3 + t_l__E8__4 + t_l__E8__5
# t_l__E9__1 + t_l__E9__2 + t_l__E9__3

# %%
# Wall segment lengths
l_zone__1 = l_EF__EG
l_zone__2 = l_EH__EI
l_stairwell = 18.75

l_E8__1 = 25
l_E8__2 = 12.5
l_E8__3 = 6
l_E8__4 = 11.5
l_E8__5 = 14.75
l_E8__6 = 7

l_E9__1 = 6
l_E9__2 = 12
l_E9__3 = 8.5
l_E9__4 = 24

# zone 3 walls
t_w__EB = l_EA__EB - float(L_jog) + (l_EB__EC / 2)
t_w__EC = (l_EB__EC / 2) + (l_EC__ED / 2)
t_w__ED = (l_EC__ED / 2) + l_ED__EF

# zone 4 walls
t_w__EB1 = l_EA__EB1 + (l_EB1__EC2 / 2)
t_w__EC2 = (l_EB1__EC2 / 2) + (l_EC2__EC5 / 2)
t_w__EC5 = (l_EC2__EC5 / 2) + (l_EC5__EF2 / 2)
t_w__EF2 = (l_EC5__EF2 / 2) + l_EF2__EH

# zone 1 and 2 walls
t_w__G = l_EF__EG + l_EG__EH / 2
t_w__H = l_EG__EH / 2 + l_EH__EI

# zone 3 walls
t_l__EB = d_trib_ew_1
t_l__EC = d_trib_ew_1
t_l__ED = d_trib_ew_1

# zone 4 walls
t_l__EB1 = d_trib_ew_1
t_l__EC2 = d_trib_ew_1
t_l__EC5 = d_trib_ew_1
t_l__EF2 = d_trib_ew_1

# zone 1 and 2 walls
t_l__EG__1 = 40.75
t_l__EG__2 = 14
t_l__EG__3 = 16
t_l__EG__4 = 12.5
t_l__EG__5 = 24.75
t_l__EG__6 = 21.25
t_l__EG__7 = 36
t_l__EH__1 = 27.33
t_l__EH__2 = 18.66
t_l__EH__3 = 25.66
t_l__EH__4 = 33
t_l__EH__5 = 18.66
t_l__EH__6 = 18.75
t_l__EH__7 = 27.5

# %%
# Wall segment lengths
l_zone__3 = l_E7__E8
l_zone__4 = l_E9__E10
l_stairwell = 18.75

l_EG__1 = 11.75
l_EG__2 = 6.75
l_EG__3 = 7.75
l_EG__4 = 6
l_EG__5 = 12.25
l_EG__6 = 10.5
l_EG__7 = 7.5

l_EH__1 = 23.25
l_EH__2 = 12.33
l_EH__3 = 20
l_EH__4 = 27
l_EH__5 = 12
l_EH__6 = 13
l_EH__7 = 23.33

# %%
# Set up diaphragm dataframe
walls_input = {
    # N-S
    # zone 1 walls
    "E-2": {
        "direction": "y",
        "x": float(l_E1__E2),
        "y": 0,
        "tributary width": float(t_w__E2),
        "tributary length": t_l__E2,
        "wall length": l_zone__1,
    },
    "E-3": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3),
        "y": 0,
        "tributary width": float(t_w__E3),
        "tributary length": t_l__E3,
        "wall length": l_zone__1,
    },
    "E-3.3": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E33),
        "y": 0,
        "tributary width": float(t_w__E33),
        "tributary length": t_l__E33,
        "wall length": l_stairwell,
    },
    "E-4": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4),
        "y": 0,
        "tributary width": float(t_w__E4),
        "tributary length": t_l__E4,
        "wall length": l_zone__1,
    },
    "E-5": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5),
        "y": 0,
        "tributary width": float(t_w__E5),
        "tributary length": t_l__E5,
        "wall length": l_zone__1,
    },
    # zone 2 walls
    "E-2.4": {
        "direction": "y",
        "x": float(l_E1__E24),
        "y": 0,
        "tributary width": float(t_w__E24),
        "tributary length": t_l__E24,
        "wall length": l_zone__2,
    },
    "E-3.4": {
        "direction": "y",
        "x": float(l_E1__E24 + l_E24__E34),
        "y": 0,
        "tributary width": float(t_w__E34),
        "tributary length": t_l__E34,
        "wall length": l_zone__2,
    },
    "E-4.8": {
        "direction": "y",
        "x": float(l_E1__E24 + l_E24__E34 + l_E34__E48),
        "y": 0,
        "tributary width": float(t_w__E48),
        "tributary length": t_l__E48,
        "wall length": l_zone__2,
    },
    "E-5.9": {
        "direction": "y",
        "x": float(l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59),
        "y": 0,
        "tributary width": float(t_w__E59),
        "tributary length": t_l__E59,
        "wall length": l_zone__2,
    },
    "E-7.4": {
        "direction": "y",
        "x": float(l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59 + l_E59__E74),
        "y": 0,
        "tributary width": float(t_w__E74),
        "tributary length": t_l__E74,
        "wall length": l_zone__2,
    },
    # zone 3 and 4 walls
    "E-8--E-A.2": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__1,
        "wall length": l_E8__1,
    },
    "E-8--E-B": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__2,
        "wall length": l_E8__2,
    },
    "E-8--E-B.7": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__3,
        "wall length": l_E8__3,
    },
    "E-8--E-C.3": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__4,
        "wall length": l_E8__4,
    },
    "E-8--E-F": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__5,
        "wall length": l_E8__5,
    },
    "E-8--E-F.7": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8),
        "y": 0,
        "tributary width": float(t_w__E8),
        "tributary length": t_l__E8__6,
        "wall length": l_E8__6,
    },
    "E-9--E-B.1": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
        "tributary width": float(t_w__E9),
        "tributary length": t_l__E9__1,
        "wall length": l_E9__1,
    },
    "E-9--E-B.6": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
        "tributary width": float(t_w__E9),
        "tributary length": t_l__E9__2,
        "wall length": l_E9__2,
    },
    "E-9--E-C.5": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
        "tributary width": float(t_w__E9),
        "tributary length": t_l__E9__3,
        "wall length": l_E9__3,
    },
    "E-9--E-F.4": {
        "direction": "y",
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
        "tributary width": float(t_w__E9),
        "tributary length": t_l__E9__4,
        "wall length": l_E9__4,
    },
    # E-W
    # zone 3 walls
    "E-B": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB),
        "tributary width": float(t_w__EB),
        "tributary length": t_l__EB,
        "wall length": l_zone__3,
    },
    "E-C": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC),
        "tributary width": float(t_w__EC),
        "tributary length": t_l__EC,
        "wall length": l_zone__3,
    },
    "E-D": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED),
        "tributary width": float(t_w__ED),
        "tributary length": t_l__ED,
        "wall length": l_zone__3,
    },
    # zone 4 walls
    "E-B.1": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB1),
        "tributary width": float(t_w__EB1),
        "tributary length": t_l__EB1,
        "wall length": l_zone__4,
    },
    "E-C.2": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB1 + l_EB1__EC2),
        "tributary width": float(t_w__EC2),
        "tributary length": t_l__EC2,
        "wall length": l_stairwell,
    },
    "E-C.5": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB1 + l_EB1__EC2 + l_EC2__EC5),
        "tributary width": float(t_w__EC5),
        "tributary length": t_l__EC5,
        "wall length": l_stairwell,
    },
    "E-F.2": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB1 + l_EB1__EC2 + l_EC2__EC5 + l_EC5__EF2),
        "tributary width": float(t_w__EF2),
        "tributary length": t_l__EF2,
        "wall length": l_zone__4,
    },
    # zones 1 and 2 walls
    "E-G--E-2": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__1,
        "wall length": l_EG__1,
    },
    "E-G--E-2.7": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__2,
        "wall length": l_EG__2,
    },
    "E-G--E-3.4": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__3,
        "wall length": l_EG__3,
    },
    "E-G--E-3.8": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__4,
        "wall length": l_EG__4,
    },
    "E-G--E-4.4": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__5,
        "wall length": l_EG__5,
    },
    "E-G--E-7": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG),
        "tributary width": float(t_w__G),
        "tributary length": t_l__EG__7,
        "wall length": l_EG__7,
    },
    "E-H--E-1.2": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__1,
        "wall length": l_EH__1,
    },
    "E-H--E-2.4": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__2,
        "wall length": l_EH__2,
    },
    "E-H--E-3.1": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__3,
        "wall length": l_EH__3,
    },
    "E-H--E-4.1": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__4,
        "wall length": l_EH__4,
    },
    "E-H--E-7": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__6,
        "wall length": l_EH__6,
    },
    "E-H--E-7.6": {
        "direction": "x",
        "x": 0,
        "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH),
        "tributary width": float(t_w__H),
        "tributary length": t_l__EH__7,
        "wall length": l_EH__7,
    },
}

shear_walls = create_shear_walls_dataframe_from_dict(walls_input, seismic_loads)

shear_walls["level seismic force per area"] = shear_walls["level seismic force"] / float(A_level)
E_endpost = 1_600_000
A_endpost = 2 * 5.25
Delta_A = 0.25

x_1 = (131 + 10 / 12) / 2
x_2 = ((170 + 2 / 12) / 2) + float(L_jog)
x_3 = ((30 + (9 / 12) + 5 + (6 / 12)) / 2) + (131 + 10 / 12)
x_4 = ((30 + (9 / 12)) / 2) + (131 + (10 / 12) + 30 + (9 / 12) + 5 + (6 / 12))

y_1 = ((30 + (9 / 12) + 5 + (6 / 12)) / 2) + (83 + 10 / 12)
y_2 = ((30 + (9 / 12)) / 2) + (83 + (10 / 12) + 30 + (9 / 12) + 5 + (6 / 12))
y_3 = (115 + 1 / 12) / 2 + float(L_jog)
y_4 = (120 + 1 / 12) / 2

A_1 = L_1 * w_1
A_2 = L_2 * w_2
A_3 = L_3 * w_3
A_4 = L_4 * w_4

CM_x = (x_1 * A_1 + x_2 * A_2 + x_3 * A_3 + x_4 * A_4) / A_level
CM_y = (y_1 * A_1 + y_2 * A_2 + y_3 * A_3 + y_4 * A_4) / A_level

X_plan = fi(198, 10)
Y_plan = fi(150, 10)
plan_dimensions = (X_plan, Y_plan)
center_of_mass = (CM_x, CM_y)
shear_walls = design_shear_walls_envelope(
    shear_walls,
    sheathing,
    end_post_youngs_modulus=E_endpost,
    end_post_area=A_endpost,
    Delta_A=Delta_A,
    center_of_mass=center_of_mass,
    plan_dimensions=plan_dimensions,
)

trib_area_reduction_dict = {
    "E-2": 1 / 4,
    "E-2.4": 1 / 4,
    "E-3.4": 1 / 4,
    "E-5": 1 / 4,
    "E-7.4": 1 / 4,
    "E-8--E-A.2": 0,
    "E-8--E-B": 0,
    "E-8--E-B.7": 0,
    "E-8--E-C.3": 0,
    "E-8--E-F": 0,
    "E-8--E-F.7": 0,
    "E-9--E-B.1": 0,
    "E-9--E-B.6": 0,
    "E-9--E-C.5": 0,
    "E-9--E-F.4": 0,
    "E-B.1": 1 / 4,
    "E-D": 1 / 4,
    "E-F.2": 1 / 16,
    "E-G--E-2": 0,
    "E-G--E-2.7": 0,
    "E-G--E-3.4": 0,
    "E-G--E-3.8": 0,
    "E-G--E-4.4": 0,
    "E-G--E-7": 0,
    "E-H--E-1.2": 0,
    "E-H--E-2.4": 0,
    "E-H--E-3.1": 0,
    "E-H--E-4.1": 0,
    "E-H--E-7": 0,
    "E-H--E-7.6": 0,
}


wall_arm_shortening_length = 0.5  # feet due to holddown positioning
shear_walls = calculate_end_post_forces(
    shear_walls, float(U_floor * 1000), float(seismic_params.s_ds), wall_arm_shortening_length, trib_area_reduction_dict
)
shear_walls, shear_wall_schedule = assign_force_schedule(shear_walls)
shear_walls.to_csv("shear_walls_E_new.csv")
print(shear_walls["tension force"].to_string())

assumed_diaphragm_capacity = 215
shear_walls["required strap capacity"] = (
    shear_walls["adjusted diaphragm floor unit shear demand"] * shear_walls["wall length"]
    - 2 * assumed_diaphragm_capacity * shear_walls["wall length"]
)

shear_walls["required strap capacity"] = np.where(
    shear_walls["required strap capacity"] < 0, 0, shear_walls["required strap capacity"]
)
print(shear_walls["required strap capacity"].to_string())

# %% [markdown]
# ### Flexible Diaphragm Assumption
#
# We will uniformly distribute the shear forces across the diaphragms based on tributary areas of each level. This removes some nuance in the distribution of floor weight distribution (corridors are 17% heavier than the rest of the floors, for instance), but the vast majority of the area of each level is typical floor weight (30 psf) instead of corridor weight (35 psf).

# %% [markdown]
# $$
# f_{p,x} = F_x / A_{level}
# $$

# %%
# %%render
A_level
f_p__roof = seismic_loads.loc["Roof", "F_x"] * 1000 * lb / A_level
f_p__5 = seismic_loads.loc["Level 5", "F_x"] * 1000 * lb / A_level
f_p__4 = seismic_loads.loc["Level 4", "F_x"] * 1000 * lb / A_level
f_p__3 = seismic_loads.loc["Level 3", "F_x"] * 1000 * lb / A_level
f_p__2 = seismic_loads.loc["Level 2", "F_x"] * 1000 * lb / A_level

# %% [markdown]
# #### North-South Direction

# %% [markdown]
# Most of the shear walls are not in line which presents an issue: if we assume a diaphragm is the full depth, there will be an extremely long collector element without a shear wall on both sides. We will avoid this by just assuming the flexible diaphragm does not extend the full depth and instead is just tributary to the nearest wall (\autoref{fig:diaphragm_trib_area}). This means that there are essentially two diaphragms through the depth of the building. This is a rough approximation to ensure all the shear load is resolved in the shear walls. \
#
# Proper detailing will be done to ensure continuity through the depth of the diaphragm.
#
# ![Shear walls (thick red lines) in plan view with their (colored) tributary areas.](images/shear_wall_tributary_area.png){width=60% #fig:diaphragm_trib_area}

# %% [markdown]
# Shear wall shear forces are tabulated in \autoref{tab:shear_wall_forces_roof_ns} through \autoref{tab:shear_wall_forces_level2_ns}.
#
# Shear wall sheathing will be assumed to have the following properties:
#
# 1. WSP Sheathing
# 2. 15/32" thickness
# 3. 10d common nails
# 4. Plywood panels
# 5. Framing uses Douglas-Fir-Larch (no specific gravity adjustment)
#
# Shear capacity and stiffness values come from **AWC SDPWS 2021 Table 4.3A**.
#
# Notes:
#
# - Cumulative shear force, $F_{x,cum}$, is cumulative from the Roof downwards. For example, Level 3 includes the shear forces from the Roof, Level 4, and Level 3.

# %%
from structural_tools.seismic.sheathing import (
    DesignMethod,
    LoadCase,
    Nail,
    PanelType,
    Sheathing,
    SheathingMaterial,
    get_sheathing_properties,
)

# Set baseline sheathing properties. We set the number of sheathed sides later
sheathing = Sheathing(
    sheathing_material=SheathingMaterial.WSP_SHEATHING,
    minimum_nominal_panel_thickness=15 / 32,
    nail=Nail.COMMON_10D,
    panel_type=PanelType.PLY,
)

# %%
# Plan dimensions
l_E1__E2 = 26 + 1 / 12
l_E2__E3 = 24 + 8 / 12
l_E3__E4 = 26 + 4 / 12
l_E4__E5 = 21 + 4 / 12
l_E5__E6 = 24 + 8 / 12
l_E6__E7 = 8 + 9 / 12
l_E7__E8 = 30 + 9 / 12
l_E8__E9 = 5 + 6 / 12
l_E9__E10 = 30 + 9 / 12

l_E1__E24 = 35 + 11 / 12
l_E24__E34 = 24 + 8 / 12
l_E34__E48 = 34 + 4 / 12
l_E48__E59 = 24 + 8 / 12
l_E59__E74 = 24 + 8 / 12
l_E74__E9 = 23 + 10 / 12
l_E74__E92 = l_E74__E9 + float(Overlap)

l_E2__E24 = l_E1__E24 - l_E1__E2
l_E24__E3 = l_E2__E3 - l_E2__E24
l_E3__E33 = 8 + 10 / 12
l_E33__E4 = l_E3__E4 - l_E3__E33
l_E59__E6 = l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 - (l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59)
l_E59__E7 = (
    l_E1__E24
    + l_E24__E34
    + l_E34__E48
    + l_E48__E59
    + l_E59__E74
    - (l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7)
)
l_E7__E74 = l_E7__E8 + l_E8__E9 - l_E74__E9

l_EA__EB = 35 + 11 / 12
l_EB__EC = 21 + 4 / 12
l_EC__ED = 17 + 11 / 12
l_ED__EF = 8 + 8 / 12
l_EF__EG = 30 + 9 / 12
l_EG__EH = 5 + 6 / 12
l_EH__EI = 30 + 9 / 12

l_EA__EB1 = 38 + 10.25 / 12
l_EB1__EC2 = 24 + 8.25 / 12
l_EC2__EF = 20 + 3.5 / 12
l_EF__EF2 = 6

l_EC2__EC5 = 8 + 10 / 12
l_EC5__EF = l_EC2__EF - l_EC2__EC5
l_EC2__EF2 = l_EC2__EF + l_EF__EF2
l_EC5__EF2 = l_EC5__EF + l_EF__EF2
l_EF2__EG = l_EF__EG - l_EF__EF2
l_EF2__EH = l_EF2__EG + l_EG__EH

d_trib_ns_1 = 67 / 2

# %%
# Wall tributary widths and lengths

# Zone 1 walls
t_w__E2 = l_E1__E2 + (l_E2__E3 / 2)
t_w__E3 = (l_E2__E3 / 2) + (l_E3__E33 / 2)
t_w__E33 = (l_E3__E33 / 2) + (l_E33__E4 / 2)
t_w__E4 = (l_E33__E4 / 2) + (l_E4__E5 / 2)
t_w__E5 = (l_E4__E5 / 2) + (l_E5__E6 / 2)
t_w__E6 = (l_E5__E6 / 2) + l_E6__E7

# Zone 2 walls
t_w__E24 = l_E1__E24 + (l_E24__E34 / 2) - float(L_jog)
t_w__E34 = (l_E24__E34 / 2) + (l_E34__E48 / 2)
t_w__E48 = (l_E34__E48 / 2) + (l_E48__E59 / 2)
t_w__E59 = (l_E48__E59 / 2) + l_E59__E7  # to reentrant corner
t_w__E74 = l_E7__E74 + l_E74__E92

# Zones 3 & 4 walls
t_w__E8 = l_E7__E8 + l_E8__E9 / 2
t_w__E9 = l_E8__E9 / 2 + l_E9__E10

# Zone 1 walls
t_l__E2 = d_trib_ns_1
t_l__E3 = d_trib_ns_1
t_l__E33 = d_trib_ns_1
t_l__E4 = d_trib_ns_1
t_l__E5 = d_trib_ns_1
t_l__E6 = d_trib_ns_1

# Zone 2 walls
t_l__E24 = d_trib_ns_1
t_l__E34 = d_trib_ns_1
t_l__E48 = d_trib_ns_1
t_l__E59 = d_trib_ns_1
t_l__E74 = l_EH__EI  # specific case for this bottom wall

# Zones 3 & 4 walls
t_l__E8__1 = 28
t_l__E8__2 = 18.33
t_l__E8__3 = 12.25 + 1 / 12  # make trib areas match
t_l__E8__4 = 17.75
t_l__E8__5 = 20
t_l__E8__6 = 18.75

t_l__E9__1 = 47.67
t_l__E9__2 = 20.67
t_l__E9__3 = 20.25 + 2 / 12  # make trib areas match
t_l__E9__4 = 31.5 + 0.5 / 12  # make trib areas match
# t_l__E8__1 + t_l__E8__2 + t_l__E8__3 + t_l__E8__4 + t_l__E8__5
# t_l__E9__1 + t_l__E9__2 + t_l__E9__3

# %%
# Wall segment lengths
l_zone__1 = l_EF__EG
l_zone__2 = l_EH__EI
l_stairwell = 18.75

l_E8__1 = 25
l_E8__2 = 12.5
l_E8__3 = 6
l_E8__4 = 11.5
l_E8__5 = 14.75
l_E8__6 = 7

l_E9__1 = 6
l_E9__2 = 12
l_E9__3 = 8.5
l_E9__4 = 24

# %%
flexible_ns.loc["Level 5", "tributary weight"] = float(f_p__roof)
flexible_ns.loc["Level 4", "tributary weight"] = float(f_p__5)
flexible_ns.loc["Level 3", "tributary weight"] = float(f_p__4)
flexible_ns.loc["Level 2", "tributary weight"] = float(f_p__3)
flexible_ns.loc["Level 1", "tributary weight"] = float(f_p__2)

# Shear wall height is for the floor below
flexible_ns.loc["Level 5", "wall height"] = float(seismic_loads.loc["Level 5", "floor height"])
flexible_ns.loc["Level 4", "wall height"] = float(seismic_loads.loc["Level 4", "floor height"])
flexible_ns.loc["Level 3", "wall height"] = float(seismic_loads.loc["Level 3", "floor height"])
flexible_ns.loc["Level 2", "wall height"] = float(seismic_loads.loc["Level 2", "floor height"])
flexible_ns.loc["Level 1", "wall height"] = float(seismic_loads.loc["Level 1", "floor height"])

# Add wall tributary widths and wall lengths to df
wall_tributary = {
    # zone 1 walls
    "E-2": {"trib_width": float(t_w__E2), "trib_depth": t_l__E2, "wall_length": l_zone__1},
    "E-3": {"trib_width": float(t_w__E3), "trib_depth": t_l__E3, "wall_length": l_zone__1},
    "E-3.3": {"trib_width": float(t_w__E33), "trib_depth": t_l__E33, "wall_length": l_stairwell},
    "E-4": {"trib_width": float(t_w__E4), "trib_depth": t_l__E4, "wall_length": l_zone__1},
    "E-5": {"trib_width": float(t_w__E5), "trib_depth": t_l__E5, "wall_length": l_zone__1},
    # "E-6": {"trib_width": float(t_w__E6), "trib_depth": t_l__E6, "wall_length": l_zone__1},
    # zone 2 walls
    "E-2.4": {"trib_width": float(t_w__E24), "trib_depth": t_l__E24, "wall_length": l_zone__2},
    "E-3.4": {"trib_width": float(t_w__E34), "trib_depth": t_l__E34, "wall_length": l_zone__2},
    "E-4.8": {"trib_width": float(t_w__E48), "trib_depth": t_l__E48, "wall_length": l_zone__2},
    "E-5.9": {"trib_width": float(t_w__E59), "trib_depth": t_l__E59, "wall_length": l_zone__2},
    "E-7.4": {"trib_width": float(t_w__E74), "trib_depth": t_l__E74, "wall_length": l_zone__2},
    # zone 3 and 4 walls
    "E-8--E-A.2": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__1, "wall_length": l_E8__1},
    "E-8--E-B": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__2, "wall_length": l_E8__2},
    "E-8--E-B.7": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__3, "wall_length": l_E8__3},
    "E-8--E-C.3": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__4, "wall_length": l_E8__4},
    "E-8--E-F": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__5, "wall_length": l_E8__5},
    "E-8--E-F.7": {"trib_width": float(t_w__E8), "trib_depth": t_l__E8__6, "wall_length": l_E8__6},
    "E-9--E-B.1": {"trib_width": float(t_w__E9), "trib_depth": t_l__E9__1, "wall_length": l_E9__1},
    "E-9--E-B.6": {"trib_width": float(t_w__E9), "trib_depth": t_l__E9__2, "wall_length": l_E9__2},
    "E-9--E-C.5": {"trib_width": float(t_w__E9), "trib_depth": t_l__E9__3, "wall_length": l_E9__3},
    "E-9--E-F.4": {"trib_width": float(t_w__E9), "trib_depth": t_l__E9__4, "wall_length": l_E9__4},
}

flexible_ns["wall length"] = flexible_ns.index.get_level_values("Wall").map(
    {key: value["wall_length"] for key, value in wall_tributary.items()}
)
flexible_ns["tributary width"] = flexible_ns.index.get_level_values("Wall").map(
    {key: value["trib_width"] for key, value in wall_tributary.items()}
)
flexible_ns["tributary depth"] = flexible_ns.index.get_level_values("Wall").map(
    {key: value["trib_depth"] for key, value in wall_tributary.items()}
)
flexible_ns["tributary area"] = flexible_ns["tributary depth"] * flexible_ns["tributary width"]

# Fix trib area of certain walls on all levels
flexible_ns.loc[flexible_ns.index.get_level_values("Wall") == "E-5", "tributary area"] += 306
flexible_ns.loc[flexible_ns.index.get_level_values("Wall") == "E-5.9", "tributary area"] += 400.2

# Ensure that the calculated areas from earlier and tributary areas match
trib_area = flexible_ns["tributary area"].sum() / len(flexible_ns.index.get_level_values("Level").unique())
assert np.isclose(trib_area, float(A_level)), (
    f"Shear wall tributary areas ({trib_area}) and level area ({float(A_level)}) do not match up. Ensure you are including all areas in your calculations"
)

# Automatically extract the shear line by splitting at "_" and taking the first section
flexible_ns["line"] = flexible_ns.index.get_level_values("Wall").str.split("--").str[0]
# Get tributary area and wall length for the shear line
flexible_ns["line tributary area"] = flexible_ns.groupby(["Level", "line"])["tributary area"].transform("sum")
flexible_ns["line wall length"] = flexible_ns.groupby(["Level", "line"])["wall length"].transform("sum")

flexible_ns["line shear demand"] = flexible_ns["line tributary area"] * flexible_ns["tributary weight"]
flexible_ns["line unit shear"] = flexible_ns["line shear demand"] / flexible_ns["line wall length"]
flexible_ns["line cumulative shear demand"] = flexible_ns.groupby(level="Wall")["line shear demand"].cumsum()
flexible_ns["line cumulative unit shear"] = flexible_ns["line cumulative shear demand"] / flexible_ns["line wall length"]

flexible_ns["shear demand"] = flexible_ns["line unit shear"] * flexible_ns["wall length"]
flexible_ns["cumulative shear demand"] = flexible_ns.groupby(level="Wall")["shear demand"].cumsum()
flexible_ns["unit shear demand"] = flexible_ns["cumulative shear demand"] / flexible_ns["wall length"]
flexible_ns["adjusted unit shear demand"] = flexible_ns["unit shear demand"] * LOAD_FACTOR_SEISMIC_ASD

flexible_ns["sheathed sides"] = 1
flexible_ns["aspect ratio"] = flexible_ns["wall height"] / flexible_ns["wall length"]

# %% [markdown]
# #### Check aspect ratios
#
# We must check the aspect ratios of the shear walls to ensure we don't need to apply the aspect ratio factor in SDPWS Sec. 4.3.3.2. The maximum aspect ratio of all N-S walls is:

# %%
# %%render
ratio_max = check_value(flexible_ns["aspect ratio"].max(), 2, "<")

# %%
if "OK" in ratio_max:
    display_text(r"$\therefore$ All aspect ratios are OK")
else:
    display_text("The aspect ratio knockdown factor must be applied.")

# %%
flexible_ns[["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"]] = get_sheathing_properties(
    flexible_ns[["adjusted unit shear demand", "sheathed sides"]], sheathing=sheathing
)

shear_dcr_check_value = 1

# Assign 2-sided sheathing to walls requiring it, then recalc capacity for just those walls (more efficient than passing the full wall dataframe)
if (flexible_ns["shear dcr"] > shear_dcr_check_value).any():
    flexible_ns.loc[flexible_ns["shear dcr"] > shear_dcr_check_value, "sheathed sides"] = 2
    flexible_ns.loc[
        flexible_ns["shear dcr"] > shear_dcr_check_value,
        ["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"],
    ] = get_sheathing_properties(
        flexible_ns.loc[flexible_ns["shear dcr"] > shear_dcr_check_value, ["adjusted unit shear demand", "sheathed sides"]],
        sheathing=sheathing,
    )

# %% editable=true slideshow={"slide_type": ""}
# Update dataframe
flexible = flexible.reindex(columns=flexible_ns.columns)
flexible["line"] = flexible["line"].astype("string")
flexible.update(flexible_ns)

# %%
# Rename columns for display with LaTeX formatting for units and subscripts/superscripts
column_name_map = {
    "tributary weight": "$w_{trib}$ [psf]",
    "tributary area": "$A$ [ft$^2$]",
    "shear demand": "$F_x$ [lbf]",
    "cumulative shear demand": "$F_{x,cum}$ [lbf]",
    "wall length": "$l_{wall}$ [ft]",
    "unit shear demand": "$v$ [plf]",
    "adjusted unit shear demand": "$0.7 v$ [lbf]",
    "adjusted unit shear capacity": "$v_{cap}$ [plf]",
    "sheathed sides": "sides",
    "nail spacing": "$s_{nail}$ [in]",
}


# %%
def highlight_insufficient_capacity(row):
    styles = [""] * len(row)

    if row["$0.7 v$ [plf]"] > row["$v_{cap}$ [plf]"]:
        demand_idx = row.index.get_loc("$0.7 v$ [plf]")
        capacity_idx = row.index.get_loc("$v_{cap}$ [plf]")

        styles[demand_idx] = "background-color: red; font-weight: bold"
        styles[capacity_idx] = "background-color: red; font-weight: bold"

    return styles


# %%
display_table(
    dataframe=flexible_ns,
    levels="Level 5",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 5",
    label="tab:shear_wall_forces_roof_ns",
)
display_table(
    dataframe=flexible_ns,
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 4",
    label="tab:shear_wall_forces_level4_ns",
)
display_table(
    dataframe=flexible_ns,
    levels="Level 3",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 3",
    label="tab:shear_wall_forces_level3_ns",
)
display_table(
    dataframe=flexible_ns,
    levels="Level 2",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 2",
    label="tab:shear_wall_forces_level2_ns",
)
display_table(
    dataframe=flexible_ns,
    levels="Level 1",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 1",
    label="tab:shear_wall_forces_level1_ns",
)

# %% [markdown]
# #### East-West Direction

# %% [markdown]
# Shear wall shear forces are tabulated in \autoref{tab:shear_wall_forces_roof_ew} through \autoref{tab:shear_wall_forces_level2_ew}.

# %%
# Plan dimensions
d_trib_ew_1 = 67 / 2

# %%
# Wall tributary widths and lengths

# zone 3 walls
t_w__EB = l_EA__EB - float(L_jog) + (l_EB__EC / 2)
t_w__EC = (l_EB__EC / 2) + (l_EC__ED / 2)
t_w__ED = (l_EC__ED / 2) + l_ED__EF

# zone 4 walls
t_w__EB1 = l_EA__EB1 + (l_EB1__EC2 / 2)
t_w__EC2 = (l_EB1__EC2 / 2) + (l_EC2__EC5 / 2)
t_w__EC5 = (l_EC2__EC5 / 2) + (l_EC5__EF2 / 2)
t_w__EF2 = (l_EC5__EF2 / 2) + l_EF2__EH

# zone 1 and 2 walls
t_w__G = l_EF__EG + l_EG__EH / 2
t_w__H = l_EG__EH / 2 + l_EH__EI

# zone 3 walls
t_l__EB = d_trib_ew_1
t_l__EC = d_trib_ew_1
t_l__ED = d_trib_ew_1

# zone 4 walls
t_l__EB1 = d_trib_ew_1
t_l__EC2 = d_trib_ew_1
t_l__EC5 = d_trib_ew_1
t_l__EF2 = d_trib_ew_1

# zone 1 and 2 walls
t_l__EG__1 = 40.75
t_l__EG__2 = 14
t_l__EG__3 = 16
t_l__EG__4 = 12.5
t_l__EG__5 = 24.75
t_l__EG__6 = 21.25
t_l__EG__7 = 36
t_l__EH__1 = 27.33
t_l__EH__2 = 18.66
t_l__EH__3 = 25.66
t_l__EH__4 = 33
t_l__EH__5 = 18.66
t_l__EH__6 = 18.75
t_l__EH__7 = 27.5

# %%
# Wall segment lengths
l_zone__3 = l_E7__E8
l_zone__4 = l_E9__E10
l_stairwell = 18.75

l_EG__1 = 11.75
l_EG__2 = 6.75
l_EG__3 = 7.75
l_EG__4 = 6
l_EG__5 = 12.25
l_EG__6 = 10.5
l_EG__7 = 7.5

l_EH__1 = 23.25
l_EH__2 = 12.33
l_EH__3 = 20
l_EH__4 = 27
l_EH__5 = 12
l_EH__6 = 13
l_EH__7 = 23.33

# %%
flexible_ew.loc["Level 5", "tributary weight"] = float(f_p__roof)
flexible_ew.loc["Level 4", "tributary weight"] = float(f_p__5)
flexible_ew.loc["Level 3", "tributary weight"] = float(f_p__4)
flexible_ew.loc["Level 2", "tributary weight"] = float(f_p__3)
flexible_ew.loc["Level 1", "tributary weight"] = float(f_p__2)

# Shear wall height is for the floor below
flexible_ew.loc["Level 5", "wall height"] = float(seismic_loads.loc["Level 5", "floor height"])
flexible_ew.loc["Level 4", "wall height"] = float(seismic_loads.loc["Level 4", "floor height"])
flexible_ew.loc["Level 3", "wall height"] = float(seismic_loads.loc["Level 3", "floor height"])
flexible_ew.loc["Level 2", "wall height"] = float(seismic_loads.loc["Level 2", "floor height"])
flexible_ew.loc["Level 1", "wall height"] = float(seismic_loads.loc["Level 1", "floor height"])

# Add wall tributary widths and wall lengths to df
wall_tributary = {
    # E-W
    # zone 3 walls
    "E-B": {"trib_width": float(t_w__EB), "trib_depth": t_l__EB, "wall_length": l_zone__3},
    "E-C": {"trib_width": float(t_w__EC), "trib_depth": t_l__EC, "wall_length": l_zone__3},
    "E-D": {"trib_width": float(t_w__ED), "trib_depth": t_l__ED, "wall_length": l_zone__3},
    # zone 4 walls
    "E-B.1": {"trib_width": float(t_w__EB1), "trib_depth": t_l__EB1, "wall_length": l_zone__4},
    "E-C.2": {"trib_width": float(t_w__EC2), "trib_depth": t_l__EC2, "wall_length": l_stairwell},
    "E-C.5": {"trib_width": float(t_w__EC5), "trib_depth": t_l__EC5, "wall_length": l_stairwell},
    "E-F.2": {"trib_width": float(t_w__EF2), "trib_depth": t_l__EF2, "wall_length": l_zone__4},
    # zones 1 and 2 walls
    "E-G--E-2": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__1, "wall_length": l_EG__1},
    "E-G--E-2.7": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__2, "wall_length": l_EG__2},
    "E-G--E-3.4": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__3, "wall_length": l_EG__3},
    "E-G--E-3.8": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__4, "wall_length": l_EG__4},
    "E-G--E-4.4": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__5, "wall_length": l_EG__5},
    # "E-G--E-5.7": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__6, "wall_length": l_EG__6},
    "E-G--E-7": {"trib_width": float(t_w__G), "trib_depth": t_l__EG__7, "wall_length": l_EG__7},
    "E-H--E-1.2": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__1, "wall_length": l_EH__1},
    "E-H--E-2.4": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__2, "wall_length": l_EH__2},
    "E-H--E-3.1": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__3, "wall_length": l_EH__3},
    "E-H--E-4.1": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__4, "wall_length": l_EH__4},
    # "E-H--E-5.5": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__5, "wall_length": l_EH__5},
    "E-H--E-7": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__6, "wall_length": l_EH__6},
    "E-H--E-7.6": {"trib_width": float(t_w__H), "trib_depth": t_l__EH__7, "wall_length": l_EH__7},
}

flexible_ew["wall length"] = flexible_ew.index.get_level_values("Wall").map(
    {key: value["wall_length"] for key, value in wall_tributary.items()}
)
flexible_ew["tributary width"] = flexible_ew.index.get_level_values("Wall").map(
    {key: value["trib_width"] for key, value in wall_tributary.items()}
)
flexible_ew["tributary depth"] = flexible_ew.index.get_level_values("Wall").map(
    {key: value["trib_depth"] for key, value in wall_tributary.items()}
)
flexible_ew["tributary area"] = flexible_ew["tributary depth"] * flexible_ew["tributary width"]

# Ensure that the calculated areas from earlier and tributary areas match
trib_area = flexible_ns["tributary area"].sum() / len(flexible_ns.index.get_level_values("Level").unique())
assert np.isclose(trib_area, float(A_level)), (
    f"Shear wall tributary areas ({trib_area}) and level area ({float(A_level)}) do not match up.\n\
    Ensure you are including all areas in your calculations"
)

# Automatically extract the shear line by splitting at "_" and taking the first section
flexible_ew["line"] = flexible_ew.index.get_level_values("Wall").str.split("--").str[0]

# Get tributary area and wall length for the shear line
flexible_ew["line tributary area"] = flexible_ew.groupby(["Level", "line"])["tributary area"].transform("sum")
flexible_ew["line wall length"] = flexible_ew.groupby(["Level", "line"])["wall length"].transform("sum")

flexible_ew["line shear demand"] = flexible_ew["line tributary area"] * flexible_ew["tributary weight"]
flexible_ew["line unit shear"] = flexible_ew["line shear demand"] / flexible_ew["line wall length"]
flexible_ew["line cumulative shear demand"] = flexible_ew.groupby(level="Wall")["line shear demand"].cumsum()
flexible_ew["line cumulative unit shear"] = flexible_ew["line cumulative shear demand"] / flexible_ew["line wall length"]

flexible_ew["shear demand"] = flexible_ew["line unit shear"] * flexible_ew["wall length"]
flexible_ew["cumulative shear demand"] = flexible_ew.groupby(level="Wall")["shear demand"].cumsum()
flexible_ew["unit shear demand"] = flexible_ew["cumulative shear demand"] / flexible_ew["wall length"]
flexible_ew["adjusted unit shear demand"] = flexible_ew["unit shear demand"] * LOAD_FACTOR_SEISMIC_ASD

flexible_ew["sheathed sides"] = 1
flexible_ew["aspect ratio"] = flexible_ew["wall height"] / flexible_ew["wall length"]
flexible_ew["aspect ratio factor"] = np.where(
    flexible_ew["aspect ratio"] > 2, 1.25 - 0.125 * flexible_ew["wall height"] / flexible_ew["wall length"], 1
)

# %% [markdown]
# #### Check aspect ratios
#
# We will also check the aspect ratios of the E-W walls.

# %%
# %%render
ratio_max = check_value(flexible_ew["aspect ratio"].max(), 2, "<=")

# %%
if "OK" in ratio_max:
    display_text(r"$\therefore$ All aspect ratios are OK")
else:
    display_text("The aspect ratio knockdown factor must be applied.")

# %%
# using sheathing from the NS calc in memory

flexible_ew[["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"]] = get_sheathing_properties(
    flexible_ew[["adjusted unit shear demand", "sheathed sides"]],
    sheathing=sheathing,
)

# Assign 2-sided sheathing to walls requiring it, then recalc capacity for just those walls (more efficient than passing the full wall dataframe)
if (flexible_ew["shear dcr"] > shear_dcr_check_value).any():
    flexible_ew.loc[flexible_ew["shear dcr"] > shear_dcr_check_value, "sheathed sides"] = 2
    flexible_ew.loc[
        flexible_ew["shear dcr"] > shear_dcr_check_value,
        ["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"],
    ] = get_sheathing_properties(
        flexible_ew.loc[
            flexible_ew["shear dcr"] > shear_dcr_check_value,
            ["adjusted unit shear demand", "sheathed sides"],
        ],
        sheathing=sheathing,
    )

# %%
# Update dataframe
flexible.update(flexible_ew)

# %%
# Rename columns for display with LaTeX formatting for units and subscripts/superscripts
column_name_map = {
    "tributary weight": "$W_{trib}$ [psf]",
    "tributary area": "$A$ [ft$^2$]",
    "shear demand": "$F_x$ [lbf]",
    "cumulative shear demand": "$F_{x,cum}$ [lbf]",
    "wall length": "$l_{wall}$ [ft]",
    "unit shear demand": "$v$ [plf]",
    "adjusted unit shear demand": "$0.7 v$ [plf]",
    "adjusted unit shear capacity": "$v_{cap}$ [plf]",
    "sheathed sides": "sides",
    "nail spacing": "$s_{nail}$ [in]",
}

# %%
display_table(
    dataframe=flexible_ew,
    levels="Level 5",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces EW - Level 5",
    label="tab:shear_wall_forces_roof_ew",
)
display_table(
    dataframe=flexible_ew,
    levels="Level 3",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces EW - Level 3",
    label="tab:shear_wall_forces_level3_ew",
)
display_table(
    dataframe=flexible_ew,
    levels="Level 2",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces EW - Level 2",
    label="tab:shear_wall_forces_level2_ew",
)
display_table(
    dataframe=flexible_ew,
    levels="Level 1",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces EW - Level 1",
    label="tab:shear_wall_forces_level1_ew",
)

# %% [markdown]
# ### Rigid Diaphragm Assumption

# %% [markdown]
# With a rigid diaphragm assumption, we now must calculate the following in each direction:
#
# 1. The relative stiffnesses of the shear walls to obtain a center of rigidity.
# 2. The center of mass of the diaphragm with the center of rigidity to obtain an eccentricity.
#
# The relative stiffnesses of the shear walls can be calculated by using the deflection equation from NDS SDPWS Sec. 4.3.4
#
# \begin{align*}
# k &= \frac{F}{\delta},\, \text{assume }F = \text{induced load demand in the shear wall} \\
# \delta_{sw} &= \frac{8vh^3}{EAb} + \frac{vh}{1000 G_a} + \frac{h \Delta_A}{b}\quad \text{(SDPWS Eq. 4.3-1)}
# \end{align*}
#
# where
#
# - $b =$ the shear wall length [ft]
# - $\Delta_A =$ vertical deformation of the wall overturning anchorage system + vertical compression deformation [in]
# - $E =$ modulus of elasticity of end posts [psi]
# - $A =$ area of end post cross-section [in$^2$]
# - $G_a =$ apparent shear wall shear stiffness from nail slip and panel shear deformation [kip/in]
# - $h =$ shear wall height [ft]
# - $v =$ unit shear force induced by the design load [plf]
# - $\delta_{sw} =$ maximum shear wall deflection determined by elastic analysis [in]
#
# Note that for the following calculations, we will conservatively use the unit shear force induced by the nominal load, not the design load, for the value of $v$. This is because the stiffnesses obtained from this use will allow us to use the stiffnesses for deflection limits and story drift calculations
#
# We will assume that the end posts are (2) 2x4 Douglas Fir No. 2; this means:

# %%
# %%render
E_endpost = 1_600_000 * psi
A_endpost = 2 * 5.25 * inch**2

# %% [markdown]
# We will assume a continuous tiedown rod system. A conservative first-pass value for $\Delta_A$ can be:

# %%
# %%render params
Delta_A = 0.25 * inch

# %% [markdown]
# #### Shear Wall Rigidities
#
# With the deflections of each shear wall, along with the force demand of each shear wall, we can obtain the stiffnesses (and therefore, relative stiffnesses) of all the walls.
#
# Each relative stiffness is relative to the level. The sum of all relative stiffnesses on each level is $1.0$.
#
# \begin{align*}
# k &= \frac{F_x}{\delta_{sw}} \\
# R &= \frac{k_{wall}}{\sum {k_{wall}}}
# \end{align*}

# %% [markdown]
# ##### North-South Direction

# %% [markdown]
# Rigidities in the N-S direction can be found in \autoref{tab:wall_rigidities_ns_roof} through \autoref{tab:wall_rigidities_ns_level2}.

# %%
rigid_ns["unit shear demand"] = flexible_ns["unit shear demand"]
rigid_ns["wall length"] = flexible_ns["wall length"]
rigid_ns["wall height"] = flexible_ns["wall height"]
rigid_ns["shear stiffness"] = flexible_ns["shear stiffness"]
rigid_ns["sheathed sides"] = flexible_ns["sheathed sides"]
rigid_ns["nail spacing"] = flexible_ns["nail spacing"]
rigid_ns["end post youngs modulus"] = float(E_endpost)
rigid_ns["end post area"] = float(A_endpost)
rigid_ns["Delta_A"] = float(Delta_A)

rigid_ns["delta_sw"] = (
    8
    * rigid_ns["unit shear demand"]
    * rigid_ns["wall height"] ** 3
    / (rigid_ns["end post youngs modulus"] * rigid_ns["end post area"] * rigid_ns["wall length"])
    + rigid_ns["unit shear demand"] * rigid_ns["wall height"] / (1000 * rigid_ns["shear stiffness"] * rigid_ns["sheathed sides"])
    + rigid_ns["wall height"] * rigid_ns["Delta_A"] / rigid_ns["wall length"]
)

# %%
rigid_ns["cumulative shear demand"] = flexible_ns["cumulative shear demand"]
rigid_ns["wall stiffness"] = rigid_ns["cumulative shear demand"] / rigid_ns["delta_sw"]
rigid_ns["total level stiffness"] = rigid_ns.groupby("Level")["wall stiffness"].transform("sum")
rigid_ns["total level cumulative shear"] = rigid_ns.groupby("Level")["cumulative shear demand"].transform("sum")
rigid_ns["relative wall stiffness"] = rigid_ns["wall stiffness"] / rigid_ns["total level stiffness"]
rigid_ns["direct shear force"] = rigid_ns["total level cumulative shear"] * rigid_ns["relative wall stiffness"]

# %%
column_name_map = {
    "cumulative shear demand": "$F_x$ [lbf]",
    "delta_sw": r"$\delta_{sw}$ [in]",
    "wall stiffness": "$k$ [lbf/in]",
    "relative wall stiffness": "$R$ [-]",
}

# %%
display_table(
    dataframe=rigid_ns,
    levels="Level 5",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 5",
    label="tab:wall_rigidities_ns_level5",
)
display_table(
    dataframe=rigid_ns,
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 4",
    label="tab:wall_rigidities_ns_level4",
)
display_table(
    dataframe=rigid_ns,
    levels="Level 3",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 3",
    label="tab:wall_rigidities_ns_level3",
)
display_table(
    dataframe=rigid_ns,
    levels="Level 2",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 2",
    label="tab:wall_rigidities_ns_level2",
)
display_table(
    dataframe=rigid_ns,
    levels="Level 1",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 1",
    label="tab:wall_rigidities_ns_level1",
)

# %% [markdown]
# ##### East-West Direction

# %% [markdown]
# Rigidities in the E-W direction can be found in \autoref{tab:wall_rigidities_ew_roof}, \autoref{tab:wall_rigidities_ew_level4}, \autoref{tab:wall_rigidities_ew_level3}, \autoref{tab:wall_rigidities_ew_level2},

# %%
rigid_ew["unit shear demand"] = flexible_ew["unit shear demand"]
rigid_ew["wall length"] = flexible_ew["wall length"]
rigid_ew["wall height"] = flexible_ew["wall height"]
rigid_ew["shear stiffness"] = flexible_ew["shear stiffness"]
rigid_ew["sheathed sides"] = flexible_ew["sheathed sides"]
rigid_ew["nail spacing"] = flexible_ew["nail spacing"]
rigid_ew["end post youngs modulus"] = float(E_endpost)
rigid_ew["end post area"] = float(A_endpost)
rigid_ew["Delta_A"] = float(Delta_A)

rigid_ew["delta_sw"] = (
    8
    * rigid_ew["unit shear demand"]
    * rigid_ew["wall height"] ** 3
    / (rigid_ew["end post youngs modulus"] * rigid_ew["end post area"] * rigid_ew["wall length"])
    + rigid_ew["unit shear demand"] * rigid_ew["wall height"] / (1000 * rigid_ew["shear stiffness"] * rigid_ew["sheathed sides"])
    + rigid_ew["wall height"] * rigid_ew["Delta_A"] / rigid_ew["wall length"]
)

# %%
rigid_ew["cumulative shear demand"] = flexible_ew["cumulative shear demand"]
rigid_ew["wall stiffness"] = rigid_ew["cumulative shear demand"] / rigid_ew["delta_sw"]
rigid_ew["total level stiffness"] = rigid_ew.groupby("Level")["wall stiffness"].transform("sum")
rigid_ew["total level cumulative shear"] = rigid_ew.groupby("Level")["cumulative shear demand"].transform("sum")
rigid_ew["relative wall stiffness"] = rigid_ew["wall stiffness"] / rigid_ew["total level stiffness"]
rigid_ew["direct shear force"] = rigid_ew["total level cumulative shear"] * rigid_ew["relative wall stiffness"]

# %%
column_name_map = {
    "cumulative shear demand": "$F_x$ [lbf]",
    "delta_sw": r"$\delta_{sw}$ [in]",
    "wall stiffness": "$k$ [lbf/in]",
    "relative wall stiffness": "$R$ [-]",
}

# %%
display_table(
    dataframe=rigid_ew,
    levels="Level 5",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities EW - Level 5",
    label="tab:wall_rigidities_ew_level5",
)
display_table(
    dataframe=rigid_ns,
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities NS - Level 4",
    label="tab:wall_rigidities_ns_level4",
)
display_table(
    dataframe=rigid_ew,
    levels="Level 3",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities EW - Level 3",
    label="tab:wall_rigidities_ew_level3",
)
display_table(
    dataframe=rigid_ew,
    levels="Level 2",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities EW - Level 2",
    label="tab:wall_rigidities_ew_level2",
)
display_table(
    dataframe=rigid_ew,
    levels="Level 1",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities EW - Level 1",
    label="tab:wall_rigidities_ew_level1",
)

# %% [markdown]
# #### Center of Rigidity and Center of Mass
#
# We can find the center of rigidity by taking the weighted sum of all wall stiffnesses and their locations relative to a datum. In this case, we will use the north-west-most corner of the building (i.e., the intersection of lines B-1 and B-A).
#
# $$
# \bar{x}_r = \frac{\sum k_{ns} x}{\sum k_{ns}},\, \bar{y}_r = \frac{\sum k_{ew} y}{\sum k_{ew}}
# $$
#
# Note: each wall is assumed to have zero rigidity in the direction perpendicular to them.
#
# We will get the center of mass, $CM_i$, for each direction by taking the weighted sum of the centroids of each of the zones 1, 2, 3, and 4.

# %%
set_params_columns(4)

# %%
# %%render params
x_1 = (131 + 10 / 12) / 2
x_2 = ((170 + 2 / 12) / 2) + float(L_jog)
x_3 = ((30 + (9 / 12) + 5 + (6 / 12)) / 2) + (131 + 10 / 12)
x_4 = ((30 + (9 / 12)) / 2) + (131 + (10 / 12) + 30 + (9 / 12) + 5 + (6 / 12))

y_1 = ((30 + (9 / 12) + 5 + (6 / 12)) / 2) + (83 + 10 / 12)
y_2 = ((30 + (9 / 12)) / 2) + (83 + (10 / 12) + 30 + (9 / 12) + 5 + (6 / 12))
y_3 = (115 + 1 / 12) / 2 + float(L_jog)
y_4 = (120 + 1 / 12) / 2

A_1 = L_1 * w_1
A_2 = L_2 * w_2
A_3 = L_3 * w_3
A_4 = L_4 * w_4

# %%
# %%render long
CM_x = (x_1 * A_1 + x_2 * A_2 + x_3 * A_3 + x_4 * A_4) / A_level
CM_y = (y_1 * A_1 + y_2 * A_2 + y_3 * A_3 + y_4 * A_4) / A_level

# %% [markdown]
# An accidental torsion eccentricity of 5% must be added/subtracted.

# %%
# %%render
X_plan = 198 + 10 / 12
Y_plan = 150 + 10 / 12
CM_x__a__plus = CM_x + X_plan * 0.05
CM_x__a__minus = CM_x - X_plan * 0.05
CM_y__a__plus = CM_y + Y_plan * 0.05
CM_y__a__minus = CM_y - Y_plan * 0.05

# %% [markdown]
# The eccentricity used is determined by the location of the wall in relation to the center of rigidity. Since the wall rotates about the center of rigidity, we want to position the center of mass to ensure each wall is getting the maximum possible force to be conservative. If the center of mass is positioned on the opposite side of the center of rigidity relative to the wall in question, that mass will _decrease_ the shear experienced by that wall.
#
# This means we always want the center of mass to be on the same side of the center of rigidity as the wall. The eccentricities shown in the following tables are based on the worst-case scenario. This is not immediately obvious due to all torques being given a positive value.
#
# How the center of rigidity, center of mass, and eccentricity interact are illustrated in \autoref{fig:rigid_diaphragm_moments}. The moments generated by the rotation of the center of mass around the center of rigidity differ based on the chosen eccentricity.
#
# ![Moments applied by a rigid diaphragm](images/rigid_diaphragm_moments.png){width=50% #fig:rigid_diaphragm_moments}
#
# $$
# d_x = \left| x - \text{CR}_x \right|\text{ and }\, d_y = \left| y - \text{CR}_y \right|
# $$

# %%
# Update dataframe
rigid = rigid.reindex(columns=rigid_ns.columns)
rigid.update(rigid_ns)
rigid.update(rigid_ew)

# %%
# Location of wall lines of rigidity
wall_rigidity_loc = {
    # N-S
    # zone 1 walls
    "E-2": {"x": float(l_E1__E2), "y": 0},
    "E-3": {"x": float(l_E1__E2 + l_E2__E3), "y": 0},
    "E-3.3": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E33), "y": 0},
    "E-4": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4), "y": 0},
    "E-5": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5), "y": 0},
    # "E-6": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6), "y": 0},
    # zone 2 walls
    "E-2.4": {"x": float(l_E1__E24), "y": 0},
    "E-3.4": {"x": float(l_E1__E24 + l_E24__E34), "y": 0},
    "E-4.8": {"x": float(l_E1__E24 + l_E24__E34 + l_E34__E48), "y": 0},
    "E-5.9": {"x": float(l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59), "y": 0},
    "E-7.4": {"x": float(l_E1__E24 + l_E24__E34 + l_E34__E48 + l_E48__E59 + l_E59__E74), "y": 0},
    # zone 3 and 4 walls
    "E-8--E-A.2": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-8--E-B": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-8--E-B.7": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-8--E-C.3": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-8--E-F": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-8--E-F.7": {"x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8), "y": 0},
    "E-9--E-B.1": {
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
    },
    "E-9--E-B.6": {
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
    },
    "E-9--E-C.5": {
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
    },
    "E-9--E-F.4": {
        "x": float(l_E1__E2 + l_E2__E3 + l_E3__E4 + l_E4__E5 + l_E5__E6 + l_E6__E7 + l_E7__E8 + l_E8__E9),
        "y": 0,
    },
    # E-W
    # zone 3 walls
    "E-B": {"x": 0, "y": float(l_EA__EB)},
    "E-C": {"x": 0, "y": float(l_EA__EB + l_EB__EC)},
    "E-D": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED)},
    # zone 4 walls
    "E-B.1": {"x": 0, "y": float(l_EA__EB1)},
    "E-C.2": {"x": 0, "y": float(l_EA__EB1 + l_EB1__EC2)},
    "E-C.5": {"x": 0, "y": float(l_EA__EB1 + l_EB1__EC2 + l_EC2__EC5)},
    "E-F.2": {"x": 0, "y": float(l_EA__EB1 + l_EB1__EC2 + l_EC2__EC5 + l_EC5__EF2)},
    # zones 1 and 2 walls
    "E-G--E-2": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-G--E-2.7": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-G--E-3.4": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-G--E-3.8": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-G--E-4.4": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    # "E-G--E-5.7": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-G--E-7": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG)},
    "E-H--E-1.2": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    "E-H--E-2.4": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    "E-H--E-3.1": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    "E-H--E-4.1": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    # "E-H--E-5.5": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    "E-H--E-7": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    "E-H--E-7.6": {"x": 0, "y": float(l_EA__EB + l_EB__EC + l_EC__ED + l_ED__EF + l_EF__EG + l_EG__EH)},
    # N-S
}

# %%
# Calc centers of rigidity
rigid["x"] = rigid.index.get_level_values("Wall").map({key: value["x"] for key, value in wall_rigidity_loc.items()})
rigid["y"] = rigid.index.get_level_values("Wall").map({key: value["y"] for key, value in wall_rigidity_loc.items()})

rigid["kx"] = rigid["relative wall stiffness"] * rigid["x"]
rigid["CRx"] = rigid.groupby(["Level", "Direction"])["kx"].transform("sum")  # all relative wall rigidities sum to 1
rigid["ky"] = rigid["relative wall stiffness"] * rigid["y"]
rigid["CRy"] = rigid.groupby(["Level", "Direction"])["ky"].transform("sum")  # all relative wall rigidities sum to 1

# x eccentricity. Use the "+" eccentricity for walls on the + side of the CR. "-" for - side
plan_dim_x = float(X_plan)
CMx = CM_x
CMxa_plus = CM_x__a__plus
CMxa_minus = CM_x__a__minus
rigid["ex"] = np.where(rigid["x"] <= rigid["CRx"], rigid["CRx"] - CMxa_minus, CMxa_plus - rigid["CRx"])
rigid["ex"] = np.where(rigid["ex"] <= 0, 0.05 * plan_dim_x, rigid["ex"])  # assign 0s to negatives

# y eccentricity. Use the "+" eccentricity for walls on the + side of the CR. "-" for - side
plan_dim_y = float(Y_plan)
CMy = CM_y
CMya_plus = CM_y__a__plus
CMya_minus = CM_y__a__minus
rigid["ey"] = np.where(rigid["y"] <= rigid["CRy"], rigid["CRy"] - CMya_minus, CMya_plus - rigid["CRy"])
rigid["ey"] = np.where(rigid["ey"] <= 0, 0.05 * plan_dim_y, rigid["ey"])  # assign 0s to negatives

# distance from CR
rigid["dx"] = abs(rigid["x"] - rigid["CRx"])
rigid["dy"] = abs(rigid["y"] - rigid["CRy"])

# Torque
rigid["torque x"] = rigid["total level cumulative shear"] * rigid["ey"]
rigid["torque y"] = rigid["total level cumulative shear"] * rigid["ex"]

# Polar moment of inertia
rigid["Jx wall"] = rigid["relative wall stiffness"] * rigid["dx"] ** 2
rigid["Jy wall"] = rigid["relative wall stiffness"] * rigid["dy"] ** 2
rigid["J"] = (rigid["Jx wall"] + rigid["Jy wall"]).groupby(["Level"]).transform("sum")

rigid["torsional shear force x"] = rigid["torque x"] * rigid["relative wall stiffness"] * rigid["dx"] / rigid["J"]
rigid["torsional shear force y"] = rigid["torque y"] * rigid["relative wall stiffness"] * rigid["dy"] / rigid["J"]
rigid["torsional shear force"] = np.where(
    rigid["torsional shear force x"] == 0, rigid["torsional shear force y"], rigid["torsional shear force x"]
)
rigid["shear force demand"] = rigid["torsional shear force"] + rigid["direct shear force"]

# %% [markdown]
# The centers of rigidity for each floor are given in \autoref{tab:center_of_rigidity_roof} through \autoref{tab:center_of_rigidity_level2}.

# %%
column_name_map = {
    "x": "$x$ [ft]",
    "y": "$y$ [ft]",
    "CRx": "CR$_x$ [ft]",
    "dx": "$d_x$ [ft]",
    "ex": "e$_x$ [ft]",
    "CRy": "CR$_y$ [ft]",
    "dy": "$d_y$ [ft]",
    "ey": "e$_y$ [ft]",
}
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 5",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Center of Rigidity - Level 5",
    label="tab:center_of_rigidity_level5",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 4",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Center of Rigidity - Level 4",
    label="tab:center_of_rigidity_level4",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 3",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Center of Rigidity - Level 3",
    label="tab:center_of_rigidity_level3",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 2",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Center of Rigidity - Level 2",
    label="tab:center_of_rigidity_level2",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 1",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Center of Rigidity - Level 1",
    label="tab:center_of_rigidity_level1",
)

# %% [markdown]
# And now, we can get the forces in the shear walls using the rigid diaphragm assumption in \autoref{tab:rigid_forces_roof} through \autoref{tab:rigid_forces_level2}. These forces can be compared to the flexible shear wall forces obtained earlier, $F_{flex}$.

# %%
rigid["flexible shear force demand"] = flexible["cumulative shear demand"]

# %%
column_name_map = {
    "torque x": "$T_x$ [lbf-ft]",
    "torque y": "$T_y$ [lbf-ft]",
    "dx": "$d_x$ [ft]",
    "dy": "$d_y$ [ft]",
    "relative wall stiffness": "$R$ [-]",
    "torsional shear force": "$F_t$ [lbf]",
    "direct shear force": "$F_v$ [lbf]",
    "shear force demand": "$F_{tot}$ [lbf]",
    "flexible shear force demand": "$F_{flex}$ [lbf]",
}
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 5",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Shear wall forces with the rigid diaphragm assumption - Level 5",
    label="tab:rigid_forces_level5",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 4",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Shear wall forces with the rigid diaphragm assumption - Level 4",
    label="tab:rigid_forces_level4",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 3",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Shear wall forces with the rigid diaphragm assumption - Level 3",
    label="tab:rigid_forces_level3",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 2",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Shear wall forces with the rigid diaphragm assumption - Level 2",
    label="tab:rigid_forces_level2",
)
display_table(
    dataframe=rigid,
    column_names_filter_and_map=column_name_map,
    levels="Level 1",
    position_float="centering",
    formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
    caption="Shear wall forces with the rigid diaphragm assumption - Level 1",
    label="tab:rigid_forces_level1",
)

# %% [markdown]
# ### Final Shear Wall Nailing Selection
#
# With the envelope procedure complete, we can now select the shear wall sheathing and nailing specifications in \autoref{tab:shear_walls_roof} through \autoref{tab:shear_walls_level2}.

# %%
shear_walls["wall length"] = rigid["wall length"]
shear_walls["rigid shear force demand"] = rigid["shear force demand"]
shear_walls["flexible shear force demand"] = rigid["flexible shear force demand"]
shear_walls["shear force demand"] = np.where(
    shear_walls["flexible shear force demand"] >= shear_walls["rigid shear force demand"],
    shear_walls["flexible shear force demand"],
    shear_walls["rigid shear force demand"],
)
shear_walls["unit shear demand"] = shear_walls["shear force demand"] / shear_walls["wall length"]
shear_walls["adjusted unit shear demand"] = (
    shear_walls["shear force demand"] / shear_walls["wall length"] * LOAD_FACTOR_SEISMIC_ASD
)
shear_walls["sheathed sides"] = 1

# using sheathing from the NS calc in memory

shear_walls[["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"]] = get_sheathing_properties(
    shear_walls[["adjusted unit shear demand", "sheathed sides"]],
    sheathing=sheathing,
)

# Assign 2-sided sheathing to walls requiring it, then recalc capacity for just those walls (more efficient than passing the full wall dataframe)
if (shear_walls["shear dcr"] > shear_dcr_check_value).any():
    shear_walls.loc[shear_walls["shear dcr"] > shear_dcr_check_value, "sheathed sides"] = 2
    shear_walls.loc[
        shear_walls["shear dcr"] > shear_dcr_check_value,
        ["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"],
    ] = get_sheathing_properties(
        shear_walls.loc[
            shear_walls["shear dcr"] > shear_dcr_check_value,
            ["adjusted unit shear demand", "sheathed sides"],
        ],
        sheathing=sheathing,
    )

column_name_map = {
    "shear force demand": "$F_{x,cum}$ [lbf]",
    "wall length": "$l_{wall}$ [ft]",
    "unit shear demand": "$v$ [plf]",
    "adjusted unit shear demand": "$0.7 v$ [plf]",
    "adjusted unit shear capacity": "$v_{cap}$ [plf]",
    "sheathed sides": "sides",
    "nail spacing": "$s_{nail}$ [in]",
}

display_table(
    dataframe=shear_walls,
    column_names_filter_and_map=column_name_map,
    levels="Level 5",
    position_float="centering",
    caption="Final shear wall design values - Level 5",
    label="tab:shear_walls_level5",
)
display_table(
    dataframe=shear_walls,
    column_names_filter_and_map=column_name_map,
    levels="Level 4",
    position_float="centering",
    caption="Final shear wall design values - Level 4",
    label="tab:shear_walls_level4",
)
display_table(
    dataframe=shear_walls,
    column_names_filter_and_map=column_name_map,
    levels="Level 3",
    position_float="centering",
    caption="Final shear wall design values - Level 3",
    label="tab:shear_walls_level3",
)
display_table(
    dataframe=shear_walls,
    column_names_filter_and_map=column_name_map,
    levels="Level 2",
    position_float="centering",
    caption="Final shear wall design values - Level 2",
    label="tab:shear_walls_level2",
)
display_table(
    dataframe=shear_walls,
    column_names_filter_and_map=column_name_map,
    levels="Level 1",
    position_float="centering",
    caption="Final shear wall design values - Level 1",
    label="tab:shear_walls_level1",
)

# %% [markdown]
# ### Shear Wall Axial Forces

# %% [markdown]
# #### Tension Forces

# %% [markdown]
# The shear forces get converted to overturning forces in the end posts of the shear walls. This overturning moment is counteracted by the dead load tributary to each shear wall according to ASCE 7-22 Sec. 2.4.5. The applicable equation is:
#
# $$
# 0.6 D - 0.7 E_v + 0.7 E_h \quad\text{(ASCE 7-22 Eq. 2.4.5-10)}
# $$
#
# There are no vertical seismic effects, $E_v$, so we will only consider the horizontal seismic effects, $E_h$, and dead loads, $D$.
#
# The overturning moment is:
#
# $$
# M_{OT} = V h_{wall}
# $$
#
# Using the area dead loads from earlier, along with the tributary areas for each shear wall, we can get the dead load that is applied to each wall. This resists the overturning moment at the centroid of the wall.
#
# $$
# M_R = \frac{D l_{wall}}{2}
# $$
#
# We will assume that the holddowns are 6" inset from the end posts. The tension caused by horizontal load effects counteracted by dead loads is:
#
# $$
# T_{seismic} = \frac{M_{OT} - M_R}{l_{wall} - 0.5}\quad\text{(units in feet)}
# $$

# %%
shear_walls["line"] = flexible["line"]
shear_walls["wall height"] = rigid["wall height"]
shear_walls["tributary area"] = flexible["tributary area"]
shear_walls["floor OTM"] = shear_walls["shear force demand"] * shear_walls["wall height"]
shear_walls["OTM"] = shear_walls.groupby(level="Wall")["floor OTM"].cumsum()
shear_walls["dead load"] = 0.6 * float(U_floor * 1000) * shear_walls["tributary area"]

# Reduce resisting moment dead load in some walls that definitely are not correct
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "E-2", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "E-2.4", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "E-3.4", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "E-5", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "E-7.4", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls["line"] == "E-8", "dead load"] *= 0
shear_walls.loc[shear_walls["line"] == "E-9", "dead load"] *= 0
shear_walls.loc[shear_walls["line"] == "E-B.1", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls["line"] == "E-D", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls["line"] == "E-F.2", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls["line"] == "E-G", "dead load"] *= 0
shear_walls.loc[shear_walls["line"] == "E-H", "dead load"] *= 0

shear_walls["RM"] = shear_walls["dead load"] * shear_walls["wall length"] / 2

shear_walls["total moment"] = shear_walls["OTM"] - shear_walls["RM"]
shear_walls["total moment"] = np.where(shear_walls["RM"] <= shear_walls["OTM"], shear_walls["OTM"] - shear_walls["RM"], 0)
wall_arm_shortening_length = 0.5  # feet due to holddown positioning
shear_walls["tension force"] = shear_walls["total moment"] / (shear_walls["wall length"] - wall_arm_shortening_length)
num_bins = 5
shear_walls["binned tension force"] = (
    shear_walls["tension force"]
    .groupby(level="Level")
    .transform(lambda s: pd.qcut(s, q=num_bins, duplicates="drop").map(lambda x: x.right if pd.notna(x) else 0))
)

# %% [markdown]
# #### Compression Forces

# %% [markdown]
# We will calculate the compression forces in a very similar way. However, we will ignore the resisting moment.
#
# The live load, $L$, is $40$ psf, while the snow load, $S$, is $25$ psf and applies only to corridor walls as that is how the roof trusses are supported.
#
# $$
# C_{seismic} = \frac{M_{OT}}{l_{wall} - 0.5}\quad\text{(units in feet)}
# $$

# %%
shear_walls["compression force"] = shear_walls["OTM"] / (shear_walls["wall length"] - wall_arm_shortening_length)
shear_walls["binned compression force"] = (
    shear_walls["compression force"]
    .groupby(level="Level")
    .transform(lambda s: pd.qcut(s, q=num_bins).map(lambda x: x.right if pd.notna(x) else 0))
    # .transform(lambda s: pd.qcut(s, q=num_bins, duplicates="drop").map(lambda x: x.right if pd.notna(x) else pd.NA))
)

# %% [markdown]
# #### Holddown Forces Table
#
# With the tension and compression forces calculated, we can construct a table of the forces for each level. We will default to 5 bins for each level to give us options A through E.
#
# However, there is only one table in the plans for all the holddown forces, so these will all get compared together as an aggregate of all buildings and simplified further.

# %%
NUM_OPTIONS = 5
LABELS = list("ABCDE")

schedule_rows = []


def assign_schedule(group):
    group = group.copy()

    level = group.index.get_level_values("Level")[0]

    n = len(group)
    group["_bin"] = np.floor(np.arange(n) * min(NUM_OPTIONS, n) / n)

    schedule = (
        group.groupby("_bin")
        .agg(
            tension=("tension force", "max"),
            compression=("compression force", "max"),
        )
        .reset_index()
    )

    # convert to kips
    schedule["tension"] /= 1000
    schedule["compression"] /= 1000

    # sort ONLY schedule
    schedule["governing"] = schedule[["tension", "compression"]].max(axis=1)
    schedule = schedule.sort_values("governing").reset_index(drop=True)

    # assign A–E
    schedule["option"] = LABELS[: len(schedule)]

    # IMPORTANT: keep level in schedule
    schedule["Level"] = level

    # map bins → options
    bin_to_option = dict(zip(schedule["_bin"], schedule["option"]))
    group["option"] = group["_bin"].map(bin_to_option)

    schedule_map = schedule.set_index("option")[["tension", "compression"]]

    group["binned tension force"] = group["option"].map(schedule_map["tension"])
    group["binned compression force"] = group["option"].map(schedule_map["compression"])

    schedule_rows.append(schedule[["Level", "option", "tension", "compression"]])

    return group.drop(columns=["_bin"])


shear_walls = shear_walls.groupby(level="Level", group_keys=False).apply(assign_schedule)

schedule_df = (
    pd.concat(schedule_rows, ignore_index=True)
    .rename(
        columns={
            "tension": "binned tension force",
            "compression": "binned compression force",
        }
    )
    .set_index(["Level", "option"])
    .sort_index()
)

# %%
column_name_map = {
    "binned tension force": "$T$ [kip]",
    "binned compression force": "$C$ [kip]",
}
display_table(
    schedule_df,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Holddown Schedule",
    label="tab:holddown_schedule",
)

# %% [markdown]
# ## Diaphragm Design

# %% [markdown]
# We must first get the diaphragm inertial design forces. These are distinct from the seismic forces determined in \autoref{tab:seismic_forces_calculation_table}.
#
# \begin{align*}
# F_{px} &= \frac{\sum_{i=x}^n F_i}{\sum_{i=x}^n w_i} w_{px} \quad \text{(ASCE Eq. 12.10-1)} \\
# F_{px} &\ge 0.2 S_{DS} I_e w_{px} \quad \text{(ASCE Eq. 12.10-2)} \\
# F_{px} &\le 0.4 S_{DS} I_e w_{px}  \quad \text{(ASCE Eq. 12.10-3)}
# \end{align*}
#
# These forces are shown in \autoref{tab:diaphragm_forces}

# %%
seismic_loads["cumulative floor weight"] = seismic_loads.iloc[::-1]["floor weight"].cumsum()
seismic_loads["F_x,cum"] = seismic_loads.iloc[::-1]["F_x"].cumsum()
seismic_loads["F_px lower bound"] = 0.2 * S_DS * I_e * seismic_loads["floor weight"]
seismic_loads["F_px upper bound"] = 0.4 * S_DS * I_e * seismic_loads["floor weight"]
seismic_loads["F_px initial"] = (
    seismic_loads["F_x,cum"] / seismic_loads["cumulative floor weight"] * seismic_loads["floor weight"]
)

# Assign F_px with limits from ASCE
seismic_loads["F_px"] = np.where(
    seismic_loads["F_px initial"] <= seismic_loads["F_px lower bound"],
    seismic_loads["F_px lower bound"],
    np.where(
        seismic_loads["F_px initial"] >= seismic_loads["F_px upper bound"],
        seismic_loads["F_px upper bound"],
        seismic_loads["F_px initial"],
    ),
)

# %%
column_name_map = {
    "floor weight": "$w_{px}$ [kip]",
    "cumulative floor weight": r"$\sum w_x$ [kip]",
    "F_x,cum": r"$\sum F_x$ [kip]",
    "F_px initial": "$F_{px,init}$ [kip]",
    "F_px lower bound": "$F_{px,lower}$ [kip]",
    "F_px upper bound": "$F_{px,upper}$ [kip]",
    "F_px": "$F_{px}$ [kip]",
}
display_table(
    dataframe=seismic_loads.iloc[:0:-1],
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Diaphragm Forces",
    label="tab:diaphragm_forces",
)

# %% [markdown]
# With these diaphragm forces, we can design the diaphragm for the worst-case cantilever and simply-supported conditions.
#
# The longest simply-supported span in the N-S direction is about 34.5 feet, while the longest cantilever span is about 32 feet. The depth of the diaphragm in the N-S direction is 67 feet. The longest simply-supported span in the E-W direction is about 5.5 feet, while the longest cantilever span is about 31 feet. The depth of the diaphragm in the E-W direction is 230 feet.
#
# Diaphragm unit shear demands are shown in \autoref{tab:diaphragm_unit_shears}

# %%
# Set up diaphragm dataframe
levels = ["Level 5", "Level 4", "Level 3", "Level 2", "Level 1"]
directions = ["N-S", "E-W"]
index = pd.MultiIndex.from_product(
    [levels, directions],
    names=["Level", "Direction"],
)
diaphragms = pd.DataFrame(index=index)
diaphragms_ns = diaphragms.xs("N-S", level="Direction").copy()
diaphragms_ew = diaphragms.xs("E-W", level="Direction").copy()

# %%
diaphragms_ns["F_px"] = seismic_loads["F_px"] * 1000
diaphragms_ns["area load"] = diaphragms_ns["F_px"] / float(A_level)
diaphragms_ns["L_simp"] = 35
diaphragms_ns["L_cant"] = 35
diaphragms_ns["depth"] = 67
diaphragms_ns["V_simp"] = diaphragms_ns["area load"] * diaphragms_ns["depth"] * diaphragms_ns["L_simp"] / 2
diaphragms_ns["V_cant"] = diaphragms_ns["area load"] * diaphragms_ns["depth"] * diaphragms_ns["L_cant"]
diaphragms_ns["v_simp"] = diaphragms_ns["V_simp"] / diaphragms_ns["depth"]
diaphragms_ns["v_cant"] = diaphragms_ns["V_cant"] / diaphragms_ns["depth"]

# %% [markdown]
# We will assume the diaphragm sheathing has the following properties:
#
# - Unblocked
# - WSP Sheathing
# - 15/32" thick
# - 10d common nails
# - Plywood
# - Nailed face is 2"
# - Edge adjoining cases 2 through 6
#
# The diaphragm capacity checks for this chosen sheathing are shown in \autoref{tab:diaphragm_capacities}

# %%
# Set baseline sheathing properties. We set the number of sheathed sides later
from structural_tools.seismic.sheathing import SheathingApplication

diaphragm_sheathing = Sheathing(
    sheathing_material=SheathingMaterial.WSP_SHEATHING,
    minimum_nominal_panel_thickness=15 / 32,
    nail=Nail.COMMON_10D,
    panel_type=PanelType.PLY,
    minimum_nominal_width_of_nailed_face=2,
    adjoining_panel_edge_location=(2, 3, 4, 5, 6),
)

diaphragms_ns[
    ["adjusted unit shear capacity simply supported", "shear stiffness simply supported", "shear dcr simply supported"]
] = get_sheathing_properties(diaphragms_ns["v_simp"], sheathing=sheathing, sheathing_application=SheathingApplication.DIAPHRAGM)

diaphragms_ns[["adjusted unit shear capacity cantilever", "shear stiffness cantilever", "shear dcr cantilever"]] = (
    get_sheathing_properties(diaphragms_ns["v_cant"], sheathing=sheathing, sheathing_application=SheathingApplication.DIAPHRAGM)
)

# %%
diaphragms_ew["F_px"] = seismic_loads["F_px"] * 1000
diaphragms_ew["area load"] = diaphragms_ew["F_px"] / float(A_level)
diaphragms_ew["L_simp"] = 25
diaphragms_ew["L_cant"] = 35
diaphragms_ew["depth"] = 67
diaphragms_ew["V_simp"] = diaphragms_ew["area load"] * diaphragms_ew["depth"] * diaphragms_ew["L_simp"] / 2
diaphragms_ew["V_cant"] = diaphragms_ew["area load"] * diaphragms_ew["depth"] * diaphragms_ew["L_cant"]
diaphragms_ew["v_simp"] = diaphragms_ew["V_simp"] / diaphragms_ew["depth"]
diaphragms_ew["v_cant"] = diaphragms_ew["V_cant"] / diaphragms_ew["depth"]

# %%
diaphragms_ew[
    ["adjusted unit shear capacity simply supported", "shear stiffness simply supported", "shear dcr simply supported"]
] = get_sheathing_properties(diaphragms_ew["v_simp"], sheathing=sheathing, sheathing_application=SheathingApplication.DIAPHRAGM)

diaphragms_ew[["adjusted unit shear capacity cantilever", "shear stiffness cantilever", "shear dcr cantilever"]] = (
    get_sheathing_properties(diaphragms_ew["v_cant"], sheathing=sheathing, sheathing_application=SheathingApplication.DIAPHRAGM)
)

# %%
idx = pd.IndexSlice
diaphragms = diaphragms.reindex(columns=diaphragms_ns.columns)
diaphragms_ns["Direction"] = "N-S"
diaphragms_ew["Direction"] = "E-W"
diaphragms_ns = diaphragms_ns.set_index("Direction", append=True)
diaphragms_ew = diaphragms_ew.set_index("Direction", append=True)
diaphragms.update(diaphragms_ns)
diaphragms.update(diaphragms_ew)

# %%
column_name_map = {
    "F_px": "$F_{px}$ [lbf]",
    "area load": "$w$ [psf]",
    "V_simp": "$V_{simp}$ [lbf]",
    "V_cant": "$V_{cant}$ [lbf]",
    "v_simp": "$v_{simp}$ [plf]",
    "v_cant": "$v_{cant}$ [plf]",
}
display_table(
    dataframe=diaphragms,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Diaphragm Unit Shear Demands",
    label="tab:diaphragm_unit_shears",
)

# %%
column_name_map = {
    "v_simp": "$v_{simp}$ [plf]",
    "adjusted unit shear capacity simply supported": "$v_{cap,simp}$ [plf]",
    "shear dcr simply supported": "DCR$_{simp}$ [-]",
    "v_cant": "$v_{cant}$ [plf]",
    "adjusted unit shear capacity cantilever": "$v_{cap,cant}$ [plf]",
    "shear dcr cantilever": "DCR$_{cant}$ [-]",
}
display_table(
    dataframe=diaphragms,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Diaphragm Capacity Checks",
    label="tab:diaphragm_capacities",
)

# %%
if (diaphragms[["shear dcr simply supported", "shear dcr cantilever"]] <= 1).all().all():  # double .all() because 2 columns
    display_text(r"$\therefore$ All shear capacities are OK")
else:
    display_text("You must check for new diaphragm nailing.")

# %% [markdown]
# ## Story Drift Determination

# %% [markdown]
# The inelastic design story drift is calculated as the difference in deflections between stories at their centers of mass.

# %% [markdown]
# We can find the shear wall line that is closest to the center of mass, then take the cumulative force on that line and divide it by the cumulative wall stiffness on that line. This will give us an approximate elastic story drift that will then be converted to inelastic story drift with $C_d$ and $I_e$.
#
# Note that this is a conservative estimation of inelastic story drift due to assuming the deflection of a single shear wall line, rather than the full deflection of the diaphragm as a rigid unit.

# %% [markdown]
# \begin{align*}
#     \delta_{xe} &= F/k \\
#     \delta_x &= \frac{C_d \delta_{xe}}{I_e}
# \end{align*}

# %%
# %%render
C_d
I_e

# %%
shear_walls["line"] = flexible["line"]
shear_walls["wall stiffness"] = rigid["wall stiffness"]
shear_walls["cumulative shear demand"] = flexible["cumulative shear demand"]
shear_walls["x"] = rigid["x"]
shear_walls["y"] = rigid["y"]
shear_walls["delta_sw"] = rigid["delta_sw"]


def line_deflection(group, line):
    line_walls = group[group["line"] == line]

    total_stiffness = line_walls["wall stiffness"].sum()
    total_shear = line_walls["cumulative shear demand"].sum()

    return total_shear / total_stiffness


def get_cm_line_deflection(group, CM_x, CM_y, C_d, I_e):
    x_wall = group.loc[(group["x"] - CM_x).abs().idxmin()]

    y_wall = group.loc[(group["y"] - CM_y).abs().idxmin()]

    x_line = x_wall["line"]
    y_line = y_wall["line"]

    x_delta_sw = line_deflection(group, x_line)
    y_delta_sw = line_deflection(group, y_line)

    drift_factor = C_d / I_e

    return pd.Series(
        {
            "x line": x_line,
            "x delta_sw": x_delta_sw,
            "x inelastic drift": x_delta_sw * drift_factor,
            "y line": y_line,
            "y delta_sw": y_delta_sw,
            "y inelastic drift": y_delta_sw * drift_factor,
        }
    )


story_drift = shear_walls.groupby(level="Level", sort=False).apply(
    get_cm_line_deflection,
    CM_x=CM_x,
    CM_y=CM_y,
    C_d=C_d,
    I_e=I_e,
)
story_drift["allowable story drift"] = seismic_loads["floor height"] * 0.02 * 12

# %%
column_name_map = {
    "x line": "N-S line",
    "x delta_sw": r"$\delta_{xe,ns}$ [in]",
    "x inelastic drift": r"$\delta_{x,ns}$ [in]",
    "y line": "E-W line",
    "y delta_sw": r"$\delta_{xe,ew}$ [in]",
    "y inelastic drift": r"$\delta_{x,ew}$ [in]",
    "allowable story drift": r"$\Delta_a$ [in]",
}
display_table(
    story_drift,
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Story Drifts",
    label="tab:story_drifts",
)

# %%
# Save dataframes to CSV
shear_walls.to_csv("shear_walls_E.csv")
