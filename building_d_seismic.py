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
# title: Building D Seismic Design
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
set_params_columns(4)

# %%
# %%render params
U_corridor = 35 / 1000 * ksf
U_floor = 35 / 1000 * ksf
U_roof = 25 / 1000 * ksf
U_wall = 12 / 1000 * ksf

# %% [markdown]
# ## Building Geometry

# %% [markdown]
# The lengths and widths of the weight areas are consistent across levels 2, 3, and 4.

# %%
set_params_columns(5)

# %%
# %%render params
L_floor = 204 + (10.25 / 12) * ft
w_floor = (30 + 9 / 12) * ft
L_corridor = L_floor
w_corridor = 5.5 * ft
w_building = (w_floor * 2) + w_corridor

# %% [markdown]
# The wall heights vary across the levels.

# %%
# %%render params
h_wall__1 = 13.5 * ft
h_wall__2 = 10.5 * ft
h_wall__3 = 10.5 * ft
h_wall__4 = 10 * ft

# %% [markdown]
# There is a small jog in the building floorplan. This does not affect the area, but the perimeter will be longer.

# %%
# %%render params
L_jog = 5 * ft

# %%
# %%render
P_wall = (2 * L_floor) + (2 * w_building) + (2 * L_jog)

# %% [markdown]
# Areas can now be calculated. Note that the areas of the walls are the actual wall areas; they have not been distributed to each floor based on tributary area, yet.

# %%
# %%render
A_floor = 2 * (L_floor * w_floor)
# This is a slight overestimate as the corridor doesn't stretch the full length of the building
A_corridor = L_corridor * w_corridor
A_level = L_floor * w_building
A_wall__1 = h_wall__1 * P_wall
A_wall__2 = h_wall__2 * P_wall
A_wall__3 = h_wall__3 * P_wall
A_wall__4 = h_wall__4 * P_wall

# %% [markdown]
# ## Component Weights

# %% [markdown]
# The typical floor and corridor weights are

# %%
# %%render
W_floor__typ = U_floor * A_floor
W_corridor__typ = U_corridor * A_corridor

# %% [markdown]
# This applies to levels 2, 3, and 4.

# %%
set_params_columns(3)

# %%
# %%render params
W_floor__2 = W_floor__typ
W_floor__3 = W_floor__typ
W_floor__4 = W_floor__typ
W_corridor__2 = W_corridor__typ
W_corridor__3 = W_corridor__typ
W_corridor__4 = W_corridor__typ

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
W_wall__roof = U_wall * (A_wall__4 / 2)

# %% [markdown]
# ## Level Weights

# %% [markdown]
# Each level's weight is

# %%
# %%render
W_level__2 = W_floor__2 + W_corridor__2 + W_wall__2
W_level__3 = W_floor__3 + W_corridor__3 + W_wall__3
W_level__4 = W_floor__4 + W_corridor__4 + W_wall__4
# be sure to add the weight of the truss later
W_level__roof = W_roof + W_wall__roof

# %% [markdown]
# ## Total Weight

# %% [markdown]
# The total weight of the structure is

# %%
# %%render long
W_total = W_level__2 + W_level__3 + W_level__4 + W_level__roof

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
h_top__roof = (60 + 3 / 12) * ft
h_bottom__roof = (48) * ft
h_n = np.mean([h_top__roof, h_bottom__roof])  # ASCE 7-22 Sec. 11.2

# %%
# %%render
h_n__check = check_value(h_n, 65 * ft, "<=")

# %%
h_n = float(h_n)

# %%
structure = Structure(
    lateral_system_x=LateralSystem(system_index="A16"),
    lateral_system_y=LateralSystem(system_index="A16"),
    levels_input={
        1: Level(height=h_wall__1, weight=0),  # ground level weight not considered
        2: Level(height=h_wall__2, weight=W_level__2),
        3: Level(height=h_wall__3, weight=W_level__3),
        4: Level(height=h_wall__4, weight=W_level__4),
        5: Level(height=0, weight=W_level__roof),  # roof level height not considered
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

# %%
# Set up diaphragm dataframe
# %%
# Plan dimensions
l_DA__DB = (30 + 9 / 12) * ft
l_DB__DC = (5 + 6 / 12) * ft
l_DC__DD = (30 + 9 / 12) * ft

d_trib_ew = L_floor

# %%
# Wall tributary widths and lengths
t_w__DB = l_DA__DB + l_DB__DC / 2
t_w__BC = l_DB__DC / 2 + l_DC__DD

t_l__DB__1 = 46.5
t_l__DB__2 = 28.25
t_l__DB__3 = 27.67
t_l__DB__4 = 27.75
t_l__DB__5 = 23
t_l__DB__6 = 18.33 + 1.24 / 12  # add 1.24 inches to pencil out 204'-10.25" length
t_l__DB__7 = 33.25

t_l__DC__1 = 37
t_l__DC__2 = 21.66
t_l__DC__3 = 16
t_l__DC__4 = 11.66
t_l__DC__5 = 33.67
t_l__DC__6 = 33.66 + 2.5 / 12  # add 2.5 inches to pencil out 204'-10.25" length
t_l__DC__7 = 51
# t_l__BC4__1 = 19

# %%
# Wall segment lengths
l_DB__1 = 9.67
l_DB__2 = 13.5
l_DB__3 = 16.75
l_DB__4 = 22.75
l_DB__5 = 18.25
l_DB__6 = 8.75
l_DB__7 = 14.67

l_DC__1 = 23.5
l_DC__2 = 10.25
l_DC__3 = 9.25
l_DC__4 = 6.75
l_DC__5 = 28.75
l_DC__6 = 14
l_DC__7 = 14.5
# l_DC4__1 = 19

# Add wall tributary widths and wall lengths to df
l_da__db = 29
l_dc__dd = 29
l_stairwall = 20
l_elevator = 10 + 7 / 12

# %%
# Plan dimensions
l_D1__D2 = (35 + 1.25 / 12) * ft
l_D2__D3 = (23 + 8 / 12) * ft
l_D3__D4 = (34 + 4 / 12) * ft
l_D4__D5 = (24 + 8 / 12) * ft
l_D5__D6 = (18 + 7 / 12) * ft
l_D6__D7 = (37 + 1 / 12) * ft
l_D7__D8 = (36 + 5 / 12) * ft

l_D1__D21 = (36 + 5.25 / 12) * ft
l_D21__D28 = (17 + 11 / 12) * ft
l_D28__D36 = (26 + 4 / 12) * ft
l_D36__D43 = (21 + 4 / 12) * ft
l_D43__D6 = (34 + 4 / 12) * ft
l_D6__D67 = (26 + 1 / 12) * ft
l_D67__D73 = (21 + 4 / 12) * ft
l_D73__D8 = (26 + 1 / 12) * ft

l_D2__D27 = (18 + 8.75 / 12) * ft
l_D2__D28 = l_D1__D21 + l_D21__D28 - l_D1__D2
l_D28__D32 = (8 + 9.75 / 12) * ft
l_D32__D4 = l_D1__D2 + l_D2__D3 + l_D3__D4 - (l_D1__D21 + l_D21__D28 + l_D28__D32)
l_D6__D62 = (7 + 2 / 12) * ft
l_D6__D68 = (28 + 2.5 / 12) * ft
l_D62__D67 = l_D6__D67 - l_D6__D62
l_D62__D68 = l_D6__D68 - l_D6__D62
l_D67__D7 = l_D6__D7 - l_D6__D67
l_D68__D7 = l_D6__D7 - l_D6__D68
l_D67__D68 = l_D6__D68 - l_D6__D67

l_DA__DB = 30 + 9 / 12
l_DB__DC = 5 + 6 / 12
l_DC__DD = 30 + 9 / 12

d_trib_ns = 67 / 2 * ft

# %%
# Wall tributary widths and lengths

# Average of the top half and bottom half as we will use the full building depth for the trib area for this wall
t_w__D2_left = np.average(
    [l_D1__D2, l_D1__D2 - L_jog], weights=[(l_DA__DB + l_DB__DC) / (2 * d_trib_ns), l_DC__DD / (2 * d_trib_ns)]
)
t_w__D2_right = l_D2__D28 / 2
t_w__D2 = t_w__D2_left + t_w__D2_right

# t_w__D27 = (l_D4__D5 - L_jog) + ((l_D5__D6 + l_D6__D7) / 2)
t_w__D28 = (l_D2__D28 / 2) + (l_D28__D32 / 2)
t_w__D32 = (l_D28__D32 / 2) + (l_D32__D4 / 2)
t_w__D4 = (l_D32__D4 / 2) + (l_D4__D5 / 2)
t_w__D5 = (l_D4__D5 / 2) + (l_D5__D6 / 2)
t_w__D6 = (l_D5__D6 / 2) + (l_D6__D62 / 2)
t_w__D62 = (l_D6__D62 / 2) + (l_D62__D67 / 2)
t_w__D67 = (l_D62__D67 / 2) + (l_D67__D68 / 2)
t_w__D68 = (l_D67__D68 / 2) + (l_D68__D7 / 2)

t_w__D7_left = l_D68__D7 / 2
# Average of the top half and bottom half as we will use the full building depth for the trib area for this wall
t_w__D7_right = np.average(
    [l_D7__D8 - L_jog, l_D7__D8], weights=[(l_DA__DB + l_DB__DC) / (2 * d_trib_ns), l_DC__DD / (2 * d_trib_ns)]
)
t_w__D7 = t_w__D7_left + t_w__D7_right

t_l__D2 = float(2 * d_trib_ns)
t_l__D28 = float(2 * d_trib_ns)
t_l__D32 = float(2 * d_trib_ns)
t_l__D4 = float(2 * d_trib_ns)
t_l__D5 = float(2 * d_trib_ns)
t_l__D6 = float(2 * d_trib_ns)
t_l__D62 = float(2 * d_trib_ns)
t_l__D67 = float(2 * d_trib_ns)
t_l__D68 = float(2 * d_trib_ns)
t_l__D7 = float(2 * d_trib_ns)

walls_input = {
    "D-2": {
        "direction": "y",
        "tributary width": float(t_w__D2),
        "tributary length": t_l__D2,
        "wall length": l_da__db,
        "x": float(l_D1__D2),
        "y": 0,
    },
    "D-2.8": {
        "direction": "y",
        "tributary width": float(t_w__D28),
        "tributary length": t_l__D28,
        "wall length": l_dc__dd,
        "x": float(l_D1__D2 + l_D2__D28),
        "y": 0,
    },
    "D-3.2": {
        "direction": "y",
        "tributary width": float(t_w__D32),
        "tributary length": t_l__D32,
        "wall length": l_stairwall,
        "x": float(l_D1__D2 + l_D2__D28 + l_D28__D32),
        "y": 0,
    },
    "D-4": {
        "direction": "y",
        "tributary width": float(t_w__D4),
        "tributary length": t_l__D4,
        "wall length": l_da__db,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4),
        "y": 0,
    },
    "D-5": {
        "direction": "y",
        "tributary width": float(t_w__D5),
        "tributary length": t_l__D5,
        "wall length": l_da__db,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5),
        "y": 0,
    },
    "D-6": {
        "direction": "y",
        "tributary width": float(t_w__D6),
        "tributary length": t_l__D6,
        "wall length": l_da__db,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6),
        "y": 0,
    },
    "D-6.2": {
        "direction": "y",
        "tributary width": float(t_w__D62),
        "tributary length": t_l__D62,
        "wall length": l_elevator,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D62),
        "y": 0,
    },
    "D-6.7": {
        "direction": "y",
        "tributary width": float(t_w__D67),
        "tributary length": t_l__D67,
        "wall length": l_elevator,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D67),
        "y": 0,
    },
    "D-6.8": {
        "direction": "y",
        "tributary width": float(t_w__D68),
        "tributary length": t_l__D68,
        "wall length": l_stairwall,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D68),
        "y": 0,
    },
    "D-7": {
        "direction": "y",
        "tributary width": float(t_w__D7),
        "tributary length": t_l__D7,
        "wall length": l_da__db,
        "x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D7),
        "y": 0,
    },
    "D-B--D-1.8": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__1,
        "wall length": l_DB__1,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-3": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__2,
        "wall length": l_DB__2,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-3.7": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__3,
        "wall length": l_DB__3,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-4.5": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__4,
        "wall length": l_DB__4,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-5.8": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__5,
        "wall length": l_DB__5,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-6.5": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__6,
        "wall length": l_DB__6,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-B--D-7.2": {
        "direction": "x",
        "tributary width": float(t_w__DB),
        "tributary length": t_l__DB__7,
        "wall length": l_DB__7,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-C--D-1.2": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__1,
        "wall length": l_DC__1,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-C--D-1.8": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__2,
        "wall length": l_DC__2,
        "x": 0,
        "y": float(l_DA__DB),
    },
    "D-C--D-3.2": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__3,
        "wall length": l_DC__3,
        "x": 0,
        "y": float(l_DA__DB + l_DB__DC),
    },
    "D-C--D-3.6": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__4,
        "wall length": l_DC__4,
        "x": 0,
        "y": float(l_DA__DB + l_DB__DC),
    },
    "D-C--D-3.9": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__5,
        "wall length": l_DC__5,
        "x": 0,
        "y": float(l_DA__DB + l_DB__DC),
    },
    "D-C--D-5.2": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__6,
        "wall length": l_DC__6,
        "x": 0,
        "y": float(l_DA__DB + l_DB__DC),
    },
    "D-C--D-6.7": {
        "direction": "x",
        "tributary width": float(t_w__BC),
        "tributary length": t_l__DC__7,
        "wall length": l_DC__7,
        "x": 0,
        "y": float(l_DA__DB + l_DB__DC),
    },
}

shear_walls = create_shear_walls_dataframe_from_dict(walls_input, seismic_loads)

shear_walls["level seismic force per area"] = shear_walls["level seismic force"] / float(A_level)
E_endpost = 1_600_000
A_endpost = 2 * 5.25
Delta_A = 0.25

X_plan = float(L_floor + L_jog)
Y_plan = float(w_building)
CM_x = X_plan / 2
CM_y = Y_plan / 2
plan_dimensions = (X_plan, Y_plan)
center_of_mass = (CM_x, CM_y)
sheathing = Sheathing(
    sheathing_material=SheathingMaterial.WSP_SHEATHING,
    minimum_nominal_panel_thickness=15 / 32,
    nail=Nail.COMMON_10D,
    panel_type=PanelType.PLY,
)
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
    "D-2": 1 / 8,
    "D-2.1": 1 / 2,
    "D-2.8": 1 / 4,
    "D-3": 1 / 4,
    "D-3.2": 1 / 4,
    "D-4": 1 / 16,
    "D-5": 1 / 16,
    "D-6": 1 / 16,
    "D-6.2": 1 / 16,
    "D-6.7": 1 / 16,
    "D-6.8": 1 / 16,
    "D-7": 1 / 16,
    "D-B--D-1.8": 0,
    "D-B--D-3": 0,
    "D-B--D-3.7": 0,
    "D-B--D-4.5": 0,
    "D-B--D-5.8": 0,
    "D-B--D-6.5": 0,
    "D-B--D-7.2": 0,
    "D-C--D-1.2": 0,
    "D-C--D-1.8": 0,
    "D-C--D-3.2": 0,
    "D-C--D-3.6": 0,
    "D-C--D-3.9": 0,
    "D-C--D-5.2": 0,
    "D-C--D-6.7": 0,
}

wall_arm_shortening_length = 0.5  # feet due to holddown positioning
shear_walls = calculate_end_post_forces(
    shear_walls, float(U_floor * 1000), float(seismic_params.s_ds), wall_arm_shortening_length, trib_area_reduction_dict
)
shear_walls, shear_wall_schedule = assign_force_schedule(shear_walls)
shear_walls.to_csv("shear_walls_D_new.csv")

# %% [markdown]
r"""
## Collectors

We must transfer excess shear demand into the shear walls that have a higher unit shear capacity than that of the diaphragm. This will be done by straps.

Strap capacity is measured in force instead of force per length, so the necessary strap capacity can be calculated with

$$
v_s l_s - 2 v_d l_s = P_{strap}
$$
 - $v_s =$ ASD-adjusted unit shear capacity of the shear wall
 - $v_d =$ ASD-adjusted unit shear capacity of the diaphragm
 - $l_s =$ length of the shear wall
 - $P_{strap} =$ required strap capacity
"""

assumed_diaphragm_capacity = 215
shear_walls["required strap capacity"] = (
    shear_walls["adjusted diaphragm floor unit shear demand"] * shear_walls["wall length"]
    - 2 * assumed_diaphragm_capacity * shear_walls["wall length"]
)

shear_walls["required strap capacity"] = np.where(
    shear_walls["required strap capacity"] < 0, 0, shear_walls["required strap capacity"]
)
print(shear_walls["required strap capacity"].to_string())

import sys

sys.exit()

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
l_D1__D2 = (35 + 1.25 / 12) * ft
l_D2__D3 = (23 + 8 / 12) * ft
l_D3__D4 = (34 + 4 / 12) * ft
l_D4__D5 = (24 + 8 / 12) * ft
l_D5__D6 = (18 + 7 / 12) * ft
l_D6__D7 = (37 + 1 / 12) * ft
l_D7__D8 = (36 + 5 / 12) * ft

l_D1__D21 = (36 + 5.25 / 12) * ft
l_D21__D28 = (17 + 11 / 12) * ft
l_D28__D36 = (26 + 4 / 12) * ft
l_D36__D43 = (21 + 4 / 12) * ft
l_D43__D6 = (34 + 4 / 12) * ft
l_D6__D67 = (26 + 1 / 12) * ft
l_D67__D73 = (21 + 4 / 12) * ft
l_D73__D8 = (26 + 1 / 12) * ft

l_D2__D27 = (18 + 8.75 / 12) * ft
l_D2__D28 = l_D1__D21 + l_D21__D28 - l_D1__D2
l_D28__D32 = (8 + 9.75 / 12) * ft
l_D32__D4 = l_D1__D2 + l_D2__D3 + l_D3__D4 - (l_D1__D21 + l_D21__D28 + l_D28__D32)
l_D6__D62 = (7 + 2 / 12) * ft
l_D6__D68 = (28 + 2.5 / 12) * ft
l_D62__D67 = l_D6__D67 - l_D6__D62
l_D62__D68 = l_D6__D68 - l_D6__D62
l_D67__D7 = l_D6__D7 - l_D6__D67
l_D68__D7 = l_D6__D7 - l_D6__D68
l_D67__D68 = l_D6__D68 - l_D6__D67

l_DA__DB = 30 + 9 / 12
l_DB__DC = 5 + 6 / 12
l_DC__DD = 30 + 9 / 12

d_trib_ns = 67 / 2 * ft

# %%
# Wall tributary widths and lengths

# Average of the top half and bottom half as we will use the full building depth for the trib area for this wall
t_w__D2_left = np.average(
    [l_D1__D2, l_D1__D2 - L_jog], weights=[(l_DA__DB + l_DB__DC) / (2 * d_trib_ns), l_DC__DD / (2 * d_trib_ns)]
)
t_w__D2_right = l_D2__D28 / 2
t_w__D2 = t_w__D2_left + t_w__D2_right

# t_w__D27 = (l_D4__D5 - L_jog) + ((l_D5__D6 + l_D6__D7) / 2)
t_w__D28 = (l_D2__D28 / 2) + (l_D28__D32 / 2)
t_w__D32 = (l_D28__D32 / 2) + (l_D32__D4 / 2)
t_w__D4 = (l_D32__D4 / 2) + (l_D4__D5 / 2)
t_w__D5 = (l_D4__D5 / 2) + (l_D5__D6 / 2)
t_w__D6 = (l_D5__D6 / 2) + (l_D6__D62 / 2)
t_w__D62 = (l_D6__D62 / 2) + (l_D62__D67 / 2)
t_w__D67 = (l_D62__D67 / 2) + (l_D67__D68 / 2)
t_w__D68 = (l_D67__D68 / 2) + (l_D68__D7 / 2)

t_w__D7_left = l_D68__D7 / 2
# Average of the top half and bottom half as we will use the full building depth for the trib area for this wall
t_w__D7_right = np.average(
    [l_D7__D8 - L_jog, l_D7__D8], weights=[(l_DA__DB + l_DB__DC) / (2 * d_trib_ns), l_DC__DD / (2 * d_trib_ns)]
)
t_w__D7 = t_w__D7_left + t_w__D7_right

t_l__D2 = float(2 * d_trib_ns)
t_l__D28 = float(2 * d_trib_ns)
t_l__D32 = float(2 * d_trib_ns)
t_l__D4 = float(2 * d_trib_ns)
t_l__D5 = float(2 * d_trib_ns)
t_l__D6 = float(2 * d_trib_ns)
t_l__D62 = float(2 * d_trib_ns)
t_l__D67 = float(2 * d_trib_ns)
t_l__D68 = float(2 * d_trib_ns)
t_l__D7 = float(2 * d_trib_ns)

# %%
flexible_ns.loc["Level 4", "tributary weight"] = float(f_p__roof)
flexible_ns.loc["Level 3", "tributary weight"] = float(f_p__4)
flexible_ns.loc["Level 2", "tributary weight"] = float(f_p__3)
flexible_ns.loc["Level 1", "tributary weight"] = float(f_p__2)

# Shear wall height is for the floor below
flexible_ns.loc["Level 4", "wall height"] = float(seismic_loads.loc["Level 4", "floor height"])
flexible_ns.loc["Level 3", "wall height"] = float(seismic_loads.loc["Level 3", "floor height"])
flexible_ns.loc["Level 2", "wall height"] = float(seismic_loads.loc["Level 2", "floor height"])
flexible_ns.loc["Level 1", "wall height"] = float(seismic_loads.loc["Level 1", "floor height"])

# Add wall tributary widths and wall lengths to df
l_da__db = 29
l_dc__dd = 29
l_stairwall = 20
l_elevator = 10 + 7 / 12
wall_tributary = {
    "D-2": {"trib_width": float(t_w__D2), "trib_depth": t_l__D2, "wall_length": l_da__db},
    # "D-2.7": {"trib_width": float(t_w__D27), "trib_depth": t_l__D27, "wall_length": l_dc__dd},
    "D-2.8": {"trib_width": float(t_w__D28), "trib_depth": t_l__D28, "wall_length": l_dc__dd},
    "D-3.2": {"trib_width": float(t_w__D32), "trib_depth": t_l__D32, "wall_length": l_stairwall},
    "D-4": {"trib_width": float(t_w__D4), "trib_depth": t_l__D4, "wall_length": l_da__db},
    "D-5": {"trib_width": float(t_w__D5), "trib_depth": t_l__D5, "wall_length": l_da__db},
    "D-6": {"trib_width": float(t_w__D6), "trib_depth": t_l__D6, "wall_length": l_da__db},
    "D-6.2": {"trib_width": float(t_w__D62), "trib_depth": t_l__D62, "wall_length": l_elevator},
    "D-6.7": {"trib_width": float(t_w__D67), "trib_depth": t_l__D67, "wall_length": l_elevator},
    "D-6.8": {"trib_width": float(t_w__D68), "trib_depth": t_l__D68, "wall_length": l_stairwall},
    "D-7": {"trib_width": float(t_w__D7), "trib_depth": t_l__D7, "wall_length": l_da__db},
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
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces NS - Level 4",
    label="tab:shear_wall_forces_roof_ns",
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
l_DA__DB = (30 + 9 / 12) * ft
l_DB__DC = (5 + 6 / 12) * ft
l_DC__DD = (30 + 9 / 12) * ft

d_trib_ew = L_floor

# %%
# Wall tributary widths and lengths
t_w__DB = l_DA__DB + l_DB__DC / 2
t_w__BC = l_DB__DC / 2 + l_DC__DD

t_l__DB__1 = 46.5
t_l__DB__2 = 28.25
t_l__DB__3 = 27.67
t_l__DB__4 = 27.75
t_l__DB__5 = 23
t_l__DB__6 = 18.33 + 1.24 / 12  # add 1.24 inches to pencil out 204'-10.25" length
t_l__DB__7 = 33.25

t_l__DC__1 = 37
t_l__DC__2 = 21.66
t_l__DC__3 = 16
t_l__DC__4 = 11.66
t_l__DC__5 = 33.67
t_l__DC__6 = 33.66 + 2.5 / 12  # add 2.5 inches to pencil out 204'-10.25" length
t_l__DC__7 = 51
# t_l__BC4__1 = 19

# %%
# Wall segment lengths
l_DB__1 = 9.67
l_DB__2 = 13.5
l_DB__3 = 16.75
l_DB__4 = 22.75
l_DB__5 = 18.25
l_DB__6 = 8.75
l_DB__7 = 14.67

l_DC__1 = 23.5
l_DC__2 = 10.25
l_DC__3 = 9.25
l_DC__4 = 6.75
l_DC__5 = 28.75
l_DC__6 = 14
l_DC__7 = 14.5
# l_DC4__1 = 19

# %%
flexible_ew.loc["Level 4", "tributary weight"] = float(f_p__roof)
flexible_ew.loc["Level 3", "tributary weight"] = float(f_p__4)
flexible_ew.loc["Level 2", "tributary weight"] = float(f_p__3)
flexible_ew.loc["Level 1", "tributary weight"] = float(f_p__2)

# Shear wall height is for the floor below
flexible_ew.loc["Level 4", "wall height"] = float(seismic_loads.loc["Level 4", "floor height"])
flexible_ew.loc["Level 3", "wall height"] = float(seismic_loads.loc["Level 3", "floor height"])
flexible_ew.loc["Level 2", "wall height"] = float(seismic_loads.loc["Level 2", "floor height"])
flexible_ew.loc["Level 1", "wall height"] = float(seismic_loads.loc["Level 1", "floor height"])

# Add wall tributary widths and wall lengths to df
wall_tributary = {
    "D-B--D-1.8": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__1, "wall_length": l_DB__1},
    "D-B--D-3": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__2, "wall_length": l_DB__2},
    "D-B--D-3.7": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__3, "wall_length": l_DB__3},
    "D-B--D-4.5": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__4, "wall_length": l_DB__4},
    "D-B--D-5.8": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__5, "wall_length": l_DB__5},
    "D-B--D-6.5": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__6, "wall_length": l_DB__6},
    "D-B--D-7.2": {"trib_width": float(t_w__DB), "trib_depth": t_l__DB__7, "wall_length": l_DB__7},
    "D-C--D-1.2": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__1, "wall_length": l_DC__1},
    "D-C--D-1.8": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__2, "wall_length": l_DC__2},
    "D-C--D-3.2": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__3, "wall_length": l_DC__3},
    "D-C--D-3.6": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__4, "wall_length": l_DC__4},
    "D-C--D-3.9": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__5, "wall_length": l_DC__5},
    "D-C--D-5.2": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__6, "wall_length": l_DC__6},
    "D-C--D-6.7": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC__7, "wall_length": l_DC__7},
    # "D-C.4--D-6.2": {"trib_width": float(t_w__BC), "trib_depth": t_l__DC4__1, "wall_length": l_DC4__1},
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
trib_area = flexible_ew["tributary area"].sum() / len(flexible_ns.index.get_level_values("Level").unique())
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
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Shear wall forces EW - Level 4",
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
    levels="Level 4",
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Wall rigidities EW - Level 4",
    label="tab:wall_rigidities_ew_level4",
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
# The building is symmetric in each direction when projected to that direction, so the centers of mass are just half of each total plan dimension.

# %%
# %%render
X_plan = L_floor + L_jog
Y_plan = w_building
CM_x = X_plan / 2
CM_y = Y_plan / 2

# %% [markdown]
# An accidental torsion eccentricity of 5% must be added/subtracted.

# %%
# %%render
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
    "D-2": {"x": float(l_D1__D2), "y": 0},
    "D-2.8": {"x": float(l_D1__D2 + l_D2__D28), "y": 0},
    "D-3.2": {"x": float(l_D1__D2 + l_D2__D28 + l_D28__D32), "y": 0},
    "D-4": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4), "y": 0},
    "D-5": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5), "y": 0},
    "D-6": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6), "y": 0},
    "D-6.2": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D62), "y": 0},
    "D-6.7": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D67), "y": 0},
    "D-6.8": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D68), "y": 0},
    "D-7": {"x": float(l_D1__D2 + l_D2__D3 + l_D3__D4 + l_D4__D5 + l_D5__D6 + l_D6__D7), "y": 0},
    # E-W
    "D-B--D-1.8": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-3": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-3.7": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-4.5": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-5.8": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-6.5": {"x": 0, "y": float(l_DA__DB)},
    "D-B--D-7.2": {"x": 0, "y": float(l_DA__DB)},
    "D-C--D-1.2": {"x": 0, "y": float(l_DA__DB)},
    "D-C--D-1.8": {"x": 0, "y": float(l_DA__DB)},
    "D-C--D-3.2": {"x": 0, "y": float(l_DA__DB + l_DB__DC)},
    "D-C--D-3.6": {"x": 0, "y": float(l_DA__DB + l_DB__DC)},
    "D-C--D-3.9": {"x": 0, "y": float(l_DA__DB + l_DB__DC)},
    "D-C--D-5.2": {"x": 0, "y": float(l_DA__DB + l_DB__DC)},
    "D-C--D-6.7": {"x": 0, "y": float(l_DA__DB + l_DB__DC)},
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
CMx = plan_dim_x / 2
CMxa_plus = CMx + plan_dim_x * 0.05
CMxa_minus = CMx - plan_dim_x * 0.05
rigid["ex"] = np.where(rigid["x"] <= rigid["CRx"], rigid["CRx"] - CMxa_minus, CMxa_plus - rigid["CRx"])
rigid["ex"] = np.where(rigid["ex"] <= 0, 0.05 * plan_dim_x, rigid["ex"])  # assign 5% accidental eccentricity to negatives

# y eccentricity. Use the "+" eccentricity for walls on the + side of the CR. "-" for - side
plan_dim_y = float(Y_plan)
CMy = plan_dim_y / 2
CMya_plus = CMy + plan_dim_y * 0.05
CMya_minus = CMy - plan_dim_y * 0.05
rigid["ey"] = np.where(rigid["y"] <= rigid["CRy"], rigid["CRy"] - CMya_minus, CMya_plus - rigid["CRy"])
rigid["ey"] = np.where(rigid["ey"] <= 0, 0.05 * plan_dim_y, rigid["ey"])  # assign 5% accidental eccentricity to negatives

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
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-2", "dead load"] *= 1 / 8
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-2.1", "dead load"] *= 1 / 2
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-2.8", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-3", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-3.2", "dead load"] *= 1 / 4
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-4", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-5", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-6", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-6.2", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-6.7", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-6.8", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls.index.get_level_values("Wall") == "D-7", "dead load"] *= 1 / 16
shear_walls.loc[shear_walls["line"] == "D-B", "dead load"] *= 0
shear_walls.loc[shear_walls["line"] == "D-C", "dead load"] *= 0

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
levels = ["Level 4", "Level 3", "Level 2", "Level 1"]
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
diaphragms_ns["L_simp"] = 30
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
diaphragms_ew["L_simp"] = 5.5
diaphragms_ew["L_cant"] = 31
diaphragms_ew["depth"] = float(L_floor)
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
shear_walls.to_csv("shear_walls_D.csv")
