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
# title: Building A Seismic Design
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
import warnings
from pathlib import Path
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
from structural_tools.notebook.runtime import image_path, runtime
from structural_tools.structure import LateralSystem, Level, Structure
from structural_tools.wood.shear_walls import (
    assign_force_schedule,
    assign_values_for_all_levels_per_wall,
    assign_values_for_all_walls_per_level,
    calculate_deflections,
    calculate_end_post_forces,
    calculate_required_strap_capacity_interior_walls,
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

warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
)
initialize_notebook()

# %% [markdown]
r"""
# Effective Seismic Weight Calculation

We calculate the effective seismic weight with unit weights and square footages.

## Unit Weights

Weights will be assigned with area unit weights.
"""

# %%
set_params_columns(4)

# %%
# %%render params
U_corridor = 35 / 1000 * ksf
U_floor = 35 / 1000 * ksf
U_roof = 25 / 1000 * ksf
U_extwall = 12 / 1000 * ksf  # ASCE 7-22 Table C3.1-1a

# %% [markdown]
r"""
## Building Geometry

The lengths and widths of the weight areas are consistent across levels 2 and up.
"""

# %%
set_params_columns(5)

# %%
# %%render params
L_floor = 230 * ft
w_floor = (30 + 9 / 12) * ft
L_corridor = 230 * ft
w_corridor = 5.5 * ft
w_building = (w_floor * 2) + w_corridor

# %% [markdown]
r"""
The wall heights vary across the levels.
"""

# %%
# %%render params
h_wall__1 = 11.5 * ft
h_wall__2 = 10.5 * ft
h_wall__3 = 10.5 * ft
h_wall__4 = 10 * ft

# %% [markdown]
r"""
The building perimeter will be used to calculate exterior wall weights.
"""

# %%
# %%render
L_jog = 5 * ft
P_wall = (2 * L_floor) + (2 * w_building) + (2 * L_jog)

# %% [markdown]
r"""
Areas can now be calculated. Note that the areas of the walls are the actual wall areas per level; they have not been distributed to each floor based on tributary area, yet (divided in half to the floor above and below).
"""

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
r"""
## Component Weights

The typical floor and corridor weights are
"""

# %%
# %%render
W_floor__typ = U_floor * A_floor
W_corridor__typ = U_corridor * A_corridor

# %% [markdown]
r"""
This applies to levels 2 and up.
"""

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
r"""
The roof weight is
"""

# %%
# %%render
W_roof = U_roof * A_level

# %% [markdown]
r"""
The exterior walls are divided in half to the floor above and below.
"""

# %%
# %%render
W_wall__2 = U_extwall * (A_wall__2 / 2 + A_wall__1 / 2)
W_wall__3 = U_extwall * (A_wall__3 / 2 + A_wall__2 / 2)
W_wall__4 = U_extwall * (A_wall__4 / 2 + A_wall__3 / 2)
W_wall__roof = U_extwall * (A_wall__4 / 2)

# %% [markdown]
r"""
## Level Weights

Each level's weight is
"""

# %%
# %%render
W_level__2 = W_floor__2 + W_corridor__2 + W_wall__2
W_level__3 = W_floor__3 + W_corridor__3 + W_wall__3
W_level__4 = W_floor__4 + W_corridor__4 + W_wall__4
# be sure to add the weight of the truss later
W_level__roof = W_roof + W_wall__roof

# %% [markdown]
r"""
## Total Weight

The total weight of the structure is
"""

# %%
# %%render long
W_total = W_level__2 + W_level__3 + W_level__4 + W_level__roof

# %% [markdown]
r"""
# Seismic Equivalent Lateral Force (ELF) Analysis

We will be using the equivalent lateral force method prescribed by ASCE 7-22 Section 12.8. We will envelope the forces with a fully flexible diaphragm assumption and a fully rigid diaphragm assumption.

## Seismic Parameters

Seismic design parameters are obtained from the ASCE Hazard Tool and ASCE 7-22 Sec. 11 and 12.
"""

# %%
latitude = 45.501
longitude = -122.834
seismic_params = SeismicParameters(latitude=latitude, longitude=longitude, risk_category=RiskCategory.II, site_class=SiteClass.D)

# %%
display_text(f"For latitude $= {latitude}$ and longitude $= {longitude}$:")

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
r"""
## Base Shear

## Structural seismic parameters

The lateral force resisting systems in each direction (E-W/x and N-S/y) have the following seismic design coefficients.
"""

# %%
set_params_columns(1)

# %%
# %%render params
h_top__roof = (58 + 3 / 12) * ft
h_bottom__roof = 42.5 * ft
h_n = np.mean([h_top__roof, h_bottom__roof])  # ASCE 7-22 Sec. 11.2

# %%
# %%render
h_n__check = check_value(h_n, 65 * ft, "<=")

# %%
h_n = float(h_n)

# %%
plan_dimensions = (float(L_floor + L_jog), float(w_building))
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
    plan_dimensions=plan_dimensions,
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
We will approximate the building period with

\begin{equation*}
    T_a = C_t h_n^x \quad\text{(ASCE 7-22 Eq. 12.8-8)}
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
seismic_loads.seismic_loads_x.to_csv(f"{runtime.identifier}_seismic_x.csv")
seismic_loads.seismic_loads_y.to_csv(f"{runtime.identifier}_seismic_y.csv")

# %%
set_params_columns(2)

# %%
# %%render params
C_s__x = seismic_loads.c_s_x
C_s__y = seismic_loads.c_s_y
V_x = seismic_loads.v_x
V_y = seismic_loads.v_y

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
r"""
# Diaphragm and Shear Wall Design

## Redundancy Factor

The redundancy factor, $\rho$, is either 1.3 or 1.0 according to ASCE 7-22 12.3.4.

No single story resists more than 35% of the shear force (see $C_{vx}$ in \autoref{tab:seismic_forces_calculation_table_ew} and \autoref{tab:seismic_forces_calculation_table_ns}), so all values of $\rho$ are $1.0$, initially.

## Shear Wall Design

We will envelope the forces with a fully flexible diaphragm assumption and a fully rigid diaphragm assumption.

The shear wall naming goes _[E-W line]-[N-S line intersection of wall left edge]_. For example, wall "A-C--A-3.8" is the wall on line "A-C" whose left edge begins at "A-4.2". If there is only one wall on the line, then the name is just the grid line.

### Flexible Diaphragm Assumption

We will uniformly smear the shear forces across the diaphragms based on tributary areas of each level. This removes some nuance in the distribution of floor weight distribution, but other areas of the design are conservative and make up for this assumption.

$$
f_{p,x} = F_x / A_{level}
$$
"""

# %%
display_figure(
    image_path("shear_wall_tributary_area"),
    caption="Shear walls (thick red lines) in plan view with their (colored) tributary areas",
    label="fig:diaphragm_trib_area",
    width="60%",
)

# %% [markdown]
"""
Shear wall shear forces are tabulated below.

Shear wall sheathing will be assumed to have the following properties:

1. WSP Sheathing grade
2. 15/32" thickness
3. 10d common nails
4. Plywood panels
5. Framing uses Douglas-Fir-Larch (no specific gravity adjustment)

Shear capacity and stiffness values come from **AWC SDPWS 2021 Table 4.3A**.

Notes:

- Cumulative shear force, $F_{x,cum}$, is cumulative from the Roof downwards.
"""

# %%
# Set baseline sheathing properties. We set the number of sheathed sides later
sheathing = Sheathing(
    sheathing_material=SheathingMaterial.WSP_SHEATHING,
    minimum_nominal_panel_thickness=15 / 32,
    nail=Nail.COMMON_10D,
    panel_type=PanelType.PLY,
)

# %%
# Plan dimensions for wall lengths
l_stairwall = 20
l_AA__AB = 30 + 9 / 12
l_AB__AC = 5 + 6 / 12
l_AC__AD = 30 + 9 / 12

l_A1__A2 = 31 + 11 / 12
l_A2__A3 = 17 + 11 / 12
l_A3__A4 = 26 + 4 / 12
l_A4__A5 = 21 + 4 / 12
l_A5__A6 = 24 + 8 / 12
l_A6__A7 = 34 + 4 / 12
l_A7__A8 = 26 + 1 / 12
l_A8__A9 = 21 + 4 / 12
l_A9__A10 = 31 + 1 / 12

l_A1__A24 = 40 + 1 / 12
l_A24__A35 = 23 + 8 / 12
l_A35__A51 = 34 + 4 / 12
l_A51__A61 = 24 + 8 / 12
l_A61__A68 = 24 + 8 / 12
l_A68__A82 = 38 + 3 / 12
l_A82__A89 = 17 + 11 / 12
l_A89__A10 = 31 + 5 / 12

l_A37__A4 = 8 + 10 / 12
l_A3__A37 = l_A3__A4 - l_A37__A4
l_A78__A82 = 8 + 10 / 12
l_A68__A78 = l_A68__A82 - l_A78__A82

d_trib_ew = float(L_floor)
d_trib_ns = float(67 / 2)

# Wall tributary widths and lengths
t_w__AB = l_AA__AB + l_AB__AC / 2
t_w__AC = l_AB__AC / 2 + l_AC__AD

t_w__A2 = l_A1__A2 + (l_A2__A3) / 2
t_w__A3 = (l_A2__A3 / 2) + (l_A3__A37 / 2)
t_w__A37 = (l_A3__A37 / 2) + (l_A37__A4 / 2)
t_w__A4 = (l_A37__A4 / 2) + (l_A4__A5 / 2)
t_w__A5 = (l_A4__A5 / 2) + (l_A5__A6 / 2)
t_w__A6 = (l_A5__A6 / 2) + (l_A6__A7 / 2)
t_w__A7 = (l_A6__A7 / 2) + (l_A7__A8 / 2)
t_w__A8 = (l_A7__A8 / 2) + (l_A8__A9 / 2)
t_w__A9 = (l_A8__A9 / 2) + (l_A9__A10 - float(L_jog))

t_w__A24 = (l_A1__A24 - float(L_jog)) + (l_A24__A35 / 2)
t_w__A35 = (l_A24__A35 / 2) + (l_A35__A51 / 2)
t_w__A51 = (l_A35__A51 / 2) + (l_A51__A61 / 2)
t_w__A61 = (l_A51__A61 / 2) + (l_A61__A68 / 2)
t_w__A68 = (l_A61__A68 / 2) + (l_A68__A78 / 2)
t_w__A78 = (l_A68__A78 / 2) + (l_A78__A82 / 2)
t_w__A82 = (l_A78__A82 / 2) + (l_A82__A89 / 2)
t_w__A89 = (l_A82__A89 / 2) + l_A89__A10

t_l__AB__1 = 33
t_l__AB__2 = 21.5
t_l__AB__3 = 16.75
t_l__AB__4 = 19.5
t_l__AB__5 = 16
t_l__AB__6 = 34
t_l__AB__7 = 28.75
t_l__AB__8 = 28
t_l__AB__9 = 32.5


t_l__AC__1 = 46.67
t_l__AC__2 = 14
t_l__AC__3 = 21.5
t_l__AC__4 = 20.33
t_l__AC__5 = 30.5
t_l__AC__6 = 16
t_l__AC__7 = 29
t_l__AC__8 = 23.5
t_l__AC__9 = 28.5

t_l__A2 = float(d_trib_ns)
t_l__A24 = float(d_trib_ns)
t_l__A3 = float(d_trib_ns)
t_l__A35 = float(d_trib_ns)
t_l__A37 = float(d_trib_ns)
t_l__A4 = float(d_trib_ns)
t_l__A5 = float(d_trib_ns)
t_l__A51 = float(d_trib_ns)
t_l__A6 = float(d_trib_ns)
t_l__A61 = float(d_trib_ns)
t_l__A68 = float(d_trib_ns)
t_l__A7 = float(d_trib_ns)
t_l__A78 = float(d_trib_ns)
t_l__A8 = float(d_trib_ns)
t_l__A82 = float(d_trib_ns)
t_l__A89 = float(d_trib_ns)
t_l__A9 = float(d_trib_ns)

l_AB__1 = 20
l_AB__2 = 13
l_AB__3 = 9
l_AB__4 = 12
l_AB__5 = 10.67
l_AB__6 = 29
l_AB__7 = 13
l_AB__8 = 11.67
l_AB__9 = 10

l_AC__1 = 16
l_AC__2 = 9.34
l_AC__3 = 16.75
l_AC__4 = 15
l_AC__5 = 26
l_AC__6 = 11.5
l_AC__7 = 16
l_AC__8 = 9.75
l_AC__9 = 24

# %%
# Shear walls input
walls_input = {
    # N-S
    "A-2": {
        "direction": "y",
        "tributary width": float(t_w__A2),
        "tributary length": t_l__A2,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2),
        "y": 0,
    },
    "A-2.4": {
        "direction": "y",
        "tributary width": float(t_w__A24),
        "tributary length": t_l__A24,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24),
        "y": 0,
    },
    "A-3": {
        "direction": "y",
        "tributary width": float(t_w__A3),
        "tributary length": t_l__A3,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3),
        "y": 0,
    },
    "A-3.5": {
        "direction": "y",
        "tributary width": float(t_w__A35),
        "tributary length": t_l__A35,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24 + l_A24__A35),
        "y": 0,
    },
    "A-3.7": {
        "direction": "y",
        "tributary width": float(t_w__A37),
        "tributary length": t_l__A37,
        "wall length": l_stairwall,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A37),
        "y": 0,
    },
    "A-4": {
        "direction": "y",
        "tributary width": float(t_w__A4),
        "tributary length": t_l__A4,
        "wall length": l_stairwall,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4),
        "y": 0,
    },
    "A-5": {
        "direction": "y",
        "tributary width": float(t_w__A5),
        "tributary length": t_l__A5,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4 + l_A4__A5),
        "y": 0,
    },
    "A-5.1": {
        "direction": "y",
        "tributary width": float(t_w__A51),
        "tributary length": t_l__A51,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51),
        "y": 0,
    },
    "A-6": {
        "direction": "y",
        "tributary width": float(t_w__A6),
        "tributary length": t_l__A6,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4 + l_A4__A5 + l_A5__A6),
        "y": 0,
    },
    "A-6.1": {
        "direction": "y",
        "tributary width": float(t_w__A61),
        "tributary length": t_l__A61,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51 + l_A51__A61),
        "y": 0,
    },
    "A-6.8": {
        "direction": "y",
        "tributary width": float(t_w__A68),
        "tributary length": t_l__A68,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51 + l_A51__A61 + l_A61__A68),
        "y": 0,
    },
    "A-7": {
        "direction": "y",
        "tributary width": float(t_w__A7),
        "tributary length": t_l__A7,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4 + l_A4__A5 + l_A5__A6 + l_A6__A7),
        "y": 0,
    },
    "A-7.8": {
        "direction": "y",
        "tributary width": float(t_w__A78),
        "tributary length": t_l__A78,
        "wall length": l_stairwall,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51 + l_A51__A61 + l_A61__A68 + l_A68__A78),
        "y": 0,
    },
    "A-8": {
        "direction": "y",
        "tributary width": float(t_w__A8),
        "tributary length": t_l__A8,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4 + l_A4__A5 + l_A5__A6 + l_A6__A7 + l_A7__A8),
        "y": 0,
    },
    "A-8.2": {
        "direction": "y",
        "tributary width": float(t_w__A82),
        "tributary length": t_l__A82,
        "wall length": l_stairwall,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51 + l_A51__A61 + l_A61__A68 + l_A68__A78 + l_A78__A82),
        "y": 0,
    },
    "A-8.9": {
        "direction": "y",
        "tributary width": float(t_w__A89),
        "tributary length": t_l__A89,
        "wall length": l_AC__AD - 1,
        "x": float(l_A1__A24 + l_A24__A35 + l_A35__A51 + l_A51__A61 + l_A61__A68 + l_A68__A78 + l_A78__A82 + l_A82__A89),
        "y": 0,
    },
    "A-9": {
        "direction": "y",
        "tributary width": float(t_w__A9),
        "tributary length": t_l__A9,
        "wall length": l_AA__AB - 1,
        "x": float(l_A1__A2 + l_A2__A3 + l_A3__A4 + l_A4__A5 + l_A5__A6 + l_A6__A7 + l_A7__A8 + l_A8__A9),
        "y": 0,
    },
    # E-W
    "A-B--A-1": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__1,
        "wall length": l_AB__1,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-2.5": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__2,
        "wall length": l_AB__2,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-3.5": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__3,
        "wall length": l_AB__3,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-4": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__4,
        "wall length": l_AB__4,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-4.8": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__5,
        "wall length": l_AB__5,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-5.5": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__6,
        "wall length": l_AB__6,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-6.5": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__7,
        "wall length": l_AB__7,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-7.9": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__8,
        "wall length": l_AB__8,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-B--A-8.8": {
        "direction": "x",
        "tributary width": float(t_w__AB),
        "tributary length": t_l__AB__9,
        "wall length": l_AB__9,
        "x": 0,
        "y": float(l_AA__AB),
    },
    "A-C--A-1.8": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__1,
        "wall length": l_AC__1,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-3.2": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__2,
        "wall length": l_AC__2,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-3.8": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__3,
        "wall length": l_AC__3,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-4.6": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__7,
        "wall length": l_AC__7,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-5.5": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__4,
        "wall length": l_AC__4,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-6.5": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__5,
        "wall length": l_AC__5,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-6.9": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__6,
        "wall length": l_AC__6,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-8.5": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__8,
        "wall length": l_AC__8,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
    "A-C--A-9.2": {
        "direction": "x",
        "tributary width": float(t_w__AC),
        "tributary length": t_l__AC__9,
        "wall length": l_AC__9,
        "x": 0,
        "y": float(l_AA__AB + l_AB__AC),
    },
}

# %%
# Shear walls design
shear_walls = create_shear_walls_dataframe_from_dict(walls_input, seismic_loads, tributary_area_check=float(A_level))
shear_walls["level seismic force per area"] = shear_walls["level seismic force"] / float(A_level)
E_endpost = 1_600_000
A_endpost = 2 * 5.25
Delta_A = 0.2
center_of_mass = (plan_dimensions[0] / 2, plan_dimensions[1] / 2)
shear_walls = design_shear_walls_envelope(
    shear_walls,
    sheathing,
    end_post_youngs_modulus=E_endpost,
    end_post_area=A_endpost,
    Delta_A=Delta_A,
    center_of_mass=center_of_mass,
    plan_dimensions=plan_dimensions,
    c_d_x=C_d__x,
    c_d_y=C_d__y,
    i_e=I_e,
)
maximum_inelastic_story_drift_x = shear_walls["inelastic story drift x"].max() * 12
maximum_inelastic_story_drift_y = shear_walls["inelastic story drift y"].max() * 12

# %%
# Tension and compression forces
trib_area_reduction_dict = {
    "A-2": 1 / 2,
    "A-2.4": 1 / 4,
    "A-3.5": 1 / 4,
    "A-5.1": 1 / 2,
    "A-6": 1 / 2,
    "A-6.8": 1 / 4,
    "A-7": 1 / 4,
    "A-7.8": 1 / 4,
    "A-8": 1 / 2,
    "A-8.9": 1 / 2,
    "A-9": 1 / 4,
    "A-B--A-1": 0,
    "A-B--A-2.5": 0,
    "A-B--A-3.5": 0,
    "A-B--A-4": 0,
    "A-B--A-4.8": 0,
    "A-B--A-5.5": 0,
    "A-B--A-6.5": 0,
    "A-B--A-7.9": 0,
    "A-B--A-8.8": 0,
    "A-C--A-1.8": 0,
    "A-C--A-3.2": 0,
    "A-C--A-3.8": 0,
    "A-C--A-4.6": 0,
    "A-C--A-5.5": 0,
    "A-C--A-6.5": 0,
    "A-C--A-6.9": 0,
    "A-C--A-8.5": 0,
    "A-C--A-9.2": 0,
}
wall_arm_shortening_length = 0.5  # feet due to holddown positioning
shear_walls = calculate_end_post_forces(
    shear_walls, float(U_floor * 1000), float(seismic_params.s_ds), wall_arm_shortening_length, trib_area_reduction_dict
)
shear_walls, shear_wall_schedule = assign_force_schedule(shear_walls)

# Straps design
shear_walls = calculate_required_strap_capacity_interior_walls(shear_walls, 600)

# %%
# Rename columns for display with LaTeX formatting for units and subscripts/superscripts
column_name_map = {
    "level seismic force per area": "$w_{trib}$ [psf]",
    "tributary area": "$A$ [ft$^2$]",
    "shear demand": "$F_x$ [lbf]",
    "flexible shear force demand": "$F_{x,cum}$ [lbf]",
    "wall length": "$l_{wall}$ [ft]",
    "unit shear demand": "$v$ [plf]",
    "adjusted flexible unit shear demand": "$0.7 v$ [lbf]",
    "adjusted flexible unit shear capacity": "$v_{cap}$ [plf]",
    "sheathed sides flexible": "sides",
    "nail spacing flexible": "$s_{nail}$ [in]",
}

# %%
for direction in ["x", "y"]:
    for level in seismic_loads.structure.shear_wall_levels[::-1]:
        display_table(
            dataframe=shear_walls[shear_walls["Direction"] == direction],
            levels=level,
            column_names_filter_and_map=column_name_map,
            position_float="centering",
            caption=f"Shear wall forces {direction} - Level {level}",
            label=f"tab:shear_wall_forces_{level}_{direction}",
            position="H",
        )

# %% [markdown]
r"""
#### Check aspect ratios

We will also check the aspect ratios. Ensuring an aspect ratio under 2 means no strength reduction is necessary.
"""

# %%
# %%render
ratio_max = check_value(shear_walls["aspect ratio"].max(), 2, "<")

# %%
if "OK" in ratio_max:
    display_text(r"$\therefore$ All aspect ratios are OK")
else:
    display_text("The aspect ratio knockdown factor must be applied.")

# %% [markdown]
r"""
### Rigid Diaphragm Assumption

With a rigid diaphragm assumption, we now must calculate the following in each direction:

1. The relative stiffnesses of the shear walls to obtain a center of rigidity.
2. The center of mass of the diaphragm with the center of rigidity to obtain an eccentricity.

The relative stiffnesses of the shear walls can be calculated by using the deflection equation from NDS SDPWS Sec. 4.3.4

\begin{align*}
k &= \frac{F}{\delta},\, \text{assume }F = \text{induced load demand in the shear wall} \\
\delta_{sw} &= \frac{8vh^3}{EAb} + \frac{vh}{1000 G_a} + \frac{h \Delta_A}{b}\quad \text{(SDPWS Eq. 4.3-1)}
\end{align*}

where

- $b =$ the shear wall length [ft]
- $\Delta_A =$ vertical deformation of the wall overturning anchorage system + vertical compression deformation [in]
- $E =$ modulus of elasticity of end posts [psi]
- $A =$ area of end post cross-section [in$^2$]
- $G_a =$ apparent shear wall shear stiffness from nail slip and panel shear deformation [kip/in]
- $h =$ shear wall height [ft]
- $v =$ unit shear force induced by the design load [plf]
- $\delta_{sw} =$ maximum shear wall deflection determined by elastic analysis [in]

Note that for the following calculations, we will conservatively use the unit shear force induced by the nominal load, not the design load, for the value of $v$. This is because the stiffnesses obtained from this use will allow us to use the stiffnesses for deflection limits and story drift calculations

We will assume that the end posts are (2) 2x4 Douglas Fir No. 2; this means:
"""

# %%
# %%render
E_endpost = E_endpost  # psi
A_endpost = A_endpost  # in$^2$

# %% [markdown]
r"""
We will assume a continuous tiedown rod system. A conservative first-pass value for $\Delta_A$ can be:
"""

# %%
# %%render params
Delta_A = Delta_A  # in

# %% [markdown]
r"""
#### Shear Wall Rigidities

With the deflections of each shear wall, along with the force demand of each shear wall, we can obtain the stiffnesses (and therefore, relative stiffnesses) of all the walls.

Each relative stiffness is relative to the level. The sum of all relative stiffnesses on each level is $1.0$.

\begin{align*}
k &= \frac{F_x}{\delta_{sw}} \\
R &= \frac{k_{wall}}{\sum {k_{wall}}}
\end{align*}
"""

# %%
column_name_map = {
    "cumulative shear demand": "$F_x$ [lbf]",
    "delta_sw": r"$\delta_{sw}$ [in]",
    "wall stiffness": "$k$ [lbf/in]",
    "relative wall stiffness": "$R$ [-]",
}
for direction in ["x", "y"]:
    for level in seismic_loads.structure.shear_wall_levels[::-1]:
        display_table(
            dataframe=shear_walls[shear_walls["Direction"] == direction],
            levels=level,
            column_names_filter_and_map=column_name_map,
            position_float="centering",
            caption=f"Wall rigidities {direction} - Level {level}",
            label=f"tab:wall_rigidities_{direction}_{level}",
            position="H",
        )

# %% [markdown]
r"""
#### Center of Rigidity and Center of Mass

We can find the center of rigidity by taking the weighted sum of all wall stiffnesses and their locations relative to a datum.

$$
\bar{x}_r = \frac{\sum k_{ns} x}{\sum k_{ns}},\, \bar{y}_r = \frac{\sum k_{ew} y}{\sum k_{ew}}
$$

Note: each wall is assumed to have zero rigidity in the direction perpendicular to them.

An accidental torsion eccentricity of 5% must be added/subtracted.
"""

# %%
CM_x = plan_dimensions[0] / 2
CM_y = plan_dimensions[1] / 2
X_plan = plan_dimensions[0]
Y_plan = plan_dimensions[1]

# %%
# %%render
CM_x__a__plus = CM_x + X_plan * 0.05
CM_x__a__minus = CM_x - X_plan * 0.05
CM_y__a__plus = CM_y + Y_plan * 0.05
CM_y__a__minus = CM_y - Y_plan * 0.05

# %% [markdown]
r"""
The eccentricity used is determined by the location of the wall in relation to the center of rigidity. Since the wall rotates about the center of rigidity, we want to position the center of mass to ensure each wall is getting the maximum possible force to be conservative. If the center of mass is positioned on the opposite side of the center of rigidity relative to the wall in uestion, that mass will _decrease_ the shear experienced by that wall.

This means we always want the center of mass to be on the same side of the center of rigidity as the wall. The eccentricities shown in the following tables are based on the worst-case scenario. This is not immediately obvious due to all torues being given a positive value.

How the center of rigidity, center of mass, and eccentricity interact are illustrated in \autoref{fig:rigid_diaphragm_moments}. The moments generated by the rotation of the center of mass around the center of rigidity differ based on the chosen eccentricity.

$$
d_x = \left| x - \text{CR}_x \right|\text{ and }\, d_y = \left| y - \text{CR}_y \right|
$$
"""

# %%
display_figure(
    image_path("rigid_diaphragm_moments"),
    caption="Moments applied to a rigid diaphragm",
    label="fig:rigid_diaphragm_moments",
    width="50%",
)

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
for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        dataframe=shear_walls,
        column_names_filter_and_map=column_name_map,
        levels=level,
        position_float="centering",
        formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
        caption=f"Center of Rigidity - Level {level}",
        label=f"tab:center_of_rigidity_{level}",
        position="H",
    )

# %% [markdown]
r"""
And now, we can get the forces in the shear walls using the rigid diaphragm assumption. These forces can be compared to the flexible shear wall forces obtained earlier, $F_{flex}$.
"""

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
for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        dataframe=shear_walls,
        column_names_filter_and_map=column_name_map,
        levels=level,
        position_float="centering",
        formatter_functions=[lambda value: sig_figs(value, sig_figs=4)],
        caption=f"Shear wall forces with the rigid diaphragm assumption - Level {level}",
        label=f"tab:rigid_forces_{level}",
        position="H",
    )

# %% [markdown]
r"""
### Final Shear Wall Nailing Selection

With the envelope procedure complete, we can now select the shear wall sheathing and nailing specifications.
"""

# %%
column_name_map = {
    "shear force demand": "$F_{x,cum}$ [lbf]",
    "wall length": "$l_{wall}$ [ft]",
    "unit shear demand": "$v$ [plf]",
    "adjusted unit shear demand": "$0.7 v$ [plf]",
    "adjusted unit shear capacity": "$v_{cap}$ [plf]",
    "sheathed sides": "sides",
    "nail spacing": "$s_{nail}$ [in]",
}

for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        dataframe=shear_walls,
        column_names_filter_and_map=column_name_map,
        levels=level,
        position_float="centering",
        caption=f"Final shear wall design values - Level {level}",
        label=f"tab:shear_walls_{level}",
        position="H",
    )

# %% [markdown]
r"""
### Shear Wall Axial Forces

#### Tension Forces

The shear forces get converted to overturning forces in the end posts of the shear walls. This overturning moment is counteracted by the dead load tributary to each shear wall according to ASCE 7-22 Sec. 2.4.5. The governing load combination equation is:

\begin{align*}
&0.6 D - 0.7 E_v + 0.7 E_h \qquad\text{(ASCE 7-22 Eq. 2.4.5-10)}
\end{align*}

Using the area dead loads from earlier, along with the tributary areas for each shear wall, we can get the dead load that is applied to each wall. This resists the overturning moment at the centroid of the wall.

For high-aspect-ratio walls, we can assume that the wall acts rigidly, thus:

\begin{align*}
M_R &= D \frac{l_{wall}}{2}\quad\text{(units in kips and feet)} \\
D &= 0.6 D_{nominal}
\end{align*}

However, for longer (low aspect ratio) walls, the wall will not behave rigidly. We can only assume a portion of the vertical loading is transferred to the end posts; the rest is resolved through the wall studs. We will assume that a high aspect ratio is anything over 1:1; therefore, we will take the length of dead load being resisted as equal to the height of the wall.

\begin{align*}
M_R &= \left( D * \text{min}\left[ \frac{h_{wall}}{l_{wall}},\quad 1 \right] \right) \frac{l_{wall}}{2}\quad\text{(units in kips and feet)} \\
D &= 0.6 D_{nominal}
\end{align*}

The vertical seismic effect is taken as, $E_v = 0.2 S_{DS} D$ (units in kips), so the dead load length for the low aspect-ratio walls is also used.

\begin{align*}
P_v &= 0.7 (0.2 S_{DS} D_{nominal})
\end{align*}

The overturning moment is:

\begin{align*}
M_{OT} &= V h_{wall} + P_v \frac{l_{wall}}{2} \quad\text{(units in kips and feet)} \\
V &= 0.7 V_{nominal}
\end{align*}

We will assume that the holddowns are 6" inset from the end posts. The ASD tension caused by horizontal load effects counteracted by dead loads is:

$$
T_{seismic} = \frac{M_{OT} - M_R}{l_{wall} - 0.5}\quad\text{(units in kips and feet)}
$$
"""

# %%
column_name_map = {
    "tension wall vertical load length": "$l_{grav}$ [ft]",
    "horizontal effects moment lc10": "$M_{Eh}$ [lbf-ft]",
    "vertical effects moment lc10": "$M_{Ev}$ [lbf-ft]",
    "dead effects moment lc10": "$M_{D}$ [lbf-ft]",
    "tension force": "$T_{seismic}$ [lbf]",
}
for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        dataframe=shear_walls,
        column_names_filter_and_map=column_name_map,
        levels=level,
        position_float="centering",
        caption=f"End post tension forces - Level {level}",
        label=f"tab:tension_forces_{level}",
        position="H",
    )

# %% [markdown]
r"""
#### Compression Forces

We will calculate the ASD compression forces in a very similar way. 

One significant change will be the additional gravity loads (D, L, and S). The bearing studs in the shear wall are designed to take the compressive gravity load across the length of the wall, so the tributary width for the gravity loads into the compression stud will come from the width of one stud bay (i.e., the end stud bay tributary to the end post in compression).

The governing load combination for compression is one of the following:

\begin{align*}
1.0 D + 0.7 E_v + 0.7 E_h \qquad\text{(ASCE 7-22 Eq. 2.4.5-8)} \\
1.0 D + 0.525 E_v + 0.525 E_h + 0.75 L + 0.1 S \qquad\text{(ASCE 7-22 Eq. 2.4.5-9)}
\end{align*}

The vertical seismic effect is taken as, $E_v = 0.2 S_{DS} D$.

The live load is taken as $L = 40$ psf, while the snow load is taken as $S = 25$ psf.
"""

# %%
column_name_map = {
    "compression wall vertical load length": "$l_{grav}$ [ft]",
    "net moment lc8": "$M_{eq8}$ [lbf-ft]",
    "net moment lc9": "$M_{eq9}$ [lbf-ft]",
    "compression force": "$T_{seismic}$ [lbf]",
}
for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        dataframe=shear_walls,
        column_names_filter_and_map=column_name_map,
        levels=level,
        position_float="centering",
        caption=f"End post compression forces - Level {level}",
        label=f"tab:compression_forces_{level}",
        position="H",
    )

# %% [markdown]
r"""
## Diaphragm Design

We must first get the diaphragm inertial design forces. These are distinct from the seismic forces determined earlier.

\begin{align*}
F_{px} &= \frac{\sum_{i=x}^n F_i}{\sum_{i=x}^n w_i} w_{px} \quad \text{(ASCE Eq. 12.10-1)} \\
F_{px} &\ge 0.2 S_{DS} I_e w_{px} \quad \text{(ASCE Eq. 12.10-2)} \\
F_{px} &\le 0.4 S_{DS} I_e w_{px}  \quad \text{(ASCE Eq. 12.10-3)}
\end{align*}

These forces are shown in \autoref{tab:diaphragm_forces}
"""

# %%
column_name_map = seismic_loads.column_name_map
del column_name_map["level height"]
del column_name_map["level elevation"]
del column_name_map["level weighting parameter"]
del column_name_map["vertical distribution factor"]
del column_name_map["lateral seismic force"]
del column_name_map["seismic design story shear"]
del column_name_map["overturning moment"]
diaphragm_loads_x = seismic_loads.seismic_loads_x
diaphragm_loads_y = seismic_loads.seismic_loads_y
display_table(
    dataframe=diaphragm_loads_x.iloc[::-1],
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Diaphragm Forces - x",
    label="tab:diaphragm_forces",
    position="H",
)
display_table(
    dataframe=diaphragm_loads_y.iloc[::-1],
    column_names_filter_and_map=column_name_map,
    position_float="centering",
    caption="Diaphragm Forces - y",
    label="tab:diaphragm_forces",
    position="H",
)

# %% [markdown]
r"""
With these diaphragm forces, we can design the diaphragm for the worst-case cantilever and simply-supported conditions.

The longest simply-supported span in the N-S direction is about 34.5 feet, while the longest cantilever span is about 32 feet. The depth of the diaphragm in the N-S direction is 67 feet. The longest simply-supported span in the E-W direction is about 5.5 feet, while the longest cantilever span is about 31 feet. The depth of the diaphragm in the E-W direction is 230 feet.

Diaphragm unit shear demands are shown in \autoref{tab:diaphragm_unit_shears}
"""

# %%
# Set up diaphragm dataframe
levels = seismic_loads.structure.diaphragm_levels[::-1]
diaphragm_loads_y = seismic_loads.seismic_loads_y
directions = ["N-S", "E-W"]
index = pd.MultiIndex.from_product(
    [levels, directions],
    names=["Level", "Direction"],
)
diaphragms = pd.DataFrame(index=index)
diaphragms_ns = diaphragms.xs("N-S", level="Direction").copy()
diaphragms_ew = diaphragms.xs("E-W", level="Direction").copy()

# %%
diaphragms_ns["diaphragm design force"] = diaphragm_loads_y["diaphragm design force"] * 1000
diaphragms_ns["area load"] = diaphragms_ns["diaphragm design force"] / float(A_level)
diaphragms_ns["L_simp"] = 34.5
diaphragms_ns["L_cant"] = 32
diaphragms_ns["depth"] = 67
diaphragms_ns["V_simp"] = diaphragms_ns["area load"] * diaphragms_ns["depth"] * diaphragms_ns["L_simp"] / 2
diaphragms_ns["V_cant"] = diaphragms_ns["area load"] * diaphragms_ns["depth"] * diaphragms_ns["L_cant"]
diaphragms_ns["v_simp"] = diaphragms_ns["V_simp"] / diaphragms_ns["depth"]
diaphragms_ns["v_cant"] = diaphragms_ns["V_cant"] / diaphragms_ns["depth"]

# %% [markdown]
r"""
We will assume the diaphragm sheathing has the following properties:

- Unblocked
- WSP Sheathing
- 15/32" thick
- 10d common nails
- Plywood
- Nailed face is 2"
- Edge adjoining cases 2 through 6

The diaphragm capacity checks for this chosen sheathing are shown in \autoref{tab:diaphragm_capacities}
"""

# %%
# Set baseline sheathing properties. We set the number of sheathed sides later
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
diaphragm_loads_x = seismic_loads.seismic_loads_x
diaphragms_ew["diaphragm design force"] = diaphragm_loads_x["diaphragm design force"] * 1000
diaphragms_ew["area load"] = diaphragms_ew["diaphragm design force"] / float(A_level)
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
    "diaphragm design force": "$F_{px}$ [lbf]",
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
    position="H",
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
    position="H",
)

# %%
if (diaphragms[["shear dcr simply supported", "shear dcr cantilever"]] <= 1).all().all():  # double .all() because 2 columns
    display_text(r"$\therefore$ All shear capacities are OK")
else:
    display_text("You must check for new diaphragm nailing.")

# %% [markdown]
r"""
## Story Drift Determination

The inelastic design story drift is calculated as the difference in deflections between stories at their centers of mass.

We can get the elastic story drift at the center of mass by simply dividing the total story shear by the total wall stiffness in the relevant direction. Then, we can use the equation for inelastic story drift from ASCE 7.

\begin{align*}
    \delta_{xe} &= F/k \\
    \delta_{DE} &= \frac{C_d \delta_{xe}}{I_e} + \delta_{di}
\end{align*}

For buildings with a torsional irregularity ratio of over 1.2, we must calculate the accidental torsional ampliciation factor, $A_x$, in each direction, then amplify the accidental moment by that factor. All shear design forces values are recalculated with this value.

We must also look at the edges of the rigid diaphragm for these buildings with the $TIR>2$. The story drift for these buildings is the greatest difference between the edges of the structure at each level and not just at the center of mass. There are 8 different possitibilities that will be simplified into one number (max):

1. far left edge, counter-clockwise rotation, EW forces
2. far right edge, counter-clockwise rotation, EW forces
3. far left edge, counter-clockwise rotation, NS forces
4. far right edge, counter-clockwise rotation, NS forces
5. far left edge, clockwise rotation, EW forces
6. far right edge, clockwise rotation, EW forces
7. far left edge, clockwise rotation, NS forces
8. far right edge, clockwise rotation, NS forces

Note: it is taken into account that $TIR>1.4$ in both directions results in $\rho = 1.3$. The value of $\rho$ is increased to 1.3 when necessary.
"""

# %%
column_name_map = {
    "Direction": "Direction",
    "torsional irregularity ratio": r"TIR [-]",
    "accidental torsional amplification factor x": r"$A_x$ [-]",
    "accidental torsional amplification factor y": r"$A_y$ [-]",
    "elastic story drift x": r"max $\delta_{elastic}$ [ft]",
    "inelastic story drift x": r"max $\delta_{DE}$ [ft]",
    "allowable story drift": r"$\Delta_{allow}$ [ft]",
}
for level in seismic_loads.structure.shear_wall_levels[::-1]:
    display_table(
        shear_walls,
        levels=level,
        column_names_filter_and_map=column_name_map,
        position_float="centering",
        caption=f"Story Drifts - Level {level}",
        label=f"tab:story_drifts_{level}",
        position="H",
    )

# %%
# Save dataframes to CSV
shear_walls.to_csv(f"{runtime.identifier}.csv")
