from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from pathlib import Path
from importlib import resources
import pandas as pd

from ..asce.parameters import DesignMethod

sheathing_material = ["Wood structural Panels - Structural I"]

SDPWS_TABLE_4_3_A = pd.read_csv(
    filepath_or_buffer=resources.files("structural_tools.data")
    .joinpath("sdpws_2021_table_4_3_a.csv")
    .open("r", encoding="utf-8"),
)


class LoadCase(Enum):
    SEISMIC = "SEISMIC"
    WIND = "WIND"


class SheathingMaterial(Enum):
    WSP_STRUCTURAL_I = "Wood Structural Panels - Structural I"
    WSP_SHEATHING = "Wood Structural Panels - Sheathing"
    PLYWOOD_SIDING = "Plywood Siding"
    PARTICLEBOARD_SHEATHING = "Particleboard Sheathing"
    SFS = "Structural Fiberboard Sheathing"


class Nail(Enum):
    COMMON_6D = "6d common"
    COMMON_8D = "8d common"
    COMMON_10D = "10d common"
    GALV_CASING_6D = "6d galvanized casing"
    GALV_CASING_8D = "8d galvanized casing"
    GALV_ROOFING_11GA = "11 gauge galvanized roofing"


class PanelType(Enum):
    OSB = "OSB"
    PLY = "PLY"


VALID_NOMINAL_PANEL_THICKNESSES = {
    5 / 16,
    3 / 8,
    7 / 16,
    15 / 32,
    1 / 2,
    19 / 32,
    25 / 32,
}

VALID_NAIL_BEARING_LENGTHS = {
    1 + 1 / 4,
    1 + 3 / 8,
    1 + 1 / 2,
}

VALID_NAIL_SPACING = {
    2,
    3,
    4,
    6,
}


VALID_SHEATHED_SIDES = {
    1,
    2,
}


@dataclass
class Sheathing:
    nail_spacing: int | None = None
    sheathing_material: SheathingMaterial | None = None
    minimum_nominal_panel_thickness: float | None = None
    minimum_nail_bearing_length: float | None = None
    nail: Nail | None = None
    panel_type: PanelType | None = None
    sheathed_sides: int = 1

    def __post_init__(self):
        # Value checks
        if (
            self.minimum_nominal_panel_thickness
            and self.minimum_nominal_panel_thickness
            not in VALID_NOMINAL_PANEL_THICKNESSES
        ):
            raise ValueError(
                f"Invalid minimum nominal panel thickness: {self.minimum_nominal_panel_thickness}.\nValid values include: {VALID_NOMINAL_PANEL_THICKNESSES}"
            )
        if (
            self.minimum_nail_bearing_length
            and self.minimum_nail_bearing_length not in VALID_NAIL_BEARING_LENGTHS
        ):
            raise ValueError(
                f"Invalid minimum nail bearing length: {self.minimum_nail_bearing_length}.\nValid values include: {VALID_NAIL_BEARING_LENGTHS}"
            )
        if self.sheathed_sides and self.sheathed_sides not in VALID_SHEATHED_SIDES:
            raise ValueError(
                f"Invalid number of sheathed sides: {self.sheathed_sides}.\nValid values include: {VALID_SHEATHED_SIDES}"
            )
        if self.nail_spacing and self.nail_spacing not in VALID_NAIL_SPACING:
            raise ValueError(
                f"Invalid nail spacing: {self.nail_spacing}.\nValid values include: {VALID_NAIL_SPACING}"
            )


def select_nail_spacing(demand: float, sheathing: Sheathing):
    possible_sheathing_options: pd.DataFrame = SDPWS_TABLE_4_3_A[
        SDPWS_TABLE_4_3_A["Unit Shear"] >= demand
    ]
    return possible_sheathing_options
