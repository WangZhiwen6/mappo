import os
import sys
import importlib
from datetime import datetime

_PACKAGE_NAME = __package__

# Allow `python rppo.py` direct execution while keeping relative imports.
if __package__ in (None, ""):
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_THIS_DIR)
    _COMMON_DIR = os.path.join(_THIS_DIR, "common")
    for _path in (_PARENT_DIR, _THIS_DIR, _COMMON_DIR):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    _PACKAGE_NAME = os.path.basename(_THIS_DIR)
    __package__ = _PACKAGE_NAME
else:
    _PACKAGE_NAME = __package__

# Unify callback module identity for absolute/relative imports inside local SB3 copy.
# Some files import `callbacks` while others import `<package>.common.callbacks`.
if _PACKAGE_NAME and "callbacks" not in sys.modules:
    sys.modules["callbacks"] = importlib.import_module(f"{_PACKAGE_NAME}.common.callbacks")


import gymnasium as gym
import numpy as np
import mygym
import mygym.utils.gcloud as gcloud
from gymnasium.wrappers.stateful_reward import NormalizeReward
from mygym.utils.callbacks import *
from mygym.utils.constants import *
from mygym.utils.logger import CSVLogger, WandBOutputFormat
from mygym.utils.rewards import *
from mygym.utils.wrappers import *

from myrppo.common.callbacks import CallbackList
from myrppo.common.logger import HumanOutputFormat
from myrppo.common.logger import Logger as SB3Logger
from myrppo.ppo_recurrent import RecurrentPPO

new_variables = {
        # 环境变量 (温度、湿度、风速、风向、太阳辐射、太阳位置等)
        'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'), # [℃]
        'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),      # [%]
        'wind_speed': ('Site Wind Speed', 'Environment'),           # [m/s]
        'wind_direction': ('Site Wind Direction', 'Environment'),   # [deg/°], 风向指的是风从哪个方向吹来的
        'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),   # [W/m2]
        'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),     # [W/m2]
        'air': ('Air System Outdoor Air Flow Fraction', 'VAV with Reheat'),     # []
        'fan_air_mass_flow_rate': ('Fan Air Mass Flow Rate', 'FAN VARIABLE VOLUME 1'),     # [kg/s]
        'fan_electricity_energy': ('Fan Electricity Energy', 'FAN VARIABLE VOLUME 1'),     # [J]
        'chiller_electricity_energy': ('Chiller Electricity Energy', 'CHILLER ELECTRIC EIR 1'),     # [J]

        # 热区观测变量 (56个热区的温度)
        "thermal_zone:_hall-1-1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-1'),
        "thermal_zone:_hall-1-10_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-10'),
        "thermal_zone:_hall-1-11_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-11'),
        "thermal_zone:_hall-1-12_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-12'),
        "thermal_zone:_hall-1-13_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-13'),
        "thermal_zone:_hall-1-2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-2'),
        "thermal_zone:_hall-1-3_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-3'),
        "thermal_zone:_hall-1-4_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-4'),
        "thermal_zone:_hall-1-5_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-5'),
        "thermal_zone:_hall-1-6_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-6'),
        "thermal_zone:_hall-1-7_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-7'),
        "thermal_zone:_hall-1-8_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-8'),
        "thermal_zone:_hall-1-9_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-1-9'),
        "thermal_zone:_hall-2-1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-2-1'),
        "thermal_zone:_hall-2-2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-2-2'),
        "thermal_zone:_hall-2-3_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-2-3'),
        "thermal_zone:_hall-3-1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-3-1'),
        "thermal_zone:_hall-3-2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-3-2'),
        "thermal_zone:_hall-3-3_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-3-3'),
        "thermal_zone:_hall-4-1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-4-1'),
        "thermal_zone:_hall-4-2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-4-2'),
        "thermal_zone:_hall-4-3_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-4-3'),
        "thermal_zone:_hall-4-4_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: HALL-4-4'),
        "thermal_zone:_p1-1-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-1-COMMERCE 1'),
        "thermal_zone:_p1-1-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-1-DINING 1'),
        "thermal_zone:_p1-1-office_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-1-OFFICE 1'),
        "thermal_zone:_p1-1-office_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-1-OFFICE 2'),
        "thermal_zone:_p1-10-restroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-10-RESTROOM 1'),
        "thermal_zone:_p1-11-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-11-COMMERCE 1'),
        "thermal_zone:_p1-11-commerce_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-11-COMMERCE 2'),
        "thermal_zone:_p1-11-office_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-11-OFFICE 1'),
        "thermal_zone:_p1-11-office_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-11-OFFICE 2'),
        "thermal_zone:_p1-2-breakroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-2-BREAKROOM 1'),
        "thermal_zone:_p1-2-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-2-DINING 1'),
        "thermal_zone:_p1-2-restroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-2-RESTROOM 1'),
        "thermal_zone:_p1-3-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-3-COMMERCE 1'),
        "thermal_zone:_p1-4-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-4-COMMERCE 1'),
        "thermal_zone:_p1-4-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-4-DINING 1'),
        "thermal_zone:_p1-5-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-5-COMMERCE 1'),
        "thermal_zone:_p1-6-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-6-COMMERCE 1'),
        "thermal_zone:_p1-7-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-7-DINING 1'),
        "thermal_zone:_p1-7-office_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-7-OFFICE 1'),
        "thermal_zone:_p1-8-restroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-8-RESTROOM 1'),
        "thermal_zone:_p1-9-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P1-9-COMMERCE 1'),
        "thermal_zone:_p2-1-commerce_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P2-1-COMMERCE 2'),
        "thermal_zone:_p2-1-commerce_3_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P2-1-COMMERCE 3'),
        "thermal_zone:_p2-2-restroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P2-2-RESTROOM 1'),
        "thermal_zone:_p3-1-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P3-1-COMMERCE 1'),
        "thermal_zone:_p3-1-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P3-1-DINING 1'),
        "thermal_zone:_p3-2-dining_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P3-2-DINING 1'),
        "thermal_zone:_p3-2-restroom_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P3-2-RESTROOM 1'),
        "thermal_zone:_p4-1-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P4-1-COMMERCE 1'),
        "thermal_zone:_p4-2-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P4-2-COMMERCE 1'),
        "thermal_zone:_p4-2-commerce_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P4-2-COMMERCE 2'),
        "thermal_zone:_p4-3-commerce_1_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P4-3-COMMERCE 1'),
        "thermal_zone:_p4-3-commerce_2_air_temperature": ('Zone Air Temperature', 'THERMAL ZONE: P4-3-COMMERCE 2'),
        # 热区观测变量 (56个热区的二氧化碳数据)
        "thermal_zone:_hall-1-1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-1'),
        "thermal_zone:_hall-1-10_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-10'),
        "thermal_zone:_hall-1-11_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-11'),
        "thermal_zone:_hall-1-12_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-12'),
        "thermal_zone:_hall-1-13_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-13'),
        "thermal_zone:_hall-1-2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-2'),
        "thermal_zone:_hall-1-3_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-3'),
        "thermal_zone:_hall-1-4_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-4'),
        "thermal_zone:_hall-1-5_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-5'),
        "thermal_zone:_hall-1-6_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-6'),
        "thermal_zone:_hall-1-7_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-7'),
        "thermal_zone:_hall-1-8_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-8'),
        "thermal_zone:_hall-1-9_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-1-9'),
        "thermal_zone:_hall-2-1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-2-1'),
        "thermal_zone:_hall-2-2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-2-2'),
        "thermal_zone:_hall-2-3_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-2-3'),
        "thermal_zone:_hall-3-1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-3-1'),
        "thermal_zone:_hall-3-2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-3-2'),
        "thermal_zone:_hall-3-3_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-3-3'),
        "thermal_zone:_hall-4-1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-4-1'),
        "thermal_zone:_hall-4-2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-4-2'),
        "thermal_zone:_hall-4-3_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-4-3'),
        "thermal_zone:_hall-4-4_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: HALL-4-4'),
        "thermal_zone:_p1-1-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-1-COMMERCE 1'),
        "thermal_zone:_p1-1-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-1-DINING 1'),
        "thermal_zone:_p1-1-office_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-1-OFFICE 1'),
        "thermal_zone:_p1-1-office_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-1-OFFICE 2'),
        "thermal_zone:_p1-10-restroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-10-RESTROOM 1'),
        "thermal_zone:_p1-11-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-11-COMMERCE 1'),
        "thermal_zone:_p1-11-commerce_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-11-COMMERCE 2'),
        "thermal_zone:_p1-11-office_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-11-OFFICE 1'),
        "thermal_zone:_p1-11-office_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-11-OFFICE 2'),
        "thermal_zone:_p1-2-breakroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-2-BREAKROOM 1'),
        "thermal_zone:_p1-2-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-2-DINING 1'),
        "thermal_zone:_p1-2-restroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-2-RESTROOM 1'),
        "thermal_zone:_p1-3-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-3-COMMERCE 1'),
        "thermal_zone:_p1-4-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-4-COMMERCE 1'),
        "thermal_zone:_p1-4-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-4-DINING 1'),
        "thermal_zone:_p1-5-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-5-COMMERCE 1'),
        "thermal_zone:_p1-6-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-6-COMMERCE 1'),
        "thermal_zone:_p1-7-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-7-DINING 1'),
        "thermal_zone:_p1-7-office_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-7-OFFICE 1'),
        "thermal_zone:_p1-8-restroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-8-RESTROOM 1'),
        "thermal_zone:_p1-9-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P1-9-COMMERCE 1'),
        "thermal_zone:_p2-1-commerce_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P2-1-COMMERCE 2'),
        "thermal_zone:_p2-1-commerce_3_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P2-1-COMMERCE 3'),
        "thermal_zone:_p2-2-restroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P2-2-RESTROOM 1'),
        "thermal_zone:_p3-1-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P3-1-COMMERCE 1'),
        "thermal_zone:_p3-1-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P3-1-DINING 1'),
        "thermal_zone:_p3-2-dining_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P3-2-DINING 1'),
        "thermal_zone:_p3-2-restroom_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P3-2-RESTROOM 1'),
        "thermal_zone:_p4-1-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P4-1-COMMERCE 1'),
        "thermal_zone:_p4-2-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P4-2-COMMERCE 1'),
        "thermal_zone:_p4-2-commerce_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P4-2-COMMERCE 2'),
        "thermal_zone:_p4-3-commerce_1_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P4-3-COMMERCE 1'),
        "thermal_zone:_p4-3-commerce_2_co": ('Zone Air CO2 Concentration', 'THERMAL ZONE: P4-3-COMMERCE 2'),
    }






def build_actuators() -> dict:
    new_actuators = {}
    num_zones = 56

    flow = (
        "DIFFUSER","DIFFUSER 1","DIFFUSER 2","DIFFUSER 3","DIFFUSER 4","DIFFUSER 5","DIFFUSER 6","DIFFUSER 7","DIFFUSER 8","DIFFUSER 9",
        "DIFFUSER 10","DIFFUSER 11","DIFFUSER 12","DIFFUSER 13","DIFFUSER 14","DIFFUSER 15","DIFFUSER 16","DIFFUSER 17","DIFFUSER 18",
        "DIFFUSER 19","DIFFUSER 20","DIFFUSER 21","DIFFUSER 22","DIFFUSER 23","DIFFUSER 24","DIFFUSER 25","DIFFUSER 26","DIFFUSER 27",
        "DIFFUSER 28","DIFFUSER 29","DIFFUSER 30","DIFFUSER 31","DIFFUSER 32","DIFFUSER 33","DIFFUSER 34","DIFFUSER 35","DIFFUSER 36",
        "DIFFUSER 37","DIFFUSER 38","DIFFUSER 39","DIFFUSER 40","DIFFUSER 41","DIFFUSER 42","DIFFUSER 43","DIFFUSER 44","DIFFUSER 45",
        "DIFFUSER 46","DIFFUSER 47","DIFFUSER 48","DIFFUSER 49","DIFFUSER 50","DIFFUSER 51","DIFFUSER 52","DIFFUSER 53","DIFFUSER 54","DIFFUSER 55")
    
    for i in range(1, num_zones + 1):
        air_terminal_key = f"{flow[i-1]}"
        new_actuators[air_terminal_key] = (
            'AirTerminal:SingleDuct:ConstantVolume:NoReheat',
            'Mass Flow Rate',
            flow[i-1]

    )
    return new_actuators


def main() -> None:
    new_action_space = gym.spaces.Box(
        low=np.array([0] * 56, dtype=np.float32),
        high=np.array([1] * 56, dtype=np.float32),
        dtype=np.float32,
    )

    new_meters = {
                "Electricity:HVAC" : "total_electricity_HVAC",
    }



    # ------------------------------ definition of new_reward ------------------------------ #
    reward_class = Myreward
    new_reward_kwargs = {
        "temperature_variables" : [
            "thermal_zone:_hall-1-1_air_temperature", "thermal_zone:_hall-1-10_air_temperature", "thermal_zone:_hall-1-11_air_temperature", "thermal_zone:_hall-1-12_air_temperature",
            "thermal_zone:_hall-1-13_air_temperature", "thermal_zone:_hall-1-2_air_temperature", "thermal_zone:_hall-1-3_air_temperature", "thermal_zone:_hall-1-4_air_temperature",
            "thermal_zone:_hall-1-5_air_temperature", "thermal_zone:_hall-1-6_air_temperature", "thermal_zone:_hall-1-7_air_temperature", "thermal_zone:_hall-1-8_air_temperature",
            "thermal_zone:_hall-1-9_air_temperature", "thermal_zone:_hall-2-1_air_temperature", "thermal_zone:_hall-2-2_air_temperature", "thermal_zone:_hall-2-3_air_temperature",
            "thermal_zone:_hall-3-1_air_temperature", "thermal_zone:_hall-3-2_air_temperature", "thermal_zone:_hall-3-3_air_temperature", "thermal_zone:_hall-4-1_air_temperature",
            "thermal_zone:_hall-4-2_air_temperature", "thermal_zone:_hall-4-3_air_temperature", "thermal_zone:_hall-4-4_air_temperature", "thermal_zone:_p1-1-commerce_1_air_temperature",
            "thermal_zone:_p1-1-dining_1_air_temperature", "thermal_zone:_p1-1-office_1_air_temperature", "thermal_zone:_p1-1-office_2_air_temperature", "thermal_zone:_p1-10-restroom_1_air_temperature",
            "thermal_zone:_p1-11-commerce_1_air_temperature", "thermal_zone:_p1-11-commerce_2_air_temperature","thermal_zone:_p1-11-office_1_air_temperature","thermal_zone:_p1-11-office_2_air_temperature",
            "thermal_zone:_p1-2-breakroom_1_air_temperature","thermal_zone:_p1-2-dining_1_air_temperature", "thermal_zone:_p1-2-restroom_1_air_temperature","thermal_zone:_p1-3-commerce_1_air_temperature",
            "thermal_zone:_p1-4-commerce_1_air_temperature","thermal_zone:_p1-4-dining_1_air_temperature", "thermal_zone:_p1-5-commerce_1_air_temperature","thermal_zone:_p1-6-commerce_1_air_temperature",
            "thermal_zone:_p1-7-dining_1_air_temperature","thermal_zone:_p1-7-office_1_air_temperature", "thermal_zone:_p1-8-restroom_1_air_temperature","thermal_zone:_p1-9-commerce_1_air_temperature",
            "thermal_zone:_p2-1-commerce_2_air_temperature","thermal_zone:_p2-1-commerce_3_air_temperature", "thermal_zone:_p2-2-restroom_1_air_temperature","thermal_zone:_p3-1-commerce_1_air_temperature",
            "thermal_zone:_p3-1-dining_1_air_temperature", "thermal_zone:_p3-2-dining_1_air_temperature", "thermal_zone:_p3-2-restroom_1_air_temperature","thermal_zone:_p4-1-commerce_1_air_temperature",
            "thermal_zone:_p4-2-commerce_1_air_temperature", "thermal_zone:_p4-2-commerce_2_air_temperature","thermal_zone:_p4-3-commerce_1_air_temperature", "thermal_zone:_p4-3-commerce_2_air_temperature"
        ],
        "energy_variables"      : ["total_electricity_HVAC"],
        "co2_variables"         : [	"thermal_zone:_hall-1-1_co","thermal_zone:_hall-1-10_co","thermal_zone:_hall-1-11_co","thermal_zone:_hall-1-12_co","thermal_zone:_hall-1-13_co",
									"thermal_zone:_hall-1-2_co","thermal_zone:_hall-1-3_co","thermal_zone:_hall-1-4_co","thermal_zone:_hall-1-5_co","thermal_zone:_hall-1-6_co",
									"thermal_zone:_hall-1-7_co","thermal_zone:_hall-1-8_co","thermal_zone:_hall-1-9_co","thermal_zone:_hall-2-1_co","thermal_zone:_hall-2-2_co",
									"thermal_zone:_hall-2-3_co","thermal_zone:_hall-3-1_co","thermal_zone:_hall-3-2_co","thermal_zone:_hall-3-3_co","thermal_zone:_hall-4-1_co",
									"thermal_zone:_hall-4-2_co","thermal_zone:_hall-4-3_co","thermal_zone:_hall-4-4_co","thermal_zone:_p1-1-commerce_1_co","thermal_zone:_p1-1-dining_1_co",
									"thermal_zone:_p1-1-office_1_co","thermal_zone:_p1-1-office_2_co","thermal_zone:_p1-10-restroom_1_co","thermal_zone:_p1-11-commerce_1_co",
                                    "thermal_zone:_p1-11-commerce_2_co","thermal_zone:_p1-11-office_1_co","thermal_zone:_p1-11-office_2_co","thermal_zone:_p1-2-breakroom_1_co",
									"thermal_zone:_p1-2-dining_1_co","thermal_zone:_p1-2-restroom_1_co","thermal_zone:_p1-3-commerce_1_co","thermal_zone:_p1-4-commerce_1_co","thermal_zone:_p1-4-dining_1_co",
									"thermal_zone:_p1-5-commerce_1_co","thermal_zone:_p1-6-commerce_1_co","thermal_zone:_p1-7-dining_1_co","thermal_zone:_p1-7-office_1_co","thermal_zone:_p1-8-restroom_1_co",
									"thermal_zone:_p1-9-commerce_1_co","thermal_zone:_p2-1-commerce_2_co","thermal_zone:_p2-1-commerce_3_co","thermal_zone:_p2-2-restroom_1_co","thermal_zone:_p3-1-commerce_1_co",
									"thermal_zone:_p3-1-dining_1_co","thermal_zone:_p3-2-dining_1_co","thermal_zone:_p3-2-restroom_1_co","thermal_zone:_p4-1-commerce_1_co",
									"thermal_zone:_p4-2-commerce_1_co","thermal_zone:_p4-2-commerce_2_co","thermal_zone:_p4-3-commerce_1_co","thermal_zone:_p4-3-commerce_2_co"
                                    ],
        "range_comfort_winter"  : [20, 24],
        "range_comfort_summer"  : [20, 24],
        "summer_start"          : [6, 1],
        "summer_final"          : [9, 30],
        "energy_weight"         : 0.32,
        "temp_violation_weight" : 0.38,
        "co2_weight"            : 0.3,
        "lambda_energy"         : 1e-8,
        "lambda_temperature"    : 1e-4,
		"lambda_co2"            :1e-4,
		"range_comfort_co2"     :[0, 800],
		"co2_threshold"         :800
    }

    environment = "Eplus-carbon-mixed-continuous-stochastic-v1"
    episodes = 200
    experiment_date = datetime.today().strftime("%Y-%m-%d-%H_%M")
    experiment_name = f"RPPO-{environment}-episodes-{episodes}_{experiment_date}"

    extra_params = {"timesteps_per_hour": 6, "runperiod": (1, 7, 2006, 31, 7, 2006)}

    env = gym.make(
        environment,
        env_name=experiment_name,
        variables = new_variables,
        reward = reward_class,
        reward_kwargs = new_reward_kwargs,
        config_params=extra_params,
        actuators=build_actuators(),
        action_space=new_action_space,
    )

    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = NormalizeReward(env)
    env = LoggerWrapper(env)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        batch_size=64,
        n_steps=1024,
        verbose=1,
        policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=3),
    )

    callbacks = []
    logger = SB3Logger(
        folder=None, output_formats=[HumanOutputFormat(sys.stdout, max_length=120)]
    )
    model.set_logger(logger)

    timesteps = episodes * (env.get_wrapper_attr("timestep_per_episode") - 1)
    callbacks.append(LoggerCallback())
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=timesteps, callback=callback, log_interval=100)

    model.save(f"{env.get_wrapper_attr('timestep_per_episode')}/{experiment_name}")

    if hasattr(env, "mean") and hasattr(env, "var"):
        training_mean = env.get_wrapper_attr("mean")
        training_var = env.get_wrapper_attr("var")
        _ = (training_mean, training_var)

    env.close()


if __name__ == "__main__":
    main()
