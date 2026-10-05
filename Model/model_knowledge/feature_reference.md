# Feature Reference

This document provides a full reference for every feature used in the regression model, including its data type, source, and domain interpretation.

---

## Raw Features from battery_observations.csv

| Feature | Type | Description |
| --- | --- | --- |
| `cycle_index` | int | Sequential index of the charge-discharge cycle for a given battery |
| `battery_age_days` | float | Calendar age of the battery in days at the time of the cycle |
| `time_since_previous_cycle_hours` | float | Hours elapsed since the previous recorded cycle |
| `equivalent_full_cycles` | float | Cumulative equivalent full discharge cycles, weighted by depth of discharge |
| `soc_start_pct` | float | State of charge at the beginning of the cycle (0-100%) |
| `soc_end_pct` | float | State of charge at the end of the cycle (0-100%) |
| `depth_of_discharge_pct` | float | Fraction of usable capacity discharged during the cycle |
| `high_soc_exposure_fraction` | float | Fraction of cycle time spent at high state of charge (accelerates degradation) |
| `low_soc_exposure_fraction` | float | Fraction of cycle time spent at low state of charge |
| `charge_c_rate` | float | Charge current normalized by nominal capacity (C-rate) |
| `discharge_c_rate` | float | Discharge current normalized by nominal capacity (C-rate) |
| `charge_current_a` | float | Absolute charge current in amperes |
| `discharge_current_a` | float | Absolute discharge current in amperes |
| `charge_duration_min` | float | Duration of the charge phase in minutes |
| `discharge_duration_min` | float | Duration of the discharge phase in minutes |
| `energy_discharged_kwh` | float | Energy delivered to the load during the cycle in kilowatt-hours |
| `ambient_temp_c` | float | External ambient temperature in Celsius |
| `pack_temp_c` | float | Internal battery pack temperature in Celsius (1% missing; imputed) |
| `cumulative_high_temp_hours` | float | Total hours the pack has operated above a high-temperature threshold |
| `cooling_active` | int | Binary flag (0/1) indicating whether active cooling was engaged |
| `pack_voltage_mean_v` | float | Mean pack voltage over the cycle in volts |
| `pack_voltage_charge_v` | float | Pack voltage during the charge phase in volts |
| `pack_voltage_discharge_v` | float | Pack voltage during the discharge phase in volts |
| `state_of_health_pct` | float | Current capacity relative to nominal (100% = new, degrades over time) |
| `degradation_status` | str | Ordinal severity label: Normal, Moderate, Severe, Critical |
| `internal_resistance_mOhm` | float | Measured internal resistance in milli-ohms (increases with degradation) |

---

## Features from battery_metadata.csv

| Feature | Type | Values | Description |
| --- | --- | --- | --- |
| `chemistry` | str | LFP, NMC, NCA | Electrochemical composition of the battery cells |
| `manufacturer` | str | Various | Battery pack manufacturer identifier |
| `climate_zone` | str | Temperate, Desert, Nordic, Equatorial | Operational environment of the battery |
| `nominal_capacity_kwh` | float | Continuous | Design capacity of the battery pack |
| `usage_strategy` | str | Balanced, Conservative, Aggressive, Fast-Charge Heavy | Charging and discharging behavior profile |

---

## Engineered Features

| Feature | Formula | Domain Rationale |
| --- | --- | --- |
| `soc_swing` | `soc_start_pct - soc_end_pct` | Captures effective utilization width per cycle. Larger swings correlate with deeper cycling stress and faster capacity fade. |
| `charge_discharge_ratio` | `charge_c_rate / (discharge_c_rate + 1e-6)` | A ratio greater than 1 indicates faster charging relative to discharge. High charge rates accelerate lithium plating and SEI growth. |
| `temp_stress` | `abs(pack_temp_c - 25) * (1 + cumulative_high_temp_hours / 100)` | Combines instantaneous thermal deviation from the optimal 25 C operating point with cumulative heat history. Captures both acute and chronic thermal stress. |
| `energy_per_efc` | `energy_discharged_kwh / (equivalent_full_cycles + 1e-6)` | Normalizes energy output by the number of equivalent full cycles, giving a per-cycle energy efficiency metric that declines as degradation progresses. |
| `voltage_window` | `pack_voltage_charge_v - pack_voltage_discharge_v` | The voltage spread between charge and discharge endpoints. As internal resistance increases with aging, this spread narrows due to greater voltage losses. |
| `cycle_age_ratio` | `cycle_index / (battery_age_days + 1e-6)` | Cycles per day ratio. Batteries cycled more frequently per calendar day face higher overall stress for a given age. |
| `soh_degradation_rate` | `state_of_health_pct / (cycle_index + 1)` | Average SoH per cycle, approximating the trajectory of capacity retention. A battery with SoH 90% at cycle 1000 is degrading more slowly than one at 90% at cycle 200. |

---

## Excluded Features

| Feature | Reason for Exclusion |
| --- | --- |
| `remaining_useful_efc` | Direct leakage: it is a linear transformation of the target variable |
| `eol_reached` | Direct leakage: it is set to True when `remaining_useful_cycles` reaches 0 |
| `battery_id` | Non-predictive identifier used only for split assignment and metadata join |
| `timestamp` | Non-predictive identifier; temporal ordering is captured implicitly by `cycle_index` and `battery_age_days` |
