from dataclasses import dataclass
from typing import Dict, Any
from src.metric_util_OLD import *

@dataclass
class InfrastructureData:
    name: str
    units_per_day: float
    materials: Dict[str, float]
    vessels_required: int
    crew_per_vessel: int

class Logistics:
    def __init__(self):
        self.infrastructure_types = {
            'monopile': InfrastructureData(
                name="Monopile Wind Turbines",
                units_per_day=0.5,
                materials={'steel': 400, 'concrete': 1200, 'copper': 8},
                vessels_required=3,
                crew_per_vessel=50
            ),
            'jacket': InfrastructureData(
                name="Jacket Foundation Wind Turbines",
                units_per_day=0.5,
                materials={'steel': 600, 'concrete': 800, 'copper': 10},
                vessels_required=4,
                crew_per_vessel=60
            ),
            'fpv': InfrastructureData(
                name="Floating Solar (FPV)",
                units_per_day=500,
                materials={'plastic': 0.02, 'steel': 0.01},
                vessels_required=2,
                crew_per_vessel=30
            ),
            'mussel': InfrastructureData(
                name="Mussel Long Lines",
                units_per_day=5,
                materials={'rope': 200, 'buoys': 20},
                vessels_required=1,
                crew_per_vessel=10
            )
        }

    def calculate_logistics(self, infra_type: str, num_units: int) -> Dict[str, Any]:
        data = self.infrastructure_types[infra_type]
        days_required = num_units / data.units_per_day
        materials_required = {material: num_units * amount for material, amount in data.materials.items()}
        total_crew = data.vessels_required * data.crew_per_vessel

        result = {
            'infrastructure_type': data.name,
            'num_units': num_units,
            'days_required': round(days_required, 2),
            'vessels_required': data.vessels_required,
            'total_crew': total_crew,
            **materials_required
        }
        return result

    def print_logistics_report(self, logistics_result: Dict[str, Any]) -> str:
        output = f"\n*** {logistics_result['infrastructure_type']} ***\n"
        for key, value in logistics_result.items():
            if key != 'infrastructure_type':
                output += f"{key.replace('_', ' ').title()}: {format_number_eu(value)}\n"
        output += "\n"
        return output