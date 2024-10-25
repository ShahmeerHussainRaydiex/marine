# This file contains the configuration of the substations including:
# - max_power: the maximum power that the substation can deliver
# - cost_per_km: the cost of the substation per km
# - cost_energy_lost_per_km: the cost of the energy lost per km
# - dev_cost: the development cost of the substation
# - insurance: the insurance cost of the substation
# - install_uninstall: the cost of installing and uninstalling the substation


# ADD CABLE PPOWERES

sub_CONFIG = {
    'x1900_x1900': {
        'theoretical_max_power': 864.02,
        'max_turbines': 57,
        'max_power': 855,
        'cost_per_km': 3_400_000,
        'cost_energy_lost_per_km': 293_067.9024,
        'dev_cost': 17_777_225.94,
        'insurance': 17_777_225.94,
        'install_uninstall': 360_818_633.8
    },
    'x1600_x1600': {
        'theoretical_max_power': 693.7,
        'max_turbines': 46,
        'max_power': 690,
        'cost_per_km': 3_000_000,
        'cost_energy_lost_per_km': 207_033.84,
        'dev_cost': 16_451_416.94,
        'insurance': 16_451_416.94,
        'install_uninstall': 346_146_347.4
    },

    'x1200_x1200': {
        'theoretical_max_power': 523.39,
        'max_turbines': 34,
        'max_power': 510,
        'cost_per_km': 2_600_000,
        'cost_energy_lost_per_km': 137_491.704,
        'dev_cost': 15_125_607.93,
        'insurance': 15_125_607.93,
        'install_uninstall': 331_474_061.1
    },

    'x1600_x1900': {
        'theoretical_max_power': 778.86,
        'max_turbines': 51,
        'max_power': 775,
        'cost_per_km': 3_200_000,
        'cost_energy_lost_per_km': 250_050.8712,
        'dev_cost': 17_114_321.44,
        'insurance': 17_114_321.44,
        'install_uninstall': 353_482_490.6
    },

    'x1200_x1600': {
        'theoretical_max_power': 608.54,
        'max_turbines': 40,
        'max_power': 600,
        'cost_per_km': 2_800_000,
        'cost_energy_lost_per_km': 172_262.772,
        'dev_cost': 15_888_512.43,
        'insurance': 15_888_512.43,
        'install_uninstall': 338_810_204.2 
    },

    # 'x1200_x1900': {
    #     'max_power': 693.7,
    #     'cost_per_km': 3_000_000,
    #     'cost_energy_lost_per_km': 215_279.8032,
    #     'dev_cost': 16_451_416.94,
    #     'insurance': 16_451_416.94,
    #     'install_uninstall': 346_146_347.4
    # }

    'x185': 
    {
        'cost_per_km': 200_000,
        'cost_energy_lost_per_km': 223_844.28,
        'max_turbines': 0,
    },
    'x630':
    {
        'cost_per_km': 400_000,
        'cost_energy_lost_per_km': 212_342.40,
        'max_turbines': 0,
    },
}


