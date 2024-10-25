NID = {
    'NID1' : {
        'variable1' : 0
    },
    'NID2' : {
        'energy_reduction_curtialment' : 0.04, #2-5% reduciton in energy due to curtailment
        'coloured_blade_days' : 3,
        'blade_recolouring_frequency' : 1/5, #recoloring every 5 years
        'paint_cost' : 36_000,
        'NID_sebaed_area' : 5000, #m^2
        'rock_gravel_partition' : 0.25, #fraction of NID surface reserved for rock/gravel
        'shell_partition' : 0.25, #fraction of NID surface reserved for shell
        'tree_partition' : 0.25, #fraction of NID surface reserved for trees
        'oyster_partition' : 0.25, #fraction of NID surface reserved for oysters
        'tree_cost' : 150, #euro/tree
        'tree_weight' : 500, #kg/tree
        'area_per_tree' : 15, #area/tree
        'shell_density' : 1000, #kg/m^3
        'shell_layer_thickness' : 0.2, #m
        'rock_gravel_layer_thickness' : 0.5, #m
        'oyster_cost_per_juvenile' : 0.03, #euro including substrate cost
        'oyster_density' : 500, #kg/m^3
        'oyster_surface_density' : 500, #oysters/m^2
        'shell_emissions' : 10, #kgCO2eq/tonne
        'tree_emissions' : 75,
        'oyster_emissions' : 35,
        'NID_seabed_placement_time_factor' : 3.6 #multiplicator for placement time of seabed NID compared to scour protection
    },
    'NID3' : {
        'bird_attraction_cost' : 1_000_000, #flat cost for bird attraciton measure
        'bird_attraction_emissions' : 600_000, #flat emissions for bird attraction measure
        'NID_seabed_area_factor' : 5 #multiplactor for increase in NID seabed area compared to NID2
    }
}