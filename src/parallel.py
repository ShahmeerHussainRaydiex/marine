# Functions to run processes in parallel
# Functions need to be imported and not created in Jupyter notebooks (for an unknown reason)
from src.master_util import sheet_processing, add_geodata, loop
import warnings
warnings.filterwarnings("ignore")


def run_marine_plan(instance, name):
    instance.prepare_optimization(msg=0, name=name)
    instance.run_linear_optimization()


# For running when testing the substations -- just doesnt prepare the optimization as that is already done
def run_marine_plan_substations(instance, name):
    instance.run_linear_optimization()
    return instance # Return the instance so that the results can be extracted

# For the precompute
def process_sheet(excel, sheet, shape, m, geo_data, lo, scale):
    tech_process = sheet_processing(excel, sheet, shape, m)
    tech_geo = add_geodata(geo_data, lo, tech_process, scale)

    print(tech_geo.keys())

    print(f'\n{sheet} has been successfully prepared for calculations.')

    # This sets off the calculations
    calculated = loop(tech_geo)

    print(f'Calculations for {sheet} have been completed.')

    return sheet, calculated
