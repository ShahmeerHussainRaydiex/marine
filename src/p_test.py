from src.master_util import sheet_processing, add_geodata, loop
import warnings
warnings.filterwarnings("ignore")


# For the precompute
def prep_sheet(excel, sheet, shape, m, geo_data, lo, scale):
    tech_process = sheet_processing(excel, sheet, shape, m)
    tech_geo = add_geodata(geo_data, lo, tech_process, scale)

    print(f'\n{sheet} has been successfully prepared for calculations.')

    # This sets off the calculations
    calculated = loop(tech_geo)

    print(f'Calculations for {sheet} have been completed.')

    return sheet, calculated



