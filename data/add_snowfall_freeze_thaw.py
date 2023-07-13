import pandas as pd
from collections import defaultdict

def main():
    # Read dataframes
    ne_nbi = pd.read_csv('nebraska_deep.csv')
    ne_snowfall_freezethaw = pd.read_csv('nebraska-mean-snowfall-freezethaw.csv')

    # Create maps for snowfall and freezethaw
    struct_snow_map = dict(zip(ne_snowfall_freezethaw['Unnamed: 0'],
                               ne_snowfall_freezethaw['snowfall'] ))
    struct_freeze_map = dict(zip(ne_snowfall_freezethaw['Unnamed: 0'],
                               ne_snowfall_freezethaw['freezethaw'] ))

    ne_nbi['snowfall'] = ne_nbi['structureNumber'].map(struct_snow_map)
    ne_nbi['freezethaw'] = ne_nbi['structureNumber'].map(struct_freeze_map)

    ne_nbi.to_csv('nebraska_deep_new.csv')

main()
