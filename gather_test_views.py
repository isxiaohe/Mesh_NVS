import os 
import json

scans = ('alameda', 'london', 'berlin', 'nyc')
methods = ('2dgs', 'our_2dgs', 'our_pgsr')

for scan in scans:
    paths = [f'./mesh_output/{method}/{scan}/mesh_nvs_fixed/mesh/test/renders' for method in methods]
    shared_views = set(os.listdir(paths[0]))
    for p in paths[1:]:
        shared_views = shared_views.intersection(set(os.listdir(p)))
    shared_views = sorted(list(shared_views))
    with open(f'{scan}_shared_views.json', 'w') as f:
        json.dump(shared_views, f, indent=4)