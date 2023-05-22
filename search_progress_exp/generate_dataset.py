import os
import glob
import json

# Run this script from the search_progress_exp directory to generate the list of solved problem instances

for filename in glob.iglob('data/search_progress_exp/runs-*/*/properties'): 
    with open(filename, "r") as f:
        properties = json.load(f)
        if properties.get("coverage") == 1:
            sproperties_filename = os.path.join(os.path.dirname(filename), "static-properties")
            with open(sproperties_filename, "r") as fs:
                static_properties = json.load(fs)
                expansions = properties.get("expansions")
                algorithm = static_properties["algorithm"]
                search = algorithm.split("_")[0]
                heuristic = algorithm.split("_")[1]
                search_dump = glob.glob(os.path.join(os.path.dirname(filename), "search_dump_*"))
                print(search_dump[0], expansions, search, heuristic, static_properties["domain"], static_properties["problem"], sep=",")
            
