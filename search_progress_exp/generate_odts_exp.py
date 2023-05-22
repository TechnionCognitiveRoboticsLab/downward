import pandas
from collections import defaultdict
import os

df = pandas.read_csv("dataset.csv")
domains = defaultdict(lambda: [])

for i, row in df.iterrows():
    if row.expansions > 10**3 and row.expansions < 10**6:
        domains[row.domain].append(row.filename)

for domain in domains:    
    with open(os.path.join("odts", domain + "_test.txt"), "w") as test:  
        for f in domains[domain]:
            print(f, file=test)        
    with open(os.path.join("odts", domain + "_train.txt"), "w") as train:
        for other_domain in domains:
            if other_domain != domain:
                for f in domains[other_domain]:
                    print(f, file=train)        

