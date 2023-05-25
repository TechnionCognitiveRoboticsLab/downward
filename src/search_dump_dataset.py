import pandas
import sys
import gzip
from torch.utils.data import Dataset
import numpy

class SearchDumpDataset(Dataset):
    def __init__(self, datafilename, height = 3, min_expansions = 1000, max_expansions = 1000000):        
        self.datafilename = datafilename
        self.height = height
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        
        df = pandas.read_csv(datafilename)    
        self.data_df = df[(df.expansions <= self.max_expansions) & (df.expansions >= self.min_expansions)]
        self.data = {}

        self.headers = []
        self.basic_header_names = []

        files_to_train = self.data_df.search_dump_file
        for filename in files_to_train:
            with gzip.open(filename, "r") as f:
                dfr = pandas.read_csv(f, sep='\t')
                df = dfr.set_index('id')
                self.data[filename] = df

                if len(self.headers) == 0:                    
                    self.headers = ['search_algorithm','heuristic','domain','problem', 'filename', 'search_progress']     
                    for c in dfr.columns:
                        if c != 'id' and c != 'path':
                            self.headers.append(c)
                            self.basic_header_names.append(c)
                    for i in range(self.height):
                        for c in dfr.columns:
                            if c != 'id' and c != 'path':
                                self.headers.append('a' + str(i) + '_' + c)
                

    def __len__(self):        
        return self.data_df.expansions.sum()
    
    def get_search_dump(self, filename):
        return self.data[filename]
    
    def generate_row(self, data_row, relative_idx):
        df = self.get_search_dump(data_row.search_dump_file)
        row = df.iloc[relative_idx]
        label = row.N / (len(df.index) - 1)
        path = row.path
        crow = row[self.basic_header_names].values
        megarow_list = [[data_row.search_algorithm, data_row.heuristic, data_row.domain, data_row.problem, data_row.search_dump_file],[label],crow]
        if isinstance(path, str):                    
            nodes = list(filter(lambda node: node != "", path.split(",")[::-1]))[:self.height]                    
        else:
            nodes = []
        for node in nodes:                    
            node_row = df.loc[node][self.basic_header_names].values
            if node_row.ndim > 1:
                node_row = node_row[0,:]
            megarow_list.append(node_row)                        
        megarow_list.append([0] * (len(df.columns) - 2) * (self.height - len(nodes)))                                                    
        megarow = numpy.concatenate(megarow_list,axis=None) 
        return megarow
    
    def __getitem__(self, idx):
        current_idx = 0
        for i, row in self.data_df.iterrows():
            if idx >= current_idx and idx < current_idx + row.expansions:
                relative_idx = idx - current_idx
                return self.generate_row(row, relative_idx)                                
            else:
                current_idx = current_idx + row.expansions

        return None

def main():
    filename="/home/karpase/git/downward/experiments/search_progress_estimate/search_progress_exp-eval/data.csv"
    #filename=sys.argv[1]
    ds = SearchDumpDataset(filename)
    print(ds)
    print(len(ds))
    print(ds[0])
    print(ds[1])
    print(ds[3])
    print(ds[3000000])
    
    

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()