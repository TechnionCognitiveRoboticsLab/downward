import pandas
import numpy
import sys
import gzip
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class SearchDumpDataset(Dataset):    
    def __init__(self, datafilename : str, 
                 height : int = 3, 
                 seq_len : int = 10,
                 min_expansions : int = 1000, 
                 max_expansions : int = 1000000, 
                 search_algorithm : str = "",
                 heuristic : str = "",
                 domain : str = "",
                 not_domain : bool = False,
                 transform = None):
        self.datafilename = datafilename
        self.height = height
        self.seq_len = seq_len
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        self.transform = transform
        self.search_algorithm = search_algorithm
        self.heuristic = heuristic
        self.domain = domain
        self.not_domain = not_domain
        
        df = pandas.read_csv(datafilename)    
        df = df[(df.expansions <= self.max_expansions) & (df.expansions >= self.min_expansions)]
        if self.domain != "":
            if self.not_domain:
                df = df[df.domain != self.domain]
            else:
                df = df[df.domain == self.domain]
        if self.search_algorithm != "":
            df = df[df.search_algorithm == self.search_algorithm]
        if self.heuristic != "":
            df = df[df.heuristic == self.heuristic]
        self.data_df = df
        

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
    
    def generate_row_X(self, row, df):
        path = row.path
        crow = row[self.basic_header_names].to_numpy(dtype=numpy.float64)
        megarow_list = [#[data_row.search_algorithm, data_row.heuristic, data_row.domain, data_row.problem, data_row.search_dump_file],
                        crow]
        if isinstance(path, str):                    
            nodes = list(filter(lambda node: node != "", path.split(",")[::-1]))[:self.height]                    
        else:
            nodes = []
        for node in nodes:                    
            node_row = df.loc[node][self.basic_header_names].to_numpy(dtype=numpy.float64)
            if node_row.ndim > 1:
                node_row = node_row[0,:].to_numpy(dtype=numpy.float64)
            megarow_list.append(node_row)                        
        megarow_list.append([0.0] * len(self.basic_header_names) * (self.height - len(nodes)))
        megarow = torch.as_tensor(numpy.concatenate(megarow_list,axis=None))
        return megarow

    def generate_row(self, data_row, relative_idx):
        df = self.get_search_dump(data_row.search_dump_file)
        row = df.iloc[relative_idx]
        label = torch.as_tensor(row.N / (len(df.index) - 1))
        Xs = []
        current_index = relative_idx
        while current_index >= 0 and current_index > relative_idx - self.seq_len:
            row = df.iloc[current_index]
            Xs.append(self.generate_row_X(row, df))
            current_index = current_index - 1
        while current_index > relative_idx - self.seq_len:
            Xs.append(torch.as_tensor([0.0] * len(self.basic_header_names) * (self.height + 1)))
            current_index = current_index - 1
        return torch.stack(Xs), label
    
    def __getitem__(self, idx):
        current_idx = 0
        for i, row in self.data_df.iterrows():
            if idx >= current_idx and idx < current_idx + row.expansions:
                relative_idx = idx - current_idx
                return self.generate_row(row, relative_idx)                                
            else:
                current_idx = current_idx + row.expansions

        return None

import torch

def main():
    filename="/home/karpase/git/downward/experiments/search_progress_estimate/data/search_progress_exp-eval/data.csv"    
    #filename=sys.argv[1]
    ds = SearchDumpDataset(filename, height=2, seq_len = 4, min_expansions=5, domain="depot", not_domain=True)
    ds2 = SearchDumpDataset(filename, height=2, seq_len = 1, min_expansions=5, domain="depot", not_domain=False)

    
    print(len(ds), len(ds2))
    print(ds[0])
    print(ds[1])
    print(ds[3])
    print(ds[3000000])

    

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()