import pandas
import numpy
import torch
from torch.utils.data import Dataset, ConcatDataset, RandomSampler, DataLoader, WeightedRandomSampler
import random

#from speechbrain.dataio.sampler import ConcatDatasetBatchSampler

class SingleFileSearchDumpDataset(Dataset):    
    def __init__(self, datafilename : str, 
                 height : int = 3, 
                 seq_len : int = 10):
        self.datafilename = datafilename
        self.height = height
        self.seq_len = seq_len
        
        self.basic_header_names = []
        self.df = pandas.read_csv(self.datafilename, sep='\t', compression="infer", index_col="id")
               
        for c in self.df.columns:
            if c != 'id' and c != 'path':                
                self.basic_header_names.append(c)

    def generate_row_X(self, row):
        path = row.path
        crow = row[self.basic_header_names].to_numpy(dtype=numpy.float64)
        megarow_list = [#[data_row.search_algorithm, data_row.heuristic, data_row.domain, data_row.problem, data_row.search_dump_file],
                        crow]
        if isinstance(path, str):                    
            nodes = list(filter(lambda node: node != "", path.split(",")[::-1]))[:self.height]                    
        else:
            nodes = []
        for node in nodes:                    
            node_row = self.df.loc[node][self.basic_header_names].to_numpy(dtype=numpy.float64)
            if node_row.ndim > 1:
                node_row = node_row[0,:]
            megarow_list.append(node_row)                        
        megarow_list.append([0.0] * len(self.basic_header_names) * (self.height - len(nodes)))
        megarow = torch.as_tensor(numpy.concatenate(megarow_list,axis=None)).float()
        return megarow

    def __getitem__(self, idx):        
        row = self.df.iloc[idx]
        label = torch.as_tensor( row.N / (len(self.df.index) - 1) ).float()
        Xs = []  
        current_index = idx      
        while current_index >= 0 and current_index > idx - self.seq_len:
            row = self.df.iloc[current_index]
            Xs.append(self.generate_row_X(row))
            current_index = current_index - 1
        while current_index > idx - self.seq_len:
            Xs.append(torch.as_tensor([0.0] * len(self.basic_header_names) * (self.height + 1)))
            current_index = current_index - 1
        return torch.stack(Xs), label
    
    def __len__(self):        
        return len(self.df)

class SearchDumpDataset(ConcatDataset):    
    def __init__(self, datafilename : str, 
                 height : int = 3, 
                 seq_len : int = 10,
                 min_expansions : int = 1000, 
                 max_expansions : int = 1000000, 
                 search_algorithm : str = "",
                 heuristic : str = "",
                 domain : str = "",
                 not_domain : bool = False):
        self.datafilename = datafilename
        self.height = height
        self.seq_len = seq_len
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
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

        files_to_train = self.data_df.search_dump_file
        datasets = []
        for filename in files_to_train:
            datasets.append(SingleFileSearchDumpDataset(filename, height, seq_len))
        ConcatDataset.__init__(self, datasets)

# class SearchDumpDatasetSampler(ConcatDatasetBatchSampler):
#     def __init__(self, ds : SearchDumpDataset, batch_size_per_dump : int = 1):        
#         ConcatDatasetBatchSampler.__init__(self,
#             [RandomSampler(dataset) for dataset in ds.datasets],
#             batch_sizes=[batch_size_per_dump for _ in ds.datasets]
#         )


# class that will take in multiple samplers and output batches from a single dataset at a time
class SearchDumpDatasetSampler(WeightedRandomSampler):    
    def __init__(self, ds : SearchDumpDataset, num_samples : int, replacement : bool = True, generator = None):
        weights = torch.concat( [torch.tensor([1 / len(dataset)] * len(dataset), dtype=torch.float) for dataset in ds.datasets])
        WeightedRandomSampler.__init__(self, torch.tensor(weights,dtype=torch.float), num_samples, replacement, generator)
        



def main():
    random.seed(42)
    #filename="/home/karpase/git/downward/experiments/search_progress_estimate/data/search_progress_exp-eval/data.csv"    
    filename="/home/karpase/git/downward/experiments/search_progress_estimate/search_progress_exp-eval/data.csv"    
    #filename=sys.argv[1]
    ds = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=1000, domain="depot", not_domain=True)
    ds2 = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=1000, domain="depot", not_domain=False)

    
    print(len(ds), len(ds2))
    print(ds[0])
    print(ds[1])
    print(ds[3])

    sampler = SearchDumpDatasetSampler(ds, num_samples=100)
    dataloader = DataLoader(ds, sampler=sampler)
    
    for batch in sampler:
        print(batch)

    for X, y in dataloader:
        print(X,y)

    

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()