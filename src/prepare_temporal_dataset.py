import pandas
import sys
import numpy
import gzip

def read_training_set(ts_filename : str, domain_name : str, height_limit : int = 3, min_expansions = 1000, max_expansions = 1000000):    
    training_df = pandas.read_csv(ts_filename)    
    files_to_train = training_df[(training_df.domain != domain_name) & (training_df.expansions <= max_expansions) & (training_df.expansions >= min_expansions)].search_dump_file
        

    headers = []
    basic_header_names = []
    data = []

    for filename in files_to_train:
        with gzip.open(filename, "r") as f:
            dfr = pandas.read_csv(f, sep='\t') 
            df = dfr.set_index('id')

            if len(headers) == 0:
                headers = ['filename', 'search_progress']     
                for c in dfr.columns:
                    if c != 'id' and c != 'path':
                        headers.append(c)
                        basic_header_names.append(c)
                for i in range(height_limit):
                    for c in dfr.columns:
                        if c != 'id' and c != 'path':
                            headers.append('a' + str(i) + '_' + c)
                #print('\t'.join(headers))

            for i, row in df.iterrows():     
                label = row.N / (len(df.index) - 1)
                path = row.path        
                crow = row[basic_header_names].values
                megarow_list = [[filename, label],crow]
                if isinstance(path, str):                    
                    nodes = list(filter(lambda node: node != "", path.split(",")[::-1]))[:height_limit]                    
                else:
                    nodes = []
                for node in nodes:                    
                    node_row = df.loc[node][basic_header_names].values
                    if node_row.ndim > 1:
                        node_row = node_row[0,:]
                    megarow_list.append(node_row)                        
                megarow_list.append([0] * (len(dfr.columns) - 2) * (height_limit - len(nodes)))                                                    
                megarow = numpy.concatenate(megarow_list,axis=None)            
                data.append(megarow)
                #print('\t'.join(map(str,megarow)))
                if row.N > 100:  # This is just for efficiency now, delete this later
                    break
    return pandas.DataFrame(data, columns=headers)
                
            
def main():
    #read_training_set(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    df = read_training_set("/data/karpase/search_progress_estimate/data/search_progress_exp-eval/data.csv", "blocks", 3)
    print(df)
    df.to_csv("blocks_training.csv")

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    # $2 - name of domain to predict (exclude from training set)
    # $3 - how high up to got on the path to the root
    main()