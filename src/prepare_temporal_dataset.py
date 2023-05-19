import pandas
import sys
import numpy


def get_list_of_filenames(ts_filename : str):
    f = open(ts_filename, "r")
    return [s.strip() for s in f.readlines()]

def read_training_set(ts_filename : str, height_limit : int = 3):
    ts_filename = sys.argv[1]
    train_files = get_list_of_filenames(ts_filename)

    dfr = pandas.read_csv(train_files[0], sep='\t') 
    headers = ['filename', 'search_progress']     
    for c in dfr.columns:
        if c != 'id' and c != 'path':
            headers.append(c)
    for i in range(height_limit):
        for c in dfr.columns:
            if c != 'id' and c != 'path':
                headers.append('a' + str(i) + '_' + c)
    print('\t'.join(headers))



    for train_filename in train_files:
        dfr = pandas.read_csv(train_filename, sep='\t') 
        df = dfr.set_index('id')

        for i, row in df.iterrows():     
            label = row.N / (len(df.index) - 1)
            path = row.path        
            crow = row.drop(['path']).values
            megarow_list = [[train_filename, label],crow]
            if isinstance(path, str):
                nodes = path.split(",")[::-1]
                height = 0
                for node in nodes:
                    if node != "":
                        if height < height_limit:
                            node_row = df.loc[node].values[:-1]
                            megarow_list.append(node_row)               
                            height = height + 1
                        else:
                            break
            megarow = numpy.concatenate(megarow_list,axis=None)            
            print('\t'.join(map(str,megarow)))
            
def main():
    read_training_set(sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    # $1 - name of a file which contains a list of search dump filenames
    # $2 - how high up to got on the path to the root
    main()