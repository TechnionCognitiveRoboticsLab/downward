import pandas
import sys
import numpy

def main():
    filename = sys.argv[1]
    outf = open(filename + "_pp.txt", 'w')

    df = pandas.read_csv(filename, sep='\t')    

    for i, row in df.iterrows():     
        label = row.N / (len(df.index) - 1)
        path = row.path        
        crow = row.drop(['id', 'path']).values
        megarow_list = [crow]
        if isinstance(path, str):
            nodes = path.split(",")[::-1]        
            #print(path, nodes)
            for node in nodes:
                if node != "":
                    node_row = df[df.id == node]                                        
                    cnr = node_row.drop(['id', 'path'],axis=1).values
                    megarow_list.append(cnr)               
        megarow = numpy.concatenate(megarow_list,axis=None)
        megarow_with_label = numpy.append(megarow, [label])
        print('\t'.join(map(str,megarow_with_label)),file=outf)
        
    

if __name__ == "__main__":
    main()