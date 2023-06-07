import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pandas

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from search_dump_dataset import SearchDumpDataset, SearchDumpDatasetSampler
from torch.utils.data import DataLoader, BatchSampler

learning_rate = 1e-2
num_hidden_units = 128
num_layers = 6

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev) 
print("Pytorch CUDA Version is ", torch.version.cuda)
cuda_id = torch.cuda.current_device()
print("CUDA Device ID: ", torch.cuda.current_device())
print("Name of the current CUDA Device: ", torch.cuda.get_device_name(cuda_id))

#filename_train="/home/karpase/git/downward/experiments/search_progress_estimate/data/search_progress_exp-eval/data.csv"    
#filename="/home/karpase/static/data.csv"
#filename="/home/karpase/git/downward/experiments/search_progress_estimate/search_progress_exp-eval/data.csv"
filename="/data/karpase/search_progress_estimate/data/search_progress_exp-eval/data.csv"
#filename=sys.argv[1]

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().cuda()        
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().cuda()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        #X.to(device)
        #y.to(device)
        #print("device", X.device, y.device)
        output = model(X.cuda())
        loss = loss_function(output, y.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function):    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:            
            output = model(X.cuda())
            total_loss += loss_function(output, y.cuda()).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")    

def run_sklearn_model(model, data, train = True, sample_size = 1024):
    sklearn_sampler = SearchDumpDatasetSampler(data, num_samples=sample_size)
    sklearn_loader = DataLoader(data, sampler=sklearn_sampler)
    X = []
    Y = []    
    for x, y in sklearn_loader:
        X.append(torch.flatten(x).numpy())
        Y.append(y[0])
    XX = numpy.stack(X)
    YY = numpy.stack(Y)
    if train:
        model.fit(XX,YY)
    else:
        YYpred = model.predict(XX)
        print(f"Test loss: {mean_squared_error(YY, YYpred)}")    
    

def evaluate_domain(domain):    
    train = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=1000, domain=domain, not_domain=True)
    test = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=1000, domain=domain, not_domain=False)
    print("Train", len(train), " Test", len(test))


    print("Random Forest Regression")
    clf = RandomForestRegressor(n_estimators=10)
    run_sklearn_model(clf, train, train=True)
    run_sklearn_model(clf, test, train=False)

    print("kNN Regression")
    knr = KNeighborsRegressor()
    run_sklearn_model(knr, train, train=True)
    run_sklearn_model(knr, test, train=False)

    #train = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=10) 
    train_sampler = BatchSampler(SearchDumpDatasetSampler(train, num_samples=256), batch_size=256, drop_last=True)
    train_loader = DataLoader(train, batch_sampler=train_sampler)

    # test = SearchDumpDataset(filename_test, height=3, seq_len = 10, min_expansions=10)
    test_sampler = BatchSampler(SearchDumpDatasetSampler(test, num_samples=256), batch_size=256, drop_last=True)
    test_loader = DataLoader(test, batch_sampler=test_sampler)

    model = ShallowRegressionLSTM(num_sensors=train[0][0].shape[1], hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for ix_epoch in range(100):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)        
        test_model(test_loader, model, loss_function)



def main():
    df = pandas.read_csv(filename)

    for domain in numpy.unique(df.domain.values):
        print("Evaluating Domain: ", domain)
        evaluate_domain(domain)


if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()

# from sklearn.model_selection import cross_val_score

# # from sklearn import linear_model
# # reg = linear_model.LinearRegression()
# # scores = cross_val_score(reg, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# # print("Linear Regression", scores.mean())

# # from sklearn.gaussian_process import GaussianProcessRegressor
# # gpr = GaussianProcessRegressor()
# # scores = cross_val_score(gpr, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# # print("Gaussian Process Regression", scores.mean())


# from sklearn.neighbors import KNeighborsRegressor
# knr = KNeighborsRegressor()
# scores = cross_val_score(knr, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# print("KNN Regression", scores.mean())

# #clf = clf.fit(Xtrain, ytrain)
    
#     # if X is not None:
#     #     y_pred = model(X)
#     #     loss = loss_fn(y_pred, y)
#     #     optimizer.zero_grad()
#     #     loss.backward()
#     #     optimizer.step()
