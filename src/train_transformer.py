import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

from search_dump_dataset import SearchDumpDataset, SearchDumpDatasetSampler
from torch.utils.data import DataLoader

# if torch.cuda.is_available(): 
#     dev = "cuda:0" 
# else: 
#     dev = "cpu" 
# device = torch.device(dev) 
# print("Pytorch CUDA Version is ", torch.version.cuda)
# cuda_id = torch.cuda.current_device()
# print("CUDA Device ID: ", torch.cuda.current_device())
# print("Name of the current CUDA Device: ", torch.cuda.get_device_name(cuda_id))

#filename_train="/home/karpase/git/downward/experiments/search_progress_estimate/data/search_progress_exp-eval/data.csv"    
filename="/home/karpase/static/data.csv"
#filename=sys.argv[1]



train = SearchDumpDataset(filename, height=3, seq_len = 10, min_expansions=10) 
train_sampler = SearchDumpDatasetSampler(train, batch_size_per_dump=1)
train_loader = DataLoader(train, batch_sampler=train_sampler)

# test = SearchDumpDataset(filename_test, height=3, seq_len = 10, min_expansions=10)
# test_sampler = SearchDumpDatasetSampler(test, batch_size_per_dump=1)
# test_loader = DataLoader(test, batch_sampler=test_sampler)



class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
    
learning_rate = 5e-5
num_hidden_units = 16

model = ShallowRegressionLSTM(num_sensors=train[0][0].shape[1], hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#model.to(device)

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        #X.to(device)
        #y.to(device)
        #print("device", X.device, y.device)
        output = model(X)
        loss = loss_function(output, y)

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
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")    

for ix_epoch in range(100):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    #test_model(test_loader, model, loss_function)




# from sklearn.model_selection import cross_val_score

# from sklearn.metrics import mean_squared_error, make_scorer

# from sklearn.ensemble import RandomForestRegressor
# clf = RandomForestRegressor(n_estimators=10)
# scores = cross_val_score(clf, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# print("Random Forest", scores.mean())

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