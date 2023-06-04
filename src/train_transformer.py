import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

from search_dump_dataset import SearchDumpDataset


#filename="/home/karpase/git/downward/experiments/search_progress_estimate/data/search_progress_exp-eval/data.csv"    
filename="/home/karpase/static/data.csv"
#filename=sys.argv[1]
train = SearchDumpDataset(filename, height=2, seq_len = 5, min_expansions=5)#, domain="gripper") 
#test = SearchDumpDataset(filename, height=2, seq_len = 5, min_expansions=5, transform=xy_transform, domain="gripper", not_domain=True)
print(len(train))#, len(test))

# X,y = train[4122]

# for i in range(len(train)):
#     print(i)
#     print(train[i])


dim = train[0][0].shape[1] * train[0][0].shape[0]

# class SearchProgressModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=dim, hidden_size=50, num_layers=1, batch_first=True)
#         self.linear = nn.Linear(50, 1)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x
# model = SearchProgressModel()    

model = nn.Sequential(
    nn.Linear(dim, 100, dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(100, 50, dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(50, 1, dtype=torch.float64)
)    



optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

#n_epochs = 100
#for epoch in range(n_epochs):
#    model.train()
#    print(epoch)

Xtrain = []
ytrain = []

i = 0
while i < len(train):
#for i in range(len(train)):
    X = torch.flatten(train[i][0])
    y = train[i][1]
    assert X is not None
    Xtrain.append(X.numpy())
    ytrain.append(y[0].numpy())
    i = i + 100


from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=10)
scores = cross_val_score(clf, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
print("Random Forest", scores.mean())

# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# scores = cross_val_score(reg, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# print("Linear Regression", scores.mean())

# from sklearn.gaussian_process import GaussianProcessRegressor
# gpr = GaussianProcessRegressor()
# scores = cross_val_score(gpr, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
# print("Gaussian Process Regression", scores.mean())


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
scores = cross_val_score(knr, Xtrain, ytrain, scoring=make_scorer(mean_squared_error), cv=5)
print("KNN Regression", scores.mean())

#clf = clf.fit(Xtrain, ytrain)
    
    # if X is not None:
    #     y_pred = model(X)
    #     loss = loss_fn(y_pred, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()