import torch 
import torch.nn as nn
from torch.autograd import Variable 

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) 
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes) 

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        hn = hn.view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out)
        return out
    

if __name__ == "__main__":
    # wget https://query1.finance.yahoo.com/v7/finance/download/SBUX?period1=1576063151&period2=1607685551&interval=1d&events=history&includeAdjustedClose=true
    import pandas as pd
    df = pd.read_csv('SBUX.csv', index_col = 'Date', parse_dates=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, 5:6]
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    mm = MinMaxScaler()
    ss = StandardScaler()


    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    #first 200 for training

    X_train = X_ss[:200, :]
    X_test = X_ss[200:, :]

    y_train = y_mm[:200, :]
    y_test = y_mm[200:, :]

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape) 

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    #reshaping to rows, timestamps, features

    X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

    X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

    print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)


    num_epochs = 1000 
    learning_rate = 0.001 

    input_size = 5 
    hidden_size = 2 
    num_layers = 1 
    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        
        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)
        
        loss.backward() #calculates the loss of the loss function
        
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    df_X_ss = ss.transform(df.iloc[:, :-1])
    df_y_mm = mm.transform(df.iloc[:, -1:]) 

    df_X_ss = Variable(torch.Tensor(df_X_ss)) 
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    #reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

    train_predict = lstm(df_X_ss)#forward pass
    data_predict = train_predict.data.numpy() #numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=200, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual plot
    plt.plot(data_predict, label='Predicted Data') #predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show() 