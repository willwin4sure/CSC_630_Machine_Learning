import torch
from tqdm.contrib import tenumerate
import numpy as np
import pandas as pd

class ChessDataset(torch.utils.data.Dataset):
    '''Chess dataset'''

    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.input = torch.Tensor(df['Encoding'].apply(lambda x: [int(char) for char in str(x)]))
        self.output = torch.Tensor(df['EvaluationFiltered'])

        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.output = self.output.cuda()

        self.output = torch.minimum(self.output, torch.quantile(self.output, 0.95))
        self.output = torch.maximum(self.output, torch.quantile(self.output, 0.05))

        self.output = self.output - torch.mean(self.output)
        self.output = self.output / torch.std(self.output)

        self.output = self.output.reshape((-1,1))
        input(self.output[:25])

    def __getitem__(self, index):
        return {'input': self.input[index], 'output': self.output[index]}

    def __len__(self):
        return len(self.input)

def main():
    '''Trains a neural network model that takes in an input layer of 518 nodes, three hidden layers of 128 nodes, and an output layer with a single node, with RELU activation'''

    dataset = ChessDataset('data/smallerChessDataEncoded.csv')

    train_len = int(len(dataset)*0.8) 
    test_len = len(dataset) - train_len

    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])


    # Load data
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=256, shuffle=True)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(518, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)
    )
    # Loss and optimization functions
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Train model
    for epoch in range(1, 501):
        sum_loss = 0
        for batch_idx, elem in tenumerate(train_loader):
            # Forward pass
            output = model(elem['input'])
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, elem['output'])
            sum_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        avg_loss = sum_loss / len(train_loader)
        print(f'Average Loss Epoch {epoch}: {avg_loss}')

        if epoch % 5 == 0:
            # Test model
            model.eval()
            with torch.no_grad():
                sum_loss = 0
                for _,elem in tenumerate(test_loader):
                    output = model(elem['input'])
                    loss = loss_fn(output, elem['output'])
                    sum_loss += loss.item()
                avg_loss = sum_loss / len(test_loader)
                print(f'Average Test Loss Epoch {epoch}: {avg_loss}')
    

if __name__ == "__main__":
    main()