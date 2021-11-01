import torch
from torch.serialization import save
from tqdm.contrib import tenumerate
import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt

def convert_fen_to_encoding(fen_string):
    one_hot_dict = {'P': '100000000000', 'N': '010000000000', 'B': '001000000000', 'R': '000100000000', 'Q': '000010000000', 'K': '000001000000', 'p': '000000100000', 'n': '000000010000', 'b': '000000001000', 'r': '000000000100', 'q': '000000000010', 'k': '000000000001', '.': '000000000000'}
    fen_string_props = fen_string.split(' ')
    rows = chess.Board(fen_string).__str__().split('\n')
    squares_encoding = []
    for row in rows:
        squares_encoding.append(list(map(lambda x: [int(char) for char in one_hot_dict[x]], row.split(' '))))

    if fen_string_props[1] == 'w':
        turn = [1, 0]
    elif fen_string_props[1] == 'b':
        turn = [0, 1]
    else:
        turn = [0, 0]

    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = [int(x in castle_privileges) for x in ['K', 'Q', 'k', 'q']]

    row_encoding = []
    for row in squares_encoding:
        for square in row:
            row_encoding = row_encoding + square

    full_encoding = row_encoding + turn + castle_privileges_encoding
    
    return full_encoding

def filter_mates(eval):
    if '#' in str(eval):
        if '+' in str(eval):
            return 20000
        if '-' in str(eval):
            return -20000
        else:
            return 0
    return int(eval)

class ChessDataset(torch.utils.data.Dataset):
    '''Chess dataset'''

    def __init__(self, file_path, encoded=False, save_path=None):
        if not encoded:
            df = pd.read_csv(file_path)
            row_encodings = []
            for _, row in df.iterrows():
                row_encoding = [filter_mates(row['Evaluation'])]
                row_encoding = row_encoding + convert_fen_to_encoding(row['FEN'])
                row_encodings.append(row_encoding)

            row_encodings = np.array(row_encodings)
            columns_list = ['Evaluation']
            for i in range(774):
                columns_list.append('Encoding_' + str(i))
            df_encoded = pd.DataFrame(row_encodings, columns=columns_list)

            if save_path is None:
                df_encoded.to_csv('data/EncodedDataset.csv', index=False)
            else:
                df_encoded.to_csv(save_path, index=False)
        
        else:
            df_encoded = pd.read_csv(file_path)

        self.input = torch.Tensor(df_encoded.iloc[:, 1:].to_numpy())
        # input(self.input)
        self.output = torch.Tensor(df_encoded['Evaluation'])

        # self.output = torch.minimum(self.output, torch.quantile(self.output, 0.88))
        # self.output = torch.maximum(self.output, torch.quantile(self.output, 0.10))

        self.output = 1/(1+10 ** (self.output / (-400)))

        self.output = self.output - torch.mean(self.output)
        self.output = self.output / torch.std(self.output)

        self.output = self.output.reshape((-1,1))

        # input(self.input.size())
        # input(self.output.size())
        
        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.output = self.output.cuda()

        # input(self.output[:25])

    def __getitem__(self, index):
        return {'input': self.input[index], 'output': self.output[index]}

    def __len__(self):
        return len(self.input)

def main():
    dataset = ChessDataset('data/smallChessData3White.csv', encoded=False, save_path='data/smallChessData3WhiteEncoded.csv')

    train_len = int(len(dataset)*0.8) 
    test_len = len(dataset) - train_len

    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])

    # Load data
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=512, shuffle=True)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(774, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 1),
    )
    # Loss and optimization functions
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    # Train model
    for epoch in range(1, 101):
        sum_loss = 0
        for _, elem in tenumerate(train_loader):
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

        if epoch % 10 == 0:
            # Save model
            torch.save(model.state_dict(), f'model_{epoch}.pt')

def test_models():
    dataset = ChessDataset('data/smallChessDataEncoded.csv', encoded=True)

    model = torch.nn.Sequential(
        torch.nn.Linear(774, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 1),
    )

    model.load_state_dict(torch.load('models/smallChessData2_512sizeModels/model_70.pt'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        model()

if __name__ == "__main__":
    main()
    # test_models()