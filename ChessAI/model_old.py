import torch
from tqdm.contrib import tenumerate
import numpy as np
import pandas as pd
import chess

def convert_fen_to_encoding(fen_string):
    one_hot_dict = {'P': '10100000', 'N': '10010000', 'B': '10001000', 'R': '10000100', 'Q': '10000010', 'K': '10000001', 'p': '01100000', 'n': '01010000', 'b': '01001000', 'r': '01000100', 'q': '01000010', 'k': '01000001', '.': '00000000'}
    fen_string_props = fen_string.split(' ')
    rows = chess.Board(fen_string).__str__().split('\n')
    squares_encoding = []
    for row in rows:
        squares_encoding.append(list(map(lambda x: one_hot_dict[x], row.split(' '))))

    if fen_string_props[1] == 'w':
        turn = '10'
    elif fen_string_props[1] == 'b':
        turn = '01'
    else:
        turn = '00'

    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = ''.join([str(int(x in castle_privileges)) for x in ['K', 'Q', 'k', 'q']])

    row_encoding = []
    for row in squares_encoding:
        row_encoding.append(''.join(row))

    board_encoding = ''.join(row_encoding)
    full_encoding = ''.join([board_encoding, turn, castle_privileges_encoding])
    
    return full_encoding

class ChessDataset(torch.utils.data.Dataset):
    '''Chess dataset'''

    def __init__(self, file_path, encoded=False):
        if not encoded:
            df = pd.read_csv(file_path)
            df['Encoding'] = df['FEN'].apply(convert_fen_to_encoding)

            def filter_mates(eval):
                if '#' in str(eval):
                    if '+' in str(eval):
                        return 20000
                    if '-' in str(eval):
                        return -20000
                    else:
                        return 0
                return int(eval)

            df['EvaluationFiltered'] = df['Evaluation'].apply(filter_mates)

            df.to_csv('data/smallerChessDataEncoded.csv', index=False)
        
        else:
            df = pd.read_csv(file_path)

        self.input = torch.Tensor(df['Encoding'].apply(lambda x: [int(char) for char in str(x)]))
        self.output = torch.Tensor(df['EvaluationFiltered'])

        self.output = torch.minimum(self.output, torch.quantile(self.output, 0.90))
        self.output = torch.maximum(self.output, torch.quantile(self.output, 0.10))

        self.output = self.output - torch.mean(self.output)
        self.output = self.output / torch.std(self.output)

        self.output = self.output.reshape((-1,1))

        input(self.input.size())
        input(self.output.size())
        
        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.output = self.output.cuda()

        input(self.output[:25])

    def __getitem__(self, index):
        return {'input': self.input[index], 'output': self.output[index]}

    def __len__(self):
        return len(self.input)

def main():
    '''Trains a neural network model that takes in an input layer of 518 nodes, three hidden layers of 128 nodes, and an output layer with a single node, with RELU activation'''

    dataset = ChessDataset('data/chessDataEncoded.csv', encoded=True)

    train_len = int(len(dataset)*0.8) 
    test_len = len(dataset) - train_len

    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])

    # Load data
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=512, shuffle=True)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(518, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)
    )
    # Loss and optimization functions
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    # Train model
    for epoch in range(1, 501):
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

    # Save model
    torch.save(model.state_dict(), 'model.pt')
    

if __name__ == "__main__":
    main()