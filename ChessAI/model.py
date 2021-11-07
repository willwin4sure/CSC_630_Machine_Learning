import torch
from torch.serialization import save
from tqdm.contrib import tenumerate
import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt
import math
import time

def convert_fen_to_encoding(fen_string):
    one_hot_dict = {'P': [1,0,0,0,0,0,0,0,0,0,0,0], 
                    'N': [0,1,0,0,0,0,0,0,0,0,0,0], 
                    'B': [0,0,1,0,0,0,0,0,0,0,0,0], 
                    'R': [0,0,0,1,0,0,0,0,0,0,0,0], 
                    'Q': [0,0,0,0,1,0,0,0,0,0,0,0], 
                    'K': [0,0,0,0,0,1,0,0,0,0,0,0], 
                    'p': [0,0,0,0,0,0,1,0,0,0,0,0], 
                    'n': [0,0,0,0,0,0,0,1,0,0,0,0], 
                    'b': [0,0,0,0,0,0,0,0,1,0,0,0], 
                    'r': [0,0,0,0,0,0,0,0,0,1,0,0], 
                    'q': [0,0,0,0,0,0,0,0,0,0,1,0], 
                    'k': [0,0,0,0,0,0,0,0,0,0,0,1], 
                    '.': [0,0,0,0,0,0,0,0,0,0,0,0]}
    fen_string_props = fen_string.split(' ')
    rows = chess.Board(fen_string).__str__().split('\n')
    squares_encoding = []
    for row in rows:
        squares_encoding += list(map(lambda x: one_hot_dict[x], row.split(' ')))

    if fen_string_props[1] == 'w':
        turn = [1, 0]
    elif fen_string_props[1] == 'b':
        turn = [0, 1]
    else:
        turn = [0, 0]

    castle_privileges = fen_string_props[2]
    castle_privileges_encoding = [int(x in castle_privileges) for x in ['K', 'Q', 'k', 'q']]

    flattened_squares_encoding = []
    for square in squares_encoding:
        flattened_squares_encoding += square

    full_encoding = flattened_squares_encoding + turn + castle_privileges_encoding
    
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

def convert_to_pawn_advantage(output):
    output *= 0.2250
    output += 0.5385
    output = max(output, 1e-10)
    output = min(output, 1-1e-10)
    return 400 * math.log10(output/(1-output))

class ChessDataset(torch.utils.data.Dataset):
    '''Chess dataset'''

    def __init__(self, file_path, encoded=False, save_path=None):
        if not encoded:
            df = pd.read_csv(file_path)
            row_encodings = []
            for _, row in df.iterrows():
                row_encoding = [1/(1+10**(filter_mates(row['Evaluation'])/(-400)))]
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

        input(torch.mean(self.output))
        input(torch.std(self.output))

        self.output = self.output - torch.mean(self.output)
        self.output = self.output / torch.std(self.output)

        self.output = self.output.reshape((-1,1))

        # input(self.input.size())
        # input(self.output.size())
        
        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.output = self.output.cuda()

        input(self.output[:25])

    def __getitem__(self, index):
        return {'input': self.input[index], 'output': self.output[index]}

    def __len__(self):
        return len(self.input)

def main():
    dataset = ChessDataset('data/smallChessDataEncoded.csv', encoded=True)

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
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 1),
    )
    # Loss and optimization functions
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train model
    for epoch in range(1, 101):
        sum_loss = 0
        for _, elem in tenumerate(train_loader):
            model.train()
            # Forward pass
            output = model(elem['input'])
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, elem['output'])
            sum_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

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
            torch.save(model.state_dict(), f'models/models11032021/model_{epoch}.pt')

def predict_model(fen):
    encoding = convert_fen_to_encoding(fen)

    model = torch.nn.Sequential(
        torch.nn.Linear(774, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 1),
    )

    model.load_state_dict(torch.load('model_70.pt', map_location='cpu'))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with torch.no_grad():
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))
        print(output.item(), convert_to_pawn_advantage(output.item()))

    return convert_to_pawn_advantage(output.item())

    # with torch.no_grad():
    #     for n in range(15):
    #         output = model(torch.unsqueeze(dataset[n]['input'], dim=0))
    #         print(output.item(), convert_to_pawn_advantage(output.item()))
    #         print(dataset[n]['output'].item(), convert_to_pawn_advantage(dataset[n]['output'].item()))
    #         print('-----')
        

if __name__ == "__main__":
    main()
    # predict_model('r1b1k2r/pppp1ppp/8/6B1/2QNn3/P1P5/2P2PPP/R3K2R b KQkq - 0 11')
