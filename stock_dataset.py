import pandas as pd
import torch


class StockDataset(torch.utils.data.Dataset):
    filter_columns = ['종가', '시가', '고가', '저가', '거래량', '거래대금', '시가총액', '상장주식수']
    y_columns = ['대비', '등락률']

    def __init__(self, csv_file, days_later=1):
        self.days_later = days_later

        csv_data = pd.read_csv(csv_file, encoding='cp949', index_col=None)
        valid_csv_data = self.filter_valid_data(csv_data)
        self.preprocess_data(valid_csv_data)
        self.make_preprocessed_data_csv(csv_file + '_x.csv',
                                        csv_file + '_y.csv')

    def filter_valid_data(self, data):
        for column in self.filter_columns:
            data = data[~(data[column] == 0)]
        data.sort_values(by=['일자'], inplace=True, ignore_index=True)
        return data

    def preprocess_data(self, data):
        self.make_x_data(data)
        self.make_y_data(data)

    def make_x_data(self, data):
        self.x_data = data[:len(data) - self.days_later]

    def make_y_data(self, data):
        y_data = data[self.days_later:]
        self.y_data = y_data[self.y_columns]
        new_data = pd.DataFrame({
            '금일 종가 - 시가 변동률':
            ((y_data['종가'] - y_data['시가']) / y_data['시가']) * 100
        })
        self.y_data = pd.concat([self.y_data, new_data], axis=1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x_tensor = torch.FloatTensor(self.x_data[index])
        y_tensor = torch.FloatTensor(self.y_data[index])
        return x_tensor, y_tensor

    def make_preprocessed_data_csv(self, csv_name_x, csv_name_y):
        self.x_data.to_csv(csv_name_x, index=False)
        self.y_data.to_csv(csv_name_y, index=False)
