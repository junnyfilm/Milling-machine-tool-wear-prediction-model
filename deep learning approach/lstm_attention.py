import torch
import torch.nn as nn

class lstmattention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=6,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0.1
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2,6)
        self.relu = torch.nn.LeakyReLU(0.1)
        
        # mix up을 적용하기 위해서 learnable parameter인 w를 설정합니다.
        w = torch.nn.Parameter(torch.FloatTensor([-0.01]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        
        self.sigmoid = torch.nn.Sigmoid()
        
        # feature attention을 위한 dense layer를 설정합니다.
        self.dense1 = torch.nn.Linear(6, 12)
        self.dense2 = torch.nn.Linear(12, 1)

    def forward(self, x):

        pool = torch.nn.AdaptiveAvgPool1d(1)
        
        attention_x = x
        attention_x = attention_x.transpose(1,2) # batch, params, window_size
        
        attention = pool(attention_x) # batch, params, 1
        
        connection = attention # 이전 정보를 저장하고 있습니다.
        connection = connection.reshape(-1,6) # batch, params
        
        # feature attention을 적용합니다.
        attention = self.relu(torch.squeeze(attention))
        attention = self.relu(self.dense1(attention))
        attention = self.sigmoid(self.dense2(attention)) # sigmoid를 통해서 (batch, params)의 크기로 확률값이 나타나 있는 attention을 생성합니다.

        x = x.transpose(0, 1)  # (batch, window_size, params) -> (window_size, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1])) # 이전 대회 코드를 보고 leaky relu를 추가했습니다.

        mix_factor = self.sigmoid(self.w) # w의 값을 비율로 만들어 주기 위해서 sigmoid를 적용합니다.

        return mix_factor * connection * attention + out * (1 - mix_factor) 