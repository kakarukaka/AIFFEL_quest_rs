import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_chars, img_height=32, img_width=100):
        super(CRNN, self).__init__()
        self.num_chars = num_chars

        # (3, H, W) -> (64, H, W)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # (64, 16, 50)

        # (64, H/2, W/2) -> (128, H/2, W/2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # pool2 스트라이드 수정: W 보존을 위해 stride=(2, 1) 사용
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))  # (128, 8, 49)

        # (128, H/4, W/4) -> (256, H/4, W/4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # H를 줄이고 W는 유지하기 위해 stride=(2, 1) 사용
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1))  # (256, 4, 48)

        # (256, H/4, W/8) -> (512, H/4, W/8)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        # H를 줄이고 W는 유지하기 위해 stride=(2, 1) 사용
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1))  # (512, 2, 47)

        # (512, 2, 23) -> (512, 1, 22)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2)) # (512, 1, 46)

        # Bi-LSTM
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # 최종 fc
        self.fc = nn.Linear(512, self.num_chars)

    def forward(self, x):
        # (B,3,32,100)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.pool4(x)
        x = F.relu(self.conv7(x))

        # x shape: (B, 512, 1, 46)
        x = x.squeeze(2)   # (B, 512, 46)
        x = x.permute(0, 2, 1)  # (B, 46, 512)

        # LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # 최종 FC
        x = self.fc(x)  # (B, 46, num_chars)

        # (T, B, C)
        x = x.permute(1, 0, 2)

        # CTCLoss 입력을 위해 log_softmax 적용 
        x = F.log_softmax(x, dim=2)

        return x
