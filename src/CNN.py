import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNNActionValue(nn.Module):
#     def __init__(self, state_dim, action_dim, activation=F.relu):
#         super(CNNActionValue, self).__init__()
#         self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
#         self.in_features = 32 * 9 * 9
#         self.fc1 = nn.Linear(self.in_features, 256)
#         self.fc2 = nn.Linear(256, action_dim)
#         self.activation = activation

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view((-1, self.in_features))
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# class CNNActionValue(nn.Module):
#     def __init__(self, state_dim, action_dim, activation=F.relu):
#         super(CNNActionValue, self).__init__()
        
        

#         self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4) # 84-> 42 21 10
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
#         # 이미지 크기가 84x84에서 시작한다고 가정
#         # 84 -> 42 -> 21 -> 11 (rounded up from 10.5)
#         # 마지막 컨볼루션 후 크기를 10x10으로 조정하기 위한 추가 층
#         # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         # self.bn4 = nn.BatchNorm2d(64)
        
#         self.fc1 = nn.Linear(2592, 256)
#         self.fc3 = nn.Linear(256, action_dim)
        
#         self.activation = activation
        

        
#     def forward(self, x):
#         x = self.activation(self.conv1(x))
#         x = self.activation(self.conv2(x))
        
#         x = x.view(x.size(0), -1)  # Flatten, 결과는 64 * 10 * 10 = 6400
        
#         x = self.activation(self.fc1(x))
#         x = self.fc3(x)
        
#         return x
    
class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1) # 84-> 42 21 10
        self.conv1_1 = nn.Conv2d(32,32,3,1,1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
        self.activation = activation
        

        
    def forward(self, x):
        x = self.activation(self.conv1_1(self.conv1(x)))
        x = self.activation(self.conv2_1(self.conv2(x)))
        x = self.activation(self.conv3_1(self.conv3(x)))
        x = self.activation(self.conv3_2(x))
        
        x = x.view(x.size(0), -1)  # Flatten, 결과는 64 * 10 * 10 = 6400
        
        x = self.activation(self.fc1(x))
        x = self.fc3(x)
        
        return x
    