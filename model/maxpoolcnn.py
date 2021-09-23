import torch
import torch.nn as nn
import torch.nn.functional as F

class GafStackNet(nn.Module):
    def __init__(self):
        super(GafStackNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(6, 12, (5, 5)),
            nn.ReLU(),            
            nn.MaxPool2d((2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(42228, 2)
        )
        
    def forward(self,x):
        return self.model(x), None

class GafAttnNet(nn.Module):
    def __init__(self, mean=False):
        super(GafAttnNet, self).__init__()
        self.base_model = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 12, (5, 5)),
            nn.ReLU(),            
            nn.MaxPool2d((2, 2)),
            #nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
            
        self.mean = mean
        if not mean:
            self.attention = nn.Sequential(
                nn.Linear(6348, 256),
                nn.Tanh(),
                nn.Linear(256, 1))
            
        self.classifier = nn.Sequential(nn.Linear(6348, 2))
        
    def forward(self,x):
        x = x.permute(1, 0, 2, 3)
        x = self.base_model(x)
                    
        if self.mean:
            x = torch.mean(x, axis=0, keepdim=True)
            return self.classifier(x), 0
        else:
            A_unnorm = self.attention(x)
            A = torch.transpose(A_unnorm, 1, 0)
            A = F.softmax(A, dim=1)        

            M = torch.mm(A, x)
            Y_prob = self.classifier(M)      
            return Y_prob, A.flatten()