import torch
import torch.nn as nn
import torch.nn.functional as F

# MULTI LAYER PERCEPTRON: all liner layers, no convolutional layers
class MLP_3layers(nn.Module):
    def __init__(self):
        super(MLP_3layers, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)   # Strato di input da 28x28 a 128 neuroni
        self.fc2 = nn.Linear(128, 64)        # Strato nascosto da 128 a 64 neuroni
        self.fc3 = nn.Linear(64, 10)         # Strato di output da 64 a 10 neuroni (corrispondenti alle 10 classi di output)

    def forward(self, x): # forward pass della rete, come i dati input (x) passano attraverso i layer per produrre l'output
        x = x.view(-1, 28 * 28)  # Cambia la forma dell'input per adattarsi all'MLP
        x = F.gelu(self.fc1(x))  # Passaggio attraverso il primo strato con GeLU
        x = F.gelu(self.fc2(x))  # Passaggio attraverso il secondo strato con GeLU
        x = self.fc3(x)          # Passaggio attraverso lo strato di output

        return x

class MLP_2layers(nn.Module):
    def __init__(self):
      super(MLP_2layers, self).__init__()
      self.fc1 = nn.Linear(28 * 28, 16)
      self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
      x = x.view(-1, 28 * 28)
      x = F.gelu(self.fc1(x))
      x = self.fc2(x)

      return x
    
class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        """
        Definisce un Multi-Layer Perceptron che si adatta a un numero di layer variabile.

        Args:
            input_size (int): La dimensione dell'input (e.g., 28*28 per MNIST).
            hidden_layers_sizes (list or tuple): Una lista con le dimensioni di ogni strato nascosto.
                                                Es: [512, 256, 128, 64] per 4 strati nascosti.
            output_size (int): Il numero di neuroni nell'output layer.
        """
        super(DynamicMLP, self).__init__()
        
        # Aggiungi la dimensione dell'input all'inizio della lista delle dimensioni
        layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
        
        # Crea un modulo lista per contenere tutti gli strati lineari
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Aggiungi una funzione di attivazione per tutti i layer tranne l'ultimo
            if i < len(layer_sizes) - 2:
                layers.append(nn.GELU())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.network(x)
        
        return x
    
class DynamicMLP_improved(nn.Module):
    def __init__(self, input_size, hidden_layers_sizes, output_size, dropout=0.1):
        super(DynamicMLP_improved, self).__init__()
        
        # Lista di dimensioni completa (input + hidden + output)
        layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # BatchNorm e Dropout solo sui layer nascosti
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.network(x)
        return x
    
# RESIDUAL MLP with skip connections
class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ResidualMLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = F.gelu(self.fc1(x))
        out = self.fc2(out)
        out = out + residual  # skip connection
        return F.gelu(out)

class ResidualMLP(nn.Module):
    def __init__(self):
        super(ResidualMLP, self).__init__()
        self.input_proj = nn.Linear(28 * 28, 128)  # Project input to 128
        # Input and output features for block1 are now 128
        self.block1 = ResidualMLPBlock(128, 128, 128)
        self.block2 = ResidualMLPBlock(128, 64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.gelu(self.input_proj(x))  # Apply initial projection
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc3(x)
        #output = F.log_softmax(x, dim=1)

        return x
    

class DynamicResidualMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.proj(x)
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + residual
        out = F.gelu(out)
        return out


class DynamicResidualMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_sizes=[128, 64], num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks = nn.Sequential(
            *[DynamicResidualMLPBlock(
                in_features=hidden_sizes[i],
                hidden_features=hidden_sizes[i],
                out_features=hidden_sizes[i+1],
                dropout=dropout
            ) for i in range(len(hidden_sizes)-1)]
        )
        self.fc_out = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.gelu(self.input_proj(x))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x
    
# CNN
class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class myCNN_improved(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        x = torch.flatten(x, 1) 
        
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
class myMidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # downsample
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # downsample
        
        self.flatten_dim = 256 * 8 * 8 
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> 32x32x32
        x = F.relu(self.conv2(x))   # -> 32x32x64
        x = F.relu(self.conv3(x))   # -> 32x32x128
        x = F.relu(self.conv4(x))   # -> 16x16x128
        x = F.relu(self.conv5(x))   # -> 8x8x256
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class myDeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # downsample
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # downsample
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # downsample

        self.flatten_dim = 512 * 4 * 4  # after 3 downsampling layers

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  

        return x
