import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def tester(model, dl_test, device):

    print("\n")
    print("STARTING TESTING of ", model.__class__.__name__)
    print("\n")
    
    model.eval() # configura rete in validation mode, richiede uso di with torch.no_grad():
    test_loss = 0
    correct = 0

    with torch.no_grad(): # il codice sottostante non avrà bisogno di calcolo del gradiente
        for data, target in tqdm(dl_test, desc="Testing", ncols=70):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # controlla quante volte le predizioni rete sono = alle ground truth

    test_loss /= len(dl_test.dataset)
    test_acc = correct / len(dl_test.dataset)

    print('\nTest set: Average loss = {:.4f}, Accuracy = {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dl_test.dataset), 100. * test_acc))

    print("\n")
    print("FINISHED TESTING of ", model.__class__.__name__)
    print("\n")

    return test_loss, test_acc