import torch
import torch.nn.functional as F

preds = torch.randn(5, 2)
labels = torch.tensor([[0], [1], [0], [1], [0]], dtype=torch.long)
try:
    loss = F.cross_entropy(preds, labels)
    print("Loss with [5,1]:", loss)
except Exception as e:
    print("Error with [5,1]:", e)

labels2 = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
try:
    loss = F.cross_entropy(preds, labels2)
    print("Loss with [5]:", loss)
except Exception as e:
    print("Error with [5]:", e)
