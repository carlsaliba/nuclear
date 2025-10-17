#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import math


# In[2]:


import numpy as np

data = np.loadtxt("ENDF_B-VIII.1_AU-197(N,G)AU-198.yaml")  # (x, y)
print(data.shape)


# In[3]:


x_raw = data[:, 0].astype(np.float32)
y_raw = data[:, 1].astype(np.float32)
x_log = np.log10(x_raw)
y_log = np.log10(y_raw)


# In[4]:


from datasets import Dataset

full_ds = Dataset.from_dict({"x": x_log.tolist(), "y": y_log.tolist()})
ds = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]


# In[5]:


# ---------- 4) Standardize using TRAIN stats ----------
x_mean, x_std = float(np.mean(train_ds["x"])), float(np.std(train_ds["x"]))
y_mean, y_std = float(np.mean(train_ds["y"])), float(np.std(train_ds["y"]))


def standardize(example):
    example["input"] = (example["x"] - x_mean) / x_std
    example["labels"] = (example["y"] - y_mean) / y_std
    del example["x"]
    del example["y"]
    return example


train_ds = train_ds.map(standardize)
eval_ds  = eval_ds.map(standardize)


# In[6]:


# import numpy as np
# import matplotlib.pyplot as plt

# x = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)  # Skewed distribution
# x_std = (x - np.mean(x)) / np.std(x)

# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.hist(x, bins=50, alpha=0.7)
# plt.title("Original Distribution")

# plt.subplot(1,2,2)
# plt.hist(x_std, bins=50, alpha=0.7)
# plt.title("Standardized Distribution")

# plt.show()


# In[7]:


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# In[8]:


class NeuralNetwork(nn.Module):
    def __init__(self,input_size=1,output_size=1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, output_size,bias=False),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input: torch.Tensor, labels: torch.Tensor = None):
        logits = self.linear_relu_stack(input)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
        
model = NeuralNetwork().to(device)
print(model)


# In[9]:


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.array(preds).reshape(-1)
    labels = eval_pred.label_ids.reshape(-1)

    mse  = float(np.mean((preds - labels) ** 2))
    rmse = float(math.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - labels)))
    ss_res = float(np.sum((labels - preds) ** 2))
    
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def collate_scalar_to_column(batch):
    inputs = torch.tensor([ex["input"] for ex in batch], dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.float32).unsqueeze(-1)
    return {"input": inputs, "labels": labels}


# In[10]:


training_args = TrainingArguments(
            output_dir='./results',
            learning_rate=1e-4,
            per_device_train_batch_size=4,  
            per_device_eval_batch_size=4,
            max_steps=10000,  # Replace with your desired number of steps
            weight_decay=0.02,
            eval_strategy='steps', 
            eval_steps=1000,  #the save step should be a multiple of eval step, savestep=500 by default
            lr_scheduler_type="cosine",
            warmup_ratio=0.1    

        )
# ---------------- 7) Trainer ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_scalar_to_column
)


# In[11]:


trainer.train()


# In[28]:


torch.tensor(train_ds["input"]).reshape(-1,1).shape


# In[29]:


out=trainer.model(torch.tensor(train_ds["input"]).reshape(-1,1))


# In[36]:


out_1=out["logits"].reshape(-1).detach().numpy()


# In[38]:


np.array(train_ds["input"])


# In[40]:


import matplotlib.pyplot as plt
import numpy as np

# Assuminx
x=np.array(train_ds["input"])
y = out_1# All rows, column 1

# Log-log plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', alpha=0.7, markersize=1)
plt.xlabel('X (log scale)')
plt.ylabel('Y (log scale)')
plt.title('Log-Log Plot')
plt.grid(True, alpha=0.3)
plt.show()


# In[ ]:




