#!/usr/bin/env python
# coding: utf-8

# In[30]:


from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import math


# In[31]:


import numpy as np

data1 = np.loadtxt("ENDF_B-VIII.1_LI-6(N,T)HE-4.txt")
data2 = np.loadtxt("ENDF_B-VIII.1_AU-197(N,G)AU-198.yaml")



# In[32]:


print(data1.shape)
print(data2.shape)


# In[54]:


from datasets import Dataset
dataset = Dataset.from_dict({
    "input": data2[:, 0].astype(np.float32).tolist(),
    "labels": data2[:, 1].astype(np.float32).tolist()
})
ds_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = ds_split["train"]
eval_dataset = ds_split["test"]


# In[55]:


ds_split


# In[56]:


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# In[57]:


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


# In[58]:


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


# In[59]:


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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_scalar_to_column
)


# In[60]:


trainer.train()


# In[ ]:




