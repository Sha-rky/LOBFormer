# DeepLOB

### Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download PyTorch (with version compatible with your GPU)
3. Download dataset
```bash
wget https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
unzip -n data.zip
```

### Training


### Inspect Tensorboard
```bash
tensorboard --logdir=checkpoints
```

### To-Do
- [ ] dropout
- [ ] learning rate scheduler
- [ ] early stopping
- [ ] gradient clipping
- [ ] dataset visualization
- [ ] confusion matrix