# DeepLOB

### Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download PyTorch (with version compatible with your GPU)
3. Manually download dataset from [Here](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master/data)
Notice, 4 txt files should be in the root directory.

### Training
```bash
.\train.bat
```

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