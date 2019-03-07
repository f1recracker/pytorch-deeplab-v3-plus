# DeepLab V3+ PyTorch
DeepLab V3+ is a state-of-the-art model for semantic segmentation.

This repository contains a PyTorch implementation of DeepLab V3+ trained for full driving scene segmentation tasks.

## Pending Tasks

- [x] Base DeepLab model
  - [x] DeepLab decoder module
  - [x] Xception feature extractor backbone
  - [x] Dataloaders, train script, metrics
  - [x] Data augmentation pipeline
- [ ] Upload pretrained weights and results
- [ ] Planned extensions
  - [x] Mixed precision training
  - [ ] Faster, MobileNet-v2 backbone
  - [ ] Pretraining all backbones on ImageNet
  - [ ] Dataloader for City-scapes
  - [ ] Dataloader for Baidu apollo dataset
  - [ ] Setup inference mode
