# MRI_translation

- Interpolation MRI T1 Image
- Figure of Flow

<img src="./figure/Net.png" width=50% height=50%>


## Network Architecture

- Gnerative Adversarial Network
    - Generator
        - Backbone : VGG16, Resnet, Xepction, MobileNet
    - Discriminator
        - Backbone : VGG16, Resnet, Xepction, MobileNet
        
## Training Method
- Contrast Loss : Mean Square Error
- Context Loss  : Perceptual Loss or Mutual Information (Not yet...)

## To do list
- [ ] Reading Data
- [ ] Build Generator Architecture
- [ ] Build Discriminator Architecture
- [ ] Define Loss Function
- [ ] Making Data Generator 
- [ ] Training Netrowk

