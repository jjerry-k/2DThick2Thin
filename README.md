# MRI_translation

- Interpolation MRI T1 Image
- Figure of Flow

<img src="./figure/Net.png" width=50% height=50%>

## Data Information
- Input
    - Low resolution T1
    - Thickness : 6mm
    - Shape
        - Batch x Height x Width x 3
- Output
    - High resolution T1
    - Thickness : 1mm
    - Shape
        - Batch x Height x Width x 9 
        - Batch x Height x Width x 12
    
## Network Architecture

- Gnerative Adversarial Network
    - Generator
        - Backbone : VGG16, Resnet, Xepction, MobileNet, **DenseNet**
    - Discriminator
        - Backbone : VGG16, Resnet, Xepction, MobileNet, **DenseNet**
        
## Training Method
- Similarity Loss : Mean Square Error


## Result

### 1st Approach

<img src="./figure/data_01.png">

- Evaluation 결과 6장 단위로 영상이 툭툭 튀는 현상 발생.

### 2nd Approach

<img src="./figure/data_02.png">

<img src="./figure/baseline_loss.png">



