This repository contains all the codes and dataset used in the *MSR 2022* paper ***'Does Configuration Encoding Matter in Learning Software Performance? An Empirical Study on Encoding Schemes'*** by Jingzhi Gong and Tao Chen*. 

# Abstract
Learning and predicting the performance of a configurable software system helps to provide better quality assurance. One important engineering decision therein is **how to encode the configuration** into the model built. Despite the presence of different encoding schemes, there is still little understanding of which is better and under what circumstances, as the community often relies on some general beliefs that inform the decision in an ad-hoc manner. To bridge this gap, in this paper, we empirically compared the widely used encoding schemes for software performance learning, namely label, scaled label, and one-hot encoding. The study covers five systems, seven models, and three encoding schemes, leading to 105 cases of investigation. 

Our **key findings** reveal that: (1) conducting trial-and-error to find the best encoding scheme in a case by case manner can be rather expensive, requiring up to 400$+$ hours on some models and systems; (2) the one-hot encoding often leads to the most accurate results while the scaled label encoding is generally weak on accuracy over different models; (3) conversely, the scaled label encoding tends to result in the fastest training time across the models/systems while the one-hot encoding is the slowest; (4) for all models studied, label and scaled label encoding often lead to relatively less biased outcomes between accuracy and training time, but the paired model varies according to the system.

We discuss the **actionable suggestions** derived from our findings, hoping to provide a better understanding of this topic for the community. For example, we recommand using neural network paired with one-hot encoding for the best accuracy, while using linear regression paired with scaled label encoding for the fastest training. 

# Documents
Specifically, the documents include:

- **Encoding.py**: 

The **main program** used for loading the dataset, learning the machine learning models, testing the models and saving the results.

 - **mlp_sparse_model.py**, **mlp_plain_model.py**, **utils**: 

We take these codes from 'https://github.com/DeepPerf/DeepPerf', which is a state-of-the-art deep learning model for performance prediction. We use their model as the neural network performance prediction model in our paper.

 - **Data**: 

The folder contains the datasets for the five subject systems used in this paper.


 - **Results**:

The folder contains the raw RMSE and training time recorded by 'Encoding.py'. 
Each txt file corresponds to a subject system, and contains the RMSE and training times for 50 runs.

<!-- --- -->

# Datasets
Five subject systems and their datasets are applied in this study as given below:
 - MongoDB
 - Lrzip
 - Trimesh
 - ExaStencils
 - x264
 
The datasets are originally from 'https://github.com/anonymous12138/multiobj' and 'https://github.com/FlashRepo/Flash-MultiConfig',
which are the repositories of two excellent studies in the domain of performance modeling.
Details of the datasets are given in our paper.

<!-- --- -->

# Prerequisites
 - Python 3.6 - 3.7
 - Tensorflow 1.x

<!-- --- -->

# Installation
 1. Download all the files into the same folder.
 2. Install the specified version of python and tensorflow.
 3. Run Encoding.py and install all missing packages according to the runtime messages. 

<!-- --- -->

# Usage

### To run the default experiment
 - **Command line**: Move to the folder with the codes, and run 
```
python Encoding.py
```

 - **Python IDE (e.g. Pycharm)**: Open the Encoding.py file on the IDE, and click 'Run'.



### To switch between subject systems

 - Comment and Uncomment the codes following the comments in Encoding.py.




### To change experiment settings:
 - Alter the codes following the comments in Encoding.py.
 
