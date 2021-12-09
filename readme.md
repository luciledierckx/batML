# Detection and Multi-Label Classification of Bats
Python code to detect and identify the bat species in audio recordings by applying multi-label Machine Learning techniques. 
This work was achieved by Lucile Dierckx, Mélanie Beauvois and Siegfried Nijssen (UCLouvain/ICTEAM).

### Challenges
The different challenges that our architecture has to face are:
- It should be able to detect at which moments in audio signals a bat is present.
- It should be able to classify which types of bats are present at those moments.
- It should be able to perform multi-label classification, i.e., it should identify
multiple bats in the scenarios in which more than one bat species emit calls
simultaneously, which happens regularly in nature.
- It should perform its task accurately.
- It should perform its task with low computational resources, as the final goal
is to analyse the recordings on the devices that collect the data.

### Our Proposed Architecture
The model we propose is called *hybrid_cnn_xgboost* in our code. It is an XGBoost model that takes as input features the output of the second to last layer of a simple CNN. The XGBoost model performs both detection and classification.

### Models for Comparison
The models to which our architecture is compared on the challenges are:
- *resnet8*: A single ResNet50 receives spectrogram features and performs both detection and classification.
- *resnet2*: A first ResNet50 receives spectrogram features and performs detection. A second ResNet50 receives spectrogram features and performs classification.
- *hybrid_resnet_xgboost*: An XGBoost model takes as input features the output of the second to last layer of a ResNet50. The XGBoost model performs both detection and classification.
- *cnn8*: A single CNN receives spectrogram features and performs both detection and classification.
- *cnn2*: A first CNN receives spectrogram features and performs detection. A second CNN receives spectrogram features and performs classification.


### Summary of the characteristics of every model
| Name  | Calculation features | Model for detection | Model for classification |
| ------------- |:-------------:|:-------------:|:-------------:|
| ResNet8      | One ResNet50     | Fully connected layer on ResNet |Fully connected layer on ResNet    |
| ResNet2      | Two ResNet50s     | Fully connected layer on ResNet1     | Fully connected layer on ResNet2    |
| ResNet XGBoost      | One ResNet50     | Fully connected layer on ResNet     | Fully connected layer on ResNet     |
| CNN8      | One simple CNN     | Fully connected layer on CNN |Fully connected layer on CNN    |
| CNN2      | Two simple CNNs     | Fully connected layer on CNN1     | Fully connected layer on CNN2    |
| CNN XGBoost      | One simple CNN     | Fully connected layer on CNN     | Fully connected layer on CNN     |

### Train Our Classifiers on Your Own Data

#### Gather the Necessary Data
The data necessary for the detection can be found on Batdetective's website which is available [here](http://visual.cs.ucl.ac.uk/pubs/batDetective).
The data we used for the classification belongs to Natagora and is therefore not made available online. However, you can use your own labelled recordings to train our models. You will need to create a .npz file with the following fields: *train_files*, *train_durations*, *train_pos*, *train_class*, *test_files*, *test_durations*, *test_pos* and *test_class*.

- *train_files* is an array containing the filename of the recordings, without the extension.
- *train_durations* is an array containing the duration of each file present in *train_files*.
- *train_pos* is an array where each line is an array of the call positions in the respective file.
- *train_class* is an array where each line is an array containing the classes of the calls in the respective file.

The same holds for the test fields.

When two calls overlap at the same position, they need to be both inserted in the arrays as separate entries having the same position.

#### Run Training and Evaluate the Model
The *run_training.py* file allows you to train and evaluate any of our available architectures with your own data.
At the beginning of the main section of the file, you can define several parameters such as the name of the model you want to train.
The performance is written in a text file and the trained model is saved in order to use it to classify new recordings.

The tuned hyperparameters we obtained when training our different architectures on Batdetective's and Natagora's data are available in the *data/* folder and will be used by default. If you want to change the hyperparameters you should modify the values in the .csv files of the *data/* folder.

Another important file where certain parameters need to be chosen is the *data_set_params.py* file.
- In this file, it is possible to perform tuning by setting the appropriate model variables to True and choosing the desired tuning time. Note that when the tuning is interrupted it will automatically resume to its last iteration when started again.
- To perform Hard Negative Mining during the training, the number of iterations should be indicated in the *num_hard_negative_mining* variable. By default, no HNM is performed.
- To gain time, the *save_features_to_file* variable can be set to True so that the features are computed and saved once. The next time the features are needed by the models, the features will be loaded if the *load_features_from_file* variable is set to True.

To compile the .pyx files, run the *setup.py* file using the following command:
```
python setup.py build_ext --inplace
```

### Run Our Classifiers on Your Own Data
The *run_classifier.py* file allows you to run an already trained model on new data to find calls and predict the corresponding species. At the beginning of the main section of the file, you can define various parameters such as the name of the model you want to use.

Our already trained models are available in the *data/models/* directory. Your own trained models will also be saved in this folder. To use one of your models, you have to change the value of the *date* and *hnm_iter* variables so that they correspond to the date present in the name of your model and to the number of HNM iterations that were used during training.


### System Requirements
The code was run on a desktop with an i7-9800X CPU @ 3.80GHz, 32GB RAM and an RTX 2080 SUPER GPU on CentOS 8.1. 

The code has been designed to run using the following packages:

`Python 3.6`  
`CUDA 10.2`  
`cuDNN 7.6.5`  
`Cython 0.29.21`  
`hyperopt 0.2.5`  
`joblib 1.0.0`  
`numpy 1.16.4`  
`pandas 1.1.5`  
`scikit-image 0.17.2`  
`scikit-learn 0.24.1`  
`scikit-multilearn 0.2.0`  
`scipy 1.4.1`  
`tensorflow 2.1.0`  
`xgboost 1.4.0`  

### Reference
In case you want to use our work as part of your research please consider citing us.

### Acknowledgements
We would like to express our gratitude to [Natagora](https://www.natagora.be/) and the [Plecotus](https://plecotus.natagora.be/index.php?id=707) team for the large amount of labelled bat call recordings they shared with us. 

We would also like to thank [Bat detective](http://visual.cs.ucl.ac.uk/pubs/batDetective) for the data they have made available and their detection tool, used as starting point in this work. 

We thank Olivier Bonaventure for joining discussions on this project.