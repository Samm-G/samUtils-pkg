class DL:
    def __init__(self):
        """
        Theory:
            ● Activations Functions:
                ● Helps in defining the output of a node when a input is given.
                ● Ex:
                    ● Sigmoid (0,1), Tanh(-1, 1), ReLU(0, max), Binary Step Function
            ● Layer Details of Feed Forward Network:
                ● Output layer
                    ● Represents the output of the neural network
                ● Hidden layer(s):
                    ● Represents the intermediary nodes.
                    ● It takes in a set of weighted input and produces output through an  activation function
                ● Input layer:
                    ● Represents dimensions of the input vector (one node for each  dimension)

            ● Steps in Training a Neural Network:
                ● Decide the structure of network
                ● Create a neural network
                ● Choose different hyper-parameters
                ● Calculate loss
                ● Reduce loss
                ● Repeat last three steps

            ● Note on Error and Loss Function:
                ● In general, error/loss for a neural network is difference between actual value and predicted  value.
                ● The goal is to minimize the error/loss.
                ● Loss Function is a function that is used to calculate the error.
                ● You can choose loss function based on the problem you have at hand.
                ● Loss functions are different for classification and regression

            ● Gradient Descent:
                ● Gradient descent is a method that defines a cost function of parameters and uses a  systematic approach to optimize the values of parameters to get minimum cost  function.

                ● Gradient is calculated by optimization function
                ● Gradient is the change in loss with change in  weights.
                ● The weights are modified according to the  calculated gradient.
                ● Same process keep on repeating until the  minima is reached

            ● Type of Gradient Descent:	
                ● Gradient descent has 3 variations, these differ in using data to calculate the gradient of the objective function
                ● Batch gradient descent
                    ● Updates the parameter by calculating gradients of whole dataset
                ● Stochastic gradient descent
                    ● Updates the parameters by calculating gradients for each training example
                ● Mini -batch gradient descent
                    ● Updates the parameters by calculating gradients for every mini batch of “n” training example
                    ● Combination of batch and stochastic gradient descent
                    
            ● Back Propagation:
                ● Backpropagation is used while training the feedforward networks
                ● It helps in efficiently calculating the gradient of the loss function w.r.t weights
                ● It helps in minimizing loss by updating the weights
                
            ● LR and Momentum:
                ● The learning rate is a hyperparameter which determines to what extent newly  acquired weights overrides old weights. In general it lies between 0 and 1.
                ● You can try different learning rates for a neural networks to improve results.
                ● Momentum is used to decide the weight on nodes from previous iterations. It  helps in improving training speed and also in avoiding local minimas.

            ● Tensors:
                ● A Tensor is a multi-dimensional array.
                ● Similar to NumPy ndarray objects, tf.Tensor objects have a data type and a shape. Additionally, tf.Tensors can reside in accelerator memory (like a GPU).
                ● TensorFlow offers a rich library of operations (tf.add, tf.matmul, tf.linalg.inv etc.) that consume and produce tf.Tensors. 
                ● These operations automatically convert native Python types, for example:

            ● Tensor vs Numpy:
                ● The most obvious differences between NumPy arrays and tf.Tensors are:
                    ● Tensors can be backed by accelerator memory (like GPU, TPU).
                    ● Tensors are immutable.
                ● NumPy Compatibility
                    ● Converting between a TensorFlow tf.Tensors and a NumPy ndarray is easy:
                        ● TensorFlow operations automatically convert NumPy ndarrays to Tensors.
                        ● NumPy operations automatically convert Tensors to NumPy ndarrays.

            ● Keras:
                ● Keras Advantages:
                    ● User-friendly: Simple and user friendly interface. Actionable feedbacks are also provided.
                    ● Modular and composable: Modules are there for every step, you can combine them to build solutions.
                    ● Easy to extend:
                        ● Gives freedom to add custom blocks to build on new ideas.
                        ● Cusrom layers, metrics, loss functions etc. can be defined easily.

            ● Neural Nertwork Architecture:
                ● Feed Forward:
                    Process of calculating expected output
                    Combines weights and activation functions with the inputs
                    Iteratively performed over training set, and classifies test input
                ● Back Propagation:
                    ● At the end of each forward pass, we have a loss (difference between expected outcome and actual)
                    ● The core of the back prop is a partial derivative of the Loss with respect to a weight - which tells us how quickly the Loss changes for any change in the weight
                    ● Back Prop follows the chain rule of derivatives, i.e. the Loss can be computed for each and every weight in the network
                    ● In practice, backward propagation is often abstracted away, because functions take care of it - but it’s important to know how it works
                ● Fully Connected Layer:
                    ● Neurons have connections to all activations of the previous layer
                    ● Number of connections add up very quickly due to all the combinations
                    ● Forward pass of a fully connected layer -> one matrix multiplication followed by a bias offset and  activation function (you’ll soon see what we mean)
                ● Activation Functions:
                    ● An activation function takes a single input value and applies a function to it - typically a ‘non-linearity’
                    ● Why non-linearity: converts a linear function (sum of weights) into a polynomial of higher degree, this is what allows non-linear decision boundaries
                    ● Activation functions decide whether a neuron is ‘switched on’, i.e. it acts as a gate, by applying a non-linear function on the input
                    ● Many different types of activation functions exist:
                        ● Sigmoid:
                            ● Ranges from 0-1
                            ● S-shaped curve
                            ● Historically popular
                            ● Interpretation as a saturating “firing rate” of a neuron
                            ● Limitations:
                                ● Its output is not zero centered.
                                ● Vanishing Gradient Problem
                                ● Slow convergence
                        ● tanh:
                            ● Ranges between -1 to +1
                            ● Output is zero centered
                            ● Generally preferred over Sigmoid function
                            ● Limitations:
                                ● Though optimisation is easier, it still suffers from the  Vanishing Gradient Problem

                        ● ReLU:
                            ● Simple
                            ● Much better convergence than tanh and sigmoid  function.
                            ● Very efficient in computation
                            ● Limitations:
                                ● Output is not necessarily zero centered.
                                ● Some gradients can be fragile during training and can ‘die’

                        ● Leaky ReLU:
                            ● Introduced to overcome the problem of dying neurons.
                            ● Introduces a small slope to keep the neurons alive
                            ● Does not saturate in the positive region

                    ● ReLU is preferred
                    ● Be careful with learning rates + Monitor the fraction of “dead” units in network
                    ● Possibly try Leaky ReLU or Maxout
                    ● Never use Sigmoid
                    ● Try tanh
                        ● Typically perform worse than ReLU though
            
                ● Softmax Function:
                    ● Softmax function is a multinomial logistic classifier, i.e. it can handle multiple classes
                    ● Softmax typically the last layer of a neural network based classifier
                    ● Softmax function is itself an activation function, so doesn’t need to be combined with an activation  function

                ● Cross Entropy Loss:
                    ● Cross-entropy loss (often called Log loss) quantifies our unhappiness for the predicted output based  on its deviation from the desired output
                    ● Perfect prediction would have a loss of 0 (we will see how)
                    ● With gradient descent, we try to reduce this (cross-entropy) loss for a classification problem

            ● Tuning Neural Networks:
                ● Weight Initialization:
                    ● Initialize all weights with 0:
                        ● This makes your model equivalent to a linear model.
                        ● When you set all weight to 0, the derivative with respect to loss function is the same for  every w in every layer.
                        ● This makes the hidden units symmetric and continues for all the n iterations you run.
                        ● Thus setting weights to zero makes your network no better than a linear model.
                    ● Initialize with random numbers
                        ● Works okay for small networks (similar to our two layer  MNIST classifier).
                        ● may lead to distributions of the activations that are not homogeneous  across the layers of network.

                ● Vanishing Geadient Problem:
                    ● The weight update is minor and results in slower convergence. 
                    ● This makes the optimization of the loss function slow. 
                    ● In the worst case, this may completely stop the  neural network from training further.
                
                ● Exploding Gradient Problem:
                    ● This is the exact opposite of vanishing gradients. 
                    ● Consider you have non-negative and large weights and small activations A (as can be the case for  sigmoid(z)).
                    ● This may result in oscillating around the minima or even overshooting  the optimum again and again and the model will never learn!


            ● Regularization Techniques:
                ● Batch Normalization:
                    ● Batch normalisation is a technique for improving the performance and  stability of neural networks
                    ● The idea is to normalise the inputs of each layer in such a way that they have  a mean output activation of zero and standard deviation of one. 
                    ● This is  analogous to how the inputs to networks are standardised.
                    ● Benefits:
                        ● Network train faster
                        ● Provides some regularisation
                    ● Two types in which Batch Normalization can be applied:
                        ● Before activation function (non-linearity)
                        ● After non-linearity
                    ● Mini Batch Normalization:
                        ● Improves gradient flow through the  network
                        ● Allows higher learning rates
                        ● Reduces the strong dependence on  initialization
                        ● Acts as a form of regularization in a funny way, and slightly reduces the  need for dropout, maybe.
                ● Dropout:
                    ● Dropout is an approach to regularization in  neural networks which helps reducing  interdependent learning amongst the neurons.
                    ● To prevent over-fitting
                    ● A fully connected layer occupies most of the parameters, and hence,  neurons develop co-dependency amongst each other during training
                    ● This curbs the individual power of each neuron leading to over-fitting of training  data
                    
            ● Computer Vision:
            
                ● Definitions:
                    ● PIXEL:
                        ● PIXELS are ATOMIC ELEMENTS of a digital image.
                        ● It is the smallest element of an image  represented on the screen.
                        ● A pixel can have value ranging from 0 to 255.
                    ● Images:
                        ● Image Formats:
                            ● GIF, JPEG, PNG,	RAW, TIF, PGM, PBM etc.
                            ● Medical Images: DICOM, Analyze, NIFTI etc.
                        ● Image transformations:
                            ● Filtering: Sharpen, Blurm Scaling, etc.
                            ● Affine Transformation (Basic Transformations):
                                ● Scale, rotate, translate, mirror 
                
                ● Convolution (Feature Extraction from Images):
                    ● Convolution is the process of adding each  element of the image to its local neighbors,  weighted by the kernel.
                    ● This is related to a form of mathematical  Convolution operation.
                    
                ● Features from Kernels:
                    ● Kernel is also called convolution matrix  or mask.
                    ● Convolution with different kernels can be  used for different image  transformations/filtering.
                    
            ● Convolutional Neural Networks:
                
                ● Components:
                    ● Convolution and Filters
                    ● Feature Map
                    ● Max-pool layers
                    ● Other pooling types
                    ● Sequential model compilation
                    ● Cass study ; Image classification using CNN

            ● Transfer Learning:
                ● Conventional machine learning and deep learning algorithms, so far, have been traditionally designed to work in isolation. 
                ● These algorithms are trained to solve specific tasks.
                ● The models have to be rebuilt from scratch once the feature-space distribution changes. 
                ● Transfer learning is the idea of overcoming the isolated learning paradigm and utilizing knowledge acquired for one task to solve related ones. 
                
                ● Traditional ML vs Transfer Learning:
                    ● Traditional ML:
                        ● Isolated, Single Task Learning
                        ● Knowledge is not retained
                        ● Learning is done without considering past learned actions
                    ● Transfer Learning:
                        ● Learning relies on previously learned tasks
                        ● Leaning process can be faster and accurate.
                        ● Needs less training data
            
            ● Object Detection:
                    
                ● Given an image we want to  detect all the object in the  image that belong to a specific classes and give  their location.
                ● Involves Image Classification + Localization
                    
                ● Challenges in Object Detection:
                    ● Two tasks - Classification and Localization
                    ● Results/prediction take lot of time but we need fast predictions for real-  time task
                    ● Variable number of boxes as output
                    ● Different scales and aspect ratios
                    ● Limited data and labelled data
                    ● Imbalanced data-classes
                    
                ● Performance Metrics:
                    ● IoU: 
                        ● IoU is a function used to evaluate the object detection algorithm.
                        ● It computes size of intersection and divide it by the union. More  generally, IoU is a measure of the overlap between two bounding  boxes.
                        ● Formula:
                        (Area of Overlap) / (Area of Union)
                    ● Precision: what percentage of your positive  predictions are correct
                    ● Recall: what percentage of ground truth  objects were found

                    ● Mean Average Precision:
                        ● Sort predictions according to  confidence (usually classifier’s output after  softmax)
                        ● Calculate IoU of every predicted box  with every ground truth box
                        ● Match predictions to ground truth  using IoU, correct predictions are those with  IoU > threshold (.5)
                        ● Calculate precision and recall at  every row
                        ● Take the mean of maximum  precision at 11 recall values (0.0, 0.1, …  1.0) to get AP
                        ● Average across all classes to get  the mAP

            ● Object Detection Approaches:
                ● Brute Force:
                    ● Run a classifier for every  possible box
                    ● This is a 15 x 10 grid, there  are 150 small boxes. How  many total boxes?
                    ● Computationally expensive
                ● Sliding Window:
                    ● Run classifier in a sliding  window fashion
                    ● Apply a CNN to many different crops of the image
                    ● CNN classifies each crop as object or background
                    ● Problem: Need to apply CNN to huge number of locations (and scales),  very computationally expensive
                
                ● Ideas how to reduce number of boxes?
                    ● Find ‘blobby’ image regions which are likely to contain objects
                    ● Run classifier for region proposals or boxes likely to contain objects
                ● Later: Class-agnostic object detector - “Region Proposals”
                
                ● State of the Art (SOTA) Approach:
                    ● State of the Art methods are generally categorised in two categories
                        ● One stage methods
                        ● Two stage methods
                    ● One-stage methods give priority to inference speed, these include SSD,  YOLO, RetinaNet. 
                    ● Whereas Faster-RCNN, Mask RCNN and Cascade RCNN are example of two-stage methods where priority is given to detection  accuracy.
                    ● Popular benchmark dataset is Microsoft COCO
                    ● Popular metric for evaluation is mAP (Mean Average Precision)
                    
            ● Object Detection Algorithms:
                ● Region Proposal Based Algorithms:
                    ● Region Proposal meaning:
                        ● Find ‘blobby’ image regions which are likely to contain objects
                        ● Run classifier for region proposals or boxes likely to contain objects
                    ● Selective Search for Objective Recognition:
                        ● Selective Search uses the best of both worlds: Exhaustive search and  segmentation.
                        ● Segmentation improve the sampling process of different boxes - reduces considerably the search space.
                        ● To improve the algorithm’s robustness a variety of strategies are used  during the bottom-up boxes’ merging.
                        ● Selective Search Advantages:
                            ● Capture All Scales - Objects can occur at any scale within the image.  Furthermore, some objects may not have clear boundaries. This is  achieved by using an hierarchical algorithm.
                            ● Diversification - Regions may form an object because of only colour, only texture, or lighting conditions etc.
                                ● Instead of a single strategy  which works well in most cases, we prefer to have a diverse set of  strategies to deal with all cases.
                            ● Fast to compute
                    ● R-CNN:
                        ● An algorithm which can also be used for object detection.
                        ● Stands for regions with Conv Nets.
                        ● It tries to pick a few windows and run a Conv net (your confident classifier) on top of them.
                        
                        ● It first generates 2K region proposals (bounding box candidates), then detect object within the each region proposal.
                        ● It uses to pick windows is called a segmentation algorithm
                        ● Disadvantage:
                            ● Separate convnet for each box  (slow)
                    ● Fast R-CNN:
                        ● Uses RoI Pooling:
                            ● RoI pooling on a single 8×8  feature map, one region of  interest and an output size  of 2x2. 
                            ● Disadvantage:
                                ● Region Proposals take time to find.
                            
                    ● Faster R-CNN:
                        ● Make CNNs do proposals!
                        ● Insert Region Proposal Network  (RPN) to predict proposals from  features.
                        ● Jointly train with 4 losses
                            ● RPN classify object/not object
                            ● RPN regress box coordinates
                            ● Final classification score (object  classes)
                            ● Final box coordinates
                            
                ● Algorithms without Region Proposal:
                    ● YOLO (You Only Look Once):
                        ● Working:
                            ● Actually Divides the image into a grid of say, 13*13 cells (S=13)
                            ● Each of these cells is responsible for predicting 5 bounding boxes (B=5) (A bounding box describes the  rectangle that encloses an object)
                            ● YOLO for each bounding box
                                ● outputs a confidence score that tells us how good is the shape of the box
                                ● the cell also predicts a class
                            ● The confidence score of bounding box and class prediction are combined into final score -> probability  that this bounding box contains a specific object
                        ● Non-Maximal Supression:
                            ● Take similarly overlapping regions of each box and make a single box out of it.
                            ● Advantages:
                                ● Reduce the computation time of calculating a lot of bounding boxes.
                            
                    ● SSD (Single Shot Detectors):
                        ● In SSD, like YOLO, only one single pass is needed to detect multiple  objects within the image.
                        ● Two passes are needed in Regional proposal network (RPN) based approaches such as R-CNN, Fast R-CNN series. 
                        ● One pass for generating  region proposals and another pass to loop over the proposals for  detecting the object of each proposal.
                        ● SSD is much faster compared to two-shot RPN-based approaches.
                        ● Working:
                            ● A feature layer of size m×n (number of locations) with p channels
                            ● For each location, we got k bounding boxes
                            ● For each of the bounding box, we will compute c class scores and 4  offsets relative to the original default bounding box shape.
                            ● Thus, we got (c+4) x k x m x n outputs.

                    ● SSD vs YOLO:
                        ● SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences.
                        ● SSD with a 300 × 300 input size significantly outperforms its 448 × 448.
                        ● YOLO counterpart in accuracy on VOC2007 test while also improving the run-  time speed, albeit YOLO customized network is faster than VGG16.

            ● Semantic Segmentation:
                ● Put label on each pixel in the image
                ● No need to specify the difference between the instances

                ● Approaches:
                    ● Sliding window (use sliding window to segment).
                    ● Fully Convolutional:
                        Run through a fully convolutional network to get all pixels at once
                        Dilated/Atrous Convolutions:
                            Atrous (or dilated) convolutions are regular convolutions with a factor that allows us to expand the filter’s field of view.
                            Consider a 3x3 convolution filter for instance. When the dilation rate is equal to 1, it behaves like a standard convolution.  
                            But, if we set the dilation factor to 2, it has the effect of enlarging the convolution kernel.

                    ● UpSampling:
                        ● Transposed convolution / Deconvolution
                        ● Fractionally strided convolution
                        ● Max-unpooling: Preserves spatial information
                        ● Learnable Upsampling (Like NVidia DLSS)
                        
                ● UNet Architecture:
                    ● An Encoder - Downsampling part. It is used to  get context in the image. It is just a stack of  convolutional and max pooling layers.
                    ● A Decoder - Symmetric Upsampling part. It is  used for precise localization. Transposed  convolution is used for upsampling.
                    ● It is a fully convolutional network (FCN). it has  Convolutional layers and it does not have any  dense layer so it can work for image of any size.
                    
                    ● Dice Coefficient:
                        ● Dice coefficient is defined as follows:
                            ● (2*|XnY|) / (|X|+|Y|)
                            ● where, X is the predicted set of pixels and Y is the ground truth.
                            ● A higher dice coefficient is better. 
                            ● A dice coefficient of 1 can be achieved when  there is perfect overlap between X and Y. 
                            ● Since the denominator is constant, the  only way to maximize this metric is to increase overlap between X and Y.

            ● Instance Segmentation:
                
                ● Why still use a two-stage object detector?
                    ● Better recall of RPN as compared to SSD/YOLO
                        ● Trained with all object instances
                        ● Generic first stage, usable for multi task
                    ● Finer control over training classifier
                        ● Custom minibatch (sampling 3:1 negative samples)
                    ● Instance-level multi task (Mask-RCNN)

                ● Mask RCNN:
                    ● Preserves pixel-to-pixel alignment.
                    ● Quantization – loss of pixel-to-pixel alignment.
                    
                    ● RoI Align (Improvement on RoI Pooling):
                        ● Input: Feature map (5x5 here) and  region proposal (normalized float  coordinates)
                        ● Output: 2x2 ‘pooled’ bins
                        ● Sample 4 points in every bin uniformly
                        ● Compute value at each bin using  bilinear interpolation
                        ● Max or average the 4 bins

                    ● Class Imbalance in Training a Classifier:
                        ● While training detectors, maximum samples are background (negatives)
                        ● Faster R -C NN: Ratio of 3 negatives to 1 positive is maintained while training  classifier head Custom minibatch
                        ● Not easy in single stage detectors
                        ● Losses:
                            ● Cross entropy loss: (-log(Pt))
                            ● Balanced cross entropy loss: (-Alpha_t * log(Pt))
                            ● Focal loss: (-(1-Pt)^gamma * log(Pt))

            ● One Shot Learnings:
                ● For some applications in real world, we neither have large enough data for each  class and the total number classes is huge and it keeps on changing.
                ● The cost and effort of data collection and periodical re -training is too high.
                ● We use One Shot Learning!
                ● One shot learning:
                    ● Requires only one training example for each class, given as input to a Siamese Network.
                    ● This outputs a Similarity Score.
                

            ● Metric Learning:
                ● Metric learning is the task of learning a distance function over objects
                ● Metric is like a distance. It follows the following properties:
                    ● Inverse of similarity
                    ● It is symmetric
                    ● It follows triangle inequality
                
                ● If distance is considered, the objective is to	minimize the distance measure:
                    ● Euclidean and Manhattan distances.
                ● If considering similarity, the objective is to maximize the similarity measure.
                    ● Dot product, RBF (Radial Basis Function).
                    
            ● Siamese Networks:
                ● Siamese network is used to find we want to compare how similar two things are.  
                ● Some examples of such cases are Verification of signature, face recognition.
                ● Any Siamese network has two identical subnetworks, which share common parameters and weights.
                ● Siamese neural networks has a unique structure to naturally rank similarity between inputs.

                ● Applications:
                    ● Signature verification
                    ● Face verification
                    ● Paraphrase scoring
                    
                ● Generally, in such tasks, two identical subnetworks are used to process the two inputs.
                ● Another module will take their outputs and produce the final output.
                
                ● Types of Metrics:
                    ● Absolute terms (Regular Siamese training)
                        ● Distance (xref , x+) = Low; Distance (xref , x−) = High
                        ● Similarity (xref , x+) = High; S imilarity (xref , x−) = Low
                    ● Relative terms (Triplet S iamese training)
                        ● Distance (xref , x−) − Distance (xref , x+) > Margin
                        ● Similarity (xref , x+) − S imilarity (xref , x−) > Margin
                    ● Class probability was based on a s ingle input
                        ● Class Prob (x,c) = High when x ∈ c; otherwise low.
                
                ● Distances and Similarity Measures used for Siamese Networks:
                    ● Distances examples
                        ● L2 norm of difference (Euclidean distance) (||(f(xi) − f(xj)||2)
                        ● L1 norm of difference (City-block/Manhattan dist.) (|(f(xi) − f(xj)|)
                    ● Similarity examples
                        ● Dot product ( f(xi)T f(xj) )
                        ● Arc cosine ( f(xi).f(xj) / (||f(xi)|| ||f(xj)||) )
                        ● Radial basis function (RBF) exp( −||xi  − xj||2/σ2 )

                ● Triplet Loss Function:
                    ● You can train the network by taking an anchor image and comparing it with  both a positive sample and a negative sample.
                    ● The dissimilarity between the anchor image and positive image must low and the dissimilarity between the anchor image and the negative image must be high.
                    ● By using this loss function we calculate the gradients and with the help of the  gradients, we update the weights and biases of the Siamese network.

                    ● Formula:
                        ● L = max(d(a,p) - d(a,n) + margin, 0)
                        ● where
                            ● “a” - represents the anchor image
                            ● “p” - represents a positive image
                            ● “n” - represents a negative image
                            ● margin - is a hyperparameter. It defines how far away the dissimilarities should be.
        """
        pass

    def st_1_eda_steps():
        """
            Train Test Split:
                from sklearn.model_selection import train_test_split
                xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=48)
                xtrain.shape, xtest.shape
            Reshape and flatten:
                Xtrain=xtrain.reshape(1649,64*64) 
                Xtest=xtest.reshape(413,64*64)
                Xtrain.shape, Xtest.shape
            Normalize Data:
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                Xtrain1=Xtrain/255.
                Xtest1=Xtest/255.

            One Hot Encode Classes( If the inputs are  like 1,2,3,4.. ):
                from tensorflow.keras.utils import to_categorical
                ytrain = to_categorical(y_train, num_classes=10)
                ytest = to_categorical(y_test, num_classes=10)

                print("Shape of y_train:", ytrain.shape)
                print("One value of y_train:", ytrain[0])
            
        """
        pass

    def st_2_model_build_steps():
        """
            Base Model Building:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense

                def train_and_test_loop(iterations, lr, Lambda, verb=True):

                    ## hyperparameters
                    iterations = iterations
                    learning_rate = lr
                    hidden_nodes = 256
                    output_nodes = 10
                        
                    model = Sequential()
                    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))
                    model.add(Dense(hidden_nodes, activation='relu'))
                    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
                    
                    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
                    # Compile model
                    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    
                    # Fit the model
                    model.fit(X_train, y_train, epochs=iterations, batch_size=1000, verbose= 1)
                    
                    [loss,score_train]=model.evaluate(X_train,y_train)
                    [loss,score_test]=model.evaluate(X_val,y_val)
                    
                    return score_train,score_test

                lr = 0.00001
                Lambda = 0
                train_and_test_loop(10, lr, Lambda)

            Improvement 1 :

                def train_and_test_loop1(iterations, lr, Lambda, verb=True):
                    ## hyperparameters
                    iterations = iterations
                    learning_rate = lr
                    hidden_nodes = 256
                    output_nodes = 10

                    model = Sequential()
                    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))
                    model.add(Dense(hidden_nodes, activation='relu'))
                    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
                    
                    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
                    # Compile model
                    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    
                    # Fit the model
                    model.fit(X_train, y_train, epochs=iterations, batch_size=1000, verbose= 1)
                    #score = model.evaluate(X_train, y_train, verbose=0)
                    [loss,score_train]=model.evaluate(X_train,y_train)
                    [loss,score_test]=model.evaluate(X_val,y_val)
                    
                    return score_train,score_test

                lr = 20
                Lambda = 0
                train_and_test_loop1(10, lr, Lambda)

            Improvement 2: (Explore Alpha And Lambda options)

                lr=np.linspace(0.1,1,10)
                lam=np.linspace(0.001,0.01,10)
                for i,j in zip(lr,lam):
                    score=train_and_test_loop1(5,i,j)
                    print('epocs:',10,'train_accuracy:',score[0],'test_accuracy:',score[1],'alpha:', i,'Regularization:',j)
                
                # Choose Best value and retrain again..
                train_and_test_loop1(10,0.1,0.005)

                # Approach 2: Try with Random lr and lambda:
                import math
                for k in range(1,5):
                    lr = math.pow(10, np.random.uniform(-4.0, -1.0))
                    Lambda = math.pow(10, np.random.uniform(-4,-2))
                    best_acc = train_and_test_loop1(100, lr, Lambda, False)
                    print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))


            Improvement 3: (Tune Model with Keras Classifier and GridSearchCV):
                def tune_model(learning_rate,activation, lamda,initializer,num_unit):
                    model = Sequential()
                    model.add(Dense(num_unit, kernel_initializer=initializer,activation=activation, input_dim=784))
                    #model.add(Dropout(dropout_rate))
                    model.add(Dense(num_unit, kernel_initializer=initializer,activation=activation))
                    #model.add(Dropout(dropout_rate)) 
                    model.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(lamda)))
                    sgd = optimizers.SGD(learning_rate=learning_rate)
                    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
                    return model

                batch_size = [20, 50, 100][:1]
                epochs = [1, 20, 50][:1]
                initializer = ['lecun_uniform', 'normal', 'he_normal', 'he_uniform'][:1]
                learning_rate = [0.1, 0.001, 0.02][:1]
                lamda = [0.001, 0.005, 0.01][:1]
                num_unit = [256, 128][:1]
                activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'][:1]
                parameters = dict(batch_size = batch_size,
                    epochs = epochs,
                    learning_rate=learning_rate,
                    lamda = lamda,
                    num_unit = num_unit,
                    initializer = initializer,
                    activation = activation)

                model =tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=tune_model, verbose=0)
                from sklearn.model_selection import GridSearchCV
                models = GridSearchCV(estimator = model, param_grid=parameters, n_jobs=1)
                best_model = models.fit(X_train, y_train)
                print('Best model :',best_model.best_params_)
                
                import pandas as pd
                pd.DataFrame(best_model.cv_results_)

        """
        pass