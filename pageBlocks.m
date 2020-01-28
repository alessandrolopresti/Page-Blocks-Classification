

% PROJECT TITLE: Page Blocks Classification with Neural Networks
% AUTHORS: Alessandro Lo Presti, Valerio Longo
% UNIVERSITY: Sapienza - University of Rome
% LINK ORIGINAL DATASET: https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification

% OPENING THE DATASET

% The original dataset is presented in a data format file, we converted 
% into a text file to allow anyone, without the use of external 
% programs, to run the application and to be able to use csvread method.

filename = 'page-blocks.txt';
dataSet = csvread(filename);


% SUBDIVISION OF THE DATASET

features = dataSet(1:end, 1:end-1);
target = dataSet(1:end, end);

% We transpose the features matrix so that it is taken properly in input
% from the Neural Network Toolbox.

features = features';


% DATA NORMALIZATION

% In neural networks, it is a best practice to pre-process input data 
% before use. Data pre-processing  makes  the  training  of  the  network  
% faster,  memory  efficient  and  yield  accurate forecast results. 
% Neural networks only work with data usually between a specified range 
% e.g. - 1 to 1 or 0 to 1, it makes it necessary then that data is scaled 
% down and normalized. 
% Normalization ensures that data is roughly uniformly distributed between
% the network inputs and the outputs. (Mendelssohn, 1993.)  
% In Matlab features scaling is performed by the Neural Network Toolbox.
% Regarding the class normalization, if we consider the original
% target vector we are introducing a metric, ie born a distance between the
% class values, for example the value 5 of the fifth class in which the 
% value 1 of the first class is subtracted can lead to the value 4, which 
% correspond to the fourth class, this is wrong! 
% A decoding that does not introduce any metric is to consider the class 1 
% as a vector of type 1 0 0 0 0, the class 2 as a vector of type
% 0 1 0 0 0 etc.

featuresDimension = size(features);
nSamples = featuresDimension(2);
nTarget = 5;
targetNorm = zeros(nTarget, nSamples);

for i=1:nSamples
    if target(i) == 1
        targetNorm(1, i) = 1;
    elseif target(i) == 2
        targetNorm(2, i) = 1;
    elseif target(i) == 3
        targetNorm(3, i) = 1;
    elseif target(i) == 4
        targetNorm(4, i) = 1;
    else
        targetNorm(5, i) = 1;
    end
end

target = targetNorm;


% CREATING MULTILAYER NEURAL NETWORK

% Our neural network has two layers: a hidden layer and an output layer.
% The hidden layer will be initially constituted by a number of neurons 
% considered optimal by the application of heuristics, as explained in 
% report file.
% The output layer has a number of neurons equal to the number of classes.
% flagNeurons variable is used to execute portions of code with a different 
% number of neurons, it can assume the values:
% - 0 = optimum number of neurons
% - 1 = 1 neuron
% - 2 = 10 neurons
% - 3 = 20 neurons
% - 4 = 50 neurons
% The number of hidden layers can be varied by means of a special flag 
% called flagHidden, it can assume the values:
% - 0 = 1 hidden layer
% - else 2 hidden layers
% flagTransfFunc variable is used to execute portions of code with a
% different hidden-layer transfer function, it can assume the values:
% - 0 = Log-Sigmoid Transfer Function (logsig)
% - 1 = Linear Transfer Function (purelin)
% - 2 = Tan-Sigmoid Transfer Function (tansig)

flagNeurons = 2;
flagTransfFunc = 0;
flagHidden = 0;

if flagNeurons == 0
    hiddenUnits = 12;
    if flagHidden == 0
        net = patternnet(hiddenUnits);
    else
        net = patternnet([hiddenUnits hiddenUnits]);
    end
    
elseif flagNeurons == 1
    hiddenUnits = 1;
    if flagHidden == 0
        net = patternnet(hiddenUnits);
    else
        net = patternnet([hiddenUnits hiddenUnits]);
    end
    
elseif flagNeurons == 2
    hiddenUnits = 10;
    if flagHidden == 0
        net = patternnet(hiddenUnits);
    else
        net = patternnet([hiddenUnits hiddenUnits]);
    end
    
   
elseif flagNeurons == 3
    hiddenUnits = 20;
    if flagHidden == 0
        net = patternnet(hiddenUnits);
    else
        net = patternnet([hiddenUnits hiddenUnits]);
    end
 
elseif flagNeurons == 4
    hiddenUnits = 50;
    if flagHidden == 0
        net = patternet(hiddenUnits);
    else
        net = patternet([hiddenUnits hiddenUnits]);
    end
    
else
    fprintf('flagNeurons value is %d, consequently will apply the default number neurons (10 neurons) with the default number of hidden layers (1 hidden layer)\n', flagNeurons);
    net = patternnet();
    
end

if flagTransfFunc == 0
    net.layers{1}.transferFcn = 'logsig';

elseif flagTransfFunc == 1
    net.layers{1}.transferFcn = 'purelin';

elseif flagTransfFunc == 2
    net.layers{1}.transferFcn = 'tansig';

else
    fprintf('flagTransfFunc value is %d, consequently will apply the default transfer function (tansig)\n',flagTransfFunc);
    
end

setdemorandstream(391418381);


% TRAINING MULTILAYER NEURAL NETWORK

% To train the network we use a training algorithm with backpropagation.
% As a note on terminology, the term "backpropagation" is sometimes used to
% refer specifically to the gradient descent algorithm, when applied to
% neural network training. That terminology is not used in Neural Network 
% Toolbox, since the process of computing the gradient and Jacobian by 
% performing calculations backward through the network is applied in all of 
% the training functions.
% flagTrainFunc variable is used to execute portions of code with a
% different training function, it can assume the values: 
% - 0 = Scaled Conjugate Gradient (trainscg = default)
% - 1 = BFGS Quasi-Newton (trainbfg)
% - 2 = Gradient Descent (traingd)
% The quasi-Netwon method is quite fast, it tend to be less efficient for
% large networks (with thousands of weights), since it require more memory
% and more computation time for these cases.

flagTrainFunc = 0;

if flagTrainFunc == 0
    net.trainFcn = 'trainscg';
    
elseif flagTrainFunc == 1
    net.trainFcn = 'trainbfg';
    
elseif flagTrainFunc == 2
    net.trainFcn = 'traingd';
    
else
    fprintf('flagTrainFunc value is %d, consequently will apply the default training function (trainscg)\n',flagTrainFunc);
    
end

[net,tr] = train(net,features,target);
nntraintool;
plotperform(tr);
testx = features(:,tr.testInd);
testt = target(:,tr.testInd);
testY = net(testx);
testIndices = vec2ind(testY);
%plotconfusion(testt,testY);
[c,cm] = confusion(testt,testY);


% DISTRIBUTION OF SAMPLES IN TRAINING, VALIDATION AND TEST SET
% To verify if the dataset is balanced, that is, when the training set and
% the test set have roughly the same percentage of distribution of the
% samples for each class, we draw the histograms of the training,
% validation and test set.

trFirst = 0;
trSecond = 0;
trThird = 0;
trFourth = 0;
trFifth = 0;
vFirst = 0;
vSecond = 0;
vThird = 0;
vFourth = 0;
vFifth = 0;
teFirst = 0;
teSecond = 0;
teThird = 0;
teFourth = 0;
teFifth = 0;
sizeTrain = size(tr.trainInd);
sizeTrain = sizeTrain(2);
sizeValidation = size(tr.valInd);
sizeValidation = sizeValidation(2);
sizeTest = size(tr.testInd);
sizeTest = sizeTest(2);
numCol = size(dataSet);
numCol = numCol(2);

% ITERATION FOR TRAINING SET

for i = 1:sizeTrain
    if dataSet(tr.trainInd(i), numCol) == 1
        trFirst = trFirst + 1;
    elseif dataSet(tr.trainInd(i), numCol) == 2
        trSecond = trSecond + 1;
    elseif dataSet(tr.trainInd(i), numCol) == 3
        trThird = trThird + 1;
    elseif dataSet(tr.trainInd(i), numCol) == 4
        trFourth = trFourth + 1;
    else
        trFifth = trFifth + 1;
    end
end

% ITERATION FOR VALIDATION SET

for i = 1:sizeValidation
    if dataSet(tr.valInd(i), numCol) == 1
        vFirst = vFirst + 1;
    elseif dataSet(tr.valInd(i), numCol) == 2
        vSecond = vSecond + 1;
    elseif dataSet(tr.valInd(i), numCol) == 3
        vThird = vThird + 1;
    elseif dataSet(tr.valInd(i), numCol) == 4
        vFourth = vFourth + 1;
    else
        vFifth = vFifth + 1;
    end
end

% ITERATION FOR TEST SET

for i = 1:sizeTest
    if dataSet(tr.testInd(i), numCol) == 1
        teFirst = teFirst + 1;
    elseif dataSet(tr.testInd(i), numCol) == 2
        teSecond = teSecond + 1;
    elseif dataSet(tr.testInd(i), numCol) == 3
        teThird = teThird + 1;
    elseif dataSet(tr.testInd(i), numCol) == 4
        teFourth = teFourth + 1;
    else
        teFifth = teFifth + 1;
    end
end

flagHist = 0;
 
 if flagHist == 0    
    % TRAINING SET HISTOGRAM 
    bar([trFirst, trSecond, trThird, trFourth, trFifth]);
    title('Number of samples in training set for each class');
    
 elseif flagHist == 1
     % VALIDATION SET HISTOGRAM
     bar([vFirst, vSecond, vThird, vFourth, vFifth]);
     title('Number of samples in validation set for each class');
     
 elseif flagHist == 2
     % TEST SET 
     bar([teFirst, teSecond, teThird, teFourth, teFifth]);
     title('Number of samples in test set for each class');
     
 else
     fprintf('[ERROR] flagHist value is %d, the only allowable values for flagHist are: 0, 1 or 2\n',flagHist);
end

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
