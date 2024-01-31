%%%%%%%%%%%Decide your input
% Specify the folder containing your images
%imageFolder_DL = 'C:\Users\acer\OneDrive\Desktop\VS_Folder\DL';
%imageFolder_DL_299 = 'C:\\Users\\acer\\OneDrive\\Desktop\\VS_Folder\\DL_299';
%imageFolder_DL_227= 'C:\\Users\\acer\\OneDrive\\Desktop\\VS_Folder\\DL_sub';
%imageFolder_DL_224 = 'C:\\Users\\acer\\OneDrive\\Desktop\\VS_Folder\\DL_224';
imageFolder_DL_224_cropped = 'C:\\Users\\acer\\OneDrive\\Desktop\\VS_Folder\\DL_crop';

%PAY ATTENTION TO THE DIMENSION OF INPUT THAT YOU CHOOSE FOR YOUR ARCHITECTURE

%%%%%%%ARCHITECTURE
% Create an imageDatastore for the images
imds = imageDatastore(imageFolder_DL_224_cropped, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
%%%%%%%%%%Decide your architecture
% SqueezeNet
net = densenet201;
analyzeNetwork(net)
net.Layers(1)
inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',4, ...
        'BiasLearnRateFactor',4);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',4, ...
        'BiasLearnRateFactor',4);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

%%%%%%%% PAY ATTENTION to freeze the layers' weight according your
%%%%%%%% architecture (watch the number of the layers and minus 3)
layers(1:705) = freezeWeights(layers(1:705));
lgraph = createLgraphUsingConnections(layers,connections);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);

%%%%%%%% Data Augmentation
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%%%%%%%%Try different options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.00005, ...
    'MiniBatchSize', 60, ...
    'MaxEpochs', 2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency', 50, ...
    'GradientThresholdMethod', 'absolute-value', ...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'parallel'); %%%%%if you have GPU put 'gpu'
net = trainNetwork(augimdsTrain,lgraph,options);
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);
confMat = confusionmat(imdsValidation.Labels, YPred);
numClasses = 4;
% Precision, Recall, and F1-score for each class
precision = zeros(1, numClasses);
recall = zeros(1, numClasses);
F1_score = zeros(1, numClasses);
for j = 1:numClasses
    TP = confMat(j, j);
    FP = sum(confMat(:, j)) - TP;
    FN = sum(confMat(j, :)) - TP;
    
    precision(j) = TP / (TP + FP);
    recall(j) = TP / (TP + FN);
    F1_score(j) = 2 * ((precision(j) * recall(j)) / (precision(j) + recall(j)));
end
names_net= {'SqueezeNet'};
disp(names_net);
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);
disp(['F1-score: ' num2str(F1_score)]);
disp('-------------------------------');
