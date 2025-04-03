clear; clc;
%% Load Data
XTrain = [];
YTrain = [];
XTest = [];
YTest = [];


% Load input and output data from the text files
data_folder = 'C:\Users\Lenovo\Documents\UNI STUFF\FYP\Part 2\Data\';
folder_names = ["Data_1k_sets\Data_1k_rng1", "Data_1k_sets\Data_1k_rng2", "Data_1k_sets\Data_1k_rng3", ...
    "Data_10k_sets\Data_10k_rng1", "Data_10k_sets\Data_10k_rng2", "Data_10k_sets\Data_10k_rng3", ...
    "Data_100k\Data_100k"];

for i = 1:length(folder_names)
    lhs_filename = fullfile(data_folder, folder_names(i), "LHS_parameters_m.txt");
    jv_filename = fullfile(data_folder, folder_names(i), "iV_m.txt");
    processed_lhs_filename = fullfile(data_folder, folder_names(i), "lhs32DataFile.txt");
    processed_jv_filename = fullfile(data_folder, folder_names(i), "iDataFile.txt");

    process_jv_lstm(lhs_filename, jv_filename, processed_lhs_filename, processed_jv_filename);
    tempX = readmatrix(processed_lhs_filename, 'Delimiter', ','); % 46 columns: 31 device parameters + 15 voltage values
    tempX = tempX(:, 1:46); % in case of NaN at end
    tempY = readmatrix(processed_jv_filename, 'Delimiter', ',');
    tempY = tempY(:, 1:15); % in case of NaN at end
    
    if contains(folder_names(i), "1k_rng1")
        XTest = [XTest; tempX];
        YTest = [YTest; tempY];
    else
        XTrain = [XTrain; tempX];
        YTrain = [YTrain; tempY];
    end
end

%% Define dimensions
disp(size(XTrain))
numCases = size(XTrain,1);
numTimesteps = 15;
numExogenousVariables = 31;

%% Separate the components from dataX
% Columns 1-31 are exogenous parameters (constant for each time step)
exog = XTrain(:, 1:numExogenousVariables);

% Columns 32-46 are the voltage values at each timestep
volt = XTrain(:, numExogenousVariables+1:end);

%% Format data
X_cell = cell(numCases, 1);
Y_cell = cell(numCases, 1);

for i = 1:numCases
X_cell{i} = [repmat(exog(i,:)', 1, numTimesteps); volt(i,:)]; % [32 x 15]
Y_cell{i} = YTrain(i, :); % [1 x 15]
end

% Create training and validation datasets
XTrain = X_cell;
YTrain = Y_cell;

%% Define LSTM Network with Normalization and Sigmoid Activation
layers = [
    sequenceInputLayer(32, ...
    'Normalization','rescale-zero-one', ...
    'NormalizationDimension','channel')
    bilstmLayer(20, ...
    'OutputMode', 'sequence', ...
    'InputWeightsInitializer','glorot', ...
    'GateActivationFunction','sigmoid')
    fullyConnectedLayer(1)
    regressionLayer
    ];

%% Training Options with Validation Data
options = trainingOptions('rmsprop', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 9, ...
    'InitialLearnRate', 0.0096226, ...
    'LearnRateSchedule', 'piecewise', ...
    'L2Regularization', 0.0011342, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment','parallel');

%% Train the biLSTM Network
net = trainNetwork(XTrain, YTrain, layers, options);
YTrain_cell = cell2mat(YTrain);

%% Predict Training and Testing Sets
YPred_tr = predict(net, XTrain);
YPred_tr = cell2mat(YPred_tr);

%% Testing and Training Metrics
% On Training set
rmse_tr = sqrt(mean((YPred_tr - YTrain_cell).^2, 'all'));
rrmse_tr = rmse_tr / mean(abs(YTrain_cell), 'all'); % Relative RMSE
r2_tr = corr(YPred_tr(:), YTrain_cell(:))^2;
fprintf('Training RMSE: %.4f\n', rmse_tr);
fprintf('Training relative RMSE: %.4f%% \n', rrmse_tr*100);
fprintf('Training R²: %.4f\n', r2_tr);

%% Using 1k_rng1 as testing
numCases = size(XTest,1);
numTimesteps = 15;
numExogenousVariables = 31;

% Separate the components from dataX
% Columns 1-31 are exogenous parameters (constant for each case)
exog = XTest(:, 1:numExogenousVariables); % Size: [657 x 31]

% Columns 32-46 are the voltage values at each timestep
volt = XTest(:, numExogenousVariables+1:end); % Size: [657 x 15]

% Form the input sequence for each case
XTest_cell = cell(numCases, 1);
YTest_cell = cell(numCases, 1);

for i = 1:numCases
    XTest_cell{i} = [repmat(exog(i,:)', 1, numTimesteps); volt(i,:)]; % [32 x 15]
    YTest_cell{i} = YTest(i, :); % [1 x 15]
end

YPred_ts = predict(net, XTest_cell);
YPred_ts = cell2mat(YPred_ts);
YVal_cell = cell2mat(YTest_cell);

rmse_ts = sqrt(mean((YPred_ts - YVal_cell).^2, 'all'));
rrmse_ts = rmse_ts / mean(abs(YVal_cell), 'all'); % Relative RMSE
r2_ts = corr(YPred_ts(:), YVal_cell(:))^2;
fprintf('Testing RMSE: %.4f\n', rmse_ts);
fprintf('Testing RRMSE: %.4f%% \n', rrmse_ts*100);
fprintf('Testing R²: %.4f\n', r2_ts);

%% visualization
caseIndex = randi([1, size(XTest_cell, 1)]); % Choose one case to visualize

figure;
plot(1:numTimesteps, YVal_cell(caseIndex, :), 'b-o', 'LineWidth', 1.5); hold on;
plot(1:numTimesteps, YPred_ts(caseIndex, :), 'r-*', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Current Density');
legend('True Y', 'Predicted Y', Location='best');
title(sprintf('True vs Predicted Output for Validation Case %d', caseIndex));
grid on;