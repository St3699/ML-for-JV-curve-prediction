%% Setup and Initialization 
clc; clear;
rng(42);

%% Load Test Data
XTrain = [];
YTrain = [];
XTest = [];
YTest = [];

data_folder = ('C:\Users\Lenovo\Documents\UNI STUFF\FYP\Part 2\Data\');
folder_names = ["Data_1k_sets\Data_1k_rng1", "Data_1k_sets\Data_1k_rng2", "Data_1k_sets\Data_1k_rng3", ...
    "Data_10k_sets\Data_10k_rng1", "Data_10k_sets\Data_10k_rng2", "Data_10k_sets\Data_10k_rng3", ...
    "Data_100k\Data_100k"];

lhs_filename = "LHS_parameters_m.txt";
jv_filename = "iV_m.txt";
processed_lhs_filename = "lhs32DataFile.txt";
processed_jv_filename = "iDataFile.txt";

for i = 1:length(folder_names)
    inputFile = fullfile(data_folder, folder_names(i), lhs_filename); % 31 input parameters
    outputFile = fullfile(data_folder, folder_names(i), jv_filename); % output in the form of current density, A/m^2

    process_jv_fnn(inputFile, outputFile, processed_lhs_filename, processed_jv_filename);
    tempX = readmatrix(processed_lhs_filename, 'Delimiter', ','); % 46 columns: 31 device parameters + 15 voltage values
    tempY = readmatrix(processed_jv_filename, 'Delimiter', ',');
    
    if contains(folder_names(i), "1k_rng1")
        XTest = [XTest; tempX];
        YTest = [YTest; tempY];
    else
        XTrain = [XTrain; tempX];
        YTrain = [YTrain; tempY];
    end
end

%% model 
net = fitrnet(XTrain, YTrain, ...
    'LayerSizes', [ 70, 50, 45], ...
    'Activations', 'sigmoid', ...
    'LayerWeightsInitializer', 'glorot', ...
    'LayerBiasesInitializer', 'zeros', ...
    'Lambda', 0.017029, ...
    'Standardize', true);

%% shapley
disp('Shapley')
queryPoint = XTest;
explainer = shapley(netModel, 'QueryPoints', queryPoint);
figure; plot(explainer);

saveas(gcf, fullfile(results_folder, sprintf('shapley %s %s rs.fig', case_name, data_set)));
disp('Shapley done')

%% Evaluation
% training rmse
YPred_train = predict(net, XTrain);
rmse_train = sqrt(mean((YTrain - YPred_train).^2));
meanYTrain = mean(YTrain);
RRMSE_train = (rmse_train / meanYTrain) * 100;  % Relative RMSE as a percentage
disp(['Training RMSE: ', num2str(rmse_train)])
disp(['Training Relative RMSE: ', num2str(RRMSE_train), '%']);

% training R-squared 
SStot_train = sum((YTrain - meanYTrain).^2);  % Total sum of squares
SSres_train = sum((YTrain - YPred_train).^2);        % Residual sum of squares
R2_train = 1 - (SSres_train / SStot_train);               % R-squared formula
disp(['Training R-squared: ', num2str(R2_train)]);


% testing rmse
rmse_test = sqrt(mean((YTest - YPred).^2));
meanYTest = mean(YTest);
RRMSE_test = (rmse_test / meanYTest) * 100;  % Relative RMSE as a percentage
disp(['Testing RMSE: ', num2str(rmse_test)])
disp(['Testing Relative RMSE: ', num2str(RRMSE_test), '%']);

% testing R-squared 
SStot_test = sum((YTest - meanYTest).^2);  % Total sum of squares
SSres_test = sum((YTest - YPred).^2);        % Residual sum of squares
R2_test = 1 - (SSres_test / SStot_test);               % R-squared formula
disp(['Testing R-squared: ', num2str(R2_test)]);

%% Visual Assessment: Predicted vs. Observed
figure;

subplot(1,2,1);
% subtitle("Training");
hold on; grid on;
scatter(YTrain*10/1000, YPred_train*10/1000, 'filled');
plot(xlim, xlim, 'r--', 'LineWidth', 1.5);  % y=x line
xlabel('Observed \it{J} \rm(10^3 A\cdot m^{-2})');
ylabel('Predicted \it{J} \rm(10^3 A\cdot m^{-2})');
axis equal;  % Equal scaling on both axes
hold off;

subplot(1,2,2); hold on; grid on;
% subtitle("Testing");
scatter(YTest*10/1000, YPred*10/1000, 'filled');
plot(xlim, xlim, 'r--', 'LineWidth', 1.5);  % y=x line
xlabel('Observed \it{J} \rm(10^3 A\cdot m^{-2})');
ylabel('Predicted \it{J} \rm(10^3 A\cdot m^{-2})');
axis equal;  % Equal scaling on both axes
% sgtitle(sprintf('(%s) Predicted vs. Observed Values', case_name), 'fontsize', 12);