%% load model
clear; clc;
netStruct = load('novalLSTM.mat');
if isfield(netStruct, 'net')
    net = netStruct.net;
else
    error('The loaded file does not contain a field named ''net''.');
end

%% Load Test Data
XTest = [];
YTest = [];


% Load input and output data from the text files
data_folder = 'C:\Users\Lenovo\Documents\UNI STUFF\FYP\Part 2\Data\';
folder_names = "Data_1k_sets\Data_1k_rng1";

lhs_filename = fullfile(data_folder, folder_names, "LHS_parameters_m.txt");
jv_filename = fullfile(data_folder, folder_names, "iV_m.txt");
processed_lhs_filename = fullfile(data_folder, folder_names, "lhs32DataFile.txt");
processed_jv_filename = fullfile(data_folder, folder_names, "iDataFile.txt");

process_jv_lstm(lhs_filename, jv_filename, processed_lhs_filename, processed_jv_filename);
tempX = readmatrix(processed_lhs_filename, 'Delimiter', ','); % 46 columns: 31 device parameters + 15 voltage values
tempX = tempX(:, 1:46); % in case of NaN at end
tempY = readmatrix(processed_jv_filename, 'Delimiter', ',');
tempY = tempY(:, 1:15); % in case of NaN at end

XTest = [XTest; tempX];
YTest = [YTest; tempY];

%% Restructure Data
numCases = size(XTest,1);
numTimesteps = 15;
numExogenousVariables = 31;  % First 31 columns
exog = XTest(:, 1:numExogenousVariables);
volt = XTest(:, numExogenousVariables+1:end);

% Create cell arrays where each cell is a sequence [32 x 15]
XTest_cell = cell(numCases, 1);
Y_cell = cell(numCases, 1);
for i = 1:numCases
    % Each sequence has 32 features (31 exogenous + 1 voltage row per timestep? 
    % In your original code it was [repmat(exog(i,:)',1,numTimesteps); volt(i,:)]
    % which gives 32 rows and 15 columns.
    XTest_cell{i} = [repmat(exog(i,:)', 1, numTimesteps); volt(i,:)];  % [32 x 15]
    Y_cell{i} = YTest(i, :);  % [1 x 15]
end

%% 
function shapleyValues = calculateShapley(net, X)
    % X is a cell array of sequences, each of size [features x timesteps]
    totalFeatures = size(X{1}, 1);  % e.g., 32 features
    % Exclude the 32nd feature from the analysis
    featuresToConsider = 1:(totalFeatures-1);  % only features 1 to 31
    numSamples = numel(X);
    shapleyValues = zeros(1, length(featuresToConsider));

    % Get predictions for the full input (cell array output)
    yFull = predict(net, X);

    for idx = 1:length(featuresToConsider)
        i = featuresToConsider(idx);
        % Create a modified copy of X with the i-th feature replaced by a baseline (0)
        X_modified = cell(numSamples, 1);
        for j = 1:numSamples
            seq = X{j};
            seq_modified = seq;
            seq_modified(i, :) = 0;  % Replace feature i with a baseline value (e.g., 0)
            X_modified{j} = seq_modified;
        end

        % Get predictions for the modified input
        yModified = predict(net, X_modified);

        % Compute the difference for each sample individually
        differences = zeros(numSamples, 1);
        for j = 1:numSamples
            differences(j) = mean(yFull{j}(:) - yModified{j}(:));
        end

        % Average over all samples to get the Shapley value for feature i
        shapleyValues(idx) = mean(abs(differences));
    end
end


%% calculate shapley values
shapleyVals = calculateShapley(net, XTest_cell);

%% Calculate Shapley Values

% labels
labels = {'\textit{$l^{\mathrm{H}}$}(nm)', '\textit{$l^{\mathrm{P}}$}(nm)', '\textit{$l^{\mathrm{E}}$}(nm)', ... % Layer thickness of H, P, E
          '$\mu^{\mathrm{H}}_{h}$($\mathrm{m^{2}\cdot V^{-1}\cdot s^{-1}})$', '$\mu^{\mathrm{P}}_{h}$($\mathrm{m^{2}\cdot V^{-1}\cdot s^{-1}})$', ... % Hole mobility in H, P
          '$\mu^{\mathrm{P}}_{e}$($\mathrm{m^{2}\cdot V^{-1}\cdot s^{-1}})$', '$\mu^{\mathrm{E}}_{e}$($\mathrm{m^{2}\cdot V^{-1}\cdot s^{-1}})$', ... % Electron mobility in P, E
          '\textit{$N^{\mathrm{H}}_{v}$}($\mathrm{m^{-3}})$', '\textit{$N^{\mathrm{H}}_{c}$}($\mathrm{m^{-3}})$', ... % Valence and Conduction band density of state in H
          '\textit{$N^{\mathrm{E}}_{v}$}($\mathrm{m^{-3}})$', '\textit{$N^{\mathrm{E}}_{c}$}($\mathrm{m^{-3}})$', ... % Valence and Conduction band density of state in E
          '\textit{$N^{\mathrm{P}}_{v}$}($\mathrm{m^{-3}})$', '\textit{$N^{\mathrm{P}}_{c}$}($\mathrm{m^{-3}})$', ... % Valence and Conduction band density of state in P
          '$\chi^{\mathrm{H}}_{h}$(eV)', '$\chi^{\mathrm{H}}_{e}$(eV)', ... % Hole ionization potential and Electron affinity in H
          '$\chi^{\mathrm{P}}_{h}$(eV)', '$\chi^{\mathrm{P}}_{e}$(eV)', ... % Hole ionization potential and Electron affinity in P
          '$\chi^{\mathrm{E}}_{h}$(eV)', '$\chi^{\mathrm{E}}_{e}$(eV)', ... % Hole ionization potential and Electron affinity in E
          '\textit{$W_{\mathrm{B}}$}(eV)', '\textit{$W_{\mathrm{F}}$}(eV)', ... % Work function of B and F
          '$\varepsilon^{\mathrm{H}}$', '$\varepsilon^{\mathrm{P}}$', '$\varepsilon^{\mathrm{E}}$', ... % Relative permittivity in H, P, E
          '\textit{$G_{\mathrm{avg}}$}($\mathrm{m^{-3}\cdot s^{-1}}$)', ... % Average charge carrier generation rate in P
          '\textit{$A_{(e, h)}$}($\mathrm{m^{6}\cdot s^{-1}}$)', ... % Auger recombination coefficient in P
          '\textit{$B_{\mathrm{rad}}$}($\mathrm{m^{3}\cdot s^{-1}}$)', ... % Radiative recombination coefficient in P
          '$\tau_{e}$(s)', '$\tau_{h}$(s)', ... % Electron and Hole lifetime in P
          '$\nu_{\mathrm{II}}$($\mathrm{m^{4}\cdot s^{-1}}$)', '$\nu_{\mathrm{III}}$($\mathrm{m^{4}\cdot s^{-1}}$)', ... % Interface recombination velocity at II and III
          '$\mathrm{{V}_{a}}$($\mathrm{V}$)'}; % Applied Voltage

% sort shapley values
[sortedVals, sortIdx] = sort(shapleyVals, 'ascend');
sortedLabels = labels(sortIdx);


figure;
barh(sortedVals);  % Horizontal bar chart
xlabel('Shapley Value', 'FontSize', 10, 'FontName', 'Times New Roman');
ylabel('Feature Index', 'FontSize', 10, 'FontName', 'Times New Roman');
title('SHAP Analysis for LSTM Model');
grid on;
yticks(1:31);

ax = gca; % Get current axes
ax.YTick = 1:31; % Ensure there are 31 ticks
set(gca,'TickLabelInterpreter','latex', 'FontSize', 10, 'FontName', 'Times New Roman'); 
ax.YTickLabel= sortedLabels; % Assign labels
