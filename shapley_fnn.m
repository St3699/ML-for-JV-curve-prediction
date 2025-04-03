%% Load model
netStruct = load('fnn.mat');
if isfield(netStruct, 'netModel')
    net = netStruct.netModel;
else
    error('The loaded file does not contain a field named ''net''.');
end

%% load data
XTest = [];
YTest = [];

% Load input and output data from the text files
data_folder = 'C:\Users\Lenovo\Documents\UNI STUFF\FYP\Part 2\Data\';
folder_names = "Data_1k_sets\Data_1k_rng1";

lhs_filename = fullfile(data_folder, folder_names, "LHS_parameters_m.txt");
jv_filename = fullfile(data_folder, folder_names, "iV_m.txt");
processed_lhs_filename = fullfile(data_folder, folder_names, "lhs32DataFile.txt");
processed_jv_filename = fullfile(data_folder, folder_names, "iDataFile.txt");

process_jv3(lhs_filename, jv_filename, processed_lhs_filename, processed_jv_filename);
tempX = readmatrix(processed_lhs_filename, 'Delimiter', ','); % 46 columns: 31 device parameters + 15 voltage values
tempY = readmatrix(processed_jv_filename, 'Delimiter', ',');

XTest = [XTest; tempX];
YTest = [YTest; tempY];
%%
disp('Shapley')
queryPoint = XTest;
explainer = shapley(net, 'QueryPoints', queryPoint);
figure; plot(explainer);

disp('Shapley done')

%% plot explainer
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

% Extract numeric Shapley values from column 2
MAshapleyVals = explainer.MeanAbsoluteShapley;
MAshapleyVals = MAshapleyVals(1:31, :); % Keep only first 31 rows

% Convert table column to an array for sorting
shapleyValues = MAshapleyVals{:, 2}; % Extract numeric column

% Sort values in ascending order and get sorting indices
[sortedVals, sortIdx] = sort(shapleyValues, 'ascend'); 

% Reorder labels based on sorting indices
sortedLabels = labels(sortIdx);

% Plot horizontal bar chart
figure;
barh(sortedVals);
xlabel('Shapley Value', 'FontSize', 10, 'FontName', 'Times New Roman');
ylabel('Feature Index', 'FontSize', 10, 'FontName', 'Times New Roman');
title('Mean Absolute Shapley Values for the FNN Model');
grid on;
yticks(1:31);

% Update Y-axis tick labels
ax = gca; % Get current axes
ax.YTick = 1:31; 
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman'); 
ax.YTickLabel = sortedLabels; % Assign sorted labels