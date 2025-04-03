function process_jv_lstm(lhs_filename, jv_filename, processed_lhs_filename, processed_jv_filename, filter)

arguments
    lhs_filename 
    jv_filename 
    processed_lhs_filename = "lhs32DataFile.txt"
    processed_jv_filename = "iDataFile.txt"
    filter = false
end

rng(42);

% Read current from file (assuming two columns: Voltage and Current)
current = load(jv_filename); 
N = length(current);

% Read LHS data from file (600 cells (rows), 31 parameters)
lines = readlines(lhs_filename);  % Reads all lines into a string array

% Create or open a file to write the current data
fid = fopen(processed_jv_filename, 'w'); % Open the file for writing

% Create a new file to write the voltage and LHS data
hid = fopen(processed_lhs_filename, 'w');

for i = 1:N
    V = [0:0.1:0.4, 0.425:0.025:1.4]; % applied voltage V
    J = current(i, :);  % Current row

    % Find Open-Circuit Voltage (Voc) by interpolation
    [V, J, Voc, ~, ~, Jsc, ~, Jmpp, FF, ~, ~] = extractPOI(V, J);

    if ~filter || FF >= 0.70
        idx_voc = find(V == Voc, 1, 'first');
        idx_jsc = find(J == Jsc, 1, 'first');
        idx_mpp = find(J == Jmpp, 1, 'first');
    
        % Select Additional Points for Reconstruction
        desired_num = 15;

        idx_selected = round(linspace(idx_jsc, idx_voc, desired_num));
        idx_combined = unique([idx_jsc, idx_mpp, idx_voc, idx_selected]);

        if length(idx_combined) > desired_num
            extra_num = length(idx_combined) - desired_num;
            idx_selected = setdiff(idx_selected, [idx_voc, idx_jsc, idx_mpp]);
            idx_remove = randsample(idx_selected, extra_num, false);
            idx_selected = setdiff(idx_selected, idx_remove);
            idx_combined = unique([idx_jsc, idx_mpp, idx_voc, idx_selected]);

        elseif desired_num > length(idx_combined)

            additional_needed = desired_num - length(idx_combined);

            % Generate random voltages in [0, Voc] excluding existing V(idx_combined)
            V_existing = V(idx_combined);

            [~, idx_unique] = unique(J);

            V_random_pool = linspace(0, Voc, 50);
            V_random_pool = setdiff(V_random_pool, [V_existing, setdiff(V, V(idx_unique))]);

            V_additional = randsample(V_random_pool, additional_needed, false);
            J_additional = interp1(V(idx_unique), J(idx_unique), V_additional, 'pchip');

            % Merge interpolated points
            V = [V, V_additional];
            J = [J, J_additional];

            % Sort V and J accordingly
            [V, sort_idx] = sort(V, 'ascend');
            J = J(sort_idx);

            % Recalculate combined indices
            V_selected_all = [V_existing, V_additional];
            idx_combined_all = [idx_jsc, idx_mpp, idx_voc]; 
            
            for point = 1:15
                idx_selected_all = find(V == V_selected_all(point), 1, 'first');
                idx_combined_all = unique([idx_combined_all, idx_selected_all]);
            end

            idx_combined = idx_combined_all; 
        end

        idx_combined = sort(idx_combined);
    
        % Get the corresponding voltages and currents from these indices
        V_selected = V(idx_combined);
        J_selected = J(idx_combined);
        
        % Save voltage data to LHS file
        line = lines{i};  % LHS parameters as a string
        
        fprintf(fid, '%.15g,', J_selected(1:end-1));
        fprintf(fid, '%.15g\n', J_selected(end));

        fprintf(hid, '%s,', line);
        fprintf(hid, '%.15g,', V_selected(1:end-1));
        fprintf(hid, '%.15g\n', V_selected(end));
    end

end

fclose(fid);
end