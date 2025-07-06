%% Step 2: BEMD Processing for Optical Vortex Data (Single Step Demo)
% Based on bemdOpticalVecXY.m approach
% Processes single time step (1005) for demonstration

clc; clear;

% Set parameters
nimfs = 3;  % Number of IMFs
target_step = 1005;  % Target time step
dataName = "loam1data";

% Data paths
data_path = './output/';
output_path = './output/';

% Add BEMD algorithm path
addpath('../bemd');

disp('=== Step 2: BEMD Processing (Single Step Demo) ===');
fprintf('Processing step %d for demonstration\n', target_step);

% Load data
try
    fprintf('Loading processed data for step %d...\n', target_step);
    
    filename = data_path + sprintf("loam1E%d.mat", target_step);
    load(filename);
    eabs = dataE;
    clear dataE;
    
    filename = data_path + sprintf("loam1V1%d.mat", target_step);
    load(filename);
    v1abs = dataV1;
    clear dataV1;
    
    filename = data_path + sprintf("loam1V2%d.mat", target_step);
    load(filename);
    v2abs = dataV2;
    clear dataV2;
    
    filename = data_path + sprintf("loam1V3%d.mat", target_step);
    load(filename);
    v3abs = dataV3;
    clear dataV3;
    
    disp('Data loaded successfully!');
    dim = size(eabs);
    fprintf('Data shape: [%d, %d, %d]\n', dim(1), dim(2), dim(3));
    
catch ME
    disp('Error loading data:');
    disp(ME.message);
    return;
end

% Start BEMD processing
disp('Starting BEMD processing...');
successful_processing = true;

% Process only single time step (first index)
i = 1;
fprintf('Processing time step index %d (original step %d)\n', i, target_step);

% Process E component
im1 = zeros(dim(2), dim(3));
im1(:,:) = eabs(i,:,:);

try
    tic;
    a = bemd(im1, nimfs);
    elapsed_E = toc;
    fprintf('E component BEMD completed in %.2f seconds\n', elapsed_E);
    filename = output_path + dataName + "_BIMF0_E.mat";
    save(filename, 'a');
    success_E = true;
catch ME
    fprintf('Warning: BEMD failed for E: %s\n', ME.message);
    % Create zero matrix as fallback
    a = zeros(size(im1,1)*size(im1,2), nimfs);
    filename = output_path + dataName + "_BIMF0_E.mat";
    save(filename, 'a');
    success_E = false;
end

% Process V1 component
im1 = zeros(dim(2), dim(3));
im1(:,:) = v1abs(i,:,:);

try
    tic;
    b = bemd(im1, nimfs);
    elapsed_V1 = toc;
    fprintf('V1 component BEMD completed in %.2f seconds\n', elapsed_V1);
    filename = output_path + dataName + "_BIMF0_V1.mat";
    save(filename, 'b');
    success_V1 = true;
catch ME
    fprintf('Warning: BEMD failed for V1: %s\n', ME.message);
    b = zeros(size(im1,1)*size(im1,2), nimfs);
    filename = output_path + dataName + "_BIMF0_V1.mat";
    save(filename, 'b');
    success_V1 = false;
end

% Process V2 component
im1 = zeros(dim(2), dim(3));
im1(:,:) = v2abs(i,:,:);

try
    tic;
    c = bemd(im1, nimfs);
    elapsed_V2 = toc;
    fprintf('V2 component BEMD completed in %.2f seconds\n', elapsed_V2);
    filename = output_path + dataName + "_BIMF0_V2.mat";
    save(filename, 'c');
    success_V2 = true;
catch ME
    fprintf('Warning: BEMD failed for V2: %s\n', ME.message);
    c = zeros(size(im1,1)*size(im1,2), nimfs);
    filename = output_path + dataName + "_BIMF0_V2.mat";
    save(filename, 'c');
    success_V2 = false;
end

% Process V3 component
im1 = zeros(dim(2), dim(3));
im1(:,:) = v3abs(i,:,:);

try
    tic;
    d = bemd(im1, nimfs);
    elapsed_V3 = toc;
    fprintf('V3 component BEMD completed in %.2f seconds\n', elapsed_V3);
    filename = output_path + dataName + "_BIMF0_V3.mat";
    save(filename, 'd');
    success_V3 = true;
catch ME
    fprintf('Warning: BEMD failed for V3: %s\n', ME.message);
    d = zeros(size(im1,1)*size(im1,2), nimfs);
    filename = output_path + dataName + "_BIMF0_V3.mat";
    save(filename, 'd');
    success_V3 = false;
end

% Calculate success/failure statistics
if success_E && success_V1 && success_V2 && success_V3
    successful_steps = 1;
    failed_steps = 0;
    total_time = elapsed_E + elapsed_V1 + elapsed_V2 + elapsed_V3;
    fprintf('All components processed successfully! Total time: %.2f seconds\n', total_time);
else
    successful_steps = 0;
    failed_steps = 1;
    disp('Some components failed to process.');
end

fprintf('BEMD processing complete!\n');
fprintf('Successful processing: %s\n', successful_steps > 0);

% Save processing summary
disp('Saving processing summary...');
summary = struct();
summary.target_step = target_step;
summary.successful_processing = successful_steps > 0;
summary.nimfs = nimfs;
summary.data_shape = dim;
summary.processing_date = datestr(now);
if exist('total_time', 'var')
    summary.total_processing_time = total_time;
end

filename = output_path + "bemd_processing_summary.mat";
save(filename, 'summary');

% Generate verification plots
disp('Generating verification plots...');
try
    % Load BEMD results for verification
    load(output_path + dataName + "_BIMF0_E.mat", 'a');
    load(output_path + dataName + "_BIMF0_V1.mat", 'b');
    load(output_path + dataName + "_BIMF0_V2.mat", 'c');
    load(output_path + dataName + "_BIMF0_V3.mat", 'd');
    
    % Reshape to image format
    e_imf1 = reshape(a(:,1), dim(2), dim(3));
    v1_imf1 = reshape(b(:,1), dim(2), dim(3));
    v2_imf1 = reshape(c(:,1), dim(2), dim(3));
    v3_imf1 = reshape(d(:,1), dim(2), dim(3));
    
    % Generate verification plots
    figure('Visible', 'off');
    
    % 2x3 subplot layout
    subplot(2,3,1);
    imagesc(squeeze(eabs(1,:,:)));
    colorbar;
    title('Original E Intensity');
    
    subplot(2,3,2);
    imagesc(e_imf1);
    colorbar;
    title('E Component IMF1');
    
    subplot(2,3,3);
    imagesc(v1_imf1);
    colorbar;
    title('V1 Component IMF1');
    
    subplot(2,3,4);
    imagesc(v2_imf1);
    colorbar;
    title('V2 Component IMF1');
    
    subplot(2,3,5);
    imagesc(v3_imf1);
    colorbar;
    title('V3 Component IMF1');
    
    % Synthesize total IMF1
    total_imf1 = sqrt(e_imf1.^2 + v1_imf1.^2 + v2_imf1.^2 + v3_imf1.^2);
    subplot(2,3,6);
    imagesc(total_imf1);
    colorbar;
    title('Total IMF1 Magnitude');
    
    sgtitle(sprintf('BEMD Results Verification - Step %d', target_step));
    
    % Save image
    saveas(gcf, output_path + "bemd_verification.png");
    close;
    
    disp('Verification plots saved!');
    
catch ME
    fprintf('Warning: Could not generate verification plots: %s\n', ME.message);
end

disp('=== Step 2 Complete ===');
fprintf('Processed step %d with %d IMFs\n', target_step, nimfs);
fprintf('Results saved to: %s\n', output_path);
disp('Next: Run Python visualization (step3_visualization.py)'); 