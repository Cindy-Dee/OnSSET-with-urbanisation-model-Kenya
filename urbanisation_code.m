%% Prepare the data.
urbanisation = readtable('urbanisationmarch.csv'); % Define fine path
Ybool = urbanisation.IsUrban; % Extract the Urban column (Logical array)

%% Extract Predictors, adjust variable names based on csv file
X_all = urbanisation(:, ["Distance_grid","Electrified_Population","Distance_to_Urban","Elevation","Major_Road_Dist","Population_density","RiverLakes_Dist"]);
X = table2array(X_all); % Convert to a pure numeric matrix

%% Handle Missing Values
% Remove rows with missing values in predictors and response variable
rows_to_keep = ~any(ismissing(X), 2) & ~ismissing(double(Ybool));
X_clean = X(rows_to_keep, :);
Y_clean = Ybool(rows_to_keep);

%% Compute correlation matrix
% Define variable names with LaTeX formatting 
varNames = {
    '$\mathrm{D_G}$', ...
    '$\mathrm{P_E}$', ...
    '$\mathrm{D_U}$', ...
    '$\mathrm{E}$', ...
    '$\mathrm{D_R}$', ...
    '$\mathrm{P_D}$', ...
    '$\mathrm{D_W}$'
};

disp("Correlation matrix")
[R,pval] = corr(X_clean, Type="Pearson");
disp(R);
disp(pval);

figure;
imagesc(abs(R));
colorbar; 
axis xy;

xticks(1:length(varNames));
yticks(1:length(varNames));
xticklabels(varNames);
yticklabels(varNames);
xlabel("Predictor");
ylabel("Predictor");

% Set tick labels and axis labels to use Times New Roman
set(gca, 'FontName', 'Times New Roman');

% Set colorbar labels to Times New Roman
set(hcb.Label, 'FontName', 'Times New Roman');

% Export with high resolution
print(gcf, 'correlation_chart.tiff', '-dtiff', '-r600');


%% Collinearity diagnostics
[sValue,condIdx,VarDecomp] = collintest(X_clean, Plot="on");

%% Check class imbalance
disp('Class distribution before balancing:');
disp(['Non-urban (0): ', num2str(sum(Y_clean == 0))]);
disp(['Urban (1): ', num2str(sum(Y_clean == 1))]);
disp(['Imbalance ratio: ', num2str(sum(Y_clean == 0)/sum(Y_clean == 1))]);

%% Apply SMOTE (class balancing)
% Convert logical to double 
Y_clean_double = double(Y_clean);

% Separate majority and minority classes
X_majority = X_clean(Y_clean_double == 0, :);
X_minority = X_clean(Y_clean_double == 1, :);
Y_majority = zeros(size(X_majority, 1), 1);
Y_minority = ones(size(X_minority, 1), 1);

% Check counts
n_minority = size(X_minority, 1);
n_majority = size(X_majority, 1);
disp(['Number of majority samples: ', num2str(n_majority)]);
disp(['Number of minority samples: ', num2str(n_minority)]);

target_ratio = 1; % Target a 1:1 ratio
n_synthetic = min(n_majority, n_minority * 100) - n_minority; % Cap at 100x original minority size

disp(['Generating ', num2str(n_synthetic), ' synthetic samples']);

% Only proceed if we have at least 5 minority samples for SMOTE
if n_minority >= 5 && n_synthetic > 0
    % Parameters for SMOTE
    k = min(5, n_minority-1); % Number of nearest neighbors to consider
    
    % Generate synthetic samples
    synthetic_samples = zeros(n_synthetic, size(X_minority, 2));
    
    for i = 1:n_synthetic
        % Random sample from minority class
        idx = randi(n_minority);
        sample = X_minority(idx, :);
        
        % Find k nearest neighbors
        distances = sum((X_minority - repmat(sample, n_minority, 1)).^2, 2);
        [~, indices] = sort(distances);
        
        % Choose a random neighbor from the k nearest (skip the first as it's the sample itself)
        neighbor_idx = indices(randi([2, min(k+1, length(indices))]));
        neighbor = X_minority(neighbor_idx, :);
        
        % Generate synthetic sample
        gap = rand();
        synthetic_samples(i, :) = sample + gap * (neighbor - sample);
    end
    
    % For extreme imbalance, downsample the majority class
    downsample_ratio = 0.1; % Keep 10% of majority samples
    n_downsample = min(n_majority, round(n_synthetic * 5)); % Target roughly 1:5 minority:majority ratio
    
    % Randomly select majority samples to keep
    random_indices = randperm(n_majority);
    selected_majority_indices = random_indices(1:n_downsample);
    X_majority_downsampled = X_majority(selected_majority_indices, :);
    Y_majority_downsampled = zeros(size(X_majority_downsampled, 1), 1);
    
    disp(['Downsampled majority class to ', num2str(n_downsample), ' samples']);
    
    % Combine downsampled majority with original and synthetic minority samples
    X_balanced = [X_majority_downsampled; X_minority; synthetic_samples];
    Y_balanced = [Y_majority_downsampled; Y_minority; ones(n_synthetic, 1)];
    
    % Shuffle the balanced dataset
    idx = randperm(length(Y_balanced));
    X_balanced = X_balanced(idx, :);
    Y_balanced = Y_balanced(idx);
    
    disp('Class distribution after SMOTE and downsampling:');
    disp(['Non-urban (0): ', num2str(sum(Y_balanced == 0))]);
    disp(['Urban (1): ', num2str(sum(Y_balanced == 1))]);
    disp(['New ratio: ', num2str(sum(Y_balanced == 0)/sum(Y_balanced == 1))]);
else
    % downsample for cases where SMOTE can't be applied
    disp('Not enough minority samples for SMOTE, applying downsampling only');
    n_downsample = min(n_majority, n_minority * 20); % Keep at most 10x the minority size
    
    % Randomly select majority samples to keep
    random_indices = randperm(n_majority);
    selected_majority_indices = random_indices(1:n_downsample);
    X_majority_downsampled = X_majority(selected_majority_indices, :);
    Y_majority_downsampled = zeros(size(X_majority_downsampled, 1), 1);
    
    % Combine
    X_balanced = [X_majority_downsampled; X_minority];
    Y_balanced = [Y_majority_downsampled; Y_minority];
    
    % Shuffle
    idx = randperm(length(Y_balanced));
    X_balanced = X_balanced(idx, :);
    Y_balanced = Y_balanced(idx);
    
    disp('Class distribution after downsampling:');
    disp(['Non-urban (0): ', num2str(sum(Y_balanced == 0))]);
    disp(['Urban (1): ', num2str(sum(Y_balanced == 1))]);
end

%% Split data into training and test sets
cv = cvpartition(length(Y_balanced), 'HoldOut', 0.2);
idx_train = cv.training;
idx_test = cv.test;

X_train = X_balanced(idx_train, :);
Y_train = Y_balanced(idx_train);
X_test = X_balanced(idx_test, :);
Y_test = Y_balanced(idx_test);

% separate set of original unbalanced data for final evaluation
cv_orig = cvpartition(length(Y_clean_double), 'HoldOut', 0.2);
idx_orig_test = cv_orig.test;
X_orig_test = X_clean(idx_orig_test, :);
Y_orig_test = Y_clean_double(idx_orig_test);

disp('Original test set class distribution:');
disp(['Non-urban (0): ', num2str(sum(Y_orig_test == 0))]);
disp(['Urban (1): ', num2str(sum(Y_orig_test == 1))]);

%% Standardize the Data (only using training data statistics)
[X_train_scaled, mu, sigma] = zscore(X_train); % Standardisation

% Apply scaling to test data
X_test_scaled = (X_test - mu) ./ sigma;
X_orig_test_scaled = (X_orig_test - mu) ./ sigma;

%% Fit Generalized Linear Model
mdl = fitglm(X_train_scaled, Y_train, ...
    'Distribution', 'binomial', ...
    'Link', 'logit', ...
    'Intercept', false);

%% Examine model coefficients
disp('Model summary:');
disp(mdl);

% Extract model coefficients
B = mdl.Coefficients.Estimate;
SE = mdl.Coefficients.SE;
pValues = mdl.Coefficients.pValue;

%% Predict using the fitted model on test set
preds_test = predict(mdl, X_test_scaled);
preds_orig = predict(mdl, X_orig_test_scaled);

%% Find optimal threshold using ROC curve on balanced test data
thresholds = linspace(0, 1, 100);
f1_scores = zeros(length(thresholds), 1);
precision_scores = zeros(length(thresholds), 1);
recall_scores = zeros(length(thresholds), 1);

for i = 1:length(thresholds)
    threshold = thresholds(i);
    pred_labels = double(preds_test >= threshold); % Convert to double to match Y_test
    
    % Compute precision and recall
    TP = sum(pred_labels == 1 & Y_test == 1);
    FP = sum(pred_labels == 1 & Y_test == 0);
    FN = sum(pred_labels == 0 & Y_test == 1);
    
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    
    f1_scores(i) = 2 * (precision * recall) / (precision + recall + eps);
    precision_scores(i) = precision;
    recall_scores(i) = recall;
end

[max_f1, best_idx] = max(f1_scores);
optimal_threshold = thresholds(best_idx);
disp(['Optimal threshold based on F1 score: ', num2str(optimal_threshold)]);

% Apply optimal threshold to get final predictions
% Convert to double to match Y_test and Y_orig_test types
pred_labels_test = double(preds_test >= optimal_threshold);
pred_labels_orig = double(preds_orig >= optimal_threshold);

%% Compute Confusion Matrix on balanced test data

conf_mat_balanced = confusionmat(Y_test, pred_labels_test);
disp('Confusion Matrix on balanced test data:');
disp(conf_mat_balanced);

%% Compute Confusion Matrix on original test data (unbalanced)
conf_mat_orig = confusionmat(Y_orig_test, pred_labels_orig);
disp('Confusion Matrix on original (unbalanced) test data:');
disp(conf_mat_orig);

%% Visualize Confusion Matrix as Heatmap for original data
figure;
h = heatmap({'Rural', 'Urban'}, {'Rural', 'Urban'}, conf_mat_balanced);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 8, ...
         'XColor', 'black', 'YColor', 'black', ...
         'TickLabelInterpreter', 'latex', ...
         'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
h.Title = ' ';
h.XLabel = 'Predicted';
h.YLabel = 'Actual';
colormap(jet);
colorbar;
% Export to TIFF with 600 DPI
print(gcf, 'confusion_matrix_percentage.tiff', '-dtiff', '-r600');

% Calculate percentages for better interpretation
conf_mat_pct = zeros(2, 2);
if sum(conf_mat_orig(1, :)) > 0
    conf_mat_pct(1, :) = conf_mat_orig(1, :) / sum(conf_mat_orig(1, :)) * 100;
end
if sum(conf_mat_orig(2, :)) > 0
    conf_mat_pct(2, :) = conf_mat_orig(2, :) / sum(conf_mat_orig(2, :)) * 100;
end

figure;
h2 = heatmap({'Rural', 'Urban'}, {'Rural', 'Urban'}, conf_mat_pct);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, ...
         'XColor', 'black', 'YColor', 'black', ...
         'TickLabelInterpreter', 'latex', ...
         'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
h2.Title = ' ';
h2.XLabel = 'Predicted';
h2.YLabel = 'Actual';
colormap(jet);
colorbar;

% Export to TIFF with 600 DPI
print(gcf, 'confusion_matrix_percentage.tiff', '-dtiff', '-r600');

%% Compute Performance Metrics for both test sets
% Function to calculate metrics
calculate_metrics = @(cm) struct(...
    'Accuracy', (cm(1,1) + cm(2,2)) / sum(cm(:)), ...
    'Precision', cm(2,2) / (cm(2,2) + cm(1,2) + eps), ...
    'Recall', cm(2,2) / (cm(2,2) + cm(2,1) + eps), ...
    'Specificity', cm(1,1) / (cm(1,1) + cm(1,2) + eps), ...
    'F1_score', 2 * cm(2,2) / (2 * cm(2,2) + cm(1,2) + cm(2,1) + eps), ...
    'BalancedAccuracy', 0.5 * (cm(2,2) / (cm(2,2) + cm(2,1) + eps) + cm(1,1) / (cm(1,1) + cm(1,2) + eps)) ...
);

% Calculate metrics for balanced test data
metrics_balanced = calculate_metrics(conf_mat_balanced);

% Calculate metrics for original test data
metrics_orig = calculate_metrics(conf_mat_orig);

% Display results for balanced test data
fprintf('\nPerformance on balanced test data:\n');
fprintf('Accuracy: %.2f%%\n', metrics_balanced.Accuracy * 100);
fprintf('Precision: %.2f\n', metrics_balanced.Precision);
fprintf('Recall/Sensitivity: %.2f\n', metrics_balanced.Recall);
fprintf('Specificity: %.2f\n', metrics_balanced.Specificity);
fprintf('F1 Score: %.2f\n', metrics_balanced.F1_score);
fprintf('Balanced Accuracy: %.2f\n', metrics_balanced.BalancedAccuracy);

% Display results for original test data
fprintf('\nPerformance on original (unbalanced) test data:\n');
fprintf('Accuracy: %.2f%%\n', metrics_orig.Accuracy * 100);
fprintf('Precision: %.2f\n', metrics_orig.Precision);
fprintf('Recall/Sensitivity: %.2f\n', metrics_orig.Recall);
fprintf('Specificity: %.2f\n', metrics_orig.Specificity);
fprintf('F1 Score: %.2f\n', metrics_orig.F1_score);
fprintf('Balanced Accuracy: %.2f\n', metrics_orig.BalancedAccuracy);

%% Plot ROC curve for original test data
[X_roc, Y_roc, T_roc, AUC] = perfcurve(Y_test, preds_test, 1);

figure;
plot(X_roc, Y_roc, 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
hold off;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, ...
         'XColor', 'black', 'YColor', 'black', ...
         'TickLabelInterpreter', 'latex', ...
         'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
xlabel('False Positive Rate (1 - Specificity)');
ylabel('True Positive Rate (Sensitivity)');
%title(['ROC Curve (AUC = ', num2str(AUC, '%.3f'), ')']);
legend('Model', 'Random Guess', 'Location', 'southeast');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, ...
         'XColor', 'black', 'YColor', 'black', ...
         'TickLabelInterpreter', 'latex', ...
         'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
% Export to TIFF with 600 DPI
print(gcf, 'roc_curve.tiff', '-dtiff', '-r600');


%% Compute Precision-Recall Curve
[precision, recall, T_pr] = perfcurve(Y_test, preds_test, 1, 'xCrit', 'reca', 'yCrit', 'prec');

% Plot Precision-Recall Curve
figure;
plot(recall, precision, 'b-', 'LineWidth', 2); % Blue line for PR curve
hold on;
y_baseline = sum(Y_test) / length(Y_test); % Baseline: proportion of positive class
plot([0, 1], [y_baseline, y_baseline], 'r--', 'LineWidth', 1.5); % Baseline reference
hold off;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, ...
         'XColor', 'black', 'YColor', 'black', ...
         'TickLabelInterpreter', 'latex', ...
         'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
%title('Precision-Recall Curve');
legend('Model', 'Baseline (Class Ratio)', 'Location', 'southwest');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

% Save the PR curve as a high-resolution image
print(gcf, 'precision_recall_curve_balanced.tiff', '-dtiff', '-r600');


%% Save the model and results
save('urbanisation_model.mat', 'mdl', 'mu', 'sigma', 'optimal_threshold');

% Create a structured result summary
results = struct();
results.metrics_balanced = metrics_balanced;
results.metrics_original = metrics_orig;
results.confusion_matrix_balanced = conf_mat_balanced;
results.confusion_matrix_original = conf_mat_orig;
results.optimal_threshold = optimal_threshold;
results.AUC = AUC;

save('model_results.mat', 'results');


%%
%show probability for the dataset
% Use the fitted model to predict probabilities for all data rows
urbanisation_new = readtable('missing del.xlsx'); % Read CSV
urbanisation_filled = fillmissing(urbanisation_new, 'constant', mean(urbanisation_new{:,:}, 'omitnan')); % Replace with mean

X_non_standard_new = table2array(urbanisation_filled(:, ["Distance_grid","Electrified_Population","Distance_to_Urban","Elevation","Major_Road_Dist","Population_density","RiverLakes_Dist"]));
Xnew = (X_non_standard_new - mu) ./ sigma;

yPredProbInitial = predict(mdl, Xnew);  % Predicted probabilities for all rows

% Add predicted probabilities to the original dataset
urbanisation_new.PredictedProbability = yPredProbInitial;  % Add as a new column

% Display the updated table
disp('Updated data with predicted probabilities:');

% Save the updated table to a CSV file for further analysis
writetable(urbanisation_new, 'updated_bi_urbanisation_with_predicted_probabilities_missing.csv');
