data = readtable('Book1.csv', 'VariableNamingRule', 'preserve');

% Extract input features (X) and target variables (Y)
x = data{:, {'alpha', 'position_of_camber', 'thickness', 'camber', 'Reynolds_number'}};
y = data{:, {'cl', 'cd', 'Reynolds_number'}};

% Normalize the input features
x(:, 1:4) = normalize(x(:, 1:4));

% Split the data 
[trainInd, testInd] = dividerand(size(x, 1), 0.8, 0.2);
X_train = x(trainInd, :);
Y_train = y(trainInd, :);
X_test = x(testInd, :);
Y_test = y(testInd, :);

% Define the neural network architecture
hiddenLayerSize = 20; 
net = feedforwardnet(hiddenLayerSize);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 10;

% k-fold cross-validation
k = 5;
cv_mse_cl = zeros(1, k);
cv_mse_cd = zeros(1, k);
Y_test_all_cl = cell(k, 1);
Y_test_pred_all_cl = cell(k, 1);
Y_test_all_cd = cell(k, 1);
Y_test_pred_all_cd = cell(k, 1);

for i = 1:k
    % Dividing the data into k subsets for cross-validation
    cv_idx = crossvalind('Kfold', size(x, 1), k);
    trainIdx = find(cv_idx ~= i);
    testIdx = find(cv_idx == i);
    
    % Spliting the data into training and testing sets for this fold
    X_train_cv = x(trainIdx, :);
    Y_train_cv = y(trainIdx, :);
    X_test_cv = x(testIdx, :);
    Y_test_cv = y(testIdx, :);
    
    % Defining and training the neural networks for cl and cd
    netCl_cv = train(net, X_train_cv(:, 1:4)', Y_train_cv(:, 1)');
    netCd_cv = train(net, X_train_cv(:, 1:4)', Y_train_cv(:, 2)');
    
    % Predictin cl and cd values for the testing set 
    Y_test_pred_cl_cv = netCl_cv(X_test_cv(:, 1:4)');
    Y_test_pred_cd_cv = netCd_cv(X_test_cv(:, 1:4)');
    
    % Storing the actual and predicted values for cl and cd for this fold
    Y_test_all_cl{i} = Y_test_cv(:, 1)';
    Y_test_pred_all_cl{i} = Y_test_pred_cl_cv';
    Y_test_all_cd{i} = Y_test_cv(:, 2)';
    Y_test_pred_all_cd{i} = Y_test_pred_cd_cv';
    
    % Calculating the mean squared error (MSE) for this fold
    cv_mse_cl(i) = mean((Y_test_pred_cl_cv - Y_test_cv(:, 1)').^2);
    cv_mse_cd(i) = mean((Y_test_pred_cd_cv - Y_test_cv(:, 2)').^2);
end

% Calculating the average MSE over all folds for cl and cd
avg_mse_cl = mean(cv_mse_cl);
avg_mse_cd = mean(cv_mse_cd);

fprintf('Average Mean Squared Error (Cl) over Cross-Validation: %.4f\n', avg_mse_cl);
fprintf('Average Mean Squared Error (Cd) over Cross-Validation: %.4f\n', avg_mse_cd);

% Concatenate the actual and predicted values from all folds into matrices
maxLength_cl = max(cellfun(@numel, Y_test_all_cl));
maxLength_cd = max(cellfun(@numel, Y_test_all_cd));
Y_test_all_cl_padded = NaN(length(Y_test_all_cl), maxLength_cl);
Y_test_pred_all_cl_padded = NaN(length(Y_test_pred_all_cl), maxLength_cl);
Y_test_all_cd_padded = NaN(length(Y_test_all_cd), maxLength_cd);
Y_test_pred_all_cd_padded = NaN(length(Y_test_pred_all_cd), maxLength_cd);

for i = 1:k
    Y_test_all_cl_padded(i, 1:numel(Y_test_all_cl{i})) = Y_test_all_cl{i};
    Y_test_pred_all_cl_padded(i, 1:numel(Y_test_pred_all_cl{i})) = Y_test_pred_all_cl{i};
    Y_test_all_cd_padded(i, 1:numel(Y_test_all_cd{i})) = Y_test_all_cd{i};
    Y_test_pred_all_cd_padded(i, 1:numel(Y_test_pred_all_cd{i})) = Y_test_pred_all_cd{i};
end

% Creating a table with actual and predicted values for cl and cd
cl_cd_table = table(Y_test_all_cl_padded(:), Y_test_pred_all_cl_padded(:), Y_test_all_cd_padded(:), Y_test_pred_all_cd_padded(:), ...
    'VariableNames', {'Actual_Cl', 'Predicted_Cl', 'Actual_Cd', 'Predicted_Cd'});
% Output
fprintf('\n------ Results Table ------\n');
disp(cl_cd_table);
fprintf('\nAverage Mean Squared Error (Cl) over Cross-Validation: %.4f\n', avg_mse_cl);
fprintf('Average Mean Squared Error (Cd) over Cross-Validation: %.4f\n', avg_mse_cd);

% Plot scatter plot for actual and predicted cl values
figure;
scatter(Y_test_all_cl_padded(:), Y_test_pred_all_cl_padded(:), 'b', 'filled');
xlabel('Actual Cl Values');
ylabel('Predicted Cl Values');
title('Actual Cl vs. Predicted Cl');

% Plot scatter plot for actual and predicted cd values
figure;
scatter(Y_test_all_cd_padded(:), Y_test_pred_all_cd_padded(:), 'r', 'filled');
xlabel('Actual Cd Values');
ylabel('Predicted Cd Values');
title('Actual Cd vs. Predicted Cd');
