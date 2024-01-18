filename = 'Book1.xlsx';
data = xlsread(filename);
X = data(:, 1:end-1);
Y = data(:, end);
X = normalize(X);

hiddenLayerSizes = [10, 5];
net = feedforwardnet(hiddenLayerSizes);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 10;

% Divide the dataset into training, validation, and testing sets
[trainInd, valInd, testInd] = divideblock(size(X,1), 0.6, 0.20, 0.20);
X_train = X(trainInd,:);
Y_train = Y(trainInd,:);
X_val = X(valInd,:);
Y_val = Y(valInd,:);
X_test = X(testInd,:);
Y_test = Y(testInd,:);

% Train the neural network
[net, tr] = train(net, X_train', Y_train');

% Predict outputs for training, validation, and testing sets
Y_train_pred = net(X_train');
Y_val_pred = net(X_val');
Y_test_pred = net(X_test');

% Calculate evaluation metrics for training set
mse_train = mean((Y_train_pred - Y_train').^2);
rmse_train = sqrt(mse_train);
r2_train = corr(Y_train_pred', Y_train)^2;
mae_train = mean(abs(Y_train_pred - Y_train'));

% Calculate evaluation metrics for validation set
mse_val = mean((Y_val_pred - Y_val').^2);
rmse_val = sqrt(mse_val);
r2_val = corr(Y_val_pred', Y_val)^2;
mae_val = mean(abs(Y_val_pred - Y_val'));

% Calculate evaluation metrics for testing set
mse_test = mean((Y_test_pred - Y_test').^2);
rmse_test = sqrt(mse_test);
r2_test = corr(Y_test_pred', Y_test)^2;
mae_test = mean(abs(Y_test_pred - Y_test'));

% Display evaluation metrics for training set
disp('Training Results:');
disp(['Mean Squared Error: ' num2str(mse_train)]);
disp(['Root Mean Squared Error: ' num2str(rmse_train)]);
disp(['R^2 (Coefficient of Determination): ' num2str(r2_train)]);
disp(['Mean Absolute Error: ' num2str(mae_train)]);

% Display evaluation metrics for validation set
disp('Validation Results:');
disp(['Mean Squared Error: ' num2str(mse_val)]);
disp(['Root Mean Squared Error: ' num2str(rmse_val)]);
disp(['R^2 (Coefficient of Determination): ' num2str(r2_val)]);
disp(['Mean Absolute Error: ' num2str(mae_val)]);

% Display evaluation metrics for testing set
disp('Testing Results:');
disp(['Mean Squared Error: ' num2str(mse_test)]);
disp(['Root Mean Squared Error: ' num2str(rmse_test)]);
disp(['R^2 (Coefficient of Determination): ' num2str(r2_test)]);
disp(['Mean Absolute Error: ' num2str(mae_test)]);
