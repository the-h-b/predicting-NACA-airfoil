% Load the dataset from 'Book1.csv' and preserve column headers as variable names
data = readtable('Book1.csv', 'VariableNamingRule', 'preserve');

% Extract input features (X) and target variable (Y - Reynolds number)
X = data{:, 1:end-1};
Y = data.Reynolds_number; % Assuming 'Reynolds_number' is the column name in the dataset

% Normalize the input features
X = normalize(X);

% Split the data into training, validation, and testing sets
[trainInd, valInd, testInd] = divideblock(size(X, 1), 0.6, 0.20, 0.20);
X_train = X(trainInd, :);
Y_train = Y(trainInd, :);
X_val = X(valInd, :);
Y_val = Y(valInd, :);
X_test = X(testInd, :);
Y_test = Y(testInd, :);

% Define the neural network architecture
hiddenLayerSizes = [10, 5];
net = feedforwardnet(hiddenLayerSizes);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 10;

% Train the neural network
[net, tr] = train(net, X_train', Y_train');

% Test the trained network on the testing set
Y_test_pred = net(X_test');
mse_test = mean((Y_test_pred - Y_test').^2);
rmse_test = sqrt(mse_test);
r2_test = corr(Y_test_pred', Y_test)^2;
mae_test = mean(abs(Y_test_pred - Y_test'));

% Displaying the testing results
disp('Testing Results:');
disp(['Mean Squared Error: ' num2str(mse_test)]);
disp(['Root Mean Squared Error: ' num2str(rmse_test)]);
disp(['R^2 (Coefficient of Determination): ' num2str(r2_test)]);
disp(['Mean Absolute Error: ' num2str(mae_test)]);

% Create a table to display the predicted and actual Reynolds numbers
ResultsTable = table(Y_test, Y_test_pred', 'VariableNames', {'Actual_ReynoldsNumber', 'Predicted_ReynoldsNumber'});
disp(ResultsTable);
