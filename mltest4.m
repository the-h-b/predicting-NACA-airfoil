data = readmatrix('Book1.csv');
X = data(:, 1:end-1);
Y = data(:, end);
X = normalize(X);

hiddenLayerSize = 10;
maxIterations = 100;
learningRate = 0.01;
net = feedforwardnet(hiddenLayerSize);

net.trainParam.epochs = maxIterations;
net.trainParam.lr = learningRate;
[trainInd, valInd, testInd] = divideblock(size(X,1), 0.6, 0.20, 0.20);
X_train = X(trainInd,:);
Y_train = Y(trainInd,:);
X_val = X(valInd,:);
Y_val = Y(valInd,:);
X_test = X(testInd,:);
Y_test = Y(testInd,:);

hiddenLayerSizes = [10, 5];
net = feedforwardnet(hiddenLayerSizes);
net.trainFcn = 'trainlm';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 10;

[net, tr] = train(net, X_train', Y_train');
Y_train_pred = net(X_train');
Y_val_pred = net(X_val');
Y_test_pred = net(X_test');
mse_train = mean((Y_train_pred - Y_train').^2);
mse_val = mean((Y_val_pred - Y_val').^2);
mse_test = mean((Y_test_pred - Y_test').^2);

rmse_train = sqrt(mse_train);
rmse_val = sqrt(mse_val);
rmse_test = sqrt(mse_test);

r2_train = corr(Y_train_pred', Y_train)^2;
r2_val = corr(Y_val_pred', Y_val)^2;
r2_test = corr(Y_test_pred', Y_test)^2;

mae_train = mean(abs(Y_train_pred - Y_train'));
mae_val = mean(abs(Y_val_pred - Y_val'));
mae_test = mean(abs(Y_test_pred - Y_test'));


figure;
histogram(Y_train_pred - Y_train', 'Normalization', 'pdf');
hold on;
histogram(Y_val_pred - Y_val', 'Normalization', 'pdf');
histogram(Y_test_pred - Y_test', 'Normalization', 'pdf');
legend('Training', 'Validation', 'Testing');
xlabel('Error');
ylabel('PDF');
title('Error Histogram');

net = train(net, X', Y');
predictions = net(X');
mse = mean((predictions - Y').^2);
disp(['Mean Squared Error: ' num2str(mse)]);



% Testing the trained network on the testing set
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
