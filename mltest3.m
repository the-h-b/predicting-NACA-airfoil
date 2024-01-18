% Load data from the CSV file
data = readmatrix('Book1.csv');
X = data(:, 1:end-2);  % Excluding the Cl and Cd columns from input features
Y = data(:, 6:7);  % Extracting the Cl and Cd columns as output

% Normalize the input features
X_normalized = normalize(X);

% Calculate the angle of attack from the input features
angle_of_attack = rad2deg(atan2(X_normalized(:, 2), X_normalized(:, 1)));

% Create the ANN model
hiddenLayerSize = 10;
maxIterations = 100;
learningRate = 0.01;
net = feedforwardnet(hiddenLayerSize);
net.trainParam.epochs = maxIterations;
net.trainParam.lr = learningRate;

% Train the neural network
net = train(net, X_normalized', Y');

% Generate predictions using the trained model
predictions = net(X_normalized');

% Extract the lift coefficient (Cl) and drag coefficient (Cd) values
CL_values = Y(:, 1);
CD_values = Y(:, 2);

% Display the first 10 rows of 'Cl' and 'Cd' values
disp('First 10 rows of Lift Coefficient (Cl) values:');
disp(CL_values(1:10));
disp('First 10 rows of Drag Coefficient (Cd) values:');
disp(CD_values(1:10));

% Plot the angle of attack with respect to lift coefficient (Cl) and the line plot for predictions
figure;
scatter(angle_of_attack, CL_values, 'o', 'DisplayName', 'Data');
hold on;
plot(angle_of_attack, predictions(1, :), 'r', 'DisplayName', 'Predictions');
xlabel('Angle of Attack (degrees)');
ylabel('Lift Coefficient (Cl)');
legend('Location', 'best');
title('Angle of Attack vs. Lift Coefficient (Cl)');
hold off;

% Plot the angle of attack with respect to drag coefficient (Cd) and the line plot for predictions
figure;
scatter(angle_of_attack, CD_values, 'o', 'DisplayName', 'Data');
hold on;
plot(angle_of_attack, predictions(2, :), 'r', 'DisplayName', 'Predictions');
xlabel('Angle of Attack (degrees)');
ylabel('Drag Coefficient (Cd)');
legend('Location', 'best');
title('Angle of Attack vs. Drag Coefficient (Cd)');
hold off;
