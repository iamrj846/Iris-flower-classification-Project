%% Initialization
clear ; close all; clc

%%load data
data = csvread("iris.txt");
X = data(:, 1:4);
y = data(:, 5);
test_X = csvread("test.txt");
num_labels = 3;

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to X and test_X is done by oneVsAll
 

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;



%% ============ Part 2: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Accuracy for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n\n', mean(double(pred == y)) * 100);

%% ================ Part 3: Predict for One-Vs-All ================

fprintf("Predictions for the test data -> \n\n");

pred = predictOneVsAll(all_theta, test_X);
for i=1:size(test_X, 1),
  if(pred(i) == 1),
    fprintf("%f ", test_X(i, :));
    fprintf("    -> Iris-setosa\n\n");
   end;
  if(pred(i) == 2),
    fprintf("%f ", test_X(i, :));
    fprintf("    -> Iris-versicolor\n\n");
   end;
  if(pred(i) == 3),
    fprintf("%f ", test_X(i, :));
    fprintf("    -> Iris-virginica\n\n");
   end;
end;

