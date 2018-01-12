function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% C_tests = [0.01 0.03 0.1 0.3 1 3 10 30];
% sigma_tests = [0.01 0.03 0.1 0.3 1 3 10 30];

C_tests = [0.01 0.1 0.5 1.0 10 50 100 1000];
sigma_tests =  [0.01 0.1 0.5 1.0 10 50 100 1000];
ctl = size(C_tests, 2);
stl = size(sigma_tests, 2);

ini_err = numel(yval);
for i=1:ctl,
    for j=1:stl,
        c_test = C_tests(i);
        s_test = sigma_tests(j);
        test_model = svmTrain(X, y, c_test, @(x1, x2) gaussianKernel(x1, x2, s_test));
        predictions = svmPredict(test_model, Xval);
        temp_err = mean(double(predictions ~= yval));
        if (ini_err> temp_err)
            % disp('yayayayaya')
        	ini_err = temp_err;
        	C = c_test;
        	sigma = s_test;
        end
    end
end


% 123123
% C
% sigma
% 12312312

% =========================================================================

end
