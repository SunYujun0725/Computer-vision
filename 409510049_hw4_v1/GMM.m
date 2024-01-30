clc;
clear;

load('TestData.mat');
load('TestLabel.mat');
load('TrainData.mat');
load('TrainLabel.mat');

dim = 10;

Train_data = PCA_Train_data';
Test_data = PCA_Test_data(1:dim, :)'; 

num_train_label = [];

for i=1:size(train_label)
    temp = strsplit(train_label(i),"man");
    num_train_label(i) = str2num(temp(2)); 
end


for class=1:10
    
    fprintf("\n===========================\n");
    fprintf("This is class no.%d\n",class);
    
    X = Train_data(((class-1)*13+1):(class*13), 1:dim);
%%====================================================
%% STEP 2: Choose initial values for the parameters.
    
    % Set m to the number of data points
    m = size(X, 1); %data points

    k = 2;   % the number of clusters
    n = size(X, 2);  %the vector lenghts

   
    indeces = randperm(m);
    mu = X(indeces(1:k), :);

    sigma = [];

    % Use the overal covariance of the dataset as the initial variance for each cluster
    for (j = 1 : k)
        sigma{j} = cov(X);
    end
    
    phi = ones(1, k) * (1 / k);

    %%===================================================
    %% STEP 3: Run Expectation Maximization


    W = zeros(m, k);

    % Loop until convergence
    for (iter = 1:900)

        fprintf('  EM Iteration %d\n', iter);

        %%===============================================
        %% STEP 3a: Expectation

        pdf = zeros(m, k);

        for (j = 1 : k)
            pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
        end

        pdf_w = bsxfun(@times, pdf, phi);

        W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));

        %%===============================================
        %% STEP 3b: Maximization
        %%
        %% Calculate the probability for each data point for each distribution.

        prevMu = mu;    

        for (j = 1 : k)

            phi(j) = mean(W(:, j), 1);

            mu(j, :) = weightedAverage(W(:, j), X);

            sigma_k = zeros(n, n);

            % Subtract the cluster mean from all data points.
            % 加入雜訊 rand
            Xm = bsxfun(@minus, X+rand(13,10), mu(j, :));

            for (i = 1 : m)
                sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
            end

            sigma{j} = sigma_k ./ sum(W(:, j));
        end

        if (mu == prevMu)
            break
        end

    end
    pdf = zeros(size(Test_data,1), k);
    for j = 1 : k
            pdf(:, j) = gaussianND(Test_data, mu(j, :), sigma{j});
    end
    pdf_w = bsxfun(@times, pdf, phi);
    prb(:,class) = sum(pdf_w,2);
end


[~ ,result] = max(prb,[],2);

accurance = length(find(result==num_train_label'))/size(Test_data,1);
fprintf('\nThe Accuracy is %f %%\n',accurance*100);
