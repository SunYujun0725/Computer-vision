
%%======================================================
%% STEP 1a: Generate data from two 2D distributions.


mu1 = [1 2];        %第一個分布的mean vector
sigma1 = [ 3 .2;    %第一個分布的convariance vector
          .2  2];
m1 = 200;           %第一個分布的data數量

mu2 = [-1 -2];      %第二個分布的mean vector
sigma2 = [2 0;      %第二個分布的convariance vector
          0 1];
m2 = 100;           %第二個分布的data數量

R1 = chol(sigma1);                      % Cholesky分解用於生成相應的data
X1 = randn(m1, 2) * R1;                 % 從標準正態分布生成data
X1 = X1 + repmat(mu1, size(X1, 1), 1);  % 將生成的data平移至目標分布

R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);

X = [X1; X2];       % 將兩個分布的data合併為X

%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

figure(1); % 繪製第一張圖

hold off;
plot(X1(:, 1), X1(:, 2), 'bo');   % 繪製第一個分布的data並將data標記為藍色圓點
hold on;
plot(X2(:, 1), X2(:, 2), 'ro');   % 繪製第二個分布的data並將data標記為紅色圓點

set(gcf,'color','white')          % 設置圖形背景為白色

gridSize = 100;                 % 定義網格大小，即每個維度上的點數
u = linspace(-6, 6, gridSize);  % 生成一個包含100個在-6到6之間等間隔的點的數組
[A B] = meshgrid(u, u);         % 使用meshgrid函數生成二維網格，A和B是矩陣，包含了u中對應x和y值的坐標
gridX = [A(:), B(:)];           % 將A和B矩陣的列堆疊，形成一個包含網格上所有點坐標的矩陣gridX

z1 = gaussianND(gridX, mu1, sigma1);    % 利用gaussianND function計算第一個分布的概率密度
z2 = gaussianND(gridX, mu2, sigma2);    % 利用gaussianND function計算第二個分布的概率密度

%將計算得到的概率密度z1, z2從一維數組轉換為二維矩陣Z1, Z2
%以便後續在圖形上進行等高線繪製
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

[C, h] = contour(u, u, Z1);     % 繪製第一個分布的概率密度等高線
[C, h] = contour(u, u, Z2);     % 繪製第二個分布的概率密度等高線

axis([-6 6 -6 6])   % 設置坐標軸範圍
title('Original Data and PDFs');


%%====================================================
%% STEP 2: Choose initial values for the parameters.

m = size(X, 1);

k = 2;  % GMM中的分量數
n = 2;  % data的維度

indeces = randperm(m);
mu = X(indeces(1:k), :);    % 從data中隨機選擇初始mean value

sigma = [];

for (j = 1 : k)
    sigma{j} = cov(X);      % 初始covariance matrix為data的協方差
end

phi = ones(1, k) * (1 / k); % 初始權重相等

%%===================================================
%% STEP 3: Run Expectation Maximization

W = zeros(m, k);    % 設置權重矩陣

for (iter = 1:1000)
    
    fprintf('  EM Iteration %d\n', iter);

    %%===============================================
    %% STEP 3a: Expectation
    pdf = zeros(m, k);
    
    for (j = 1 : k)

        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});  % 計算每個data對每個分量的概率密度
    end

    pdf_w = bsxfun(@times, pdf, phi);   % 乘以權重
    

    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));    % 歸一化權重矩陣
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    prevMu = mu;    

    for (j = 1 : k)

        phi(j) = mean(W(:, j), 1);  % 更新權重，使用每個分布的樣本權重的平均值
        
        mu(j, :) = weightedAverage(W(:, j), X); % 使用加權平均更新每個分布的mean value
        
        sigma_k = zeros(n, n);

        Xm = bsxfun(@minus, X, mu(j, :));

        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));    % 使用加權更新每個分布的協方差矩陣
        end

        sigma{j} = sigma_k ./ sum(W(:, j));     % 歸一化得到每個分布的新協方差矩陣
    end
    
    if (mu == prevMu)   % 如果均值不再變化，結束EM算法的迭代
        break
    end

end

%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.

figure(2);  % 繪製第二張圖
hold off;
plot(X1(:, 1), X1(:, 2), 'bo');
hold on;
plot(X2(:, 1), X2(:, 2), 'ro');

set(gcf,'color','white') 

plot(mu1(1), mu1(2), 'kx');    % 在data散點圖上標記第一個高斯分布的mean value位置
plot(mu2(1), mu2(2), 'kx');    % 在data散點圖上標記第二個高斯分布的mean value位置

gridSize = 100;
u = linspace(-6, 6, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% 利用更新完的mean與covariance計算兩個分布的概率密度
z1 = gaussianND(gridX, mu(1, :), sigma{1}); 
z2 = gaussianND(gridX, mu(2, :), sigma{2});

Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

[C, h] = contour(u, u, Z1);     % 繪製更新後第一個分布的概率密度等高線
[C, h] = contour(u, u, Z2);     % 繪製更新後第二個分布的概率密度等高線
axis([-6 6 -6 6])

title('Original Data and Estimated PDFs');