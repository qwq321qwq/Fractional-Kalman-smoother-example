clear; close all;


%% 初始化参数
n_iter = 200;       % 计算连续n_iter个时刻
n = 2;              % 隐变量设定两个维度：[某参数，变化间隔]
m = 1;              % 观察变量仅设定一个维度

% 生成试验数据。真实序列为xreal，观测到的温度序列为z
% 下面迭代计算时只会用到z，xreal只在最后对比结果时使用



xrate=0.1;                     % 变化率
xreal = (0:n_iter-1)+(0:n_iter-1)*xrate+10*sin(20*sin(0:n_iter-1));  
z = xreal + 100*sin(1000*randn(m,n_iter))+200*randn(m,n_iter);    % 观测


% 基于序列z，应用卡尔曼滤波和平滑后的预测
x_fwd = zeros(n, n_iter);       % 状态（隐变量），经过前向Kalman滤波后的结果
x_fwd_old = zeros(n, n_iter);       % 状态（隐变量），经过前向Kalman滤波后的结果
x_bwd = zeros(n, n_iter);       % 状态（隐变量），经过Kalman前向滤波和后向平滑后的结果
x_bwd_old = zeros(n, n_iter);       % 状态（隐变量），经过Kalman前向滤波和后向平滑后的结果
V_fwd = zeros(n, n, n_iter);    % x_fwd的方差
V_fwd_old = zeros(n, n, n_iter);    % x_fwd的方差
V_bwd = zeros(n, n, n_iter);    % x_bwd的方差
V_bwd_old = zeros(n, n, n_iter);    % x_bwd的方差
P = zeros(n, n, n_iter-1);      % 中间变量
P_old = zeros(n, n, n_iter-1);      % 中间变量

% 卡尔曼滤波器相关参数
% A：状态转移矩阵，根据上一时刻x值预测当前x值
%   这里直接使用了真实的x变化率xrate，
%   事实上，随机设定xrate同样可以得到基本相同的结果
kf_para.A=[1, xrate; 0, 1];
kf_para.H = zeros(m, n);        % 观测转换矩阵，即：z=H*x
kf_para.H(1)=1;                 % 观察x的第一个维度
kf_para.Q = [4e-4, 0; 0, 1e-4]; % 预测误差的方差
kf_para.R = 0.25;               % 观察误差的方差0.25

%% 加一个分数阶部分，主要是Pk改一改，然后别的都正常
% fractional order alpha and its corresponding GL binomial coefficient 
% b = nchoosek(n,k) 二项式系数求法，下边用。用不到惹，
% 其实用到了gamma(i+1)=gamma(i)*i的推广然后倒推
N = n_iter;
bino_fir = zeros(1,N);       % Differential order 0.7
alpha = 0.7;
bino_fir(1,1) = 1;
for i = 2:1:N
    % 改成分数阶，这里是系数，有-1的阶次了
    bino_fir(1,i) = (1-(alpha+1)/(i-1))*bino_fir(1,i-1);  
end
I = eye(2,2);

q = 1;                   % system noise mean
r = 1;                   % measure noise mean
% short memory length under GL fractional definition
L = N+1;



%% 迭代
% 前向卡尔曼滤波（Kalman Filter）
x_fwd(:,1)=[z(1); 0.9];         % 初始化
V_fwd(:,:,1)=diag(rand(n,1));   % 随机初始化

for ii = 2:1:n_iter
    
    
    rema = [0; 0]; % 改成分数阶
    if ii>L
        for i = 2:1:L+1
            rema = rema + bino_fir(1,i)*x_fwd(:,ii+1-i);
        end
    else
        for i = 2:1:ii
            rema = rema + bino_fir(1,i)*x_fwd(:,ii+1-i);
        end
    end

    xtmp = kf_para.A*x_fwd(:,ii)-rema;
    % prediction error covariance: P_pred
    rema_P = [0 0; 0 0]; % 改成分数阶
    if ii>L+1
        for i = 3:1:L+2
            rema_P = rema_P + bino_fir(1,i)*V_fwd(:,:,ii-i+1)*bino_fir(1,i)';
        end
    else
        for i = 3:1:ii
            rema_P = rema_P + bino_fir(1,i)*V_fwd(1,ii+1-i)*bino_fir(1,i)';
        end
    end

%     deata_x = kf_para.A*x_fwd(:,ii-1);

%     xtmp = kf_para.A*x_fwd(:,ii-1);


%     P(:,:,ii-1) = kf_para.A*V_fwd(:,:,ii-1)*...
%         kf_para.A' + kf_para.Q;                         % PRML式13.88
    
    P(:,:,ii-1) = (kf_para.A-bino_fir(1,2)*I)*V_fwd(:,:,ii-1)*(kf_para.A-bino_fir(1,2)*I)'+ ...
        kf_para.Q + rema_P;                               % 改成分数阶
    
    
    
    K = P(:,:,ii-1)*kf_para.H'/...
        (kf_para.H*P(:,:,ii-1)*kf_para.H'+kf_para.R);   % PRML式13.92
    % state updating
    x_fwd(:,ii) = xtmp+K*(z(ii)-kf_para.H*xtmp);        % PRML式13.89
    % estimation variance updating
    V_fwd(:,:,ii)=(eye(n)-K*kf_para.H)*P(:,:,ii-1);     % PRML式13.90
    
%   这里是整数阶次原来的
    xtmp_old = kf_para.A*x_fwd_old(:,ii-1);
    P_old(:,:,ii-1) = kf_para.A*V_fwd_old(:,:,ii-1)*...
        kf_para.A' + kf_para.Q;                         % PRML式13.88
    
    K_old = P_old(:,:,ii-1)*kf_para.H'/...
        (kf_para.H*P_old(:,:,ii-1)*kf_para.H'+kf_para.R);   % PRML式13.92
    
    x_fwd_old(:,ii) = xtmp_old+K*(z(ii)-kf_para.H*xtmp_old);        % PRML式13.89
    V_fwd_old(:,:,ii)=(eye(n)-K*kf_para.H)*P_old(:,:,ii-1);     % PRML式13.90
    
    
end
%%
% 反向卡尔曼平滑（Kalman Smoother）
% 这两个式子在PRML中没有直接列出，可以自行推导
x_bwd(:, n_iter) = x_fwd(:, n_iter);
x_bwd_old(:, n_iter) = x_fwd_old(:, n_iter);
V_bwd(:, :, n_iter) = V_fwd(:, :, n_iter);
V_bwd_old(:, :, n_iter) = V_fwd_old(:, :, n_iter);

for ii = n_iter-1:-1:1
    J=V_fwd(:,:,ii)*kf_para.A'/P(:,:,ii);               % PRML式13.102
    
    x_bwd(:,ii)=x_fwd(:,ii)+J*(x_bwd(:,ii+1)-...
        kf_para.A*x_fwd(:,ii));                         % PRML式13.100
    V_bwd(:,:,ii)=V_fwd(:,:,ii)+...
        J*(V_bwd(:,:,ii+1)-P(:,:,ii))*J';               % PRML式13.101
    
    J_old=V_fwd_old(:,:,ii)*kf_para.A'/P_old(:,:,ii);               % PRML式13.102
    
    x_bwd_old(:,ii)=x_fwd_old(:,ii)+J_old*(x_bwd_old(:,ii+1)-...
        kf_para.A*x_fwd_old(:,ii));                         % PRML式13.100
    V_bwd_old(:,:,ii)=V_fwd_old(:,:,ii)+...
        J_old*(V_bwd_old(:,:,ii+1)-P_old(:,:,ii))*J_old';               % PRML式13.101
    
end


%% 结果打印
rgb = cbrewer('seq', 'Greens', 20, 'linear');% 生成一个配色方案的RGB矩阵



figure; hold on; 
plot(xreal, 'k.'); plot(z,'bo'); plot(x_fwd(1,:),'g*'); 
% plot(q1, 'rd');
plot(x_bwd(1,:), 'rd');
% ylim([-1 1]);
leg=legend('真实数据', '观测数据', '卡尔曼滤波',...
    '卡尔曼滤波+平滑', 'Location', 'NorthWest');
set(leg,'Fontsize',15);
title('卡尔曼滤波和平滑效果图', 'FontSize', 20);
xlabel('样本点序列', 'FontSize', 20);
ylabel('数据', 'FontSize', 20);



figure;
plot(x_fwd(2,:));
title('滤波后的数据变化间隔', 'FontSize', 20);

fprintf('****************************************************************\n');
fprintf('观测误差均值为：%5.3f；均方差为：%5.3f\n', ...
    mean(abs(z-xreal)), std(z-xreal));
fprintf('滤波后误差均值为：%5.3f；均方差为：%5.3f\n', ...
    mean(abs(x_fwd(1,:)-xreal)), std(x_fwd(1,:)-xreal));
fprintf('滤波+平滑后误差均值为：%5.3f；均方差为：%5.3f\n', ...
    mean(abs(x_bwd(1,:)-xreal)), std(x_bwd(1,:)-xreal));
fprintf('****************************************************************\n');




figure; hold on; 
plot(xreal, 'k.'); 
plot(x_fwd(1,:),'Color', rgb(4,:),'linewidth',2); 
plot(x_bwd_old(1,:),'Color', rgb(10,:),'linewidth',2); 
plot(x_bwd(1,:),'Color', rgb(20,:),'linewidth',2); 

% plot(q1, 'rd');


leg=legend('Current data', ' Kalman filter',...
    'Kalman smoother','fractional Kalman smoother', 'Location', 'NorthWest');
set(leg,'Fontsize',13);
title('Kalman filtering and smoothing', 'FontSize', 18);
xlabel('Sample points sequence', 'FontSize', 18);
ylabel('Trajectory data', 'FontSize', 18);
