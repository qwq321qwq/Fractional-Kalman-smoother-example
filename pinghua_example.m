clear; close all;


%% ��ʼ������
n_iter = 200;       % ��������n_iter��ʱ��
n = 2;              % �������趨����ά�ȣ�[ĳ�������仯���]
m = 1;              % �۲�������趨һ��ά��

% �����������ݡ���ʵ����Ϊxreal���۲⵽���¶�����Ϊz
% �����������ʱֻ���õ�z��xrealֻ�����ԱȽ��ʱʹ��



xrate=0.1;                     % �仯��
xreal = (0:n_iter-1)+(0:n_iter-1)*xrate+10*sin(20*sin(0:n_iter-1));  
z = xreal + 100*sin(1000*randn(m,n_iter))+200*randn(m,n_iter);    % �۲�


% ��������z��Ӧ�ÿ������˲���ƽ�����Ԥ��
x_fwd = zeros(n, n_iter);       % ״̬����������������ǰ��Kalman�˲���Ľ��
x_fwd_old = zeros(n, n_iter);       % ״̬����������������ǰ��Kalman�˲���Ľ��
x_bwd = zeros(n, n_iter);       % ״̬����������������Kalmanǰ���˲��ͺ���ƽ����Ľ��
x_bwd_old = zeros(n, n_iter);       % ״̬����������������Kalmanǰ���˲��ͺ���ƽ����Ľ��
V_fwd = zeros(n, n, n_iter);    % x_fwd�ķ���
V_fwd_old = zeros(n, n, n_iter);    % x_fwd�ķ���
V_bwd = zeros(n, n, n_iter);    % x_bwd�ķ���
V_bwd_old = zeros(n, n, n_iter);    % x_bwd�ķ���
P = zeros(n, n, n_iter-1);      % �м����
P_old = zeros(n, n, n_iter-1);      % �м����

% �������˲�����ز���
% A��״̬ת�ƾ��󣬸�����һʱ��xֵԤ�⵱ǰxֵ
%   ����ֱ��ʹ������ʵ��x�仯��xrate��
%   ��ʵ�ϣ�����趨xrateͬ�����Եõ�������ͬ�Ľ��
kf_para.A=[1, xrate; 0, 1];
kf_para.H = zeros(m, n);        % �۲�ת�����󣬼���z=H*x
kf_para.H(1)=1;                 % �۲�x�ĵ�һ��ά��
kf_para.Q = [4e-4, 0; 0, 1e-4]; % Ԥ�����ķ���
kf_para.R = 0.25;               % �۲����ķ���0.25

%% ��һ�������ײ��֣���Ҫ��Pk��һ�ģ�Ȼ���Ķ�����
% fractional order alpha and its corresponding GL binomial coefficient 
% b = nchoosek(n,k) ����ʽϵ���󷨣��±��á��ò����ǣ�
% ��ʵ�õ���gamma(i+1)=gamma(i)*i���ƹ�Ȼ����
N = n_iter;
bino_fir = zeros(1,N);       % Differential order 0.7
alpha = 0.7;
bino_fir(1,1) = 1;
for i = 2:1:N
    % �ĳɷ����ף�������ϵ������-1�Ľ״���
    bino_fir(1,i) = (1-(alpha+1)/(i-1))*bino_fir(1,i-1);  
end
I = eye(2,2);

q = 1;                   % system noise mean
r = 1;                   % measure noise mean
% short memory length under GL fractional definition
L = N+1;



%% ����
% ǰ�򿨶����˲���Kalman Filter��
x_fwd(:,1)=[z(1); 0.9];         % ��ʼ��
V_fwd(:,:,1)=diag(rand(n,1));   % �����ʼ��

for ii = 2:1:n_iter
    
    
    rema = [0; 0]; % �ĳɷ�����
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
    rema_P = [0 0; 0 0]; % �ĳɷ�����
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
%         kf_para.A' + kf_para.Q;                         % PRMLʽ13.88
    
    P(:,:,ii-1) = (kf_para.A-bino_fir(1,2)*I)*V_fwd(:,:,ii-1)*(kf_para.A-bino_fir(1,2)*I)'+ ...
        kf_para.Q + rema_P;                               % �ĳɷ�����
    
    
    
    K = P(:,:,ii-1)*kf_para.H'/...
        (kf_para.H*P(:,:,ii-1)*kf_para.H'+kf_para.R);   % PRMLʽ13.92
    % state updating
    x_fwd(:,ii) = xtmp+K*(z(ii)-kf_para.H*xtmp);        % PRMLʽ13.89
    % estimation variance updating
    V_fwd(:,:,ii)=(eye(n)-K*kf_para.H)*P(:,:,ii-1);     % PRMLʽ13.90
    
%   �����������״�ԭ����
    xtmp_old = kf_para.A*x_fwd_old(:,ii-1);
    P_old(:,:,ii-1) = kf_para.A*V_fwd_old(:,:,ii-1)*...
        kf_para.A' + kf_para.Q;                         % PRMLʽ13.88
    
    K_old = P_old(:,:,ii-1)*kf_para.H'/...
        (kf_para.H*P_old(:,:,ii-1)*kf_para.H'+kf_para.R);   % PRMLʽ13.92
    
    x_fwd_old(:,ii) = xtmp_old+K*(z(ii)-kf_para.H*xtmp_old);        % PRMLʽ13.89
    V_fwd_old(:,:,ii)=(eye(n)-K*kf_para.H)*P_old(:,:,ii-1);     % PRMLʽ13.90
    
    
end
%%
% ���򿨶���ƽ����Kalman Smoother��
% ������ʽ����PRML��û��ֱ���г������������Ƶ�
x_bwd(:, n_iter) = x_fwd(:, n_iter);
x_bwd_old(:, n_iter) = x_fwd_old(:, n_iter);
V_bwd(:, :, n_iter) = V_fwd(:, :, n_iter);
V_bwd_old(:, :, n_iter) = V_fwd_old(:, :, n_iter);

for ii = n_iter-1:-1:1
    J=V_fwd(:,:,ii)*kf_para.A'/P(:,:,ii);               % PRMLʽ13.102
    
    x_bwd(:,ii)=x_fwd(:,ii)+J*(x_bwd(:,ii+1)-...
        kf_para.A*x_fwd(:,ii));                         % PRMLʽ13.100
    V_bwd(:,:,ii)=V_fwd(:,:,ii)+...
        J*(V_bwd(:,:,ii+1)-P(:,:,ii))*J';               % PRMLʽ13.101
    
    J_old=V_fwd_old(:,:,ii)*kf_para.A'/P_old(:,:,ii);               % PRMLʽ13.102
    
    x_bwd_old(:,ii)=x_fwd_old(:,ii)+J_old*(x_bwd_old(:,ii+1)-...
        kf_para.A*x_fwd_old(:,ii));                         % PRMLʽ13.100
    V_bwd_old(:,:,ii)=V_fwd_old(:,:,ii)+...
        J_old*(V_bwd_old(:,:,ii+1)-P_old(:,:,ii))*J_old';               % PRMLʽ13.101
    
end


%% �����ӡ
rgb = cbrewer('seq', 'Greens', 20, 'linear');% ����һ����ɫ������RGB����



figure; hold on; 
plot(xreal, 'k.'); plot(z,'bo'); plot(x_fwd(1,:),'g*'); 
% plot(q1, 'rd');
plot(x_bwd(1,:), 'rd');
% ylim([-1 1]);
leg=legend('��ʵ����', '�۲�����', '�������˲�',...
    '�������˲�+ƽ��', 'Location', 'NorthWest');
set(leg,'Fontsize',15);
title('�������˲���ƽ��Ч��ͼ', 'FontSize', 20);
xlabel('����������', 'FontSize', 20);
ylabel('����', 'FontSize', 20);



figure;
plot(x_fwd(2,:));
title('�˲�������ݱ仯���', 'FontSize', 20);

fprintf('****************************************************************\n');
fprintf('�۲�����ֵΪ��%5.3f��������Ϊ��%5.3f\n', ...
    mean(abs(z-xreal)), std(z-xreal));
fprintf('�˲�������ֵΪ��%5.3f��������Ϊ��%5.3f\n', ...
    mean(abs(x_fwd(1,:)-xreal)), std(x_fwd(1,:)-xreal));
fprintf('�˲�+ƽ��������ֵΪ��%5.3f��������Ϊ��%5.3f\n', ...
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
