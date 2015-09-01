%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  �j���[�����l�b�g���[�N(���w�p�[�Z�v�g����)�ŉ�A���s���v���O����    %%%
%%%%  �֐� sum_of_square_error.m��output.m���g�p����                   %%% 
%%%%  2014/5/16                          Programmer: Yuki Kashiwaba   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% �p�����[�^�ݒ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �P���f�[�^�����G�N�Z���t�@�C���̖��O(�p�X���ʂ��Ă�t�H���_�ɒu���Ƃ�)
% �Ƃ肠�����C�f�t�H���g�ňȉ���4��p�ӂ��Ă���(�����ŗV�т�����Γ����悤�Ȍ`����Excel�t�@�C����ǉ����Ă�������)
input_file_name = 'square.xlsx';
%input_file_name = 'sine.xlsx';
%input_file_name = 'Heaviside.xlsx';
%input_file_name = 'abs.xlsx';

% �O���t�̃����W���w��
% �f�t�H���g
x_range = [-3,3];
y_range = [-3,3];
%�e�G�N�Z���t�@�C�����ɐݒ�
if strcmp(input_file_name, 'square.xlsx')
    x_range = [-1.5,1.5];
    y_range = [0,1.5];
    
elseif strcmp(input_file_name, 'sine.xlsx')
    x_range = [-3,3];
    y_range = [-1.5,1.5];
    
elseif strcmp(input_file_name, 'Heaviside.xlsx')
    x_range = [-1.5,1.5];
    y_range = [-0.2,1.2]; 
    
elseif strcmp(input_file_name, 'abs.xlsx')
    x_range = [-1.5,1.5];
    y_range = [-0.2,1.2];
end


ETA = 0.3;          % �w�K���Ł@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�������Ă����v
max_loop = 10000;    % �w�K���[�v�񐔁@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�@�������Ă����v
num_show = 100;      % num_show�񂲂ƂɊw�K�r���̃O���t��\������         �������Ă����v
num_input = 2;      % ���͑w���j�b�g��(�o�C�A�X�܂�)�@�@                 ��A�̏ꍇ2�ŌŒ�(���͂�x�̒l�̂�)
num_hidden = 6;     % �B�ꑍ���j�b�g��(�o�C�A�X�܂�)�@�@                 �������Ă����v(�����ǁC�o�C�A�X�����܂߂�̂�2�ȏ�͕K�v)
num_output = 1;     % �o�͑w���j�b�g���@�@�@�@�@�@�@�@�@                 ��A�̏ꍇ1�Œ�(�o�͂�x�ɑ΂���y�̒l�̂�)�@�@�@�@

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% �P���f�[�^�̓ǂݍ���
[ndata, text, alldata] = xlsread(input_file_name);

% �P���f�[�^�̕`��
xlist = transpose(ndata(:,1));
tlist = transpose(ndata(:,2));
plot(xlist,tlist,'o');
title('INPUT');
axis([x_range(1),x_range(2),y_range(1),y_range(2)])

N = length(xlist);      % �P���f�[�^�̑���
% �d�݂������_��(-1,1)�ɏ�����
w1 = rand(num_hidden,num_input);
w2 = rand(num_output,num_hidden);

% �o�̓��X�g�̏�����
ylist = zeros(num_output,N);  

%%%%%%%%%%%% �w�K�J�n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for loop = 1:max_loop,
    for n = 1:N,
        % �B��w�Əo�͑w�̊���a��������
        a1 = zeros(1,num_hidden);
        a2 = zeros(1,num_output);
        
        % �B��w�Əo�͑w�̏o�͂�������
        z = zeros(1,num_hidden);
        y = zeros(1,num_output);
        
        % �덷�̏�����
        d1 = zeros(1,num_hidden);
        d2 = zeros(1,num_output);
        
        % n�Ԗڂ̌P���f�[�^���Z�b�g
        % MATLAB�̐����C�z���0�Ԗڂ��g���Ȃ��̂ŁC�o�C�A�X����x1,z1���g��
        x = zeros(1,num_input);
        x(1) = 1;                       % �o�C�A�X���͏��1�ɐݒ�
        x(:,2:num_input) = xlist(n);
        
        % ���`�d�ɂ��B��w�̏o�͂��v�Z
        for j = 1:num_hidden,
            for i = 1:num_input,
                a1(j) = a1(j) + w1(j,i) * x(i);
            end
            z(j) = tanh(a1(j));
        end
        z(1) = 1;       % �B��w�̃o�C�A�X���͏��1�ɐݒ�
        
        % ���`�d�ɂ��o�͑w�̏o�͂��v�Z
        for k = 1:num_output,
            for j = 1:num_hidden,
                a2(k) = a2(k) + w2(k,j) * z(j);
            end
            y(k) = a2(k);            
        end
        
        % �o�͑w�̌덷���v�Z
        for k = 1:num_output,
            d2(k) = y(k) - tlist(k,n);
        end
        
        % �o�͑w�̌덷���t�`�d�����āC�B��w�̌덷�����߂�
        for j = 1:num_hidden,
            temp = 0.0;
            for k = 1:num_output,
                temp = temp + w2(k,j) * d2(k);
            end
            d1(j) = (1 - z(j)*z(j)) * temp;
        end
        
        % ���͑w�@-> �B��w�̏d��w1���X�V
        for j = 1:num_hidden,
            for i = 1:num_input
                w1(j,i) = w1(j,i) - ETA * d1(j) * x(i);
            end
        end
        
        % �B��w -> �o�͑w�̏d��w2���X�V
        for k = 1:num_output,
            for j = 1:num_hidden,
                w2(k,j) = w2(k,j) - ETA * d2(k) * z(j);
            end
        end     
    end
    
    
    % num_show�񂲂Ƃɓr�����ʂ��O���t�ɕ\��
    if rem(loop,num_show) == 0 || loop == 1
       fprintf('%d���[�v��\n',loop); 
       
       % �P���f�[�^�̊ex�ɑ΂���B��w�Əo�͑w�̏o�͂��v�Z
       % �����悤�ȏ������������猩�Â炭�Ȃ����̂Ŋ֐�output�������
       % �֐�output�͌��݂̏d��w�ŌP���f�[�^x_n�̏o�͂�Ԃ�
       % ����Ă邱�Ƃ͊w�K���̏��`�d�����Ɠ���
       for n = 1:N,
           ylist(:,n) = output(xlist, n, num_input, num_hidden, num_output, w1, w2);
       end
       
       % �w�K���̍ŏ����덷���o��
       error = sum_of_square_error(ylist,tlist,N,num_output);
       fprintf('�ŏ����덷 : %d\n',error)
       fprintf('\n')
       
       % �w�K�r���̃O���t��\��
       plot(xlist,ylist);
       hold on
       plot(xlist,tlist,'o');
       title(strcat('LOOP = ',num2str(loop)));
       axis([x_range(1),x_range(2),y_range(1),y_range(2)])
       hold off
       pause(0.05);
    end
end
%%%%%%%%%%%% �w�K�I�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% �P���f�[�^�̊ex�ɑ΂���o�͑w�̏o�͂��v�Z
ylist = zeros(num_output,N);         % �o�͑w�̏o�̓��X�g�̏�����
% �w�K���̏��`�d�Ƃ���Ă邱�Ƃ͓���
for n = 1:N,
    ylist(:,n) = output(xlist, n, num_input, num_hidden, num_output, w1, w2);
end

% �w�K��̍ŏ����덷���o��
error = sum_of_square_error(ylist,tlist,N,num_output);
fprintf('�ŏ����덷 : %d\n',error)

% �w�K��̌��ʂ��O���t�ɕ`��
% ���ŕ\�����̂͌P���f�[�^�̊ex
% ���ŕ\�����̂́C�w�K��̃O���t
plot(xlist,ylist);
hold on
plot(xlist,tlist,'o');
title('OUTPUT');
axis([x_range(1),x_range(2),y_range(1),y_range(2)])

