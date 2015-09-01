function kaeri = output(xlist, n, num_input, num_hidden, num_output, w1, w2)

% �B��w�Əo�͑w�̊���a��������
a1 = zeros(1,num_hidden);
a2 = zeros(1,num_output);

% �B��w�Əo�͑w�̏o�͂�������
z = zeros(1,num_hidden);
y = zeros(1,num_output);

% n�Ԗڂ̌P���f�[�^���Z�b�g
x = zeros(1,num_input);
x(1) = 1;
x(:,2:num_input) = xlist(n);

% ���`�d�ɂ��B��w�̏o�͂��v�Z
for j = 1:num_hidden,
    for i = 1:num_input,
        a1(j) = a1(j) + w1(j,i) * x(i);
    end
    z(j) = tanh(a1(j));
end
z(1) = 1;       % �B��w�̃o�C�A�X��


% ���`�d�ɂ��o�͑w�̏o�͂��v�Z
for k = 1:num_output,
    for j = 1:num_hidden,
        a2(k) = a2(k) + w2(k,j) * z(j);
    end
    y(k) = a2(k);
end

kaeri = y;









