function kaeri = output(xlist, n, num_input, num_hidden, num_output, w1, w2)

% BęwĆoÍwĚŤađúť
a1 = zeros(1,num_hidden);
a2 = zeros(1,num_output);

% BęwĆoÍwĚoÍđúť
z = zeros(1,num_hidden);
y = zeros(1,num_output);

% nÔÚĚPűf[^đZbg
x = zeros(1,num_input);
x(1) = 1;
x(:,2:num_input) = xlist(n);

% `dÉćčBęwĚoÍđvZ
for j = 1:num_hidden,
    for i = 1:num_input,
        a1(j) = a1(j) + w1(j,i) * x(i);
    end
    z(j) = tanh(a1(j));
end
z(1) = 1;       % BęwĚoCAX


% `dÉćčoÍwĚoÍđvZ
for k = 1:num_output,
    for j = 1:num_hidden,
        a2(k) = a2(k) + w2(k,j) * z(j);
    end
    y(k) = a2(k);
end

kaeri = y;









