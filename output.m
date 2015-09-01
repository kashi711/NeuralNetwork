function kaeri = output(xlist, n, num_input, num_hidden, num_output, w1, w2)

% ‰B‚ê‘w‚Æo—Í‘w‚ÌŠˆ«a‚ğ‰Šú‰»
a1 = zeros(1,num_hidden);
a2 = zeros(1,num_output);

% ‰B‚ê‘w‚Æo—Í‘w‚Ìo—Í‚ğ‰Šú‰»
z = zeros(1,num_hidden);
y = zeros(1,num_output);

% n”Ô–Ú‚ÌŒP—ûƒf[ƒ^‚ğƒZƒbƒg
x = zeros(1,num_input);
x(1) = 1;
x(:,2:num_input) = xlist(n);

% ‡“`”d‚É‚æ‚è‰B‚ê‘w‚Ìo—Í‚ğŒvZ
for j = 1:num_hidden,
    for i = 1:num_input,
        a1(j) = a1(j) + w1(j,i) * x(i);
    end
    z(j) = tanh(a1(j));
end
z(1) = 1;       % ‰B‚ê‘w‚ÌƒoƒCƒAƒX€


% ‡“`”d‚É‚æ‚èo—Í‘w‚Ìo—Í‚ğŒvZ
for k = 1:num_output,
    for j = 1:num_hidden,
        a2(k) = a2(k) + w2(k,j) * z(j);
    end
    y(k) = a2(k);
end

kaeri = y;









