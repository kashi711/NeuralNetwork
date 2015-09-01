%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  ニューラルネットワーク(多層パーセプトロン)で回帰を行うプログラム    %%%
%%%%  関数 sum_of_square_error.mとoutput.mを使用する                   %%% 
%%%%  2014/5/16                          Programmer: Yuki Kashiwaba   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% パラメータ設定 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 訓練データを持つエクセルファイルの名前(パスが通ってるフォルダに置いとく)
% とりあえず，デフォルトで以下の4つを用意しておく(自分で遊びたければ同じような形式のExcelファイルを追加してください)
input_file_name = 'square.xlsx';
%input_file_name = 'sine.xlsx';
%input_file_name = 'Heaviside.xlsx';
%input_file_name = 'abs.xlsx';

% グラフのレンジを指定
% デフォルト
x_range = [-3,3];
y_range = [-3,3];
%各エクセルファイル毎に設定
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


ETA = 0.3;          % 学習率η　　　　　　　　　　　　　　　　　　　　　　いじっても大丈夫
max_loop = 10000;    % 学習ループ回数　　　　　　　　　　　　　　　　　　　いじっても大丈夫
num_show = 100;      % num_show回ごとに学習途中のグラフを表示する         いじっても大丈夫
num_input = 2;      % 入力層ユニット数(バイアス含み)　　                 回帰の場合2で固定(入力はxの値のみ)
num_hidden = 6;     % 隠れ総ユニット数(バイアス含み)　　                 いじっても大丈夫(だけど，バイアス項も含めるので2以上は必要)
num_output = 1;     % 出力層ユニット数　　　　　　　　　                 回帰の場合1固定(出力はxに対するyの値のみ)　　　　

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 訓練データの読み込み
[ndata, text, alldata] = xlsread(input_file_name);

% 訓練データの描写
xlist = transpose(ndata(:,1));
tlist = transpose(ndata(:,2));
plot(xlist,tlist,'o');
title('INPUT');
axis([x_range(1),x_range(2),y_range(1),y_range(2)])

N = length(xlist);      % 訓練データの総数
% 重みをランダム(-1,1)に初期化
w1 = rand(num_hidden,num_input);
w2 = rand(num_output,num_hidden);

% 出力リストの初期化
ylist = zeros(num_output,N);  

%%%%%%%%%%%% 学習開始 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for loop = 1:max_loop,
    for n = 1:N,
        % 隠れ層と出力層の活性aを初期化
        a1 = zeros(1,num_hidden);
        a2 = zeros(1,num_output);
        
        % 隠れ層と出力層の出力を初期化
        z = zeros(1,num_hidden);
        y = zeros(1,num_output);
        
        % 誤差δの初期化
        d1 = zeros(1,num_hidden);
        d2 = zeros(1,num_output);
        
        % n番目の訓練データをセット
        % MATLABの制約上，配列の0番目を使えないので，バイアス項はx1,z1を使う
        x = zeros(1,num_input);
        x(1) = 1;                       % バイアス項は常に1に設定
        x(:,2:num_input) = xlist(n);
        
        % 順伝播により隠れ層の出力を計算
        for j = 1:num_hidden,
            for i = 1:num_input,
                a1(j) = a1(j) + w1(j,i) * x(i);
            end
            z(j) = tanh(a1(j));
        end
        z(1) = 1;       % 隠れ層のバイアス項は常に1に設定
        
        % 順伝播により出力層の出力を計算
        for k = 1:num_output,
            for j = 1:num_hidden,
                a2(k) = a2(k) + w2(k,j) * z(j);
            end
            y(k) = a2(k);            
        end
        
        % 出力層の誤差を計算
        for k = 1:num_output,
            d2(k) = y(k) - tlist(k,n);
        end
        
        % 出力層の誤差を逆伝播させて，隠れ層の誤差を求める
        for j = 1:num_hidden,
            temp = 0.0;
            for k = 1:num_output,
                temp = temp + w2(k,j) * d2(k);
            end
            d1(j) = (1 - z(j)*z(j)) * temp;
        end
        
        % 入力層　-> 隠れ層の重みw1を更新
        for j = 1:num_hidden,
            for i = 1:num_input
                w1(j,i) = w1(j,i) - ETA * d1(j) * x(i);
            end
        end
        
        % 隠れ層 -> 出力層の重みw2を更新
        for k = 1:num_output,
            for j = 1:num_hidden,
                w2(k,j) = w2(k,j) - ETA * d2(k) * z(j);
            end
        end     
    end
    
    
    % num_show回ごとに途中結果をグラフに表示
    if rem(loop,num_show) == 0 || loop == 1
       fprintf('%dループ目\n',loop); 
       
       % 訓練データの各xに対する隠れ層と出力層の出力を計算
       % 同じような処理を書いたら見づらくなったので関数outputを作った
       % 関数outputは現在の重みwで訓練データx_nの出力を返す
       % やってることは学習時の順伝播部分と同じ
       for n = 1:N,
           ylist(:,n) = output(xlist, n, num_input, num_hidden, num_output, w1, w2);
       end
       
       % 学習中の最小二乗誤差を出力
       error = sum_of_square_error(ylist,tlist,N,num_output);
       fprintf('最小二乗誤差 : %d\n',error)
       fprintf('\n')
       
       % 学習途中のグラフを表示
       plot(xlist,ylist);
       hold on
       plot(xlist,tlist,'o');
       title(strcat('LOOP = ',num2str(loop)));
       axis([x_range(1),x_range(2),y_range(1),y_range(2)])
       hold off
       pause(0.05);
    end
end
%%%%%%%%%%%% 学習終了 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% 訓練データの各xに対する出力層の出力を計算
ylist = zeros(num_output,N);         % 出力層の出力リストの初期化
% 学習時の順伝播とやってることは同じ
for n = 1:N,
    ylist(:,n) = output(xlist, n, num_input, num_hidden, num_output, w1, w2);
end

% 学習後の最小二乗誤差を出力
error = sum_of_square_error(ylist,tlist,N,num_output);
fprintf('最小二乗誤差 : %d\n',error)

% 学習後の結果をグラフに描写
% ○で表されるのは訓練データの各x
% 線で表されるのは，学習後のグラフ
plot(xlist,ylist);
hold on
plot(xlist,tlist,'o');
title('OUTPUT');
axis([x_range(1),x_range(2),y_range(1),y_range(2)])

