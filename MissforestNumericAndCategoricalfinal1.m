clc;
close all;
clear;
%%
% X an n×p table
opts = detectImportOptions('AutoData.csv','NumHeaderLines',0);
X_original = readtable('AutoData.csv',opts);
%%
% n = 8*q
ratio = [8 2 1 0.5];
ind = 1;
q = size(X_original,2);
n = ratio(ind)*q;
X_true = X_original(1:n,1:q);
varNames = X_true.Properties.VariableNames;

% Extract numerical features
array_numeric = varfun(@isnumeric,X_true,'output','uniform');
index_notnumeric = find(array_numeric==0);
X_numeric = removevars(X_true,index_notnumeric);
varNames_numeric = X_numeric.Properties.VariableNames;

% Extract text features
array_numeric = varfun(@isnumeric,X_true,'output','uniform');
array_datetime = varfun(@isdatetime,X_true,'output','uniform');
index_numeric = find(array_numeric==1);
index_datetime = find(array_datetime==1);
index_nottext = [index_numeric index_datetime];
X_text = removevars(X_true,index_nottext);
varNames_text = X_text.Properties.VariableNames;
            
% Extract numerical indices and text indices
index_numeric = find(array_numeric==1);
index_text = find(array_numeric==0);

% Data Normalization
X_numeric_array = table2array(X_numeric);
indices_nonnan = find(isnan(X_numeric_array)==0);
Y = X_numeric_array;
Y(indices_nonnan) = (X_numeric_array(indices_nonnan) - min(X_numeric_array(indices_nonnan))) / ( max(X_numeric_array(indices_nonnan)) - min(X_numeric_array(indices_nonnan)));
X_updated = array2table(Y);
% Displays the original table with column names
for i = 1:size(X_updated,2)
    X_true(:,index_numeric(i)) = X_updated(:,i);
end
%%
% percentage of missing values
percentage_missing = {};

% normalized root squared mean error
NRSME = {};

sd_NRSME = {};

% percentage of falsely classified entries
PFC = {};

sd_PFC = {};
% normalize numeric features at start
for p = 1:9
    for run = 1:5
        X1 = ones(size(X_true));
        matrix_size = numel(X1);
        missingNumber = round(p*0.1*matrix_size);
        X1(randperm(matrix_size, missingNumber))= missing;
        
        X = X_true;
        
        for i = 1:length(index_numeric)
            indices = find(isnan(X1(:,index_numeric(i)))==1);
            for k = 1:length(indices)
                X{indices(k),index_numeric(i)} = NaN;
            end
        end

        for i2 = 1:length(index_text)
            indices2 = find(isnan(X1(:,index_text(i2)))==1);
            for k2 = 1:length(indices2)
                X{indices2(k2),index_text(i2)} = {''};
            end
        end
        
        percentage_missing{p} = round((length(find(ismissing(X) == 1))/numel(X))*100);
        
        % Stopping criterion gamma
        diff_old = 10^15;
        diff_new = 10^12;
        
        % Make initial guess for missing values
        X_initialguess = fillmissing(X,'nearest');
        
        % k = vector of sorted indices of columns in X w.r.t. increasing amount of missing values
        missing1 = ismissing(X);
        m = zeros(1,size(X,2));
        indexMissing = {};
        indexObserved = {};
        for i=1:size(X,2)
            indexMissing{i} = find(missing1(:,i)==1);
            indexObserved{i} = find(missing1(:,i)==0);
            m(i) = length(find(missing1(:,i)==1));
        end
        
        m1 = m';
        index1=1:size(X,2);
        t = table(index1',m1);
        t_sorted = sortrows(t,'m1');
        k = t_sorted{:,1};
        
        % while not gamma do
        % Initialize iteration
        iteration = 1;
        Delta_F = {};
        while diff_new < diff_old
            diff_old = diff_new;
            
            % Ximpold = store previously imputed matrix
            X_oldimp = X_initialguess;
            X_old = X_initialguess;
            
            % for s in k do
            for i = 1:length(k)
                
                s = k(i);
                if m1(s)~=0
                    y_obs = X_old(indexObserved{s},s);
                    
                    x_obs = X_old(indexObserved{s},:);
                    x_obs(:,s) = [];
                    
                    y_misold = X_old(indexMissing{s},s);
                    
                    x_mis = X_old(indexMissing{s},:);
                    x_mis(:,s) = [];
                    
                    % Fit a random forest: y_obs(s)~x_obs(s)
                    NumTrees = 100;
                    if any(varNames_numeric == string(X_old.Properties.VariableNames{s})) == 1
                        Mdl = TreeBagger(NumTrees,x_obs,y_obs,'Method','regression');
                    else
                        Mdl = TreeBagger(NumTrees,x_obs,y_obs,'Method','classification');
                    end
                    
                    % Predict y_mis(s) using x_mis(s)
                    y_misnew = predict(Mdl,x_mis);
                    
                    % Ximpnew = update imputed matrix, using predicted y(s)mis
                    X_newimp = X_old;
                    X_newimp{indexMissing{s},s} = y_misnew;
                    
                    % Initialize X_old again
                    X_old = X_newimp;
                end
            end
            
            % update gamma
            Delta_N = sum(sum((X_newimp{:,index_numeric} - X_oldimp{:,index_numeric}).^2))/sum(sum((X_newimp{:,index_numeric}).^2));
            
            I = 0;
            for j = 1:length(index_text)
                j_text = index_text(j);
                for i = 1:size(X,1)
                    if isequal(X_newimp{i,j_text}, X_oldimp{i,j_text}) == 0
                        I = I+1;
                    end
                end
            end
            Delta_F_denominator = sum(t{index_text,2});
            if Delta_F_denominator ~= 0
                Delta_F{iteration} = I/Delta_F_denominator;
            else
                Delta_F{iteration} = 0;
            end
            
            diff_new = Delta_N + Delta_F{iteration};
            
            % Initial guess changes
            X_initialguess = X_newimp;
            
            % Iteration increases by 1
            iteration = iteration + 1;
        end
        % return the imputed matrix Ximp
        X_imputed = X_oldimp;
        %%
        NRSME1(run) = mean(sqrt(mean((X_true{:,index_numeric} - X_imputed{:,index_numeric}).^2)./var(X_true{:,index_numeric})));
        
        I = 0;
        for j = 1:length(index_text)
            j_text = index_text(j);
            for i = 1:size(X,1)
                if isequal(X_imputed{i,j_text}, X_true{i,j_text}) == 0
                    I = I+1;
                end
            end
        end
        PFC1(run) = I/numel(X_text);
    end
    NRSME{p} = mean(NRSME1);
    sd_NRSME{p} = std(NRSME1);
    PFC{p} = mean(PFC1);
    sd_PFC{p} = std(PFC1);
end
%%
% Error plot for continuous data
errorbar(1:1:9,str2double(string(NRSME)),str2double(string(sd_NRSME)),'-k','MarkerSize',7,...
    'Marker','*','MarkerEdgeColor','black','MarkerFaceColor','black','LineWidth',1)
xlim([1 9])
xticks([1 2 3 4 5 6 7 8 9])
xticklabels({'10%','20%','30%','40%','50%','60%','70%','80%','90%'})
title({'MissForest imputer method on numerical and categorical data';'192 Rows, 24 Columns';'15 Numeric features, 9 Text features'})
xlabel('Percentage of missing data')
ylabel('NRSME')
saveas(gcf,'fig1_NRSME.png')
hold off
%%
% Error plot for categorical data
errorbar(1:1:9,str2double(string(PFC)),str2double(string(sd_PFC)),'-k','MarkerSize',7,...
    'Marker','*','MarkerEdgeColor','black','MarkerFaceColor','black','LineWidth',1)
xlim([1 9])
xticks([1 2 3 4 5 6 7 8 9])
xticklabels({'10%','20%','30%','40%','50%','60%','70%','80%','90%'})
title({'MissForest imputer method on numerical and categorical data';'192 Rows, 24 Columns';'15 Numeric features, 9 Text features'})
xlabel('Percentage of missing data')
ylabel('PEC')
saveas(gcf,'fig1_PFC.png')