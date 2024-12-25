function pev = calculate_PEV(st, trial_conds)

if any(st~=0)
    %     [~,anova_table] = anova1(st,trial_conds,'off');
    %
    %     SS_groups = anova_table{2,2};
    %     SS_total = anova_table{4,2};
    %     df_groups = anova_table{2,3};
    %     MS_error = anova_table{3,4};
    %     pev_old = (SS_groups-df_groups*MS_error)/(SS_total+MS_error);
    
    anova_table = anova_custom(st,trial_conds);
    SS_groups = anova_table(2,2);
    SS_total = anova_table(4,2);
    df_groups = anova_table(2,3);
    MS_error = anova_table(3,4);
    pev = (SS_groups-df_groups*MS_error)/(SS_total+MS_error);
else
    pev = 0;
end
return

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%
function Table = anova_custom(x, conds)


m = length(unique(conds));
lx = length(x);
xr = x - mean(x);
xm = zeros(1, m);
countx = zeros(1, m);
for i = 1:m
    ind = find(conds == i);
    countx(i) = length(ind);
    xm(i) = mean(xr(ind));
end
gm = mean(xr);                      % grand mean
df1 = sum(countx>0) - 1;            % Column degrees of freedom
df2 = lx - df1 - 1;                 % Error degrees of freedom
xc = xm - gm;                       % centered
xc(countx==0) = 0;
RSS = dot(countx, xc.^2);           % Regression Sum of Squares

TSS = (xr(:) - gm)'*(xr(:) - gm);  % Total Sum of Squares
SSE = TSS - RSS;                   % Error Sum of Squares
if (df2 > 0)
    mse = SSE/df2;
else
    mse = NaN;
end
F = (RSS/df1) / mse;
Table=zeros(3,5);               %Formatting for ANOVA Table printout
Table(:,1)=[ RSS SSE TSS]';
Table(:,2)=[df1 df2 df1+df2]';
Table(:,3)=[ RSS/df1 mse Inf ]';
Table(:,4)=[ F Inf Inf ]';

[m1,m2]=size(Table);
Table = [zeros(1,m2); Table];
Table = [zeros(m1+1,1) Table];
return
