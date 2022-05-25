clear all 
close all


%% UCR
T=readtable('results/HDC_MINIROCKET_results.xlsx','Sheet','UCR');

num_ds = 128;

% indices
idx_minirocket = 2;
idx_select_best_s_xval = 10;

% results 
results_minirocket = T{1:num_ds,idx_minirocket};
best_hdc_minirocket_xval = T{1:num_ds,idx_select_best_s_xval};


num_scale = 7;
results_hdc_minirocket = [];

for scale = 1:num_scale
    results_hdc_minirocket(:,end+1) = T{1:num_ds,scale+1};
end

optimal_hdc_minirocket = max(results_hdc_minirocket,[],2);


%% pairwise accuracy with auto and optimal selected s
figure()
X = results_minirocket;
x = linspace(0,1,50);
y = linspace(0,1,50);

scatter([X,X],[optimal_hdc_minirocket, best_hdc_minirocket_xval],100,'.')
hold on 
grid on 
plot (x,y,'--','color','#666')
xlabel('MiniROCKET')
ylabel('HDC-MiniROCKET')
legend({'optimal selection of s (oracle)','simple data-driven selection of s'},'Position',[0.55 0.23 0.15 0.06])
% title('Pairwise comparison of MiniROCKET and HDC-MiniROCKET')
text(0.05,0.95,"HDC-MiniROCKET is better")
text(0.5,0.05,"MiniROCKET is better")

xlim([0,1])
ylim([0 1])
set(gcf,'color','w')
set(gcf,'Position',[100 100 450 350])
saveas(gcf,'images/pairwise_ucr_auto_s','epsc')


%% barplot with relativ performance change
figure()
change_auto = (best_hdc_minirocket_xval - results_minirocket)./results_minirocket;
change_optimal = (optimal_hdc_minirocket - results_minirocket)./results_minirocket;

% find where is a change
idx = find((abs(change_auto)+abs(change_optimal))>0);

bar([change_optimal(idx)*100, change_auto(idx)*100])

% load dataset infos 
xticks(1:size(idx,1));
xlabels = T.UCR_idx(idx);
xticklabels(xlabels);
xtickangle(90);
legend({'optimal selection of s (oracle)','simple data-driven selection of s'})
set(gcf,'color','w')
set(gcf,'Position',[100 100 1500 350])
grid on
% title('Relative change in accuracy with optimal and simple data-driven selection of s for HDC-MiniROCKET')
xlabel('UCR Dataset index')
ylabel('relative Accuracy change [%]')

saveas(gcf,'images/ucr_results_auto_s','epsc')

%% plot acc over different s
figure()
delta = results_hdc_minirocket-results_hdc_minirocket(:,1);

z = 0.12;

colorMap =  customcolormap([0 z 1], [0 0 1; 1 1 1; 1 0 0]);

h = heatmap(delta,'Colormap',colorMap,'GridVisible','off')

x_labels = 1:num_ds;
idx = mod(x_labels,10)>0;             % index of datetime values to show as x tick
h.YDisplayLabels(idx) = {''}; 
h.XDisplayLabels = {'0','1','2','3','4','5','6'}

xlabel('parameter s')
ylabel('dataset ID')
set(gcf,'color','w')
set(gcf,'Position',[100 100 700 400])
saveas(gcf,'images/ucr_diff_surf','epsc')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% eval time effort 
T_time_hdc=readtable('results/HDC_MINIROCKET_time_results.xlsx','Sheet','UCR');
T_time_orig=readtable('results/MINIROCKET_time_results.xlsx','Sheet','UCR');

preproc_train_minirocket = T_time_orig{1:128,"train_time_preproc"};
learning_minirocket = T_time_orig{1:128,"train_time"};
preproc_test_minirocket = T_time_orig{1:128,"test_time_preproc"};
inference_minirocket = T_time_orig{1:128,"test_time"};

preproc_train__hdc_minirocket = T_time_hdc{1:128,"train_time_preproc"};
learning_hdc_minirocket = T_time_hdc{1:128,"train_time"};
preproc_test_hdc_minirocket = T_time_hdc{1:128,"test_time_preproc"};
inference_hdc_minirocket = T_time_hdc{1:128,"test_time"};

complete_inference_minirocket = inference_minirocket + preproc_test_minirocket;
complete_inference_hdc_minirocket = inference_hdc_minirocket + preproc_test_hdc_minirocket;

figure()

x = linspace(0,100,5000);
y = linspace(0,100,5000);

scatter(complete_inference_minirocket,complete_inference_hdc_minirocket,100,'.')
grid on 
hold on 
plot (x,y,'--','color','#666')
set(gca,'xscale','log')
set(gca,'yscale','log')
axis equal
xlim([-inf 10^1.5])
ylim([-inf 10^1.5])
xlabel('MiniROCKET inference time [s]')
ylabel('HDC-MiniROCKET inference time [s]')
% title('Inference Time of MiniROCKET and HDC-MiniROCKET per dataset')
text(0.025,15,"HDC-MiniROCKET is slower")
text(1,0.05,"MiniROCKET is slower")
set(gcf,'color','w')
set(gcf,'Position',[100 100 400 400])
saveas(gcf,'images/pairwise_time_ucr','epsc')

