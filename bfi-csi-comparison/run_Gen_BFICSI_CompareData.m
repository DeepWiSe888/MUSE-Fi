close all;
addpath(genpath(pwd));


rng(101);
in_ULCSI_directory = ['./Data/BFI-CSI_compare/'];
obj_Gen_Compare = Gen_CSI_CompareData(in_ULCSI_directory);

in_file_name = 'bfi_csi_compare_csi.csi';
in_file_path = [in_ULCSI_directory, in_file_name];

uiopen(in_file_path, 1);
in_variable_name = split(in_file_name, '.');
clear(in_variable_name{1});

handle_csi = CellHandle();
handle_csi.Cell = latest;
obj_Gen_Compare.handle_csi = handle_csi;

out_directory = ['./Data/BFICSI_Compare'];

in_case_name = struct('csi_readin', 'default');

%%
CSI_output = obj_Gen_Compare.generate_train_test_data(in_file_name, in_case_name);

%========================================
%  BFI
close all;
input_file_dir = './Data/BFI-CSI_compare/';
obj_GEN_BFI_CompareData = GEN_BFI_CompareData(input_file_dir);
input_file_name = 'bfi_csi_compare_bfi.json';
case_name = 'bfi_csi_compare';
BFI_output = obj_GEN_BFI_CompareData.main_process(input_file_name, case_name);

%========================================
% Power Spectrum
case_name_set = {'breath', 'gesture', 'action'};
for case_idx = 1:3
    case_name = case_name_set{case_idx};
    figure('Name',['Power Spectrum Comparison ', case_name],'NumberTitle','off');
    hold on;
    plot(CSI_output.(case_name).pspectrum.freq_vec, CSI_output.(case_name).pspectrum.pow_vec, 'LineWidth', 1, 'DisplayName', 'CSI');
    plot(BFI_output.(case_name).pspectrum.freq_vec, BFI_output.(case_name).pspectrum.pow_vec, 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'BFI');
    xlim([0,20]);
    if case_idx == 1
        xlim([0,1]);
    end
    set(gca, 'YScale', 'log');
    grid on; 
    box on;
    set(gcf, 'Position', [918,305,222,148]);
    set(gca,'FontName', 'Arial', 'FontSize',16)
    xlabel('Frequency [Hz]');
    ylabel('Power Spectrum [dB]');
    legend;
end


%========================================
% Variance
case_name_set = {'breath', 'gesture', 'action'};
f_cutoff = [1,64/3, 64/3];
time_offset = -1.2;
for case_idx = 1:3
    case_name = case_name_set{case_idx};
    figure('Name',['Variance Comparison ', case_name],'NumberTitle','off');
    hold on;
    paras = struct();
    paras.stat_interval_duration = 0.1;
    paras.resample_freq = 64;
    paras.order_butter_lp_filter = 8;
    paras.f_cutoff = f_cutoff(case_idx);
    [time_sequence_csi, std_sequence_csi] = gen_std_time_sequence(CSI_output.(case_name).time_vec + time_offset, CSI_output.(case_name).data_vec, paras);
    area(time_sequence_csi, std_sequence_csi, 'LineWidth', 1, 'DisplayName', 'CSI');
    [time_sequence_bfi, std_sequence_bfi] = gen_std_time_sequence(BFI_output.(case_name).time_vec + time_offset, BFI_output.(case_name).data_vec, paras);
    area(time_sequence_bfi, std_sequence_bfi, 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'BFI');
    xlim([0,10]);
    set(gca, 'YScale', 'log');
    grid on; 
    box on;
    set(gcf, 'Position', [918,305,222,148]);
    set(gca,'FontName', 'Arial', 'FontSize',16, 'YTick', [1e-3, 1, 1e3]);
    xlim([0,8]);
    ylim([1e-5,1e3]);
    xlabel('Time [sec]');
    ylabel('Std [degree]')
    legend;
end

%========================================
% Temporal
case_name_set = {'breath', 'gesture', 'action'};
f_cutoff = [0.8,64/3, 64/3];
order_butter_lp_filter = 8;
f_resample = 64;
resample_time_vec = 0:1/f_resample:10;
offset = [0,10,50];
time_offset = -1;
case_line_width = [1.5,1,1];
for case_idx = 1:3
    [filter_b, filter_a] = butter(order_butter_lp_filter, f_cutoff(case_idx)/(f_resample/2));
    case_name = case_name_set{case_idx};
    figure('Name',['Time Domain Comparison ', case_name],'NumberTitle','off');
    hold on;
    x_vec_csi = CSI_output.(case_name).time_vec;
    y_vec_csi = CSI_output.(case_name).data_vec;
    y_vec_csi = y_vec_csi - mean(y_vec_csi);
    y_vec_csi = interp1(x_vec_csi, y_vec_csi, resample_time_vec, 'nearest', 'extrap');
    y_vec_csi = filter(filter_b, filter_a, y_vec_csi) - offset(case_idx);
    

    x_vec_bfi = BFI_output.(case_name).time_vec;
    y_vec_bfi = BFI_output.(case_name).data_vec;
    y_vec_bfi = y_vec_bfi - mean(y_vec_bfi);
    y_vec_bfi = interp1(x_vec_bfi, y_vec_bfi, resample_time_vec, 'nearest','extrap');
    y_vec_bfi = filter(filter_b, filter_a, y_vec_bfi);
    
    plot(resample_time_vec + time_offset, y_vec_csi, 'LineWidth', case_line_width(case_idx), 'LineStyle', '-', 'DisplayName', 'CSI');
    plot(resample_time_vec + time_offset, y_vec_bfi, 'LineWidth',2, 'LineStyle', ':', 'DisplayName', 'BFI');
    xlim([0,10]);
    xlim([0,8])
    set(gca, 'YScale');
    grid on; 
    box on;
    set(gcf, 'Position', [918,305,222,148]);
    set(gca,'FontName', 'Arial', 'FontSize',16);
    if strcmp(case_name_set{case_idx}, 'breath')
        ylim([-50, 50]);
    else
        ylim([-200, 200]);
    end
    xlabel('Time [sec]');
    ylabel('Phase [deg]');
    legend;
end

%%

function [time_sequence, std_sequence] = gen_std_time_sequence(time_vec, data_vec, paras)

    resample_freq = paras.resample_freq;
    stat_interval_duration = paras.stat_interval_duration;

    [filter_b, filter_a] = butter(paras.order_butter_lp_filter, paras.f_cutoff/(resample_freq/2));

    resample_time_vec = time_vec(1):1/resample_freq:time_vec(end);
    resample_data_vec = interp1(time_vec, data_vec, resample_time_vec, 'linear','extrap');
    resample_data_vec = filter(filter_b, filter_a, resample_data_vec);

    std_sequence = [];
    time_sequence = [];
    start_time = time_vec(1);
    end_time = start_time + stat_interval_duration;
    while 1
        indicator_vec = resample_time_vec>=start_time & resample_time_vec < end_time;
        std_sequence = [std_sequence, std(detrend(resample_data_vec(indicator_vec)))];
        time_sequence = [time_sequence, mean([start_time, end_time])];
        start_time = start_time + stat_interval_duration;
        end_time = start_time + stat_interval_duration;
        if end_time > resample_time_vec(end)
            break;
        end
    end
end