function output = readin_CSI_data(method_para) 

    file_name = method_para.file_name;  
    file_path = method_para.file_path;
    read_method = method_para.read_method; 
    diff_target = method_para.diff_target;  
    time_span = method_para.time_span;  
    user_target = method_para.user_target;  
    frame_type = method_para.frame_type;  
    csi_size_type = method_para.csi_size_type; 
    handle_csi = method_para.handle_csi;
    flags = method_para.flags; 

    if flags.just_illustration 
        read_method = 'maxPower';
        time_span = [0, inf];
    end

    latest = handle_csi.Cell;

    rows = cellfun(@(x) 1 ...
            && isequal(x.StandardHeader.Addr1, user_target.rx)  ...
            && isequal(x.StandardHeader.Addr2, user_target.tx)  ...
            && ismember(Gen_Data.convert_CSIsize2Number(size(x.CSI.CSI)), csi_size_type) ...
            && ismember(Gen_Data.convert_Frametype2Number([x.StandardHeader.ControlField.Type, x.StandardHeader.ControlField.SubType]), frame_type) ...
            && 1, ...
            latest, 'UniformOutput', 0);

    rows = cell2mat(rows);
    rows_sum = sum(rows);
    data = latest(rows);

    total_time = numel(data);
    if total_time == 0
        warning("readin_CSI_data:: No data are available for the current conditions");
    end

    time_seq = [];
    index_seq = [];
    csi_seq_1 = [];
    csi_seq_2 = [];
    sequence_seq = [];

    if flags.just_illustration 
        csi_seq_t1_r1 = [];
        csi_seq_t1_r2 = [];
        csi_seq_t2_r1 = [];
        csi_seq_t2_r2 = [];
    end

    for tx = 1:rows_sum
        time_seq = [time_seq, data{tx}.RxSBasic.SystemTime];
        index_seq = [index_seq, tx];
        sequence_seq = [sequence_seq, data{tx}.StandardHeader.Sequence];
        if flags.just_illustration
            csi_seq_t1_r1 = [csi_seq_t1_r1, data{tx}.CSI.CSI(:,1, 1)];
            csi_seq_t1_r2 = [csi_seq_t1_r2, data{tx}.CSI.CSI(:,1, 2)];
            csi_seq_t2_r1 = [csi_seq_t2_r1, data{tx}.CSI.CSI(:,2, 1)];
            csi_seq_t2_r2 = [csi_seq_t2_r2, data{tx}.CSI.CSI(:,2, 2)];
        end
        if strcmp(diff_target.name, "tx_difference")
            if size(data{tx}.CSI.CSI,2) < 2
                error("readin_CSI_data:: Number of Tx STS smaller than 2");
            end
            csi_seq_1 = [csi_seq_1, data{tx}.CSI.CSI(:, 1, diff_target.para)];
            csi_seq_2 = [csi_seq_2, data{tx}.CSI.CSI(:, 2, diff_target.para)];
        elseif strcmp(diff_target.name, "rx_difference")
            if size(data{tx}.CSI.CSI,3) < 2
                error("readin_CSI_data:: Number of Rx Ant smaller than 2");
            end
            csi_seq_1 = [csi_seq_1, data{tx}.CSI.CSI(:,diff_target.para, 1)];
            csi_seq_2 = [csi_seq_2, data{tx}.CSI.CSI(:, diff_target.para, 2)];
        else
            error("readin_CSI_data::Not Found diff_target.name");
        end
    end

    unbias_time_seq = (double(time_seq) - double(time_seq(1)))./10^9;

    indicator_vec = unbias_time_seq <= time_span(2) & unbias_time_seq >= time_span(1);
    indicator_vec =  ~indicator_vec;
    unbias_time_seq(indicator_vec) = [];
    csi_seq_1(:, indicator_vec) = [];
    csi_seq_2(:, indicator_vec) = [];
    unbias_time_seq = unbias_time_seq - unbias_time_seq(1); 

    if strcmp(read_method, "pca")
        flag_jump_pca = false;
        considered_num_subcarrier = read_method.para;
    elseif strcmp(read_method, "maxPower")
        considered_num_subcarrier = 1;
        flag_jump_pca = true;
    else
        error("readin_CSI_data::Not Found read_method");
    end

    power = abs(csi_seq_1).^2 + abs(csi_seq_2).^2;
    sum_power = sum(power, 2);
    [ ~ ,subcarrier_idx] = maxk(sum_power, considered_num_subcarrier);
    if isfield(method_para,'specify_subcarrier')
        subcarrier_idx = method_para.specify_subcarrier;
    end
    csi_seq_1 = csi_seq_1(subcarrier_idx,:);
    csi_seq_2 = csi_seq_2(subcarrier_idx,:);

    if flags.just_illustration
        csi_seq_t1_r1 = csi_seq_t1_r1(subcarrier_idx,:);
        csi_seq_t1_r2 = csi_seq_t1_r2(subcarrier_idx,:);
        csi_seq_t2_r1 = csi_seq_t2_r1(subcarrier_idx,:);
        csi_seq_t2_r2 = csi_seq_t2_r2(subcarrier_idx,:);
    end


    temp_data = angle(csi_seq_1./csi_seq_2)/pi*180;
    time_seq = unbias_time_seq;
    [time_seq, temp_data] = removeNaN(time_seq, temp_data);
    if flag_jump_pca
        if strcmp(diff_target.name, "rx_difference")
            if ~isfield(method_para, 'combine_coefficient')
                output = calibrate_CSI(time_seq, temp_data, nan);
            else
                output = calibrate_CSI(time_seq, temp_data, method_para.combine_coefficient);
            end
        else
            output = temp_data;
        end
        result_data = output;
        output = struct();
        output.time_vec = time_seq;
        output.data_vec = result_data;

        if isfield(method_para, 'alter_axis_level')
            output.data_vec = process_change_axis_level(method_para.alter_axis_level, output.data_vec);
        end

        if flags.just_illustration 
            two_tx_result = struct();
            two_combined_tx_result = struct();
            two_rx_result = struct();
            [two_tx_result.tx1_time, two_tx_result.tx1] = removeNaN(unbias_time_seq, angle(csi_seq_t1_r1./csi_seq_t1_r2)/pi*180);
            two_combined_tx_result.tx1 = calibrate_CSI(two_tx_result.tx1_time, two_tx_result.tx1, nan);

            [two_tx_result.tx2_time, two_tx_result.tx2] = removeNaN(unbias_time_seq, angle(csi_seq_t2_r1./csi_seq_t2_r2)/pi*180);
            two_combined_tx_result.tx2 = calibrate_CSI(two_tx_result.tx2_time, two_tx_result.tx2, -100);

            [two_rx_result.rx1_time, two_rx_result.rx1] = removeNaN(unbias_time_seq, angle(csi_seq_t1_r1./csi_seq_t2_r1)/pi*180);
            two_rx_result.rx1 = process_change_axis_level(-90, two_rx_result.rx1);

            [two_rx_result.rx2_time, two_rx_result.rx2] = removeNaN(unbias_time_seq, angle(csi_seq_t1_r2./csi_seq_t2_r2)/pi*180);
            two_rx_result.rx2 = process_change_axis_level(-90, two_rx_result.rx2);
        end
        return;
    end
end % END OF FUNC



function output = calibrate_CSI(Rx_angle_diff_x, Rx_angle_diff_y, middle_line)
    if isnan(middle_line)
        Rx_angle_diff_y_mean = mean(Rx_angle_diff_y);
    else
        Rx_angle_diff_y_mean = middle_line;
    end
    if Rx_angle_diff_y_mean > 0
        downer_Rx_angle_range_1 = Rx_angle_diff_y_mean;
        upper_Rx_angle_range_1 = -180 + Rx_angle_diff_y_mean;

        condition_index = Rx_angle_diff_y>downer_Rx_angle_range_1 | Rx_angle_diff_y< upper_Rx_angle_range_1;
        line_1_x = Rx_angle_diff_x(condition_index);
        line_1_y = Rx_angle_diff_y(condition_index);
        line_1_y(line_1_y < upper_Rx_angle_range_1) = line_1_y(line_1_y < upper_Rx_angle_range_1) + 180 + 180;
        changed_line_1_y = line_1_y - mean(line_1_y);

        condition_index = ~condition_index;
        line_2_x = Rx_angle_diff_x(condition_index);
        line_2_y = Rx_angle_diff_y(condition_index);
        changed_line_2_y = line_2_y - mean(line_2_y);
        
        combined_line = [line_1_x, line_2_x; changed_line_1_y, changed_line_2_y].';
        sort_combined_line = sortrows(combined_line, 1);
    else
        downer_Rx_angle_range_1 = Rx_angle_diff_y_mean;
        upper_Rx_angle_range_1 = 180 + Rx_angle_diff_y_mean;

        condition_index = Rx_angle_diff_y>downer_Rx_angle_range_1 & Rx_angle_diff_y< upper_Rx_angle_range_1;
        line_1_x = Rx_angle_diff_x(condition_index);
        line_1_y = Rx_angle_diff_y(condition_index);
        changed_line_1_y = line_1_y - mean(line_1_y);

        condition_index = ~condition_index;
        line_2_x = Rx_angle_diff_x(condition_index);
        line_2_y = Rx_angle_diff_y(condition_index);
        line_2_y(line_2_y > upper_Rx_angle_range_1) = - (180 - line_2_y(line_2_y > upper_Rx_angle_range_1))- 180;
        changed_line_2_y = line_2_y - mean(line_2_y);
        
        combined_line = [line_1_x, line_2_x; changed_line_1_y, changed_line_2_y].';
        sort_combined_line = sortrows(combined_line, 1);
    end
    output = sort_combined_line(:,2).';
end

function [ox_vec, oy_vec] = removeNaN(x_vec, y_vec) 

    nanIdx = isnan(y_vec);
    ox_vec = x_vec;
    oy_vec = y_vec;
    ox_vec(nanIdx) = [];
    oy_vec(nanIdx) = [];
end

function output = process_change_axis_level(axis_level, data_vec)

    data_vec(data_vec<axis_level) = data_vec(data_vec<axis_level) + 360;
    data_vec = data_vec - mean(data_vec, 'all');
    output = data_vec;
end