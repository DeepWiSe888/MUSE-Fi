classdef Gen_CSI_CompareData < Gen_Data
    properties
        handle_csi 
        file_name
        in_ULCSI_directory
    end
    methods
        function obj = Gen_CSI_CompareData(in_ULCSI_directory)
                obj@Gen_Data();
                obj.csi_data_readin_method = 'readin_CSI_data';
                obj.in_ULCSI_directory = in_ULCSI_directory;
        end 

        function output = generate_train_test_data(obj, in_file_name, para_case)
            method_para = obj.obtain_csi_readin_method_para(obj.csi_data_readin_method, para_case.csi_readin);
            in_file_path = [obj.in_ULCSI_directory, in_file_name];

            method_para.file_name = in_file_name;
            method_para.file_path = in_file_path;
            method_para.handle_csi = obj.handle_csi;
            long_nsparse_csi_data = readin_CSI_data(method_para);
            output = obj.slicing_csi_readin_data(long_nsparse_csi_data.time_vec, long_nsparse_csi_data.data_vec, method_para);
        end 

        function output = obtain_csi_readin_method_para(obj, csi_readin_method, case_name)
            output = struct();
            case_name = split(case_name,'.');
            if strcmp(csi_readin_method, 'readin_CSI_data') && strcmp(case_name{1}, 'default')
                output.read_method = 'maxPower';
                output.diff_target = struct('name', 'tx_difference', 'para', 1); 
                output.time_span = [0, inf];
                target_mac_address = obj.JHu_mac_address;
                output.user_target = struct('tx', target_mac_address, 'rx', obj.AP_address);
                output.frame_type = [obj.frame_type2number_struct.qos_data];
                output.csi_size_type = [obj.frame_type2csi_size_number_struct.qos_data];
                output.flags.just_illustration =true; 
                output.flags.show_csi_sequence = true;
                output.breath_time = [38.72, 48.72]  + 0.5;
                output.gesture_time = [61.1, 71.1];
                output.action_time = [79.6, 89.6];
            end
        end 
        
        function output = slicing_csi_readin_data(obj, time_sequence, data_vec, paras)
            output = struct();
            output.breath = struct();
            output.gesture = struct();
            output.action = struct();

            indicator_vec = time_sequence >= paras.breath_time(1) & time_sequence <= paras.breath_time(2);
            output.breath.time_vec = time_sequence(indicator_vec);
            output.breath.time_vec = output.breath.time_vec - output.breath.time_vec(1);
            output.breath.data_vec = data_vec(indicator_vec);

            indicator_vec = time_sequence >= paras.gesture_time(1) & time_sequence <= paras.gesture_time(2);
            output.gesture.time_vec = time_sequence(indicator_vec);
            output.gesture.time_vec = output.gesture.time_vec - output.gesture.time_vec(1);
            output.gesture.data_vec = data_vec(indicator_vec);

            indicator_vec = time_sequence >= paras.action_time(1) & time_sequence <= paras.action_time(2);
            output.action.time_vec = time_sequence(indicator_vec);
            output.action.time_vec = output.action.time_vec - output.action.time_vec(1);
            output.action.data_vec = data_vec(indicator_vec);


            output.breath.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.breath.time_vec, output.breath.data_vec);
            output.gesture.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.gesture.time_vec, output.gesture.data_vec);
            output.action.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.action.time_vec, output.action.data_vec);
        end
    end 
end