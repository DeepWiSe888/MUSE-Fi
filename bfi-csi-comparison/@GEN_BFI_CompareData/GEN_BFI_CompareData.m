classdef GEN_BFI_CompareData < Gen_Data

    properties
        input_file_dir
    end

    methods
        function obj = GEN_BFI_CompareData(input_file_dir)
            obj.input_file_dir = input_file_dir;
        end
        function output = main_process(obj, input_file_name, case_name)
            input_file_path = [obj.input_file_dir, input_file_name];
            paras = obj.obtain_bfi_readin_para(case_name);
            output = readin_BFI_data(input_file_path, paras);
            output = GEN_BFI_CompareData.slicing_bfi_angle_sequence(output.time_vec, output.data_vec, paras);
        end

        function output = obtain_bfi_readin_para(obj, case_name)
            output = struct();
            if strcmp(case_name, 'bfi_csi_compare')
                output.subcarrier_idx  = 5;
                output.flag_use_order_protocol = true;
                output.breath_time = [];
                output.gesture_time = [];
                output.action_time = [];
                output.flag_draw_power_spectrum = false;
                output.axis_line = - 40;
                output.flag_show_complete_bfi_sequence = true;
                output.breath_time = [30.15, 40.15] + 0.5;
                output.gesture_time = [50.5, 60.5];
                output.action_time = [71, 81];
            end
        end
    end

    methods (Static)
        function output = slicing_bfi_angle_sequence(time_sequence, bfi_sequence, paras)
            output = struct();
            output.breath = struct();
            output.gesture = struct();
            output.action = struct();
        
        
            indicator_vec = time_sequence >= paras.breath_time(1) & time_sequence <= paras.breath_time(2);
            output.breath.time_vec = time_sequence(indicator_vec);
            output.breath.time_vec = output.breath.time_vec - output.breath.time_vec(1);
            output.breath.data_vec = bfi_sequence(indicator_vec);
        
        
            indicator_vec = time_sequence >= paras.gesture_time(1) & time_sequence <= paras.gesture_time(2);
            output.gesture.time_vec = time_sequence(indicator_vec);
            output.gesture.time_vec = output.gesture.time_vec - output.gesture.time_vec(1);
            output.gesture.data_vec = bfi_sequence(indicator_vec);
        
        
            indicator_vec = time_sequence >= paras.action_time(1) & time_sequence <= paras.action_time(2);
            output.action.time_vec = time_sequence(indicator_vec);
            output.action.time_vec = output.action.time_vec - output.action.time_vec(1);
            output.action.data_vec = bfi_sequence(indicator_vec);
        
        
            output.breath.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.breath.time_vec, output.breath.data_vec);
            output.gesture.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.gesture.time_vec, output.gesture.data_vec);
            output.action.pspectrum = GEN_BFI_CompareData.obtain_power_spectrum(output.action.time_vec, output.action.data_vec);
        end

        function output = obtain_power_spectrum(time_vec, data_vec)
            f_resample = 64;
            resample_time_vec = time_vec(1):1/f_resample:time_vec(end);
            resample_data_vec = interp1(time_vec, detrend(data_vec), resample_time_vec);
            output = struct();
            [output.pow_vec, output.freq_vec] = pspectrum(resample_data_vec, f_resample);
        end
    end
end