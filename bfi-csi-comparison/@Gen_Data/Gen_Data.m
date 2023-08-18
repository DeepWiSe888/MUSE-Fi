classdef Gen_Data

    properties        
        slicing_duration = 2 
        period_duration = 5 
        start_time_instance 
        end_time_instance 
        stable_time_duration 
        train_test_ratio = 5 
        time_length_range_of_fraction = [0.75, 0.95] 
        num_fraction_per_slice = 15 
        f_resample = 200 
        augment_factor = 15 
        upbound_freq_cwt = 50 
        clipping_level = 0.95 

        realistic_traffic_start_time 
        realistic_traffic_end_time 

        csi_data_readin_method 

        flag_illustrate_extracted_parts = true;

        
        AP_address = [20,89,192,51,139,71]; 
        JHu_mac_address =[40,2,68,84,88,207]; 
        HZhang_mac_address =[242,51,75,109,52,182]; 
        
        TZheng_sony_mac_address = [88, 72, 34, 130, 210,149]; 
        Jingyang_mac_address = [140, 122, 61, 52, 188, 83]; 
        HZhang_2_mac_address = [106, 136, 230, 255, 228, 85]; 
        
        
        JHu_mac_address_29 = [206, 69, 232, 109, 135, 33]; 
        HZhang_mac_address_29 = [182, 109, 37, 198, 90, 213]; 
        HZhang_mac_address_29_2 = [246,116,43,54,173,40]; 
        AP_address_29 = [20, 89, 192, 51, 139, 119]; 
        
        frame_type2number_struct
        frame_type2csi_size_number_struct
    end

    methods
        function obj = Gen_Data()
            obj.frame_type2number_struct = obj.construct_frame_type2number_struct();
            obj.frame_type2csi_size_number_struct = obj.construct_frame_type2csi_size_number_struct();
        end

    end

    methods(Static)
        function output = construct_frame_type2number_struct()
            output = struct();
            output.qos_data =  Gen_Data.convert_Frametype2Number([2,8]);
            output.nack = Gen_Data.convert_Frametype2Number([0,14]);
        end

        function output = construct_frame_type2csi_size_number_struct()
            output = struct();
            output.qos_data = Gen_Data.convert_CSIsize2Number([57,2,2]);
            output.nack = Gen_Data.convert_CSIsize2Number([53,1,2]);
        end
        function output = convert_CSIsize2Number(csi_size)
            output = csi_size(3) * 1e-3 + csi_size(2) + csi_size(1) * 10;
        end
        function output = convert_Frametype2Number(type_subtype)
            output = type_subtype(1)*100 + type_subtype(2);
        end
    end
end