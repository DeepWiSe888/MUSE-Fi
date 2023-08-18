function output = readin_BFI_data(file_name, paras)
    [epoch_time_seq, angle_mat] = decode_bfi(file_name, paras);

    subcarrier_idx = paras.subcarrier_idx;

    phi_seq = struct('PHI11', [],...
                                'PHI21', [],...
                                'PHI31', [],...
                                'PHI22', [],...
                                'PHI32', []...
                                );
    psi_seq =struct('PSI21', [],...
                                'PSI31', [],...
                                'PSI41', [],...
                                'PSI32', [],...
                                'PSI42', []...
                                );

    frame_length = length(epoch_time_seq);
    for idx = 1:frame_length
        phi_seq.PHI11 = [phi_seq.PHI11, angle_mat(idx, subcarrier_idx, 1)];
        phi_seq.PHI21 = [phi_seq.PHI21, angle_mat(idx, subcarrier_idx, 2)];
        phi_seq.PHI31 = [phi_seq.PHI31, angle_mat(idx, subcarrier_idx, 3)];
        psi_seq.PSI21 = [psi_seq.PSI21, angle_mat(idx, subcarrier_idx, 4)];
        psi_seq.PSI31 = [psi_seq.PSI31, angle_mat(idx, subcarrier_idx, 5)];
        psi_seq.PSI41 = [psi_seq.PSI41, angle_mat(idx, subcarrier_idx, 6)];
        phi_seq.PHI22 = [phi_seq.PHI22, angle_mat(idx, subcarrier_idx, 7)];
        phi_seq.PHI32 = [phi_seq.PHI32, angle_mat(idx, subcarrier_idx, 8)];
        psi_seq.PSI32 = [psi_seq.PSI32, angle_mat(idx, subcarrier_idx, 9)];
        psi_seq.PSI42 = [psi_seq.PSI42, angle_mat(idx, subcarrier_idx, 10)];
    end
    angle_index_mat = [phi_seq.PHI11.', phi_seq.PHI21.', phi_seq.PHI31.', psi_seq.PSI21.', psi_seq.PSI31.', psi_seq.PSI41.'];
    angle_index_mat  = [ angle_index_mat , phi_seq.PHI22.', phi_seq.PHI32.', psi_seq.PSI32.', psi_seq.PSI42.'];

    time_sequence = (epoch_time_seq - epoch_time_seq(1));
    num_frame = length(time_sequence);
    new_phi_seq = phi_seq;
    field_name_phi = fieldnames(phi_seq);
    for id_field = 1:numel(field_name_phi)
        field_name = field_name_phi{id_field};
        tar_seq = phi_seq.(field_name);
        tar_seq = ((tar_seq/64) * 2*pi)/pi * 180;
        new_phi_seq.(field_name) = tar_seq;
    end

    new_psi_seq = psi_seq;
    field_name_psi = fieldnames(psi_seq);
    for id_field = 1:numel(field_name_psi)
        field_name = field_name_psi{id_field};
        tar_seq = psi_seq.(field_name);
        tar_seq = ((tar_seq/16) * 2*pi)/pi * 180;
        new_psi_seq.(field_name) = tar_seq;
    end

    Nr = 4; Nc = 2;
    NumBitsPhi = 6; NumBitsPsi=4;

    output = struct();
    output.time_vec = time_sequence.';
    output.data_vec = zeros(1, size(angle_index_mat,1));
    for idx_time = 1:size(angle_index_mat,1)
        V = bfDecompress([angle_index_mat(idx_time,:)],Nr,Nc,NumBitsPhi,NumBitsPsi);
        output.data_vec(idx_time) = mean(angle(V(1,1,1))) / pi * 180;
    end

    [output.time_vec, output.data_vec] = calibrate_BFI(output.time_vec, output.data_vec, paras.axis_line);

end


function [time_vec, angle_vec] = decode_bfi(fname, paras)
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);

    packets_num = size(val,1);

    angle_arr = zeros(packets_num, 10);

    num_subcarrier = 52;
    bphi = 6;
    bpsi = 4;
    angle_vec = zeros(packets_num, num_subcarrier,10);
    per_subcarrier_bits = (5*bphi + 5*bpsi);
    time_vec = zeros(packets_num, 1);
    for i = 1:packets_num
        string_temp = val(i).x_source.layers.wlan_1.FixedParameters.wlan_vht_compressed_beamforming_report;
        time_vec(i) = str2num(val(i).x_source.layers.frame.frame_time_epoch);
        split_string = split(string_temp, ':');
        join_string = join(split_string(3:end),'');
        join_string = join_string{1};
        slength = length(join_string);
        bin_char_vec = [];
        for char_i = 1:slength
            temp = hex2dec(join_string(char_i));
            bin_char_vec = [bin_char_vec, dec2bin(temp, 4)];
        end
        current_bin_char_= 1;
        for subcarrier_index=1:num_subcarrier
            if paras.flag_use_order_protocol
                temp_vec = protocol_order_decode(bin_char_vec(current_bin_char_: current_bin_char_ + per_subcarrier_bits - 1), bphi, bpsi);
                angle_vec(i, subcarrier_index, :) = temp_vec;
            end
            current_bin_char_ = current_bin_char_ + per_subcarrier_bits;
        end
    end
end

function returnvec = protocol_order_decode(tar_string, bphi, bpsi)
    current_bin_char = 1;
    phi11 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi21 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi31 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    psi21 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi31 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi41 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    phi22 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi32 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    psi32 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi42 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    returnvec =  [phi11, phi21, phi31, psi21, psi31, psi41, phi22, phi32, psi32, psi42];
end


function [out_time_vec, out_data_vec] = calibrate_BFI(time_vec, data_vec, axis_line)

    long_time_vec = repmat(time_vec, [1,4]);
    long_data_vec = [data_vec, data_vec + 90,  data_vec+180, data_vec+270];


    downer_angle_range = axis_line;
    upper_angle_range = 90 + downer_angle_range;

    indicator_vec = long_data_vec >= downer_angle_range & long_data_vec < upper_angle_range;

    sifted_time_vec = long_time_vec(indicator_vec);
    sifted_data_vec = long_data_vec(indicator_vec);

    combined_line = [sifted_time_vec; sifted_data_vec].';
    sort_combined_line = sortrows(combined_line, 1);

    out_time_vec = sort_combined_line(:,1).';
    out_data_vec = sort_combined_line(:,2).';

end

function Vtilda = bfDecompress(angidx,Nr,Nc,bphi,bpsi)
    %bfDecompress re-constructs Beamforming feedback matrix
    %   V = bfDecompress(ANGIDX,NR,NC,BPSI,BPHI) reconstructs the
    %   beamforming feedback matrix, V from ANGIDX. ANGIDX are the quantized
    %   angles. NR and NC are the number of rows and number of columns in V.
    %   There are two kinds of angles in ANGIDX: phi and psi. Those angles are
    %   quantized according to the bit resolution given by BPHI and BPSI for
    %   phi and psi respectively. The size of ANGIDX should be of the form
    %   (Number of active sub-carriers)X(Number of angles for a sub-carrier).
    %
    %   References:
    %   1) IEEE Standard for Information technology--Telecommunications and
    %   information exchange between systems Local and metropolitan area
    %   networks--Specific requirements - Part 11: Wireless LAN Medium Access
    %   Control (MAC) and Physical Layer (PHY) Specifications," in IEEE Std
    %   802.11-2016 (Revision of IEEE Std 802.11-2012) , vol., no., pp.1-3534,
    %   Dec. 7 2016.

    %   Copyright 2018 The MathWorks, Inc.

    p = min([Nc,Nr-1]);
    [Nst,NumAngles] = size(angidx);

    % Perform dequantization first. See table 9-68 (Quantization of angles) in [1]
    angles = zeros(NumAngles,1,Nst);
    angcnt = 1;
    for ii = Nr-1:-1:max(Nr-Nc,1)
        for jj = 1:ii
            angles(angcnt,1,:) = (2*angidx(:,angcnt)+1)*pi/(2^bphi);
            angcnt = angcnt + 1;
        end
        
        for jj = 1:ii
            angles(angcnt,1,:) = (2*angidx(:,angcnt)+1)*pi/(2^(bpsi+2));
            angcnt = angcnt + 1;
        end
    end

    % Construction of V matrix from the angles.
    V = repmat(eye(Nr,Nc),[1,1,Nst]);
    NumAnglesCnt = NumAngles;
    for ii = p:-1:1 % Eq 19-85 in [1].
        for jj = Nr:-1:ii+1
            % for each jj, construct Givens matrix, G
            for sc = 1:Nst
                Gt = eye(Nr); % G transpose
                Gt(ii,ii) = cos(angles(NumAnglesCnt,1,sc));
                Gt(ii,jj) = -1*sin(angles(NumAnglesCnt,1,sc));
                Gt(jj,ii) = sin(angles(NumAnglesCnt,1,sc));
                Gt(jj,jj) = cos(angles(NumAnglesCnt,1,sc));
                V(:,:,sc) = Gt*V(:,:,sc);
            end
            NumAnglesCnt = NumAnglesCnt - 1;
        end
        D = [ones(ii-1,1,Nst); exp(1j*angles(NumAnglesCnt-Nr+ii+1:NumAnglesCnt,1,:)); ones(1,1,Nst)];
        NumAnglesCnt = NumAnglesCnt - Nr + ii;
        V = D.*V;
    end
    Vtilda = permute(V,[3 2 1]);
end