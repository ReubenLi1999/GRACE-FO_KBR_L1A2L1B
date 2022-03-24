 classdef Kbr
    % Kbr Summary of this class goes here
    %   Detailed explanation goes here
    
    properties 
        % time tags
        gps_time                        %% the gps time of the kbr data (1b)
        rcv_time                        %% the rcv time of the kbr data (1a)
        eps_time                        %% the eps time of the clk data (1b), interpolated
        % physical quantities about dual-one-way(dow) ranging
        dow_range                       %% dow range for the kbr
        dow_rate                        %% dow rate for the kbr
        dow_acc                         %% dow accelaration for the kbr
        dow_range_crn                   %% dow range for the kbr after crn filtering
        dow_rate_crn                    %% dow rate for the kbr after crn filtering
        dow_acc_crn                     %% dow accelaration for the kbr after crn filtering
        k_phase                         %% the phase of kbr for this satellite
        ka_phase                        %% the phase of kbra for this satellite 
        k_freq                          %% the frequency of the kbr for this satellite
        ka_freq                         %% the frequency of the kbra for this satellite        
    end
    
    methods
        function obj = Kbr(rcv_time, kbr, kbra, eps_time, freq_k, freq_ka)
            % Kbr Construct an instance of this class
            %   Detailed explanation goes here
            %   input must be Level-1A
            obj.rcv_time = rcv_time;
            obj.k_phase = kbr;
            obj.ka_phase = kbra;
            obj.eps_time = eps_time;
            obj.k_freq = freq_k;
            obj.ka_freq = freq_ka;
        end
        
        function obj = phase_wrap(obj, date, id)
            % the k and ka wave phase recorded in th level-1a data product
            % is not consistent, so the procedure of wrapping them
            % together is required.
            
            % struc_kbr_phase = readtable("../output/phase_wrap_" + date + "_" + id + ".txt", ...
            %     "FileType", "text", "ReadVariableNames", false, ...
            %     "ReadRowNames", false, "Delimiter", ',', ...
            %     'HeaderLines', 1);
            % struc_kbr_phase.Properties.VariableNames = {'k_phase', 'ka_phase'};
            % obj.k_phase = struc_kbr_phase.k_phase;
            % obj.ka_phase = struc_kbr_phase.ka_phase;
            date = date;
            index_wrap_k = find(abs(obj.k_phase(2: end) - obj.k_phase(1: end - 1)) > 10000000.0) + 1;
            index_wrap_ka = find(abs(obj.ka_phase(2: end) - obj.ka_phase(1: end - 1)) > 10000000.0) + 1;
            sign = 0.0;
            if id == 'C'
                sign = -1.0;
            else 
                sign = 1.0;
            end
            for index = 1: length(index_wrap_k)
                obj.k_phase(index_wrap_k(index): end) = obj.k_phase(index_wrap_k(index): end) + hpf(100000000.0 * sign);
            end
            for index = 1: length(index_wrap_ka)
                obj.ka_phase(index_wrap_ka(index): end) = obj.ka_phase(index_wrap_ka(index): end) + hpf(100000000.0 * sign);
            end
        end
        
        function obj = interp_phase(obj)
            % the time tags for the k wave phase in the 1a data is
            % rcv_time, but the gps_time is required.
            temp = [0.0: 0.1: 86399.9]';
            gps_time_uneq = obj.eps_time + 0.05 + temp;
            obj.gps_time = obj.rcv_time - 0.05;
            obj.k_phase = interp1(gps_time_uneq, obj.k_phase, temp, 'spline', 'extrap');
            obj.ka_phase = interp1(gps_time_uneq, obj.ka_phase, temp, 'spline', 'extrap');
        end
        
        function obj = crn_filter(obj, degree, cutoff_freq, order)

            % % % % % % CRN FILTER PARAMETRE % % % % % %
            format long;
            N = degree; %787 %747
            B = cutoff_freq;
            C = order;                    % C = 7,  9, or 11
            Nb = floor(B * N / 10); %8
            DOWR_end = [zeros(length(obj.dow_range), 1), obj.dow_range];
            % % % % % % CRN FILTER PARAMETRE % % % % % %
            
            % % % % % % Calculating   H(k) % % % % % %
            H = zeros(N, 1);
            for k = -(N - 1) / 2 : (N - 1) / 2
                z = 0;
                for j = -Nb: Nb
                    a = sin(pi * (k - j) / C);
                    b = sin(pi * (k - j) / N);
                    x = a / b;
                    if(k == j)
                        x = N / C;
                    end
                    y = x ^ C;
                    z = z + y;
                end
                H((k + (N + 1) / 2), 1) = z;
            end    
            % % % % % % Normalisation H(k) % % % % % %
            H = H / max(H);
            % % % % % % Normalisation H(k) % % % % % %

            F = zeros(N, 1);
            for j = -(N - 1) / 2 : (N - 1) / 2
                zz = 0;
                for k = -(N - 1) / 2 : (N - 1) / 2
                    zz = zz + H((k + (N + 1) / 2), 1) * cos(2 * pi * k * j / N);
                end
                F((j + (N + 1) / 2), 1) = zz;
            end
          

            Y = fft(F);%FFT
            mag = abs(Y);
            maxnumber = max(mag);
            Y = Y / maxnumber;
            F = ifft(Y);

            F1 = zeros(N, 1);
            for j = -(N - 1) / 2: (N - 1) / 2
                zz1 = 0;
                for k = -(N - 1) / 2: (N - 1) / 2
                    zz1 = zz1 + H((k + (N + 1) / 2), 1) * sin(2 * pi * k * j / N) * -1 * (2 * pi * k * 10 / N);
                end
                F1((j + (N + 1) / 2), 1) = zz1;
            end

            Y = fft(F1);%FFT
            maxnumber = N;
            Y = Y / maxnumber;
            F1 = ifft(Y);

            F2 = zeros(N, 1);
            for j = -(N - 1) / 2 : (N - 1) / 2
                zz2 = 0;
                for k = -(N - 1) / 2 : (N - 1) / 2
                    zz2 = zz2 + H((k + (N + 1) / 2), 1) * cos(2 * pi * k * j / N) * -1 * (2 * pi * k * 10 / N) ^ 2;
                end
                F2((j + (N + 1) / 2), 1) = zz2;
            end

            Y2 = fft(F2);%FFT
            maxnumber = N;
            Y2 = Y2 / maxnumber;
            F2 = ifft(Y2);

            [m, ~] = size(DOWR_end(:, 1));
            yy = zeros(m, 1);
            for i = (N + 1) / 2 : m - (N - 1) / 2   
                for j = -(N - 1) / 2 : (N - 1) / 2
                    yy(i, 1) = yy(i) + F((j + (N + 1) / 2), 1) * DOWR_end(i - j, 2);          
                end
            end 
            DOWR_end(:, 2) = yy;

            yy = zeros(m, 1);
            for i = (N + 1) / 2 : m - (N - 1) / 2     
                for j = -(N - 1) / 2 : (N - 1) / 2
                    yy(i, 1) = yy(i) + F1((j + (N + 1) / 2), 1) * DOWR_end(i - j, 2);
                end
            end

            DOWR_rate(:, 1) = DOWR_end(:, 1);
            DOWR_rate(:, 2) = yy;

            bb = zeros(m, 1);
            for i = (N + 1) / 2 : m - (N - 1) / 2     
                for j = -(N - 1) / 2 : (N - 1) / 2
                    bb(i, 1) = bb(i) + F2((j + (N + 1) / 2), 1) * DOWR_end(i - j, 2);
                end
            end
            
            writematrix([F, F1, F2], '../output/crn_filter_coeffs.txt');

            DOWR_acceleration(:, 1) = DOWR_end(:, 1);
            DOWR_acceleration(:, 2) = bb;
            obj.dow_range_crn = DOWR_end(:, 2);
            obj.dow_rate_crn = DOWR_rate(:, 2);
            obj.dow_acc_crn = DOWR_acceleration(:, 2);
        end     
        
        function obj = interp_phase_lagrange(obj)
            obj.gps_time = obj.rcv_time - 0.05;
            KBR1A_C_GPS_DATA = zeros(length(obj.rcv_time), 2);
            KBR1A_C_GPS_DATA(:, 1) = roundn(obj.gps_time - floor(obj.gps_time(1)), -1);
            KBR1A_C_GPS_DATA(:, 2) = KBR1A_C_GPS_DATA(:, 1) + obj.eps_time + 0.05;
            TIME = KBR1A_C_GPS_DATA(:, 1); % LAGRANGE POINT
            % TIME = TIME(2: end, 1);
            TIME = TIME(:, 1);
            KBR_TIME = KBR1A_C_GPS_DATA(:, 2); % ALREAD KNOW POINT
            D = obj.k_phase; % Y1 K_PHASE
            E = obj.ka_phase; % Y2 KA_PHASE

            [m, ~] = size(TIME);

            % y0 = zeros(m - 1, 1);
            % y1 = zeros(m - 1, 1);
            y0 = zeros(m, 1);
            y1 = zeros(m, 1);

            for i = 1: 1: 3
                x = [KBR_TIME(1), KBR_TIME(2), KBR_TIME(3), KBR_TIME(4)];
                y = [D(1), D(2), D(3), D(4)];
                k = [E(1), E(2), E(3), E(4)];
                y0(i) = Lagrange(x, y, TIME(i));
                y1(i) = Lagrange(x, k, TIME(i));
            end

            for i = 4: 1: m - 4
                x = [KBR_TIME(i - 3), KBR_TIME(i - 2), KBR_TIME(i - 1), KBR_TIME(i), KBR_TIME(i + 1), KBR_TIME(i + 2), KBR_TIME(i + 3), KBR_TIME(i + 4)];
                y = [D(i - 3), D(i - 2), D(i - 1), D(i), D(i + 1), D(i + 2), D(i + 3), D(i + 4)];
                k = [E(i - 3), E(i - 2), E(i - 1), E(i), E(i + 1), E(i + 2), E(i + 3), E(i + 4)];
                y0(i) = Lagrange(x, y, TIME(i));
                y1(i) = Lagrange(x, k, TIME(i));
            end

            for i = m - 3: 1: m      
                x = [KBR_TIME(m - 3), KBR_TIME(m - 2), KBR_TIME(m - 1), KBR_TIME(m)];
                y = [D(m - 3), D(m - 2), D(m - 1), D(m)];
                k = [E(m - 3), E(m - 2), E(m - 1), E(m)];
                y0(i) = Lagrange(x, y, TIME(i));
                y1(i) = Lagrange(x, k, TIME(i));
            end

            obj.k_phase = y0;
            obj.ka_phase = y1;
            % obj.k_freq(1) = []; obj.ka_freq(1) = [];
            % obj.k_freq(end) = []; obj.ka_freq(end) = [];
            % obj.gps_time(1) = []; obj.gps_time(end) = [];
        end
        
        function [residual_range, residual_rate, residual_accl] = compare_with_1b(obj, date)
            struc_kbr1b = readtable("../input/KBR1B_" + date + "_Y_04.txt", ...
                "FileType", "text", "ReadVariableNames", false, ...
                "ReadRowNames", false, "Delimiter", ' ', ...
                'HeaderLines', 162, "MultipleDelimsAsOne", true);
            struc_kbr1b.Properties.VariableNames = {'gps_time', 'biased_range', 'biased_rate', ...
                'biased_accl', 'iono_corr', 'lighttime_corr', 'lighttime_rate', 'lighttime_accl', ...
                'ant_centr_corr', 'ant_centr_rate', 'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr', ...
                'ka_b_snr', 'qualflg'};
            time_epoch_range = 449: 50: 863589;
            time_epoch_rate = 749: 50: 863289; 
            time_epoch_accl = 1449: 50: 862589;
            residual_range = zeros(length(time_epoch_range), 3);
            residual_rate = zeros(length(time_epoch_rate), 3);
            residual_accl = zeros(length(time_epoch_accl), 3);
            
            residual_range(:, 1) = struc_kbr1b.biased_range(10: end - 8);
            residual_rate(:, 1) = struc_kbr1b.biased_rate(16: end - 14);
            residual_accl(:, 1) = struc_kbr1b.biased_accl(30: end - 28);
            residual_range(:, 2) = obj.dow_range_crn(time_epoch_range);
            residual_rate(:, 2) = obj.dow_rate_crn(time_epoch_rate);
            residual_accl(:, 2) = obj.dow_acc_crn(time_epoch_accl);
            residual_range(:, 3) = struc_kbr1b.biased_range(10: end - 8) - obj.dow_range_crn(time_epoch_range);
            residual_rate(:, 3) = struc_kbr1b.biased_rate(16: end - 14) - obj.dow_rate_crn(time_epoch_rate);
            residual_accl(:, 3) = struc_kbr1b.biased_accl(30: end - 28) - obj.dow_acc_crn(time_epoch_accl);
        end
        
        function obj = shape_wave(obj, dowr)
            jump_index = find(abs(diff(dowr)) > 10000.0) + 1;
            jump_value = zeros(length(jump_index), 1);
            obj.dow_range = dowr;
            for i = 1: length(jump_index)
                jump_value(i) = dowr(jump_index(i) - 1) - dowr(jump_index(i)) + (dowr(jump_index(i) + 1) - dowr(jump_index(i))) / 2.0;
                obj.dow_range(jump_index(i): end) = obj.dow_range(jump_index(i): end) + jump_value(i);
            end
        end
        
        function [residual_range, residual_rate, residual_accl, hn, hn_dot, hn_ddot, kaiser_r] = kaiser_filter(obj, date)
            struc_kbr1b = readtable("../input/KBR1B_" + date + "_Y_04.txt", ...
                "FileType", "text", "ReadVariableNames", false, ...
                "ReadRowNames", false, "Delimiter", ' ', ...
                'HeaderLines', 162, "MultipleDelimsAsOne", true);
            struc_kbr1b.Properties.VariableNames = {'gps_time', 'biased_range', 'biased_rate', ...
                'biased_accl', 'iono_corr', 'lighttime_corr', 'lighttime_rate', 'lighttime_accl', ...
                'ant_centr_corr', 'ant_centr_rate', 'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr', ...
                'ka_b_snr', 'qualflg'};
            load('kaiser_coeff.mat');
            hn = kaiser_coeff;
            hn_dot = diff9(1.0, hn);
            hn_ddot = diff9(1.0, hn_dot) * 100.0;
            
            kaiser_r = zeros(length(obj.dow_range), 3);
            
            % range
            kaisered_range = filter(hn, [1.0 0], obj.dow_range);
            time_epoch_range = 2024: 50: (864000 - 2024);
            residual_range = zeros(length(time_epoch_range), 3);
            residual_range(:, 1) = struc_kbr1b.biased_range(23: end - 58);
            residual_range(:, 2) = kaisered_range(time_epoch_range);
            residual_range(:, 3) = struc_kbr1b.biased_range(23: end - 58) - kaisered_range(time_epoch_range);
            
            % rate
            kaisered_rate = filter(hn_dot, [1.0 0], obj.dow_range) * 10.0;
            time_epoch_rate = 5370: 50: (864000 - 5370);
            residual_rate = zeros(length(time_epoch_rate), 3);
            residual_rate(:, 1) = struc_kbr1b.biased_rate(90: end - 125);
            residual_rate(:, 2) = kaisered_rate(time_epoch_rate);
            residual_rate(:, 3) = struc_kbr1b.biased_rate(90: end - 125) - kaisered_rate(time_epoch_rate);
            
            % accl
            kaisered_accl = filter(hn_ddot, [1.0 0], obj.dow_range);
            time_epoch_accl = 3116: 50: (864000 - 3116);
            residual_accl = zeros(length(time_epoch_accl), 3);
            residual_accl(:, 1) = struc_kbr1b.biased_accl(45: end - 80);
            residual_accl(:, 2) = kaisered_accl(time_epoch_accl);
            residual_accl(:, 3) = struc_kbr1b.biased_accl(45: end - 80) - kaisered_accl(time_epoch_accl);
            
            % sum
            kaiser_r(:, 1) = kaisered_range;
            kaiser_r(:, 2) = kaisered_rate;
            kaiser_r(:, 3) = kaisered_accl;
        end
    end
end

