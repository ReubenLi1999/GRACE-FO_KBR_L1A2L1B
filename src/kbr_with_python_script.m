% timer start
tic
% typical header
clc;
clear;
close all;
warning('off');
dbstop if error;
format long;

date = "2019-01-01";
load('kaiser_coeff.mat');

struc_kbr1a_c = readtable("../output/DOWR1A_" + date + "_Y_04.txt", ...
    "FileType", "text", "ReadVariableNames", false, ...
    "ReadRowNames", false, "Delimiter", ' ', ...
    'HeaderLines', 1, "MultipleDelimsAsOne", true);
struc_kbr1a_c.Properties.VariableNames = {'gps_time', 'raw_range'};

gracefo_c = Kbr(0, 0, 0, 0, 0, 0);
gracefo_c.gps_time = struc_kbr1a_c.gps_time;
gracefo_c.dow_range = struc_kbr1a_c.raw_range;

% crn filter
length_crn = 707;
cutoff_freq = 0.1; % unit=hz
order = 7;
gracefo_c = gracefo_c.crn_filter(length_crn, cutoff_freq, order);
% compare the results
[residual_range, residual_rate, residual_accl] = gracefo_c.compare_with_1b(date);
s = generate_CRN_filter2;
viltali = filter(s.filtercoeff, 1, gracefo_c.dow_range);

% write crn results to files and using Python to plot
residual_range([1550, 1552, 1553, 6990, 9594, 9591: 9593, 9596, 9597], 3) = 0;
residual_range(13497: 13505, 3) = 0;
residual_range(17021: 17031, 3) = 0;
writematrix(residual_range, "../output/crn_range_residual_" + date + ".txt");
writematrix(residual_rate, "../output/crn_rate_residual_" + date + ".txt");
writematrix(residual_accl, "../output/crn_accl_residual_" + date + ".txt");

% crn filter coefficients display
struc_crn_filter_coeff = readtable("../output/crn_filter_coeffs.txt", ...
    "FileType", "text", "ReadVariableNames", false, ...
    "ReadRowNames", false, "Delimiter", ',', ...
    "HeaderLines", 1);
struc_crn_filter_coeff.Properties.VariableNames = {'range', 'rate', 'accl'};

% FIR filter with kaiser window
[kaisered_range, kaisered_rate, kaisered_accl, kaiser_c, kaiser_cdot, kaiser_cddot, kaiser_r] = gracefo_c.kaiser_filter(date);

% write FIR-kaiser results to files and using Python to plot
writematrix(kaisered_range, "../output/kaiser_range_residual_" + date + ".txt");
writematrix(kaisered_rate, "../output/kaiser_rate_residual_" + date + ".txt");
writematrix(kaisered_accl, "../output/kaiser_accl_residual_" + date + ".txt");
writematrix(kaiser_r, "../output/kaiser_10hz" + date + ".txt");


% display the frequency response 
% range
figure('DefaultAxesFontSize', 16, 'color', [1 1 1]);
[h_crn, f_crn] = freqz(struc_crn_filter_coeff.range, 1.0, 100000);%amplitude-frequency characteristic diagram
[h_kaiser, f_kaiser] = freqz(kaiser_coeff, 1.0, 100000);
semilogx(f_crn * 10.0 / (2 * pi), 20.0 * log10(abs(h_crn)), 'linewidth', 1.7);% parameters are respectively frequecy and amplitude
hold on;
semilogx(f_kaiser * 10.0 / (2 * pi), 20.0 * log10(abs(h_kaiser)), 'linewidth', 1.7);
hold on;
xline(0.1, 'linestyle', '--', 'color', 'green', 'linewidth', 1.9);
axis([0.01, 10.0, -400, 100]);
grid on;
box on;
legend('CRN-707', 'FIR-kaiser-1851');
xlabel('Frequency [Hz]', 'fontsize', 20); 
ylabel('Gain [dB]', 'fontsize', 20); 
set(gca, 'FontSize', 20, 'linewidth', 1.1);

% rate
figure('DefaultAxesFontSize', 16, 'color', [1 1 1]);
[h_crn, f_crn] = freqz(struc_crn_filter_coeff.rate, 1.0, 100000);%amplitude-frequency characteristic diagram
[h_kaiser, f_kaiser] = freqz(kaiser_cdot * 10.0, 1.0, 100000);
semilogx(0: 0.001: 10, 20.0 * log10(0: 0.00625: 62.5), 'linewidth', 1.7);
hold on;
semilogx(f_crn * 10.0 / (2 * pi), 20.0 * log10(abs(h_crn)), 'linewidth', 1.7);% parameters are respectively frequecy and amplitude
hold on;
semilogx(f_kaiser * 10.0 / (2 * pi), 20.0 * log10(abs(h_kaiser)), 'linewidth', 1.7);
hold on;
xline(0.1, 'linestyle', '--', 'color', 'green', 'linewidth', 1.9);
axis([0.001, 10.0, -400, 100]);
grid on;
box on;
legend('Ideal first order differentiator', 'CRN-707', 'FIR-kaiser-1851');
xlabel('Frequency [Hz]', 'fontsize', 20); 
ylabel('Gain [dB]', 'fontsize', 20); 
set(gca, 'FontSize', 20, 'linewidth', 1.1);

% accl
figure('DefaultAxesFontSize', 16, 'color', [1 1 1]);
[h_crn, f_crn] = freqz(struc_crn_filter_coeff.accl, 1.0, 100000);%amplitude-frequency characteristic diagram
[h_kaiser, f_kaiser] = freqz(kaiser_cddot, 1.0, 100000);
semilogx(0: 0.001: 10, 40.0 * log10(0: 0.00625: 62.5), 'linewidth', 1.7);
hold on;
semilogx(f_crn * 10.0 / (2 * pi), 20.0 * log10(abs(h_crn)), 'linewidth', 1.7);% parameters are respectively frequecy and amplitude
hold on;
semilogx(f_kaiser * 10.0 / (2 * pi), 20.0 * log10(abs(h_kaiser)), 'linewidth', 1.7);
hold on;
xline(0.1, 'linestyle', '--', 'color', 'green', 'linewidth', 1.9);
axis([0.001, 10.0, -400, 100]);
grid on;
box on;
legend('Ideal two order differentiator', 'CRN-707', 'FIR-kaiser-1851');
xlabel('Frequency [Hz]', 'fontsize', 20); 
ylabel('Gain [dB]', 'fontsize', 20); 
set(gca, 'FontSize', 20, 'linewidth', 1.1);

toc
