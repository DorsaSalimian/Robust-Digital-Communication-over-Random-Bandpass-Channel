clc;
clear;
close all;
rgbColors=[136, 0, 21 ; 238,93,108 ;234,112,55 ; 116,49,139; 123,192,67 ] / 255 ;

%% Parameters
M = 16;                                 % PAM order
Fs = 22050;                             % Sampling frequency
sps = 16;                               % Samples per symbol
rolloff = 0.25;                         % Rolloff factor for RRC filter
span = 10;                              % Filter span in symbols
sigma = 0.005;                          % Noise standard deviation
alpha = 1;                              % Passband amplitude
text_input = 'Bits are the basic units of information and the universal currency of communication in the digital age They carry data across networks devices and systems forming the language behind everything from simple messages to complex applications like video calls online gaming and AI In todays connected world bits are the invisible threads linking our communication infrastructure enabling fast data transfer cloud services and multimedia Without them the constant flow of information that powers innovation education and global interaction would not exist Their role keeps growing as technology advances making bits the foundation of digital life';

%% Text to bits
ascii_vals = double(text_input);
bin_data = de2bi(ascii_vals, 8, 'left-msb')';
bin_data = bin_data(:);

%% Huffman encoding
symbols = unique(bin_data);
if length(symbols) < 2
    symbols = [0; 1];
    prob = [0.5; 0.5];
else
    prob = histc(bin_data, symbols) / numel(bin_data);
end
dict = huffmandict(symbols, prob);
huff_encoded = huffmanenco(bin_data, dict);

%% Hamming (7,4) Encoding
msg_len = length(huff_encoded);
pad_size = mod(4 - mod(msg_len, 4), 4);
huff_padded = [huff_encoded; zeros(pad_size, 1)];
msg_matrix = reshape(huff_padded, 4, [])';
G = [1 0 0 0 1 1 0;
     0 1 0 0 1 0 1;
     0 0 1 0 0 1 1;
     0 0 0 1 1 1 1];
coded_matrix = mod(msg_matrix * G, 2);
coded_data = coded_matrix';
coded_data = coded_data(:);

%% Raised Cosine Filter Design
filt = rcosdesign(rolloff, span, sps, 'sqrt');

%% PAM Modulation
k = log2(M);
pad_mod = mod(k - mod(length(coded_data), k), k);
coded_data_padded = [coded_data; zeros(pad_mod, 1)];
symb = bi2de(reshape(coded_data_padded, k, []).', 'left-msb');
mod_signal = pammod(symb, M, 0, 'gray');

%% Create and Concatenate Pilots
pilot_sym_len = 32;
zero_pad_len  = 20;

%----------------------- FIX START ---------------------------
% Define pilot symbols manually as the known, non-normalized integer levels.
% This creates a solid, unambiguous reference for channel estimation.
pilot_symbols = repmat([-15; 15], pilot_sym_len/2, 1);
%------------------------ FIX END ----------------------------

pilot_shaped = upfirdn(pilot_symbols, filt, sps, 1).';
% tx_signal = upfirdn(mod_signal, filt, sps, 1);
% tx_signal_with_pilots = [pilot_shaped, tx_signal.', pilot_shaped];

tx_symbols = [pilot_symbols; zeros(zero_pad_len,1); ...
                mod_signal; zeros(zero_pad_len,1); pilot_symbols].'; 
tx_signal = upfirdn(tx_symbols, filt, sps, 1);   % single √RRC pass

% normalize signal to prevent clipping in channel
peak_amp = max(abs(tx_signal));   % includes filter overshoot
tx_signal_norm = tx_signal / peak_amp;   % now ∈ [‑1, +1]

%% Transmission Loop
fc_all = [2000, 4000, 6000, 8000];
recovered_texts = strings(1, length(fc_all));
t = (0:length(tx_signal_norm)-1) / Fs;

for b = 1:length(fc_all)
    fprintf('\n--- Transmitting on band centered at %d Hz ---\n', fc_all(b));
    band_signal = real(tx_signal_norm .* exp(1j * 2 * pi * fc_all(b) * t));
    
    % Pre-scale the signal to counteract the channel's internal amplification
    band_signal_scaled = band_signal / 8e5;
    rx_signal = simulate_channel_project(band_signal_scaled, Fs, alpha, sigma);
    
    %% POWER CHECK
    expected_len = length(tx_signal_norm);
    if length(rx_signal) < expected_len
        recovered_texts(b) = "corrupted (signal too short)";
        continue;
    end

    rx_segment = rx_signal(1:expected_len);
    noise_power_est = mean(rx_signal(1:2000).^2);
    power_threshold = 10 * noise_power_est;
    segment_power = mean(abs(rx_segment).^2);

    if segment_power < power_threshold
        recovered_texts(b) = "corrupted (low power)";
        continue;
    end

    %% Fine Timing Adjustment and Channel Estimation
    % 1. Baseband conversion
    n     = 0:length(rx_signal)-1;
    rx_bb = rx_signal .* exp(-1j*2*pi*fc_all(b)*n/Fs);
    
    % 2. Matched filter (adds another grpD samples)
    rx_mf = conv(filt,rx_bb);
    
    % Locate BOTH pilots, slice data block -----------------------
    grpD  = span*sps/2;                      % 80‑sample group delay per √RRC
    pilot_ref = upfirdn(pilot_shaped, filt, 1, 1);  % oversampled reference
    mf_2 = flipud(conj(pilot_ref(:)));     % pilot after two SRRC filters

    corr_v = abs(conv(rx_mf, mf_2, 'valid'));  % column
    
    % ---- Detect peaks -------------------------------------------------------
    [pk_max, locs] = findpeaks(corr_v, 'SortStr','descend');
%     peak_thresh = 0.9 * pk_max(1);           % ≥90 % of strongest
%     min_sep     = length(pilot_ref);         % at least one pilot apart
%     valid_peaks = locs(corr_v(locs) >= peak_thresh);
%     valid_peaks = sort(valid_peaks);         % ascending order
%     
%     % Enforce minimum separation
%     keep = true(size(valid_peaks));
%     for k = 2:numel(valid_peaks)
%         if (valid_peaks(k) - valid_peaks(k-1)) < min_sep
%             keep(k) = false;
%         end
%     end
%     pilot_peaks = valid_peaks(keep);
%     
%     if numel(pilot_peaks) ~= 2
%         recovered_texts(b) = "corrupted (pilot detect failed)";
%         continue;                               % skip this band
%     end
    
    
%     % ---- Convert peak sample -> symbol index ---------------------------------
%     start_pilot_samp = pilot_peaks(1) - length(mf_2) + 1;   % first sample of TX‑filtered pilot
%     stop_pilot_samp  = pilot_peaks(2) - length(mf_2) + 1;
%     
%     % Remove the RX filter delay so index points to the *first symbol* of pilot:
%     start_sym_samp   = start_pilot_samp - grpD;
%     stop_sym_samp    = stop_pilot_samp  - grpD;

    start_sym_samp = min(locs(1),locs(2)) + 2*grpD;
    stop_sym_samp  = max(locs(1),locs(2)) + 2*grpD;    

%     start_sym_samp = 1 + 2*grpD;
%     stop_sym_samp  = length(corr_v) + 2*grpD;
%     
    % Symbol‑spaced extraction of ENTIRE frame (pilot1 | data | pilot2)
    frame_syms = rx_mf(start_sym_samp : sps : stop_sym_samp + (pilot_sym_len-1)*sps);   % symbols between two pilots 
    
    % Split them
    rx_pilot1 = frame_syms(1:pilot_sym_len);
    rx_pilot2 = frame_syms(end-pilot_sym_len+1:end);
    data_rx   = frame_syms(pilot_sym_len+zero_pad_len+1 : end-pilot_sym_len-zero_pad_len);
    
    % ---- Channel estimate: average the two pilots ---------------------------
    H1 = dot(pilot_symbols, rx_pilot1) / dot(pilot_symbols, pilot_symbols);
    H2 = dot(pilot_symbols, rx_pilot2) / dot(pilot_symbols, pilot_symbols);
    H_est = 0.5*(H1+H2);
    
%     if abs(H_est) < 0.1
%         recovered_texts(b) = "corrupted (H too weak)";
%         continue;
%     end

    fprintf('Pilot peaks @ %d and %d – |H| = %.2f  ∠ %.1f°\n', ...
        start_sym_samp, stop_sym_samp, abs(H_est), rad2deg(angle(H_est)));
    
    data_corr = data_rx / H_est;                 % equalise gain & phase

    %% Demodulation
    rx_symbols = pamdemod(real(data_corr), M, 0, 'gray');
    
    %% Decode bitstream
    rx_bits = de2bi(rx_symbols, k, 'left-msb')';
    rx_bits = rx_bits(:);
    
    if length(rx_bits) < length(coded_data)
       recovered_texts(b) = "corrupted (length mismatch)";
       continue;
    end
    rx_bits = rx_bits(1:length(coded_data));

    %% Hamming Decoding
    rx_matrix = reshape(rx_bits, 7, [])';
    P = G(:,5:7);
    H = [P', eye(3)];
    syndromes = mod(rx_matrix * H', 2);
    syndrome_dec = bi2de(syndromes, 'left-msb');
    
    error_pos = [0, 7, 6, 3, 5, 2, 1, 4];

    for i = 1:size(rx_matrix, 1)
        err_bit_idx = error_pos(syndrome_dec(i) + 1);
        if err_bit_idx ~= 0, rx_matrix(i, err_bit_idx) = ~rx_matrix(i, err_bit_idx); end
    end
    decoded_data = rx_matrix(:, 1:4)';
    decoded_data = decoded_data(:);
    decoded_data = decoded_data(1:length(huff_padded));
    decoded_data = decoded_data(1:msg_len);
    
    %% Huffman Decoding
    try
        huff_decoded = huffmandeco(decoded_data, dict);
        bit_matrix = reshape(huff_decoded, 8, [])';
        ascii_out = bi2de(bit_matrix, 'left-msb');
        recovered_text = char(ascii_out)';
        if strcmpi(strtrim(recovered_text), strtrim(text_input))
            recovered_texts(b) = string(recovered_text);
        else
            recovered_texts(b) = "corrupted (text content mismatch)";
        end
    catch
        recovered_texts(b) = "corrupted (huffman failed)";
    end
end

%% Final Results
final_text = 'ERROR: No message could be recovered from any band.';

for i = 1:length(recovered_texts)
    if strcmpi(strtrim(recovered_texts(i)), strtrim(text_input))
        final_text = recovered_texts(i);
        fprintf('\nSuccessfully recovered text from band %d!\n', i);
        break;
    end
end
disp('-----------------------------------------');
disp('Original Text:');
disp(text_input);
disp(' ');
disp('Recovered Text per Band:');
disp(recovered_texts');
disp(' ');
disp('Final Selected Text:');
disp(final_text);

%% ================== (a) Signal Plots and Analysis =====================
% Plot 1: Channel Output in Time Domain
figure;
currenta=rgbColors(1, :);
plot((0:length(rx_signal)-1)/Fs,rx_signal,'color',currenta);
title('Received Signal from Channel (Time)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot 2: Data Signal after Removing Pilots - Time Domain
figure;
currentb=rgbColors(2, :);
plot(abs(data_corr),'color',currentb);
title('Data After Removing Pilots (Time Domain)');
xlabel('Symbol Index');
ylabel('Magnitude');
grid on;

% Plot 3: Data After Removing Pilots - Frequency Domain
N = length(data_corr);
f = Fs * (0:N-1) / N;
Y = abs(fft(data_corr));
currentc=rgbColors(3, :);
figure;
plot(f, Y , 'Color',currentc);
title('Spectrum of Data After Removing Pilots');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
xlim([0 Fs/2]);
grid on;

% Plot 4: Baseband Signal After Demodulation - Time Domain
figure;
currentd=rgbColors(4, :);
plot(real(rx_bb), 'Color',currentd);
title('Baseband Signal (Time Domain)');
xlabel('Sample Index');
ylabel('Amplitude');
grid on;

% Plot 5: Baseband Signal - Frequency Domain
N = length(rx_bb);
f = Fs * (0:N-1) / N;
Y = abs(fft(rx_bb));
currente=rgbColors(5, :);
figure;
plot(f, Y, 'Color',currente);
title('Spectrum of Baseband Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;
 
% Plot 6: Received Bitstream Before Text Conversion
figure;
stairs(rx_bits(1:min(500,end)));
title('Received Bitstream');
xlabel('Bit Index');
ylabel('Bit Value');
ylim([-0.5 1.5]);
grid on;


%% ================= (c) Effect of alpha on SNR =====================
alphas = 0.1:0.1:1;
snr_vals = zeros(size(alphas));

for i = 1:length(alphas)
    r_test = simulate_channel_project(tx_signal_norm, Fs, alphas(i), sigma);
    signal_power = mean(r_test.^2);
    noise_power = sigma^2;
    snr_vals(i) = 10*log10(signal_power / noise_power);
end


figure;
plot(alphas, snr_vals, 'm*-','LineWidth',2);
xlabel('alpha (passband gain)');
ylabel('SNR (dB)');
title('Effect of alpha on SNR');
grid on;

%% ================= (d) Effect of sigma on BER =====================
sigmas = 0.001:0.002:0.02;
ber_vs_sigma = zeros(size(sigmas));

for i = 1:length(sigmas)
    r_test = simulate_channel_project(tx_signal_norm, Fs, alpha, sigmas(i));
    r_test_bb = r_test .* exp(-1j*2*pi*fc_all(4)*(0:length(r_test)-1)/Fs);
    r_mf = conv(filt, r_test_bb);
    r_sym = r_mf(span*sps+1 : sps : span*sps+length(mod_signal)*sps);
    r_demod = pamdemod(real(r_sym), M, 0, 'gray');
    r_bits = de2bi(r_demod, k, 'left-msb')';
    r_bits = r_bits(:);
    r_bits = r_bits(1:length(coded_data));
    ber_vs_sigma(i) = sum(r_bits ~= coded_data) / length(coded_data);
end

figure;
semilogy(sigmas, ber_vs_sigma, 'kd-','LineWidth',2);
xlabel('Noise std. deviation (\sigma)');
ylabel('BER');
title('BER vs Noise Level (\sigma)');
grid on;

%% =============== (e) Channel Capacity Estimation ===================
B = 2000; % Hz (each band width)
snr_linear = 10.^(snr_vals/10);
capacity = B * log2(1 + snr_linear);

figure;
plot(alphas, capacity, 'co-','LineWidth',2);
xlabel('alpha');
ylabel('Estimated Capacity (bps)');
title('Estimated Channel Capacity vs alpha');
grid on;
%% ================= (b) BER vs. Data Rate ==========================
sps_values = [32, 24, 16, 12, 8, 6, 4]; 
data_rates = (Fs ./ sps_values) * log2(M);
ber_with_coding = zeros(size(sps_values));
ber_without_coding = zeros(size(sps_values));
fc = 4000; 

for i = 1:length(sps_values)
    current_sps = sps_values(i);
    fprintf('\nCalculating for sps = %d (Data Rate = %.2f bps)\n', current_sps, data_rates(i)); 
    filt = rcosdesign(rolloff, span, current_sps, 'sqrt');

    %with channel coding
    pad_mod = mod(log2(M) - mod(length(coded_data), log2(M)), log2(M));
    coded_padded = [coded_data; zeros(pad_mod, 1)];
    symb = bi2de(reshape(coded_padded, log2(M), []).', 'left-msb');
    mod_signal = pammod(symb, M, 0, 'gray');
    tx_signal = upfirdn(mod_signal, filt, current_sps, 1);
    peak_amp = max(abs(tx_signal));
    tx_signal_norm = tx_signal / peak_amp;

    t = (0:length(tx_signal_norm)-1) / Fs;

    exp_term = exp(1j*2*pi*fc*t);  % row vector
    tx_passband = real(tx_signal_norm.' .* exp_term);  
    tx_passband = tx_passband / 8e5;

    rx_signal = simulate_channel_project(tx_passband, Fs, alpha, sigma);

    n = 0:length(rx_signal)-1;
    rx_bb = rx_signal .* exp(-1j*2*pi*fc*n/Fs);
    rx_mf = conv(filt, rx_bb);

    grpD = span * current_sps / 2;
    rx_syms = rx_mf(grpD+1:current_sps:end);

    rx_symbols = pamdemod(real(rx_syms), M, 0, 'gray');
    rx_bits = de2bi(rx_symbols, log2(M), 'left-msb')';
    rx_bits = rx_bits(:);
    rx_bits = rx_bits(1:length(coded_padded));

    num_err = sum(rx_bits ~= coded_padded);
    ber_with_coding(i) = num_err / length(coded_padded);

    clear tx_signal tx_signal_norm t exp_term tx_passband rx_signal rx_bb rx_mf rx_syms rx_symbols rx_bits

    %without channel coding 
    pad_mod2 = mod(log2(M) - mod(length(huff_encoded), log2(M)), log2(M));
    no_coding_padded = [huff_encoded; zeros(pad_mod2,1)];
    symb_nc = bi2de(reshape(no_coding_padded, log2(M), []).', 'left-msb');
    mod_signal_nc = pammod(symb_nc, M, 0, 'gray');
    tx_signal_nc = upfirdn(mod_signal_nc, filt, current_sps, 1);
    peak_amp_nc = max(abs(tx_signal_nc));
    tx_signal_norm_nc = tx_signal_nc / peak_amp_nc;

    t_nc = (0:length(tx_signal_norm_nc)-1) / Fs;

    exp_term_nc = exp(1j*2*pi*fc*t_nc);
    tx_passband_nc = real(tx_signal_norm_nc.' .* exp_term_nc);
    tx_passband_nc = tx_passband_nc / 8e5;

    rx_signal_nc = simulate_channel_project(tx_passband_nc, Fs, alpha, sigma);

    n_nc = 0:length(rx_signal_nc)-1;
    rx_bb_nc = rx_signal_nc .* exp(-1j*2*pi*fc*n_nc/Fs);
    rx_mf_nc = conv(filt, rx_bb_nc);

    rx_syms_nc = rx_mf_nc(grpD+1:current_sps:end);

    rx_symbols_nc = pamdemod(real(rx_syms_nc), M, 0, 'gray');
    rx_bits_nc = de2bi(rx_symbols_nc, log2(M), 'left-msb')';
    rx_bits_nc = rx_bits_nc(:);
    rx_bits_nc = rx_bits_nc(1:length(no_coding_padded));

    num_err_nc = sum(rx_bits_nc ~= no_coding_padded);
    ber_without_coding(i) = num_err_nc / length(no_coding_padded);

   clear tx_signal_nc tx_signal_norm_nc t_nc exp_term_nc tx_passband_nc rx_signal_nc rx_bb_nc rx_mf_nc rx_syms_nc rx_symbols_nc rx_bits_nc

end

valid_idx = (ber_with_coding < 1);

figure;
semilogy(data_rates(valid_idx), ber_with_coding(valid_idx), 'o-r', 'LineWidth', 2); hold on;
semilogy(data_rates(valid_idx), ber_without_coding(valid_idx), 's-b', 'LineWidth', 2);
grid on;
xlabel('Data Rate (bps)');
ylabel('BER');
title('BER vs Data Rate (with and without Channel Coding)');
legend('With Coding', 'Without Coding');
