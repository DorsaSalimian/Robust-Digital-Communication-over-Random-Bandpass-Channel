
function r = simulate_channel_project(s_in, Fs, alpha, sigma)
% Inputs:
%   s_in  : input signal (vector)
%   Fs    : sampling frequency in Hz (e.g., 22050)
%   alpha : amplitude in passband (0 < alpha <= 1)
%   sigma : standard deviation of AWGN noise
% Output:
%   r     : received signal after filtering, noise, delay and clipping

if nargin < 4
    sigma = 0.05; % Default noise standard deviation
end
if nargin < 3
    alpha = 1; % Default passband amplitude
end
if nargin < 2
    Fs = 22050; % Default sampling frequency
end

n = length(s_in);
k = 0:n-1;
T = n / Fs;
freqs = k / T;

% FFT of input
S = fft(s_in) / n;

% Define band centers (in Hz)
band_centers = 2000 * (1:4);

% Initialize full filter (symmetric band-pass minus one band)
H = ones(1, n);

% Select one band to delete randomly
range_deleted = randi(4);

% Delete that band in lower frequency range
f_low = (range_deleted - 1) * 2000 + 1000; % Lower bound
f_high = f_low + 2000;                     % Upper bound
idx1 = find(freqs >= f_low, 1);
idx2 = find(freqs >= f_high, 1);
H(idx1:idx2) = 0;

% Mirror deletion on the upper frequency side
f_low_mirror = Fs - f_high;
f_high_mirror = Fs - f_low;
idx3 = find(freqs >= f_low_mirror, 1);
idx4 = find(freqs >= f_high_mirror, 1);
H(idx3:idx4) = 0;

% Apply filter
S_filtered = S .* H;

% IFFT and scale
filtered_sig = real(ifft(S_filtered) * n * 8e5);

% Generate delay samples
delay1 = randi([2000, 20000]);
delay2 = randi([2000, 50000]);

noise_delay1 = sigma * randn(1, delay1);
noise_delay2 = sigma * randn(1, delay2);
noise_sig = sigma * randn(1, length(filtered_sig));

% Apply noise to signal
noisy_sig = filtered_sig + noise_sig;

% Concatenate noise before and after
r = [noise_delay1, noisy_sig, noise_delay2];


% Clip output to range [-1, 1]
r = min(max(r, -1), 1);

% Optional debug
fprintf('Deleted band: %d kHz to %d kHz\n', f_low / 1000, f_high / 1000);
fprintf('Delay1: %d samples, Delay2: %d samples\n', delay1, delay2);

end