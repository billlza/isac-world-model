function matlab_phy_smoke(summaryCsv)
%MATLAB_PHY_SMOKE Base-MATLAB checks for the discrete MIMO-OFDM ISAC math.
% No proprietary toolbox functions are used in this minimal smoke.
% This deliberately does not claim MATLAB and Python noise-seed parity.
% Run from the repository root:
%   addpath('scripts'); matlab_phy_smoke('runs/phy-baseline/summary.csv');
% For the full MathWorks example, search the documentation for:
% "Integrated Sensing and Communication II: Communication-Centric Approach Using MIMO-OFDM".

N = 128; cp = 24; symbols = 64; tx = 2; radarRx = 8;
rangeBin = 8; angleTrue = 18; dopplerBin = -6;

% Independent check: a delayed OFDM symbol is a subcarrier phase ramp
% when its delay fits in the cyclic prefix.
k = (0:N-1).';
X = exp(1i * pi / 2 * mod(k,4));
x = ifft(X) * sqrt(N);
wave = [x(end-cp+1:end); x];
y = wave(cp+1-rangeBin : cp+N-rangeBin);
Y = fft(y) / sqrt(N);
expected = X .* exp(-1i * 2*pi * k * rangeBin / N);
assert(max(abs(Y - expected)) < 1e-10, 'OFDM CP delay / FFT check failed');

% Two simultaneous transmitted QPSK streams. Radar receives the very same
% known OFDM stream signals; no target truth is passed into the estimator.
rng(11, 'twister');
q = randi([0 3],symbols,N,tx);
S = exp(1i*pi/2*q) / sqrt(tx);
bs = exp(1i*pi*(0:tx-1).'*sind(angleTrue));
rx = exp(1i*pi*(0:radarRx-1).'*sind(angleTrue));
projected = zeros(symbols,N);
for t=1:tx
    projected = projected + conj(bs(t))*S(:,:,t);
end
phaseRange = exp(-1i*2*pi*(0:N-1)*rangeBin/N);
phaseDoppler = exp(1i*2*pi*dopplerBin*(0:symbols-1).'/symbols);
R = zeros(symbols,N,radarRx);
for r=1:radarRx
    R(:,:,r) = rx(r)*((projected .* phaseDoppler) .* phaseRange);
end

best = -inf; angleEstimate = NaN; delayEstimate = NaN; dopplerIndex = NaN;
for angle=-60:2:60
    candidateRx = exp(1i*pi*(0:radarRx-1).'*sind(angle));
    candidateTx = exp(1i*pi*(0:tx-1).'*sind(angle));
    receiveSum = zeros(symbols,N);
    transmitSum = zeros(symbols,N);
    for r=1:radarRx
        receiveSum = receiveSum + conj(candidateRx(r))*R(:,:,r);
    end
    for t=1:tx
        transmitSum = transmitSum + conj(candidateTx(t))*S(:,:,t);
    end
    cube = ifft(fft(receiveSum.*conj(transmitSum),[],1),[],2);
    magnitude = abs(cube(:,2:cp)).^2;
    magnitude(1,:)=0;  % The zero-Doppler slot is intentionally excluded.
    [peak,location] = max(magnitude(:));
    if peak > best
        best = peak;
        [dopplerIndex, d] = ind2sub(size(magnitude),location);
        angleEstimate = angle;
        delayEstimate = d; % column 1 represents delay bin 1
    end
end
signedDoppler = mod(dopplerIndex-1 + symbols/2,symbols)-symbols/2;
assert(delayEstimate == rangeBin, 'Range estimate does not match the known bin');
assert(abs(angleEstimate-angleTrue) <= 2, 'Direction estimate mismatch');
assert(signedDoppler == dopplerBin, 'Doppler FFT bin mismatch');
fprintf('MATLAB PHY smoke PASSED. Delay=%d, angle=%.1f deg, Doppler bin=%d.\n', ...
        delayEstimate, angleEstimate, signedDoppler);

% Optional: inspect actual Python Monte Carlo CSV produced by phy_sweep.py.
if nargin > 0 && ~isempty(summaryCsv)
    T = readtable(summaryCsv);
    figure('Name','ISAC communication-sensing resource tradeoff');
    plot(T.raw_payload_rate_bps/1e3,100*T.target_cell_detection_rate,'o-','LineWidth',1.5);
    grid on;
    xlabel('Raw uncoded payload bit rate (kbps)');
    ylabel('Correct range-angle-Doppler cell (%)');
    title('Synthetic matched-seed PHY experiment (NOT hardware validation)');
end
end
