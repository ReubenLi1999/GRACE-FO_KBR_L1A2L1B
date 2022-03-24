function s = generate_CRN_filter2(varargin)
% This function return a struct s containing CRN filter coefficients.
% A CRN filter is a finite impulse response filter.
%
% Literature:
%       https://www.techbriefs.com/component/content/article/tb/techbriefs/information-sciences/2242
% and    GRACE ATBD_L1B_v1.2.pdf, page 40 of 54
%
%
% Input parameters described below. The function can plot the filter
% response.
%
% Output struct:
%       s.filtercoeff = [1xNf] dimension
%       s.phasedelay_sec: the delay of the filter in second
%
%
% The filter can be applied in matlab using the filter function
% on a time-series y:
%
%   yout = filter(s.filtercoeff, 1, y);
%
% Written by Vitali Mller, AEI Hannover, 2019-02-20
% 
p = inputParser;
mislogical = @(s) (islogical(s) || s==1 || s==0);
addParameter(p, 'Fs', 10, @isnumeric); % Sampling frequency in Hz
addParameter(p, 'Nf', 707, @isnumeric); % Filter length in samples at Fs (odd number)
addParameter(p, 'Nc', 7, @isnumeric); % Convolution Number (odd integer)
addParameter(p, 'fc', 0.1, @isnumeric); % cut-off frequency of low-pass in Hz
addParameter(p, 'f0', 0.37e-3, @isnumeric); % dominant signal frequency in Hz
addParameter(p, 'plot', true, mislogical); % dominant signal frequency in Hz
parse(p, varargin{:});

Fs = p.Results.Fs;
Nc = p.Results.Nc;
Nf = p.Results.Nf;
fc = p.Results.fc;
f0 = p.Results.f0;
Tf = Nf/Fs;
NB = round(fc*Tf);
% fcnew = NB/Tf;
Nh = (Nf-1)/2;
s = struct();
s.filtercoeff = func_Fn([-Nh:Nh],Nc,Nf,NB,f0,Fs);
s.phasedelay_sec = (Nf-1)/2/Fs;

% if p.Results.plot
%     figure(); 
%     PN = 80000;
%     sp = [s.filtercoeff, zeros(1,PN)];
%     L=length(sp); 
%     Ln = ceil(L/2+1);
%     f = Fs/2*linspace(0,1,Ln);
%     fftsig = fft(sp);
%     semilogx(f, 20*log10(abs(fftsig(1:Ln))) );
%     xlabel('Fourier Frequency [Hz]');
%     ylabel('Magnitude [dB]');
%     title('Filter Response: Mangitude');
%         
%     figure(); 
%     phase = atan2(imag(fftsig(1:Ln)), real(fftsig(1:Ln)));
%     phase = unwrap(phase);
%     loglog(f,  phase );
%     xlabel('Fourier Frequency [Hz]');
%     ylabel('Phase [rad]');
%     title('Filter Response: Phase');    
%     
%     figure(); 
%     loglog(f, abs(1-abs(fftsig(1:Ln))) );
%     xlabel('Fourier Frequency [Hz]');
%     ylabel('abs(1-Magnitude)');
%     title('Filter Response: abs(1-Magnitude) ');
%     
%     figure(); 
%     dphase = phase./f/(2*pi);
%     semilogx(f,  dphase );
%     fprintf('Phase delay at DC: %.3f sec \n', mean(dphase(2:10)));
%     xlabel('Fourier Frequency [Hz]');
%     ylabel('Phase Delay [sec]');
%     title('Filter Response: Phase Delay');    
%     
% end

end


%% See GRACE ATBD_L1B_v1.2.pdf, page 40 of 54
function Hk = func_Hk(k, Nc, Nf, NB)
    Hk = zeros(size(k));
    for n=1:length(k)
        m = k(n)-[-NB:NB];
        % if m == 0, the result is NaN, however, the limit is (Nf/Nc)^Nc
        t( m ~= 0 ) = ( sin(pi*m(m ~= 0)/Nc)./sin(pi*m(m ~= 0)/Nf) ).^Nc;
        t( m == 0) = (Nf/Nc)^Nc; 
        Hk(n) = sum(t);
    end
end

function Fn = func_Fn(n, Nc, Nf, NB, f0, Fs)
    Nh = (Nf-1)/2;
    k=[-Nh:Nh];
    if (max(abs(n)) > Nh)
       error('n < Nh'); 
    end
    Fn = zeros(size(n));
    for s=1:length(n)
        Fn(s) = sum(func_Hk(k,Nc,Nf,NB).*cos(2*pi*k*n(s)/Nf));
    end
    Fnorm = sum(cos(2*pi*f0*k/Fs).*Fn); % usual gain formula for an FIR filter
    Fn = Fn / Fnorm;
end