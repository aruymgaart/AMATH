L=10; n=64;
t2=linspace(-L,L,n+1);
t=t2(1:n);
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1];

u=sech(t);
ut = fft(u);
noise = 0.5 
utn=ut + noise*(rand(1,n) + i*randn(1,n));
un = ifft(utn);

filter=exp(-0.2*(k).^2); 
unft=filter.*utn; 
unf=ifft(unft);

ks = fftshift(k); 
utns = fftshift(utn);

plot(ks, utns/max(abs(utn)));
plot(t,un); % noisy signal in time domain
plot(fftshift(k), fftshift(abs(utn)/max(abs(utn))), fftshift(k), fftshift(filter),'m');
plot(fftshift(k), fftshift(abs(unft)/max(abs(unft))))
plot(t,unf,'k', t,u,'r')
plot(t,abs(unf),'k', t,u,'r')