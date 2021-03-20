xi = linspace(-10,10,400);
t=linspace(0,4*pi,200);
dt = t(2) - t(1);
[Xgrid,T] = meshgrid(xi,t);

f1 = sech(Xgrid+3).*(1*exp(i*2.3*T));
f2 = (sech(Xgrid).*tanh(Xgrid)).*(2*exp(i*2.8*T));
f = f1 + f2;

X = f.'; % transposed.
X1 = X(:,1:end-1);
X2 = X(:,2:end); 
% usually column is time ?
% unlike PCA, with DMD it matters to get the transpose correct otherwise modes in the wrong space (space vs time)

r = 2;
[U,S,V] = svd(X1,'econ');
Ur = U(:,1:r); % Ur=400x2 
Sr = S(1:r,1:r);
Vr = V(:,1:r);

Atilde = Ur'*X2*Vr/Sr; % Atilde=2x2
[W,D] = eig(Atilde);

Phi = X2*Vr/Sr*W; % DMD modes in real space
lambda = diag(D);
omega = log(lambda)/dt;

x1 = X(:,1); % initial condition
b = Phi\x1; 

%plot(real(Phi))

time_dynamics = zeros(r,length(t));
for iter=1:length(t)
  time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
end

X_dmd = Phi * time_dynamics; % outer prod

X_dmd(1:3,1:3)

%===============
VSinv = Vr/Sr;
UtX2 = Ur'*X2;
Atest = UtX2*VSinv;
PhiTest = X2*VSinv*W;

%===============
surfl(Xgrid,T,real(X_dmd).') 







