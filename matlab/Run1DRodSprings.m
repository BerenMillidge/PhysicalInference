function [x,v,DsL,PE,KE] = Run1DRodSprings(K,L,x,v,gamma,NoSteps,dt)
%simualte spring confined on 1D ros in x-dimensioan..they are 1 unit apart
%in y
%K =is spring constants
%L = natural length
%gamma is damping
N = length(K);
for t=2:NoSteps
    deltax = x(:,t-1)- x(:,t-1)';%change in x
    sL= sqrt(ones(N)+deltax.^2);%spring lengyh
    sumv =  sum(-K.*deltax.*(ones(N)-L./sL)-gamma*v(:,t-1),2);
    sumx=  v(:,t-1);
    
    v(:,t)=v(:,t-1)+dt*sumv;
    x(:,t)=x(:,t-1)+dt*sumx;
    DsL(:,:,t) =  sL-L;%change in spring lenght for energet
end
PE  = 1/2*squeeze(sum(sum(1/2*K.*(DsL).^2,1),2))';%potential enegry
KE =  sum(1/2*v.^2);%kinetic enegry 
