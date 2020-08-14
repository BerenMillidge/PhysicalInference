od 
% simulate M1 aka rHN-s
% run many relaxatios without any learning
% then with Lenght learning
% all on the same set of random symmetric constraint problem
% Note that this is only one run of each on each problem
close all;
clear variables;
N=100;
Repeats=30; % how many differnt Ws
T=75;
dt=0.05;time=0:dt:T;
RelaxT=length(time); %time of each relaxation
LearningRate=0.005;
Exps=2; % number of experiments with each W (each problem matrix); 1=no learning, 2= Full Hebb
%Performance=ones(Exps+1,Repeats); % just a place to store summary statistics
%sprintf('%d of %d',rep, Repeats)


for i=1:1:Repeats
    % --------------- prepare W0 -------------
    topologymask=ones(N); %fully connected
    %topologymask=(rand(N)>0.8); % sparse 0.8 (20% connected) works ok for both, 0.9 doesnt work for half Hebb, 0.95 full Hebb fails too
    topologymask=topologymask-topologymask.*eye(N); % no self weights
    topologymask=triu(topologymask); topologymask=topologymask+topologymask'; % symmetric
    
    %spring lenghts
    L0=2*rand(N)+0.5; % random weights
    L0=triu(L0); L0=L0+L0'; % symmetric
    L0=L0.*topologymask;
    
    
    %spring constants
    K0=rand(N)*3; % random weights
    K0=triu(K0); K0=K0+K0'; % symmetric
    K0=K0.*topologymask;
    
    %damping
    gamma = 0.1;
    Traj=zeros(RelaxT,N); % state trajectory over one relaxation
    
    %different relaxations with learning
    Relax1 =200; %No learning
    Relax2 =800; %with learning
    
    for exp=1:2; % 1=no learning, 2= Full Hebb, 3= Half Hebb
        K=K0; % set initial weights to original values
        L=L0; % set initial weights to original values
        if(exp==2)
            Relaxations=Relax2;
            
        else
            Relaxations=Relax1;
        end
        E=zeros(RelaxT,Relaxations); % energy over one relaxation
        Eend=zeros(Relaxations,1); % energies at the end of each relaxation
        for r=1:Relaxations % many relaxations in each experiment
            r
            x=randn(N,1); % intial random states
            v=randn(N,1); % intial random states
            [x,v,DsL,PE,KE] = Run1DRodSprings(K,L,x,v,gamma,RelaxT,dt);
            Trajx=x';
            Trajv=v';
            
            
            %let spring settle from learnt position
            if(exp==2)
                [dummy1,dummy2,dummy3,PEOrig,dummy4] = Run1DRodSprings(K,L0,x(:,end),v(:,end),gamma,RelaxT/4,dt);
                Eend(r,1)=PEOrig(end);
                E(:,r) = (PE+KE)';
            else
                Eend(r,1)=PE(end);
                E(:,r) = (PE+KE)';
            end
            
            if (exp>1) % Exp 2 do some kind of learning
                deltaL=DsL(:,:,end)*LearningRate; % learning on spring lenghts ..Change in lenght*learnin rate
                L=L+deltaL;
                L(L<0.001) = 0.001; % dont let the lenght become zero
                L=L.*topologymask; % zero-out things that are not connected (inc self-weights)
            end
        end % of (all the relaxations in) the experiment
        
        if (exp==1) AllEend=Eend; else AllEend=cat(1,AllEend,Eend); end; % accumulate the points for the scatter plot
       if (exp==1)   Emin=min(Eend); Emax=max(Eend); Emean=mean(Eend); Enstd=std(Eend); end % this is the range w/o learning
       if (exp==2)    Elearn=Eend(Relaxations); end % last relaxation after Hebbian learning
        % f1=figure; imagesc(Trajx); colorbar; title('state trajectories');
        % f2=figure; plot(E); title('energy');
        % f3=figure; scatter([1:Relaxations],Eend); title('energy at end of each relaxation');
        %input('ok?')
        
    end %  of (all the experiments for) this repeat i.e. exps for this W0
    p=[Emin, Emax, Elearn];
    p=p-Emean;    p=p/Enstd; % scale between mean and +/- stdev
    Performance(:,i)=p;  % keep the scaled performance statistics
    %if (rep<5) % plot some more stuff for the first few repeats
 %   clf;
      figure; scatter([1:length(AllEend)],(AllEend-mean(AllEend))/std(AllEend)); title('1D Rod Springs');
      hold on; plot([Relax1,Relax1],[-1,3],'-');
     pause(0.1);
    
end