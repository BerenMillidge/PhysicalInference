# attempt to replicate CHris' Matlab code in julia and ultimately improve performance of it

using LinearAlgebra
using Plots
using Statistics

function run_1d_rod_springs(K,L,x_init,v_init,gamma,NoSteps,dt)
    #simulated spring confined on 1D rods in x-dimensioan. They are 1 unit apart
    #in y
    #K =is spring constants
    #L = natural length
    #gamma is damping
    x = zeros(size(x_init)[1],Int(round(NoSteps)))
    v = zeros(size(v_init)[1],Int(round(NoSteps)))
    x[:,1] = x_init
    v[:,1] = v_init
    N = length(K[:,1])
    DsL = zeros(N,N,size(x)[2])
    print("NO steps")
    print(NoSteps)
    for t = 2:Int(round(NoSteps))
        deltax = x[:,t-1] * -transpose(x[:,t-1]) #change in x
        sL = sqrt.(ones(N,N) + deltax.^2) #spring length
        sumv = -K .* deltax .* (ones(N,N) - L ./ sL) - gamma * repeat(v[:,t-1],1,N)
        sumv = sum(sumv,dims=2)
        sumx = v[:,t-1]
        v[:,t] = v[:,t-1] + dt * sumv
        x[:,t] = x[:,t-1] + dt * sumx
        DsL[:,:,t] = sL - L #change in spring length for energy
    end
    print("PE presqueeze: ")
    print(size(sum(sum(1/2*K.*(DsL).^2,dims=1),dims=2)))
    PE = (1/2) * transpose(reshape(sum(sum(1/2*K.*(DsL).^2,dims=1),dims=2),(size(DsL)[3])))
    KE = sum(0.5 * v.^2,dims=1) #kinetic energy
    return x,v,DsL,PE,KE
end

# simulate M1 aka rHN-s
# run many relaxatios without any learning
# then with length learning
# all on the same set of random symmetric constraint problem
# Note that this is only one run of each on each problem
N = 10
Repeats = 30
T = 75
dt = 0.05
time = Array(0:dt:T)
RelaxT = length(time)
lr =0.005
num_experiments = 3
performances = ones((num_experiments +1, Repeats))
experiments = ["no_learning", "Full_Hebb"]

triu_symmetrize(x) = triu(x) + transpose(triu(x))
#incidentally I can't believe I have to do this due to stupid julia's identity matrix change!
function I(N)
    mat = zeros(N,N)
    for i = 1:N
        for j = 1:N
            mat[i,j] = 1
        end
    end
    return mat
end

for i = 1:1:Repeats
    #prepare W0
    topologymask = ones(N,N)
    #print("TOPOLOGY MASK: \n")
    #print(topologymask)
    #topologymask = topologymask[rand(N,N) .>0.8] #sparsity (20% connected)
    topologymask = topologymask - (topologymask .*I(N))#Matrix{I,N,N})#Matrix{Float32}{I,N,N})
    #print("TOPOLOGY MASK: \n")
    #print(topologymask)
    topologymask = triu(topologymask)
    topologymask = topologymask + transpose(topologymask)
    #print("TOPOLOGY MASK: \n")
    #print(topologymask)

    #spring lengths
    L0 = triu(2 .* rand(N,N) .+ 0.5)
    L0 = L0 + transpose(L0)
    L0 = L0 .* topologymask
    #spring constants
    K0 = rand(N,N).* 3 # random weights
    K0 = triu(K0)
    K0 = K0 + transpose(K0)
    K0 = K0 .* topologymask
    #damping
    gamma = 0.1
    Traj = zeros(RelaxT,N) # store state trajectory over one relaxation
    #different relaxations with learning
    Relax1 = 200
    Relax2 = 800
    Emin = -1
    Emax = -1
    Emean = -1
    Enstd = -1
    Elearn = -1
    for exp_name in experiments
        K = K0
        print("SIZE K0")
        print(size(K))
        L = L0
        if exp_name == "Full_Hebb"
            Relaxations=Relax2
        elseif exp_name =="no_learning"
            Relaxations=Relax1
        else
            error("Incorrect name provided")
        end
        E = zeros(RelaxT, Relaxations) #energies over one relaxation
        Eend = zeros(Relaxations,1) # energies at end of each relaxation
        for r = 1:Relaxations
            x = randn(N,1)
            v = randn(N,1) #initial random states
            x,v,DsL,PE,KE = run_1d_rod_springs(K,L,x,v,gamma,RelaxT,dt)
            Trajx = transpose(x)
            Trajv = transpose(v)
            #let spring settle from learnt position
            if exp_name == "Full_Hebb"
                _,_,_,PEOrig,_ = run_1d_rod_springs(K,L0,reshape(x[:,end],(N,1)),reshape(v[:,end],(N,1)),gamma,RelaxT/4,dt)
                Eend[r,1] = PEOrig[end]
                E[:,r] = transpose(PE + KE)
            else
                Eend[r,1] = PE[end]
                E[:,r] = transpose(PE + KE)
            end
            if exp_name != "no_learning"
                deltaL = DsL[:,:,end] * lr #update on sptring length
                L = L + deltaL
                L[L .< 0.001] .= 0.001 #don't let lengths become 0
                L = L.*topologymask #zero out things that are not connected
            end
        end
        AllEend = Eend
        if exp_name == "no_learning"
            Emin = minimum(Eend,dims=1)
            Emax = maximum(Eend,dims=1)
            Emean = mean(Eend,dims=1)
            Enstd = std(Eend,dims=1)
        else
            AllEend = cat(AllEend,Eend,dims=1)
        end
        if exp_name == "FullHebb"
            Elearn = Eend[Relaxations] # last relaxation after hebbian
        end
        #plot(Trajx)
        plot(E)
        scatter([1:Relaxations],Eend) #title("energy at end of each relaxation)
    end
    #TODO
    #p = [Emin, Emax, Elearn]
    #p = p - Emean
    #p = p/Enstd
    #Performance[:,i] = p #keep scaled performance statistics
    #if rep < 5
#        scatter(([1:length(AllEend)], AllEend-mean(AllEend))/std(AllEend))#
#        plot([Relax1, Relax1],[-1,3])
    #end
end
