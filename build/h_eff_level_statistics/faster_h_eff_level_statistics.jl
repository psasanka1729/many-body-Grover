using NPZ
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles

L = 12;

file = raw""*string(L)*"_new_Grover_gates_data.txt" # Change for every L.
M = readdlm(file)
Gates_data_1 = M[:,1];
Gates_data_2 = M[:,2];
Gates_data_3 = M[:,3];

U_0_gate_number =  (L            # L X gate on left of MCX
                  + 1            # H gate on left of MCX
                  + 2*L^2-6*L+5  # MCX gate
                  + 1            # H gate on right of MCX
                  + L)           # L X gate on right of MCX

U_x_gate_number =  (L-1          # L-1 H gate on left of MCX
                  + L-1          # L-1 X gate on left of MCX
                  + 1            # Z gate on left of MCX
                  + 2*L^2-6*L+5  # MCX gate
                  + 1            # Z gate on right of MCX
                  + L-1          # L-1 H gate on right of MCX   
                  + L-1)          # L-1 X gate on right of MCX)             
Number_of_Gates = U_0_gate_number+U_x_gate_number

SEED = 4000+parse(Int64,ARGS[1])
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

I2 = sparse([1 0; 0 1]);
Z  = sparse([1 0;0 -1]);
X  = sparse([0 1;1 0])
H  = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)       = sparse(exp(-1im*(theta/2)*collect(I2-X)));
Hadamard(noise) = sparse(exp(-1im*(pi/2+noise)*collect(I2-H))) #Ry(pi/2+noise)*Pauli_Z;
CX(noise)       = sparse(exp(-1im*((pi/2+noise))*collect(I2-X))); # This is X gate.
Z_gate(noise)   = sparse(exp(-1im*(pi/2+noise)*collect(I2-Z))) #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise
Identity(dimension) = spdiagm(0 => ones(dimension));
int(x) = floor(Int,x);

function single_qubit_gate_matrix(single_qubit_gate, qubit)
    
    ## The case Qubit=1 is treated differently because we need to
    # initialize the matrix as U before starting the kronecker product.
    
    if qubit == 1
        
        gate_matrix = sparse(single_qubit_gate)
        for i=2:L
            gate_matrix = kron(gate_matrix, I2)
        end
        
    #=
        Single qubit gates acting on qubits othe than the first.
        =#
    else
        
        gate_matrix = I2
        for i=2:L
            if i == qubit
                gate_matrix = kron(gate_matrix, single_qubit_gate)
            else
                gate_matrix = kron(gate_matrix, I2)
            end
        end
    end
    
    return gate_matrix
end;

function single_qubit_controlled_gate_matrix(single_qubit_gate,c,t)

    # |0><0|.
    PI_0 = (I2+Z)/2
    # |1><1|.
    PI_1 = (I2-Z)/2
     
    Matrices = Dict("I" => I2,"PI_0" => PI_0,"U" => single_qubit_gate, "PI_1" => PI_1)
    
    p0 = fill("I", L)
    p1 = fill("I", L)
    
    p0[c] = "PI_0"
    p1[c] = "PI_1"
    p1[t] = "U"

    
    PI_0_matrix = Matrices[p0[1]]
    for i = 2:L
        PI_0_matrix = kron(PI_0_matrix,Matrices[p0[i]])
    end        
        
    PI_1_matrix = Matrices[p1[1]]   
    for i = 2:L
        PI_1_matrix = kron(PI_1_matrix,Matrices[p1[i]])        
    end
           
    return sparse(PI_0_matrix + PI_1_matrix)     
end;

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;
#V = py"eigu"(G_exact)[2];

function h_eff_eigenvalues(DELTA)
    
    U_list = [];
    H_list = [];
    U_x_delta = sparse(Identity(2^L));
    
    NOISE_list = []

    
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)
        
            push!(U_list,single_qubit_gate_matrix(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
        
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-H,Qubit)) #= H_had = I2-Had. =#            
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)             

            push!(U_list,single_qubit_gate_matrix(CX(0.0),Gates_data_3[i])) # Noiseless.
            
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-X,Qubit)) #= H_X = I2-X. =#   
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            
            push!(U_list,single_qubit_gate_matrix(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-Z,Qubit)) #= H_Z = I2-Z. =#  
            
        else
            #push!(ux_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

            push!(U_list,single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiselss.
           
            Angle = Gates_data_1[i]
            Control_Qubit = int(Gates_data_2[i])
            Target_Qubit = int(Gates_data_3[i])
            #= H = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
            Matrices = Dict("I" => I2,"U" => [1 -1;-1 1]/2, "PI_1" => (I2-Z)/2)
            p1 = fill("I", L)
            p1[Control_Qubit] = "PI_1"
            p1[Target_Qubit] = "U"
            H_k = Matrices[p1[1]]
            for i = 2:L
                H_k = kron(H_k,Matrices[p1[i]])
            end
            push!(H_list,H_k)
        end
    end
    

    U_0_delta = sparse(Identity(2^L));    
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

            push!(U_list,single_qubit_gate_matrix(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-H,Qubit)) #= H_had = I2-Had. =#               
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

            push!(U_list,single_qubit_gate_matrix(CX(0.0),Gates_data_3[i])) # Noiseless.
            
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-X,Qubit)) #= H_X = I2-X. =# 
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

            push!(U_list,single_qubit_gate_matrix(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
            Qubit = Gates_data_3[i] # qubit.
            push!(H_list,single_qubit_gate_matrix(I2-Z,Qubit)) #= H_Z = I2-Z. =#              
            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

            push!(U_list,single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiseless.
            
            Angle = Gates_data_1[i]
            Control_Qubit = int(Gates_data_2[i])
            Target_Qubit = int(Gates_data_3[i])
            #= H = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
            Matrices = Dict("I" => I2,"U" => [1 -1;-1 1]/2, "PI_1" => (I2-Z)/2)
            p1 = fill("I", L)
            p1[Control_Qubit] = "PI_1"
            p1[Target_Qubit] = "U"
            H_k = Matrices[p1[1]]
            for i = 2:L
                H_k = kron(H_k,Matrices[p1[i]])
            end
            push!(H_list,H_k)
        end
    end
        
    #= The following loop sums over all epsilon to get H_eff. =#

        #=
        Terms before and after the half of the number of gates will be summed seperatly.
        =#
        #= 
        For k < (Number of gates)/2
            The kth_term = 
                        G_0 * U_1^\dagger * U_2^\dagger * ... * U_k^\dagger
                        * H_k *
                        (G_0 * U_1^\dagger * U_2^\dagger * ... * U_k^\dagger)^\dagger
        For k > (Number of gates)/2
            The kth_term = 
                        (U_k+1 * ... * U_Number of Gates)^\dagger
                        * H_k *
                        (U_k+1 * ... * U_Number of Gates)

         =#
            
    h_eff = spzeros(2^L,2^L);
    
    # First time it starts with an identity.
    u_matrix_product = Identity(2^L)
    
    #=
    From k =1 to Number_of_Gates/2.
    =#
    for k = 1:int(length(U_list)/2)          
        u_matrix_product = (U_list[k])' * u_matrix_product         
        h_eff_left_side = sparse(G_exact) * u_matrix_product
        h_eff += NOISE_list[k] * h_eff_left_side * H_list[k] * (h_eff_left_side')
    end        
    
    # First time it starts with an identity.
    h_eff_left_side = Identity(2^L)
    
    # From k = length(Number_of_Gates)/2 to length(Number_of_Gates).
    for k = length(U_list):-1:int(length(U_list)/2)+2
        h_eff_left_side *= U_list[k]
        h_eff += NOISE_list[k-1]*h_eff_left_side * H_list[k-1] * (h_eff_left_side')
    end      
    
    #= The k = Number_of_Gates term. =#
    h_eff += NOISE_list[length(U_list)] * H_list[length(U_list)]   
    
    h_eff = h_eff[3:2^L,3:2^L]; # Deleting the |0> and |xbar> basis.
    E_eff_D = eigvals(collect(h_eff)) # Diagonalizing H_eff matrix.
    E_eff_D_sorted = sort(real(E_eff_D),rev = true); # Soring the eigenvalues in descending order.    
    
    return E_eff_D_sorted
end;

h_eff_eigvals = h_eff_eigenvalues(0.0);

npzwrite("h_eff_eigenvalues.npy",h_eff_eigvals)

eigenvalues_diff = Array{Float64, 1}(undef, 0)
for i = 2:2^L-2 # relative index i.e length of the eigenvector array.
    push!(eigenvalues_diff,h_eff_eigvals[i]-h_eff_eigvals[i-1])
end
npzwrite("h_diffence_eigenvalues.npy",eigenvalues_diff)

function Level_Statistics(n,Es)
    return min(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n])) / max(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))
end;

h_eff_level_statistics = Array{Float64, 1}(undef, 0)
for i = 2:2^L-3 # relative index i.e length of the eigenvector array.
    push!(h_eff_level_statistics,Level_Statistics(i,h_eff_eigvals))
end
npzwrite("h_eff_level_statistics.npy",h_eff_level_statistics)
