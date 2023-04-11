using JLD
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles

L = 8;

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

SEED = 1945
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

function u_h_JLD(DELTA)
    
    U_list = [];
    H_list = [];
    U_x_delta = sparse(Identity(2^L));
    
    NOISE_list = []

    
    # U_x
    @time for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
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
    @time for i = 1 : U_0_gate_number
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
    
    return U_list,H_list,NOISE_list
end;

U = u_h_JLD(0.0)
save("U_list.jld", "U", U[1])
save("H_list.jld", "H", U[2])
save("NOISE_list.jld", "Noise_list", U[3])
