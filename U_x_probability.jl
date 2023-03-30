L = 6;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
using JLD

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

SEED = 746
Delta = 0.0
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= sparse(exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0])));
Hadamard(noise) = sparse(exp(-1im*(pi/2+noise)*(I2-H))) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = sparse(exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0]))); # This is X gate.
Z_gate(noise) = sparse(exp(-1im*(pi/2+noise)*(I2-Z))); # noise # noise
Identity(dimension) = spdiagm(0 => ones(dimension));
int(x) = floor(Int,x);

function Matrix_Gate(Gate, Qubit) # Previously known as multi qubit gate.
    
    ## The case Qubit=1 is treated differently because we need to
    # initialize the matrix as U before starting the kronecker product.
    
    if Qubit == 1
        
        M = sparse(Gate)
        for i=2:L
            M = kron(M, Identity(2))
        end
        
    else
        
        M = sparse([1 0;0 1])
        for i=2:L
            if i == Qubit
                M = kron(M, Gate)
            else
                M = kron(M, Identity(2))
            end
        end
    end
    
    return sparse(M)
end;

function CU(U,c,t)
    
    I2 = Identity(2)
    Z = sparse([1 0;0 -1])

    PI_0 = (I2+Z)/2
    PI_1 = (I2-Z)/2
     
    Matrices = Dict("I" => I2,"PI_0" => PI_0,"U" => U, "PI_1" => PI_1)
    
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
           
    #return p0,p1
    return sparse(PI_0_matrix + PI_1_matrix)     
end;

function U_x_Matrix(DELTA)

    U_x_delta = Identity(2^L);

    # U_x
    for i = U_0_gate_number+1 : U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_x_delta *= h_matrix

            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]    
            x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= x_matrix
            
        elseif Gates_data_1[i] == "Z"

            epsilon = NOISE[i]    
            z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix
            
        else

            epsilon = NOISE[i]      
            rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            U_x_delta *= rx_matrix
            
        end
    end
end;

U_x = U_x_Matrix(0.0);

save("U_x_Matrix.jld", "U_x", U_x)