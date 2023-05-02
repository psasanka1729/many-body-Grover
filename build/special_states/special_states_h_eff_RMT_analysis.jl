using PyCall
#using NPZ
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

DELTA = 0.08
SEED = 80000+parse(Int64,ARGS[1])
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
           
    return PI_0_matrix + PI_1_matrix     
end;

# n is from 0 to 2^L-2.
function x_bar(n)
    s = zeros(2^L)
    k_n = (2*pi*n)/(2^L-1)
    for j=1:2^L-1
        ket_j      = zeros(2^L);
        ket_j[j+1] = 1 # Julia counts from 1.
        s = s+ exp(1im*(j-1)*k_n)*ket_j
    end
    return s/sqrt(2^L-1)
end;

function sigma_z_to_x_bar_basis_change_matrix(L)
    V     = spzeros(2^L,2^L)
    
    ket_0 = spzeros(2^L)
    ket_0[1] = 1
    
    ket_1    = spzeros(2^L);
    ket_1[2] = 1
    
    ket_xbar = x_bar(0)
    
    eigenstate_1 = (ket_0-1im*ket_xbar)/sqrt(2)
    eigenstate_2 = (ket_0+1im*ket_xbar)/sqrt(2)
    
    V = V+ ket_0*(eigenstate_1')
    V = V+ ket_1*(eigenstate_2')
    
    # The buk.
    for n=1:2^L-2
        # ket_(n+1) has n+2 th position as 1 in computational basis.
        ket_n    = spzeros(2^L);
        ket_n[n+2] = 1 
        
        V = V+ket_n*(x_bar(n)')
    end
    return V
end;

#basis_change_matrix = sigma_z_to_x_bar_basis_change_matrix(L);

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s>
G_exact = U_x*U_0;

function Grover_delta(DELTA)

    noise_sum = 0
    U_x_delta = Identity(2^L)
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            U_x_delta *= single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
                      
            noise_sum += epsilon
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_gate_matrix(CX(DELTA*epsilon),Gates_data_3[i])   

            noise_sum += epsilon
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_gate_matrix(Z_gate(DELTA*epsilon),Gates_data_3[i])
 
            noise_sum += epsilon
            
        else
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            
            noise_sum += epsilon/4
            
        end
    end
    

    U_0_delta = Identity(2^L);    
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]      
            U_0_delta *= single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])          
            
            noise_sum += epsilon
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]       
            U_0_delta *= single_qubit_gate_matrix(CX(DELTA*epsilon),Gates_data_3[i])
            
            noise_sum += epsilon
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]     
            U_x_delta *= single_qubit_gate_matrix(Z_gate(DELTA*epsilon),Gates_data_3[i])          
            
            noise_sum += epsilon
            
        else

            epsilon = NOISE[i]     
            U_0_delta *= single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            
            noise_sum += epsilon/4
            
        end 
    end
    GROVER_DELTA = U_x_delta*U_0_delta
    return GROVER_DELTA
end;

function h_eff_special_states(h, delta_1)

    #=Derivative of G(\delta) is calculated using forward difference.=#
    function h_eff_from_derivative(h::Float64)
        h_eff_matrix = 1im*((Grover_delta(h)*(-G_exact)')-Identity(2^L))/h
        return h_eff_matrix
    end;    

    #= Construction of block 2x2 H_eff matrix.=#
    
    # Defining the state |0> in sigma_z basis.
    ket_0    = zeros(2^L)
    ket_0[1] = 1
    
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x    = (1/sqrt(N))   * ones(N)
    ket_xbar = sqrt(N/(N-1)) * ket_x - 1/sqrt(N-1)*ket_0 # Normalization checked.
    
    # Basis is chosen as (|0> +i*|x_bar>)/sqrt(2) and (|0> -i*|x_bar>)/sqrt(2).
    basis_1 = (ket_0 + 1im*ket_xbar)/sqrt(2) # quasi-energy = -arctan(2/sqrt(N-1)).
    basis_2 = (ket_0 - 1im*ket_xbar)/sqrt(2) # quasi-energy = arctan(2/sqrt(N-1)).
    
    h_eff_matrix_whole = h_eff_from_derivative(h)
    
    # Writing the H_spec matrix in the (|0> +i*|x_bar>)/sqrt(2) and (|0> -i*|x_bar>)/sqrt(2) basis.
    h_1_1 = basis_1' * h_eff_matrix_whole * basis_1
    h_1_2 = basis_1' * h_eff_matrix_whole * basis_2
    h_2_1 = basis_2' * h_eff_matrix_whole * basis_1
    h_2_2 = basis_2' * h_eff_matrix_whole * basis_2
    
    # h_eff as 2x2 block matrix.
    h_eff_block_matrix = [ h_1_1 h_1_2; h_2_1 h_2_2]
    
    phi = -atan(2/sqrt(N-1))

    # Making the h_spec matrix tracelss to write it in terms of Pauli matrices.
    #=
        If M is any matrix, then M' = M-tr(M)/2 is a tracelss matrix.
    =#
    return delta_1*((h_eff_block_matrix) .- tr(h_eff_block_matrix)/2)#+phi*[1 0;0 -1] + 
end;

function sigma_y_to_sigma_z_basis_change(Matrix)
    
    sigma_y_n = (1/sqrt(2))*[1 -1im]'   # corresponding to -1 eigenvalue.
    sigma_y_p = (1/sqrt(2))*[1  1im]'   # corresponding to +1 eigenvalue.
    
    sigma_z_n = [0 1]'                   # corresponding to -1 eigenvalue.
    sigma_z_p = [1 0]'                   # corresponding to +1 eigenvalue.
    
    V = spzeros(2,2)
    
    V = V + sigma_z_n * sigma_y_n'
    V = V + sigma_z_p * sigma_y_p'
    
    return V*Matrix*V'
end;

# Changing the H_spec matrix from sigma_y basis to sigma_z basis.
h_spec_y_basis           = h_eff_special_states(1.e-8, DELTA)
h_spec_z_basis           = sigma_y_to_sigma_z_basis_change(h_spec_y_basis);

#=
Write the matrix B as B = B_0 * sigma_0 + B_1 * sigma_1 + B_2 * sigma_2 + B_3 * sigma_3.
=#
function Pauli_coefficients(B)
    
    sigma_x = [[0 1];
               [1 0]]
    
    sigma_y = [[0 -1im];
               [1im 0]]
    
    sigma_z = [[1 0];
               [0 -1]]
    
    B_0 = tr(B)/2
    B_1 = tr(sigma_x*B)/2
    B_2 = tr(sigma_y*B)/2
    B_3 = tr(sigma_z*B)/2
    
    return B_0,B_1,B_2,B_3
end;

pauli_coefficients_file  = open("pauli_coefficients.txt", "w")
PC = Pauli_coefficients(h_spec_z_basis)
write(pauli_coefficients_file , string(real(PC[1])))
write(pauli_coefficients_file, "\t")
write(pauli_coefficients_file , string(real(PC[2])))
write(pauli_coefficients_file, "\t")
write(pauli_coefficients_file , string(real(PC[3])))
write(pauli_coefficients_file, "\t")
write(pauli_coefficients_file , string(real(PC[4])))

h_spec_eigenvalues_file  = open("h_spec_eigenvalues.txt", "w")
h_spec_eigenvalues = eigvals(h_spec_z_basis)
write(h_spec_eigenvalues_file , string(real(h_spec_eigenvalues[1])))
write(h_spec_eigenvalues_file, "\t")
write(h_spec_eigenvalues_file , string(real(h_spec_eigenvalues[2])))


h_spec_energy_file  = open("h_spec_energy.txt", "w")
phi = -atan(2/sqrt(2^L-1))
lambda = sqrt(phi^2+2*phi*PC[3]+DELTA^2*(PC[1]^2+PC[2]^2+PC[3]^2))
write(h_spec_energy_file , string(real(lambda)))
