#using PyCall
#using NPZ
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

SEED = 1912+parse(Int64,ARGS[1])
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

basis_change_matrix = sigma_z_to_x_bar_basis_change_matrix(L);

#=
function h_eff_eigensystem(DELTA)
    
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
        
    h_eff = spzeros(2^L,2^L);
    @time for k = 1:length(U_list)
        f_k = Identity(2^L);
        for i = k:length(U_list)-1
            f_k = f_k*U_list[length(U_list)-i+k]
        end     
        h_eff += NOISE_list[k]*f_k*H_list[k]*(f_k')
    end
    
    # h_eff_xbar = V * h_eff_z * V^{\dagger}.
    h_eff_xbar_basis = (basis_change_matrix)*h_eff*(basis_change_matrix') # Matrix in |0> and |xbar> basis.
    
    # Eigenvalues.
    h_eff_bulk = h_eff_xbar_basis[3:2^L,3:2^L]; # Deleting the |0> and |xbar> basis.
    h_eff_bulk_energies = eigvals(collect(h_eff_bulk)) # Diagonalizing H_eff matrix.
    h_eff_bulk_energies = sort(real(h_eff_bulk_energies),rev = true); # Soring the eigenvalues in descending order.
    
    # Eigenvectors.
    # Defining the state |0> in sigma_z basis.
    ket_0 = zeros(2^L)
    ket_0[1] = 1
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x = (1/sqrt(N))*ones(N)
    ket_xbar = sqrt(N/(N-1))*ket_x-1/sqrt(N-1)*ket_0 # Normalization checked.    
    P_0 = ket_0*ket_0'
    P_xbar = ket_xbar*ket_xbar'
    #= 
    h_eff_truncated = (1-|xbar><xbar|)(1-|0><0|)h_eff(1-|0><0|)(1-|xbar><xbar|).
    =#
    h_eff_truncated = (Identity(2^L)-P_xbar)*(Identity(2^L)-P_0)*h_eff*(Identity(2^L)-P_0)*(Identity(2^L)-P_xbar)    
    #h_eff_z_basis = basis_change_matrix*h_eff_truncated*(basis_change_matrix')
    h_eff_eigenvectors = eigvecs(h_eff_truncated) # Diagonalizing h_eff.    
    
    return h_eff,h_eff_bulk_energies,h_eff_eigenvectors
end;
=#

#eigensystem_h_eff = h_eff_eigensystem(0.0);

#h_eff_matrix = eigensystem_h_eff[1];

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s>
G_exact = U_x*U_0;

function Grover_delta(DELTA)

    U_x_delta = Identity(2^L)
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            U_x_delta *= single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
                      
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_gate_matrix(CX(DELTA*epsilon),Gates_data_3[i])   

            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_gate_matrix(Z_gate(DELTA*epsilon),Gates_data_3[i])

            
        else
        
            epsilon = NOISE[i]       
            U_x_delta *= single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])

            
        end
    end
    

    U_0_delta = Identity(2^L);    
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]      
            U_0_delta *= single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])          

            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]       
            U_0_delta *= single_qubit_gate_matrix(CX(DELTA*epsilon),Gates_data_3[i])

            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]     
            U_x_delta *= single_qubit_gate_matrix(Z_gate(DELTA*epsilon),Gates_data_3[i])          

            
        else

            epsilon = NOISE[i]     
            U_0_delta *= single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])

            
        end 
    end
    
    GROVER_DELTA = U_x_delta*U_0_delta
    return GROVER_DELTA
end;


function h_eff_from_derivative(h)
    h_eff_matrix = 1im*((Grover_delta(h)*(-G_exact)')-Identity(2^L))/h
    # h_eff_xbar = V * h_eff_z * V^{\dagger}.
    h_eff_matrix_xbar_basis = (basis_change_matrix)*h_eff_matrix *(basis_change_matrix') # Matrix in |0> and |xbar> basis.
    return h_eff_matrix_xbar_basis
end;


function h_eff_bulk_energies(h)
    h_eff_bulk = h_eff_from_derivative(h)[3:2^L,3:2^L]; # Deleting the |0> and |xbar> basis.
    h_eff_bulk_energies = eigvals(collect(h_eff_bulk)) # Diagonalizing H_eff matrix.
    effective_energies = sort(real(h_eff_bulk_energies),rev = true) # Soring the eigenvalues in descending order.
    return effective_energies
end;

eigenvalue_file       = open("eigenvalues.txt", "w")
level_statistics_file = open("level_statistics.txt", "w")
KLd_file              = open("KLd.txt", "w");

bulk_energies = h_eff_bulk_energies(1.e-8)

for i = 1:2^L-2
    write(eigenvalue_file, string(i))
    write(eigenvalue_file, "\t")  # Add a tab indentation between the columns
    write(eigenvalue_file, string(bulk_energies[i]))
    write(eigenvalue_file, "\n")  # Add a newline character to start a new line
end

# Close the file
close(eigenvalue_file)

function Level_Statistics(n,Es)
    return min(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n])) / max(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))
end;

h_eff_level_statistics = Array{Float64, 1}(undef, 0)
for i = 2:2^L-3 # relative index i.e length of the eigenvector array.
    push!(h_eff_level_statistics,Level_Statistics(i,bulk_energies))
end


for i = 1:2^L-4
    write(level_statistics_file, string(i))
    write(level_statistics_file, "\t")  # Add a tab indentation between the columns
    write(level_statistics_file, string(h_eff_level_statistics[i]))
    write(level_statistics_file, "\n")  # Add a newline character to start a new line
end

# Close the file
close(level_statistics_file)

#h_eff_eigenvectors = eigensystem_h_eff[2]
#=
function KLd(Eigenvectors_Matrix)
    KL = []
    for n = 1:2^L-1 # Eigenvector index goes from 1 to dim(H)-1.
        #=
        Here n is the index of the eigenstate e.g n = 3 denotes the
        third eigenstate of the h_eff matrix in sigma_z basis.
        =#

        #= Calculates the p(i) = |<i|n>|^2 for a given i. This is the moduli
        squared of the i-th component of the n-th eigenstate. This is because
        in the sigma_z basis <i|n> = i-th component of |n>.
        =#

        # Initialize the sum.
        KLd_sum = 0.0
        
        # V|x_bar> = |n+1>.
        eigenvector_1_z_basis = basis_change_matrix*Eigenvectors_Matrix[:,n]
        eigenvector_2_z_basis = basis_change_matrix*Eigenvectors_Matrix[:,n+1]
        
        # The sum goes from 1 to dim(H) i.e length of an eigenvector.
        for i = 1:2^L
            p = abs(eigenvector_1_z_basis[i])^2 + 1.e-9 # To avoid singularity in log.
            q = abs(eigenvector_2_z_basis[i])^2 + 1.e-9           

            KLd_sum += p*log(p/q)
        end
        #println(KLd_sum)
        push!(KL,KLd_sum)  
    end
    return KL
end;

#=
for i = 1:2^L-1
    write(KLd_file , string(i))
    write(KLd_file , "\t")  # Add a tab indentation between the columns
    write(KLd_file , string(KLd_calculated[i]))
    write(KLd_file , "\n")  # Add a newline character to start a new line
end

# Close the file
close(KLd_file)
=#
=#
