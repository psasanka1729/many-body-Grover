L = 10;
using JLD
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
file = raw"10_new_Grover_gates_data.txt" # Change for every L.
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

SEED = 8000+parse(Int64,ARGS[1])
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

#length(NOISE)

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0]));
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0])); # This is X gate.
Z_gate(noise) = exp(-1im*(pi/2+noise)*([1 0;0 1]-[1 0;0 -1]))#Hadamard(noise)*CX(noise)*Hadamard(noise); # noise # noise
Identity(dimension) = spdiagm(0 => ones(dimension))#1* Matrix(I, dimension, dimension);
int(x) = floor(Int,x);

function Matrix_Gate(Gate, Qubit) # Previously known as multi qubit gate.
    
    ## The case Qubit=1 is treated differently because we need to
    # initialize the matrix as U before starting the kronecker product.
    
    if Qubit == 1
        
        M = sparse(Gate)
        for i=2:L
            M = kron(M, sparse([1 0;0 1]))
        end
        
    else
        
        M = sparse([1 0;0 1])
        for i=2:L
            if i == Qubit
                M = kron(M, Gate)
            else
                M = kron(M, sparse([1 0;0 1]))
            end
        end
    end
    
    return M
end;

function CU(U,c,t)
    
    I2 = sparse([1 0;0 1])
    Z = sparse([1 0;0 -1])

    PI_0 = (I2+Z)/2
    PI_1 = (I2-Z)/2
     
    #function Rx(Noise)
        #A = cos((pi+Noise)/2)
        #B = -1im*sin((pi+Noise)/2)
        #return 1im*[A B;B A]
    #end
    
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
    return PI_0_matrix + PI_1_matrix     
end;

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;

function Special_states_matrix(DELTA)
    #DELTA = 0.0
    U_list = [];

    #ux_list = []
    NOISE_list = []

    Gates_data_new_1 = []
    Gates_data_new_2 = []
    Gates_data_new_3 = []
    
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)

        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
        
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        

        
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        

        
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(ux_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
 
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
  
        
            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiselss.
            
        end
    end
    

    
    #u0_list = []
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        

        
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        

        
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        

        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        

        
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
  
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        

            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiseless.
            
        end
    end
        
    function kth_term(k)

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
        
            if k < Number_of_Gates/2
                temp_matrix = sparse(G_exact)
                for i = 1:k
                    temp_matrix *= (U_list[i])'
                end
            else
                temp_matrix = sparse(Identity(2^L))
                # Constructing the term on the right of H_k.
                for i = k+1:Number_of_Gates
                    temp_matrix *= U_list[i]'
                end
                temp_matrix = temp_matrix'
            end
        
            #= Corresponding H for the kth term. =#
            if Gates_data_new_1[k] == "H"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate(I2-H,Qubit) #= H_had = I2-Had. =#

            elseif Gates_data_new_1[k] == "X"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate([1 0;0 1]-[0 1;1 0],Qubit) #= H_X = I2-X. =#
            
            elseif Gates_data_new_1[k] == "Z"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate([1 0;0 1]-[1 0;0 -1],Qubit) #= H_Z = I2-Z. =#
            
            else
        
                Angle = Gates_data_new_1[k]
                Control_Qubit = int(Gates_data_new_2[k])
                Target_Qubit = int(Gates_data_new_3[k])
                #= H = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
                Matrices = Dict("I" => [1 0;0 1],"U" => [1 -1;-1 1]/2, "PI_1" => (I2-Z)/2)
                p1 = fill("I", L)
                p1[Control_Qubit] = "PI_1"
                p1[Target_Qubit] = "U"
                H_k = Matrices[p1[1]]
                for i = 2:L
                    H_k = kron(H_k,Matrices[p1[i]])
                end                                 
            end
        return temp_matrix*H_k*(temp_matrix')
    end;     


    #= The following loop sums over all epsilon to get H_eff. =#
    h_eff = zeros(2^L,2^L);
    for i = 1:length(U_list)
        h_eff += NOISE_list[i]*kth_term(i)
    end   
    
    #= 
    Construction of block 2x2 H_eff matrix.
    =#
    
    # Defining the state |0> in sigma_z basis.
    ket_0 = zeros(2^L)
    ket_0[1] = 1
    
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x = (1/sqrt(N))*ones(N)
    ket_xbar = sqrt(N/(N-1))*ket_x-1/sqrt(N-1)*ket_0 # Normalization checked.
    
    # Matrix elements of h_eff in |0> and |xbar> basis.
    h_0_0 = ket_0' * h_eff * ket_0
    h_0_xbar = ket_0' * h_eff * ket_xbar
    h_xbar_0 = ket_xbar' * h_eff * ket_0
    h_xbar_xbar = ket_xbar' * h_eff * ket_xbar
    
    phi = atan(2*sqrt(N-1)/(2-N))
    # h_eff block matrix.
    h_eff_block_matrix = phi*[0 -1im; 1im 0]-DELTA*[ h_0_0 h_0_xbar; h_xbar_0 h_xbar_xbar]
    

    # G_0 block matrix.
    #N = 2^L
    #G_0_block = [2/N-1 -2*sqrt(N-1)/N;2*sqrt(N-1)/N 2/N-1]
    
    return  h_eff#h_eff_block_matrix - [1 0;0 1]*tr(h_eff_block_matrix)/2# h_eff * G_0.
end;

save("h_eff_matrix.jld","h_eff",H_EFF_MATRIX)   
# Calculate the 2x2 matrix in the basis |0> and |x_bar>.
#H_eff_G_0 = Special_states_matrix();
#=
# Calculate the matrix B in sigma_y basis.
function B_matrix_y_basis()
    N = 2^L
    theta = asin(2*sqrt(N-1)/N)
    
    # Eigenstates of tau_y in the |0> and |xbar> basis.
    y_s_p = (1/sqrt(2))*[1 1im]'
    y_s_n = (1/sqrt(2))*[1 -1im]'
    
    # Value of the integration.
    I_11 = exp(-im*theta) # -1 -1.
    I_12 = (1im/(sin(theta)))*log(exp(-1im*theta)) # -1 1.
    I_21 = (1im/(sin(theta)))*log(exp(-1im*theta)) # 1 -1.
    I_22 = exp(im*theta) # 1 1.
    
    # Returns the B matrix.
    return [[(I_11*y_s_n'*H_eff_G_0*y_s_n) (I_12*y_s_n'*H_eff_G_0*y_s_p)];
            [(I_21*y_s_p'*H_eff_G_0*y_s_n) (I_22*y_s_p'*H_eff_G_0*y_s_p)]]
end;
=#
#=
Basis transformation matrix from sigma_y to sigma_z.
=#
#sigma_y_to_sigma_z(Matrix) = ((1/sqrt(2))*[[1,1] [-1im, 1im]])*Matrix*inv((1/sqrt(2))*[[1,1] [-1im,1im]]);

# Changing the B matrix from sigma_y basis to sigma_z basis.
#B_matrix_z_basis = sigma_y_to_sigma_z(B_matrix_y_basis());

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

#=
py"""
f = open('Pauli_coefficients_data'+'.txt', 'w')
def Write_file_Pauli(b_0, b_1, b_2, b_3):
    f = open('Pauli_coefficients_data'+'.txt', 'a')
    f.write(str(b_0) +'\t'+ str(b_1)+ '\t' + str(b_2) +'\t' + str(b_3) +'\n')
"""

H_EFF_MATRIX = Special_states_matrix(0.01)
PC = Pauli_coefficients(H_EFF_MATRIX)
py"Write_file_Pauli"(PC[1],PC[2],PC[3],PC[4])

py"""
f = open('special_states_eigenvalues_data'+'.txt', 'w')
def Write_file(eigenvalue_1, eigenvalue_2):
    f = open('special_states_eigenvalues_data'+'.txt', 'a')
    f.write(str(eigenvalue_1) +'\t'+ str(eigenvalue_2)+'\n')
"""

# Diagonalize the special state matrix.
Special_eigenvalues = eigvals(H_EFF_MATRIX)
# Write the two eigenvalue to the data file.
py"Write_file"(Special_eigenvalues[1],Special_eigenvalues[2])
=#
