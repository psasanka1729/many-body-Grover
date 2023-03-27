using NPZ
using JLD
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles

L = 8;

Identity(dimension) = spdiagm(0 => ones(dimension));
I2 = Identity(2);
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= exp(-1im*(theta/2)*(I2-[0 1;1 0]));
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*(I2-[0 1;1 0])); # This is X gate.
Z_gate(noise) = exp(-1im*(pi/2+noise)*(I2-Z)) #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise

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
    
    return M
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
    return PI_0_matrix + PI_1_matrix    
end;

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;
#V = py"eigu"(G_exact)[2];

list_of_U = jldopen("U_list.jld", "r") do file
    read(file, "U")
end;

Gates_data_new_1 = jldopen("Gates_new_data_1.jld", "r") do file
    read(file, "G1")
end;

Gates_data_new_2 = jldopen("Gates_new_data_2.jld", "r") do file
    read(file, "G2")
end;

Gates_data_new_3 = jldopen("Gates_new_data_3.jld", "r") do file
    read(file, "G3")
end;

NOISE_list = jldopen("NOISE_list.jld", "r") do file
    read(file, "Noise_list")
end;

function h_eff_eigenvalues(U_list,Gates_data_new_1,Gates_data_new_2,Gates_data_new_3,NOISE_list)
    
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
                H_k = Matrix_Gate(I2-[0 1;1 0],Qubit) #= H_X = I2-X. =#
            
            elseif Gates_data_new_1[k] == "Z"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate(I2-[1 0;0 -1],Qubit) #= H_Z = I2-Z. =#
            
            else
        
                Angle = Gates_data_new_1[k]
                Control_Qubit = int(Gates_data_new_2[k])
                Target_Qubit = int(Gates_data_new_3[k])
                #= H = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
                Matrices = Dict("I" => I2,"U" => [1 -1;-1 1]/2, "PI_1" => (I2-Z)/2)
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
    h_eff = spzeros(2^L, 2^L)#zeros(2^L,2^L);
    @time for i = 1:length(U_list)
        h_eff += NOISE_list[i]*kth_term(i)
    end        

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
    h_eff_trunc = (Identity(2^L)-P_xbar)*(Identity(2^L)-P_0)*h_eff*(Identity(2^L)-P_0)*(Identity(2^L)-P_xbar)    
    

    h_eff_D = h_eff_trunc[3:2^L,3:2^L]; # Deleting the |0> and |xbar> basis.
    E_eff_D = eigvals(h_eff_D) # Diagonalizing H_eff matrix.
    
    E_eff_D_sorted = sort(real(E_eff_D),rev = true); # Soring the eigenvalues in descending order.    
    return E_eff_D_sorted
end;

save("h_eff_eigenvalues.jld", "h_eff_eigenvalues", h_eff_eigenvalues(list_of_U,Gates_data_new_1,Gates_data_new_2,Gates_data_new_3,NOISE_list))
