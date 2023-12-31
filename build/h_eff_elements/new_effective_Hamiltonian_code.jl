L = 8;

using JLD
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Statistics
#using Plots
#using LaTeXStrings

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
number_of_gates = U_0_gate_number+U_x_gate_number

SEED = 7000+parse(Int64,ARGS[1])
Random.seed!(SEED)
NOISE = 2*rand(Float64,number_of_gates).-1;

I2 = sparse([1 0; 0 1]);
Pauli_Z  = sparse([1 0;0 -1]);
Pauli_X  = sparse([0 1;1 0])
H  = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)       = sparse(exp(-1im*(theta/2)*collect(I2-Pauli_X)));
Hadamard(noise) = sparse(exp(-1im*(pi/2+noise)*collect(I2-H))) #Ry(pi/2+noise)*Pauli_Z;
X(noise)        = sparse(exp(-1im*((pi/2+noise))*collect(I2-Pauli_X))); # This is X gate.
Z(noise)        = sparse(exp(-1im*(pi/2+noise)*collect(I2-Pauli_Z))) #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise
Identity(dimension) = spdiagm(0 => ones(dimension));
int(x) = floor(Int,x);

function single_qubit_gate_matrix(Gate, Qubit) # Previously known as multi qubit gate.
    
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

function single_qubit_controlled_gate_matrix(single_qubit_gate,c,t)
    
    I2 = sparse([1 0;0 1])
    Z = sparse([1 0;0 -1])

    PI_0 = (I2+Z)/2
    PI_1 = (I2-Z)/2
     
    #function Rx(Noise)
        #A = cos((pi+Noise)/2)
        #B = -1im*sin((pi+Noise)/2)
        #return 1im*[A B;B A]
    #end
    
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
           
    #return p0,p1
    return PI_0_matrix + PI_1_matrix     
end;

function eigu(U,tol=1.e-9)
    (E_1,V_1) = eigvals(U + U'),eigvecs(U + U')
    U_1=V_1'*U*V_1
    #H_1=V_1*(U+U')*V_1
    non_diag_lst=[]
    j=1
    while j <= size(U_1)[1]
        k=1
        while k <= size(U_1)[1]
            #println(j,k)
            if j!=k && abs(U_1[j,k])>tol
                if !(j in non_diag_lst)
                    push!(non_diag_lst,j)
                end
                if !(k in non_diag_lst)
                    push!(non_diag_lst,k)
                end   
            end
            k += 1 
        end
        j += 1
    end
    if length(non_diag_lst)>0
        non_diag_lst=sort(non_diag_lst)
        U_1_cut=U_1[non_diag_lst, non_diag_lst]
        (E_2_cut,V_2_cut)=eigvals(1im*(U_1_cut-U_1_cut')),eigvecs(1im*(U_1_cut-U_1_cut'))
        V_2=Identity(size(U)[1])
        V_2 = convert(Matrix{ComplexF64}, V_2)
        for j = 1:length(non_diag_lst)
             V_2[non_diag_lst[j], non_diag_lst] = V_2_cut[j, :]
        end
        V_1=V_1*V_2
        U_1=V_2'*U_1*V_2
    end
    U_1=diag(U_1)
    inds=sortperm(imag(log.(Complex.(U_1))))    
    return (U_1[inds, :], V_1[:, inds])
end;

U_0 = -Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = 1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;
#V = py"eigu"(G_exact)[2];
#V = eigu(G_exact)[2];

# x_bar_0 = x_bar_(N-1)
function x_bar(n)
    s = zeros(2^L)
    k_n = (2*pi*n)/(2^L-1)
    for j=1:2^L-1
        ket_j    = zeros(2^L);
        #=
        The formula in the sum is ket_j and the sum starts from 1. In the computational basis
        ket_j has 1 in the (j+1) position (counting from 1 instead of 0).
        =#
        ket_j[j+1] = 1 
        s = s+exp(1im*(j-1)*k_n)*ket_j
    end
    return s/sqrt(2^L-1)
end;

function sigma_z_to_x_bar_basis_change_matrix(L)
    V     = spzeros(2^L,2^L)
    ket_0 = spzeros(2^L)
    ket_0[1] = 1
    ket_1    = spzeros(2^L);
    ket_1[2] = 1
    #ket_xbar = x_bar(0)
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x    = (1/sqrt(N))   * ones(N)
    ket_xbar = sqrt(N/(N-1)) * ket_x - 1/sqrt(N-1)*ket_0 # Normalization checked.    
    eigenstate_1 = (ket_0-1im*ket_xbar)/sqrt(2)
    eigenstate_2 = (ket_0+1im*ket_xbar)/sqrt(2)
    V = V+ eigenstate_1*ket_0'
    V = V+ eigenstate_2*ket_1'
    
    # The buk.
    for n=1:2^L-2
        # ket_n has n+1 th position as 1 in comutational basis.
        ket_n    = spzeros(2^L);
        # |x_bar_1><2| + |x_bar_2><3| + .......... + |x_bar_N-2><N-1|. 
        ket_n[n+2] = 1 
        
        V = V+x_bar(n)*ket_n'
    end
    return V
end;

function grover_delta_matrix(DELTA::Float64)

    U_x_delta = sparse(Identity(2^L));
    
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            h_matrix = single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_x_delta *= h_matrix

            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]       
            x_matrix = single_qubit_gate_matrix(X(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= x_matrix
 
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]     
            z_matrix = single_qubit_gate_matrix(Z(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix

            
        else
            #push!(ux_list,"CRX")
        
            epsilon = NOISE[i]      
            rx_matrix = single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon),
             Gates_data_2[i], Gates_data_3[i])
            U_x_delta *= rx_matrix

            
        end
    end
    
    U_0_delta = sparse(Identity(2^L));
    
    #u0_list = []
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]       
            h_matrix = single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_0_delta *= h_matrix

            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]        
            x_matrix = single_qubit_gate_matrix(X(DELTA*epsilon),Gates_data_3[i])
            U_0_delta *= x_matrix
 
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]       
            z_matrix = single_qubit_gate_matrix(Z(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix

            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]       
            rx_matrix = single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), 
            Gates_data_2[i], Gates_data_3[i])
            U_0_delta *= rx_matrix
  
            
        end
    end
        
    GROVER_DELTA = U_x_delta*U_0_delta
    return GROVER_DELTA
end;

function grover_effective_Hamiltonian_matrix(DELTA)
    
    U_list = [];
    U_x_delta = sparse(Identity(2^L));
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
            #h_matrix = single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
            #U_x_delta *= h_matrix
        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
        
            push!(U_list,single_qubit_gate_matrix(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #x_matrix = single_qubit_gate_matrix(X(DELTA*epsilon),Gates_data_3[i])
            #U_x_delta *= x_matrix
        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
        
            push!(U_list,single_qubit_gate_matrix(X(0.0),Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #z_matrix = single_qubit_gate_matrix(Z(DELTA*epsilon),Gates_data_3[i])
            #U_x_delta *= z_matrix
        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
        
            push!(U_list,single_qubit_gate_matrix(Z(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(ux_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #rx_matrix = single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            #U_x_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
        
            push!(U_list,single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiselss.
            
        end
    end
    
    U_0_delta = sparse(Identity(2^L));
    
    #u0_list = []
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #h_matrix = single_qubit_gate_matrix(Hadamard(DELTA*epsilon), Gates_data_3[i])
            #U_0_delta *= h_matrix
        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
        
            push!(U_list,single_qubit_gate_matrix(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #x_matrix = single_qubit_gate_matrix(X(DELTA*epsilon),Gates_data_3[i])
            #U_0_delta *= x_matrix
        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
        
            push!(U_list,single_qubit_gate_matrix(X(0.0),Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
           # z_matrix = single_qubit_gate_matrix(Z(DELTA*epsilon),Gates_data_3[i])
            #U_x_delta *= z_matrix
        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
        
            push!(U_list,single_qubit_gate_matrix(Z(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #rx_matrix = single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            #U_0_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
        
            push!(U_list,single_qubit_controlled_gate_matrix(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiseless.
            
        end
    end
        
    GROVER_DELTA = U_x_delta*U_0_delta
    
    function kth_term(k)

            f_k = Identity(2^L);
    
            for i = k:length(U_list)-1
                f_k = f_k*collect(U_list[length(U_list)-i+k])
            end     
            #= Corresponding H for the kth term. =#
            if Gates_data_new_1[k] == "H"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = single_qubit_gate_matrix(I2-H,Qubit) #= H_had = I2-Had. =#

            elseif Gates_data_new_1[k] == "X"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = single_qubit_gate_matrix([1 0;0 1]-[0 1;1 0],Qubit) #= H_X = I2-X. =#
            
            elseif Gates_data_new_1[k] == "Z"

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = single_qubit_gate_matrix([1 0;0 1]-[1 0;0 -1],Qubit) #= H_Z = I2-Z. =#
            
            else
        
                Angle = Gates_data_new_1[k]
                Control_Qubit = int(Gates_data_new_2[k])
                Target_Qubit = int(Gates_data_new_3[k])
                #= H = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
                Matrices = Dict("I" => [1 0;0 1],"U" => (I2-Pauli_X)/2, "PI_1" => (I2-Pauli_Z)/2)
                p1 = fill("I", L)
                p1[Control_Qubit] = "PI_1"
                p1[Target_Qubit] = "U"
                H_k = Matrices[p1[1]]
                for i = 2:L
                    H_k = kron(H_k,Matrices[p1[i]])
                end                                 
            end
    
    
        return f_k*H_k*(f_k')
    end; 

    # The following loop sums over all epsilon to get H_eff.
    h_eff = zeros(ComplexF64,2^L,2^L);
    for i = 1:length(U_list)
        h_eff += (NOISE_list[i])*kth_term(i)
    end 
    return GROVER_DELTA, h_eff
    #return h_eff
end;

#=
function decimal_to_binary(decimal_number)
    binary_str = ""
    i = 1
    while decimal_number>0
        binary_str  = string(decimal_number%2)*binary_str
        decimal_number = int(decimal_number/2)
        i+=1
    end
    while length(binary_str)<L
        binary_str = "0"*binary_str
    end
    return binary_str
end;

function hamming_distance(str1, str2)
    
  if length(str1) != length(str2)
    error("Strings must be the same length.")
  end

  distance = 0
  for i in 1:length(str1)
    if str1[i] != str2[i]
      distance += 1
    end
  end

  return distance
end;

hamming_distance_lst = zeros(2^L-2,2^L-2)
for i=1:2^L-2
    for j=1:2^L-2
        hamming_distance_lst[i,j] = hamming_distance(decimal_to_binary(i),decimal_to_binary(j))
    end
end;
=#

g_h_eff = collect(grover_effective_Hamiltonian_matrix(0.0));

h_eff_compt_basis = (g_h_eff[2])
#h_eff_compt_basis_traceless = h_eff_compt_basis - Identity(2^L)*(1/2^L)*tr(h_eff_compt_basis)
save("h_eff_matrix.jld","h_eff",h_eff_compt_basis)

#=
function Level_Statistics(n,Es)
        return min(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n])) / max(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))
end;

bulk_energies = eigvals(h_eff_compt_basis)
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


h_eff_eigenvectors = eigvecs(h_eff_compt_basis)

function KLd(Eigenvectors_Matrix)
    KL = []
    for n = 1:2^L-1 # Eigenvector index goes from 1 to dim(H)-1.
        KLd_sum = 0.0
        eigenvector_1_z_basis = Eigenvectors_Matrix[:,n]
        eigenvector_2_z_basis = Eigenvectors_Matrix[:,n+1]
        for i = 1:2^L
                p = abs(eigenvector_1_z_basis[i])^2 + 1.e-9
                q = abs(eigenvector_2_z_basis[i])^2 + 1.e-9  
                KLd_sum += p*log(p/q)
        end
        push!(KL,KLd_sum) 
   end
   return KL
end;

KLd_calculated = KLd(h_eff_eigenvectors)
for i = 1:2^L-1
    write(KLd_file , string(i))
    write(KLd_file , "\t")
    write(KLd_file , string(KLd_calculated[i]))
    write(KLd_file , "\n") 
end

close(KLd_file)=#
