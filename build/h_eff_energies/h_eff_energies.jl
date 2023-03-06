L = 6;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Statistics

#= 
The data file has the list of all gates necessary to construct G. The order of the gates are U_0 and U_x.
=#
file = raw"6_new_Grover_gates_data.txt" # Change for every L.

M = readdlm(file)

Gates_data_1 = M[:,1]; # Gate name/ angle if Rx gate.
Gates_data_2 = M[:,2]; # Control qubit if any; otherwise 0.
Gates_data_3 = M[:,3]; # Target qubit if any; otherwise 0.

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

Number_of_Gates = U_0_gate_number+U_x_gate_number # Total number of gates in the Grover operator G.

SEED = 4000+parse(Int64,ARGS[1])
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

# Identity gate of size 2x2.
I2 = [1 0; 0 1];

# Pauli Z gate.
Z = [1 0;0 -1];

# Hadamard gate.
H = (1/sqrt(2))*[1 1;1 -1]

# Rx gate.
Rx(theta)= exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0]));

# Hadamard gate as a function of noise.
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;

# Control X gate as a function of noise.
CX(noise) = exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0])); # This is X gate.

# Pauli Z gate as a function of noise.
Z_gate(noise) = exp(-1im*(pi/2+noise)*(I2-Z)) #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise

# Function returns the identity matrix of size dimension.
Identity(dimension) = 1* Matrix(I, dimension, dimension);

# Function returns the integer part of a number.
int(x) = floor(Int,x);

#=
The following function returns the matrix of a gate acting on a given qubit.
=#

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

#=
The following function returns the matrix of a given gate with given control and target qubit.
=#

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

using PyCall
py"""
import numpy
import numpy.linalg
def adjoint(psi):
    return psi.conjugate().transpose()
def psi_to_rho(psi):
    return numpy.outer(psi,psi.conjugate())
def exp_val(psi, op):
    return numpy.real(numpy.dot(adjoint(psi),op.dot(psi)))
def norm_sq(psi):
    return numpy.real(numpy.dot(adjoint(psi),psi))
def normalize(psi,tol=1e-9):
    ns=norm_sq(psi)**0.5
    if ns < tol:
        raise ValueError
    return psi/ns
def is_herm(M,tol=1e-9):
    if M.shape[0]!=M.shape[1]:
        return False
    diff=M-adjoint(M)
    return max(numpy.abs(diff.flatten())) < tol
def is_unitary(M,tol=1e-9):
    if M.shape[0]!=M.shape[1]:
        return False
    diff=M.dot(adjoint(M))-numpy.identity((M.shape[0]))
    return max(numpy.abs(diff.flatten())) < tol
def eigu(U,tol=1e-9):
    (E_1,V_1)=numpy.linalg.eigh(U+adjoint(U))
    U_1=adjoint(V_1).dot(U).dot(V_1)
    H_1=adjoint(V_1).dot(U+adjoint(U)).dot(V_1)
    non_diag_lst=[]
    j=0
    while j < U_1.shape[0]:
        k=0
        while k < U_1.shape[0]:
            if j!=k and abs(U_1[j,k]) > tol:
                if j not in non_diag_lst:
                    non_diag_lst.append(j)
                if k not in non_diag_lst:
                    non_diag_lst.append(k)
            k+=1
        j+=1
    if len(non_diag_lst) > 0:
        non_diag_lst=numpy.sort(numpy.array(non_diag_lst))
        U_1_cut=U_1[non_diag_lst,:][:,non_diag_lst]
        (E_2_cut,V_2_cut)=numpy.linalg.eigh(1.j*(U_1_cut-adjoint(U_1_cut)))
        V_2=numpy.identity((U.shape[0]),dtype=V_2_cut.dtype)
        for j in range(len(non_diag_lst)):
            V_2[non_diag_lst[j],non_diag_lst]=V_2_cut[j,:]
        V_1=V_1.dot(V_2)
        U_1=adjoint(V_2).dot(U_1).dot(V_2)
    # Sort by phase
    U_1=numpy.diag(U_1)
    inds=numpy.argsort(numpy.imag(numpy.log(U_1)))
    return (U_1[inds],V_1[:,inds]) # = (U_d,V) s.t. U=V*U_d*V^\dagger
"""

#=
Following constructs the exact Grover operator for qubit numebr L.
=#

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0
V = py"eigu"(G_exact)[2]; # Basis transformation matrix between the Z basis and X_bar basis.

#DELTA = 0.01

#=
The following function returns the eigenvalues of the effective Hamiltonian.
=#
function Eigenvalues(DELTA)
    
    # This list will hold all the noiseless gates if they are created once.
    U_list = [];
    # This list will hold all the noisy gates if they are created once.
    #U_noise_list = [];
    
    #=
    G = U_x * U_0.
    
    U_x is constructed first. This is necessary for the construction of the
    efffective Hamiltonian.
    =#
    
    # Initializing the matrix U_x.
    #U_x_delta = sparse(Identity(2^L));

    #= The order of the noises are important while constructing the effective Hamiltonian.
    Therefore, all the noise values used in the gates are put in a new list so that they
    can be called while necessary.
    =#
    NOISE_list = []

    #=
    The order of the gates are different i.e U_x first and then U_0. Whereas, in the gate data
    file the order is U_0 and then U_x. Therefore, new arrays are created in order to store
    the new order of the gates.
    =#
    Gates_data_new_1 = []
    Gates_data_new_2 = []
    Gates_data_new_3 = []
    
    # U_x
    for i = U_0_gate_number+1: U_0_gate_number+U_x_gate_number
        
        # Hadamard gate.
        if Gates_data_1[i] == "H"
            
            # A noise value from the generated noise is selected.
            epsilon = NOISE[i]
            # The noise is put into the new noise list.
            push!(NOISE_list,epsilon)
            
            # The noisy matrix corresponding to the above gate is constructed.
            #h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            # The matrix is multiplied with U_x.
            #U_x_delta *= h_matrix
        
            # The new gates are put into the new gates list.
            push!(Gates_data_new_1,"H")
            # There is no inherent angle for Hadamard.
            push!(Gates_data_new_2,0.0) 
            push!(Gates_data_new_3,Gates_data_3[i])
        
            # The noisy matrix for the Hadamrd is put into a list for posterity.
            #push!(U_noise_list,h_matrix)
            # The noiselss matrix of Hadamard is put into a list for posterity.
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        # Pauli X gate.    
        elseif Gates_data_1[i] == "X"
        
            # A noise value from the generated noise is selected.
            epsilon = NOISE[i]
            # The noise is put into the new noise list.
            push!(NOISE_list,epsilon)  
            
            # The noisy matrix corresponding to the above gate is constructed.
            #x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            # The matrix is multiplied with U_x.
            #U_x_delta *= x_matrix
        
            # The new gates are put into the new gates list.
            push!(Gates_data_new_1,"X")
            # There is no inherent angle for X.
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            # The noisy matrix for the X is put into a list for posterity.
            #push!(U_noise_list,x_matrix) # Noise.
            # The noiselss matrix of X is put into a list for posterity.
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        # Pauli Z gate.    
        elseif Gates_data_1[i] == "Z"
        
            # A noise value from the generated noise is selected.
            epsilon = NOISE[i]
            # The noise is put into the new noise list.
            push!(NOISE_list,epsilon)
            
            # The noisy matrix corresponding to the above gate is constructed.
            #z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            # The matrix is multiplied with U_x.
            #U_x_delta *= z_matrix
        
            # The new gates are put into the new gates list.
            push!(Gates_data_new_1,"Z")
            # There is no inherent angle for X.
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            # The noisy matrix for the Z is put into a list for posterity.
            #push!(U_noise_list,z_matrix) # Noise.
            # The noiselss matrix of X is put into a list for posterity.
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
        # Controlled Rx gate.    
        else
            # A noise value from the generated noise is selected.
            epsilon = NOISE[i]
            # The noise is put into the new noise list.
            push!(NOISE_list,epsilon)        
            
            # The noisy matrix corresponding to the above gate is constructed.
            #rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            # The matrix is multiplied with U_x.
            #U_x_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
            # The noisy matrix for CRx is put into a list for posterity.
            #push!(U_noise_list,rx_matrix) # Noise.
            # The noiselss matrix of CRx is put into a list for posterity.
            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiselss.
            
        end
    end
    
    # Initialzing U_0 matrix.
    #U_0_delta = sparse(Identity(2^L));
    
    # U_0.
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            #U_0_delta *= h_matrix
        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
            #push!(U_noise_list,h_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            #U_0_delta *= x_matrix
        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            #push!(U_noise_list,x_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            #U_x_delta *= z_matrix
        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            #push!(U_noise_list,z_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            #rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            #U_0_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
            #push!(U_noise_list,rx_matrix) # Noise.
        
            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiseless.
            
        end
    end
    
    # G with noise.    
    #GROVER_DELTA = U_x_delta*U_0_delta
    
    # Constructing the k-th term in the effective Hamiltonian.
    function kth_term(k)

            f_k = Identity(2^L);
    
            #=
            Constructing the product to the left side of H in the
            definition of effective Hamiltonian.
            =#
            for i = k:length(U_list)-1
                f_k = f_k*collect(U_list[length(U_list)-i+k])
            end     
        
            #= Corresponding H matrix for the kth term. =#
            if Gates_data_new_1[k] == "H" # If the gate is a Hadamard gate.

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate(I2-H,Qubit) #= H_had = I2-Had. =#

            elseif Gates_data_new_1[k] == "X" # If the gate is a X gate.

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate([1 0;0 1]-[0 1;1 0],Qubit) #= H_X = I2-X. =#
            
            elseif Gates_data_new_1[k] == "Z" # If the gate is a Z gate.

                Qubit = Gates_data_new_3[k] # qubit.
                H_k = Matrix_Gate([1 0;0 1]-[1 0;0 -1],Qubit) #= H_Z = I2-Z. =#
            
        else # If the gate is a CRx gate.
        
                Angle = Gates_data_new_1[k]
                Control_Qubit = int(Gates_data_new_2[k])
                Target_Qubit = int(Gates_data_new_3[k])
            
                #= H_CRx = ((I-Z)/2)_c \otimes ((I-X)/2)_t.=#
                # Constructing the H_CRx.
                Matrices = Dict("I" => [1 0;0 1],"U" => [1 -1;-1 1]/2, "PI_1" => (I2-Z)/2)
                p1 = fill("I", L)
                p1[Control_Qubit] = "PI_1"
                p1[Target_Qubit] = "U"
                H_k = Matrices[p1[1]]
                for i = 2:L
                    H_k = kron(H_k,Matrices[p1[i]])
                end                                 
            end
    
    
        # Effective Hamiltonain.
        return f_k*H_k*(f_k')
    end; 
    
    
    #= The following loop sums over all k to get the effective Hamiltonian. =#
    h_eff = zeros(2^L,2^L);
    for i = 1:length(U_list)
        h_eff += NOISE_list[i]*kth_term(i)
    end        

    #=
    Till now the matrix of the effective Hamiltonian is in Z basis.
    In order to compare its energies with the exact Grover with noise,
    it will be written in the x_bar basis.
    =#
    h_eff_D = (V')*h_eff*(V) 

    h_eff_D = h_eff_D[3:2^L,3:2^L]; # Deleting the |0> and |xbar> basis.
    
    
    #E_eff_D = eigvals(h_eff_D) # Diagonalizing H_eff matrix.
    
    #E_eff_D_sorted = sort(real(E_eff_D)); # Soring the eigenvalues in descending order.    

    return h_eff_D
    #return E_eff_D_sorted
end;


h_eff_matrix = Eigenvalues(0.0)

py"""
f = open('trace_data'+'.txt', 'w')
def Write_file1(trace):
    f = open('trace_data'+'.txt', 'a')
    f.write(str(trace) +'\n')
"""

py"Write_file1"(real(tr(h_eff_matrix * h_eff_matrix)))

# Eigenvalues of the h_eff matrix.
E_eff_D = eigvals(h_eff_matrix)

# Sorts the eigenvalue array.
E_eff_sorted = sort(real(E_ff_D));

py"""
f = open('h_eff_energy_data'+'.txt', 'w')
def Write_file2(index, energy):
    f = open('h_eff_energy_data'+'.txt', 'a')
    f.write(str(index) + '\t'+ str(energy) +'\n')
"""

#= 
The length of the eigenvector array is 2^L-2.
=#
for i = 1:2^L-2 # relative index i.e length of the eigenvector array.
    py"Write_file2"(i,E_eff_sorted[i])
end


# Level statistics.
py"""
f = open('level_statistics_data.txt','w')
def Write_file3(index, level_stat):
	f = open('level_statistics.txt','w')
	f.write(str(index) + '\t' + str(level_stat) +'\n')
"""

function Level_Statistics(n,Es)
	return minimum(abs(Es[n]-Es[n-1),abs(Es[n+1]-Es[n])) / maximum(abs(Es[n]-Es[n-1]),abs(Es[n+1]-Es[n]))
end

for i = 2:2^L-3
	py"Write_file3"(i,Level_Statistics(i,E_eff_sorted))
end
