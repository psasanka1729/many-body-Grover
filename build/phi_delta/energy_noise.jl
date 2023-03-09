L = 12;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
file = raw"12_new_Grover_gates_data.txt" # Change for every L.
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

SEED = 764
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0]));
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0])); # This is X gate.
Z_gate(noise) = Hadamard(noise)*CX(noise)*Hadamard(noise); # noise # noise
Identity(dimension) = 1* Matrix(I, dimension, dimension);
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

function Grover_operator(DELTA)
    
    U_x_delta = sparse(Identity(2^L));

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
    
    U_0_delta = sparse(Identity(2^L));
    
 
    # U_0
    for i = 1 : U_0_gate_number
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]      
            h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_0_delta *= h_matrix
            
        elseif Gates_data_1[i] == "X"
  
            epsilon = NOISE[i]     
            x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            U_0_delta *= x_matrix
            
        elseif Gates_data_1[i] == "Z"

            epsilon = NOISE[i]    
            z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix  
            
        else
        
            epsilon = NOISE[i]     
            rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            U_0_delta *= rx_matrix
            
        end
    end
        
    GROVER_DELTA = U_x_delta*U_0_delta
    
    return collect(GROVER_DELTA)
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
The following function returns the matrix of rolling operator.
=#
function One_Roll_Operator(number_of_qubits::Int64)
    
    #= Function converts a binary number to a decimal number. =#
    Bin2Dec(BinaryNumber) = parse(Int, string(BinaryNumber); base=2);
    
    #= Function converts a decimal number to a binary number. =#
    function Dec2Bin(DecimalNumber::Int64) 
        
        init_binary = string(DecimalNumber, base = 2);
        
        #=
        While converting numbers from decimal to binary, for example, 1
        is mapped to 1, to make sure that every numbers have N qubits in them,
        the following loop adds leading zeros to make the length of the binary
        string equal to N. Now, 1 is mapped to 000.....1 (string of length N).
        =#
        
        while length(init_binary) < number_of_qubits
            init_binary = "0"*init_binary
        end
        return init_binary
    end
    
    #=
    The following function takes a binary string as input
    and rolls the qubits by one and returns the rolled binary string.
    =#
    Roll_String_Once(binary_string) = binary_string[end]*binary_string[1:end-1]
    
    #= Initializing the rolling operator. =#
    R = zeros(Float64,2^number_of_qubits,2^number_of_qubits);
    
    #= The numbers are started from 0 to 2^L-1 because for L qubits,
    binary representation goes from 0 to 2^L-1.=#
    
    for i = 0:2^number_of_qubits-1 
        
        #=
        Steps in the following loop.
        (1) The number is converted from decimal to binary.
        (2) The qubits are rolled once.
        (3) The rolled binary number is converted to decimal number.
        (4) The corresponding position in R is replaced by 1.
        =#
        
        #= The index in R will be shifted by 1 as Julia counts from 1. =#
        R[i+1,Bin2Dec(Roll_String_Once(Dec2Bin(i)))+1] = 1
    end
    
    return sparse(R)
end;
          
#=
The following function returns the von-Neumann entropy of a given
wavefunction. The sub-system size is L/2.
=#

function entanglement_entropy(Psi)
    
    sub_system_size = floor(Int,L/2)
    
    Psi = Psi/norm(Psi)
    
    function psi(s)
        return Psi[2^(sub_system_size)*s+1:2^(sub_system_size)*s+2^(sub_system_size)]
    end
    
    #= (s,s_p) element of the reduced density matrix is given by psi(s_p)^(\dagger)*psi(s). =#
    rhoA(s,s_p) = psi(s_p)' * psi(s)
        
    M = zeros(ComplexF64,2^sub_system_size,2^sub_system_size)
    
    #=
    Since the matrix is symmetric only terms above the diagonal will be calculated.
    =#
    for i = 0:2^sub_system_size-1
        for j = 0:2^sub_system_size-1
            if i <= j
                M[i+1,j+1] = rhoA(i,j)
            else
                M[i+1,j+1] = M[j+1,i+1]
            end
        end
    end 
    
    #= Eigenvalues of M. The small quantity is added to avoid singularity in log.=#
    w = eigvals(M).+1.e-10
    
    return real(-sum([w[i]*log(w[i]) for i = 1:2^(sub_system_size)]))
end;          
    
              
function average_entanglement_entropy(initial_wavefunction)
    initial_wavefunction = initial_wavefunction/norm(initial_wavefunction)
    R = One_Roll_Operator(L)
    rolled_wavefunction = R * initial_wavefunction
    rolled_entropies = [entanglement_entropy(rolled_wavefunction)]
    for i = 2:L
        rolled_wavefunction = R * rolled_wavefunction
        push!(rolled_entropies,entanglement_entropy(rolled_wavefunction))
    end
    
    return sum(rolled_entropies)/L
end;
              
py"""
f = open('plot_data'+'.txt', 'w')
def Write_file(Noise, Energy, Entropy):
    f = open('plot_data'+'.txt', 'a')
    f.write(str(Noise) +'\t'+ str(Energy)+ '\t' + str(Entropy) +'\n')
"""
# delta_index runs from 0 to 63.
delta_index = parse(Int64,ARGS[1])

Delta = LinRange(0.0,0.25,64+1)
delta_start = Delta[delta_index+1]
delta_end = Delta[delta_index+2]
Num = 10

for i=0:Num
    delta = delta_start+(i/Num)*(delta_end-delta_start)
    Op = Grover_operator(delta)
    EIGU = py"eigu"(Op)
    X = string(delta)
    Y = real(1im*log.(EIGU[1]))
    V = EIGU[2]
    
    for j=1:2^L
        py"Write_file"(delta, real(Y[j]), average_entanglement_entropy(V[1:2^L,j:j]))
    end
end
