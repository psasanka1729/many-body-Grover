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


SEED = 10
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

#length(NOISE)

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0]));
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0])); # This is X gate.
Z_gate(noise) = exp(-1im*(pi/2+noise)*(I2-Z))  #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise # noise
Identity(dimension) = 1* Matrix(I, dimension, dimension);
int(x) = floor(Int,x);

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;
#V = py"eigu"(G_exact)[2];

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

function Entropy(Psi)   
    
    LS = Int64(L/2)

    #Psi = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1im];
    
    # Normalizing Psi.
    Psi = Psi/norm(Psi) 
    
    psi(s) = Psi[(2^LS)*(s-1)+1:(2^LS)*s]
    
    #=
        psi(s_p) is a row matrix/vector. psi(s) is a column matrix/vector.      
        Dimension of rhoA is N/2 x N/2. 
        The element <s|rhoA|sp> is given by psi_sp^\dagger * psi_s.
    =#
    
    
    # psi(s_p)^\dagger * psi(s) is the element of (s,s_p) of rho_AB. 
    rhoA(s,s_p) = psi(s_p)' * psi(s)
    
    
    # Following function returns the reduced density matrix rho_A.
    function rhoA_Matrix()
        
        LS = Int64(L/2)
            
        # Creates a zero matrix to store the density matrix.
        M = zeros(ComplexF64,2^LS,2^LS)
        
        #=
        rho is Hermitian, it is sufficient to calculate the elements above the diagonal.
        The the elements below the diagonal can be replace by the complex cpnjugate of the
        elements above the diagonal.
        =#
    
        for i=1:2^LS
            for j=1:2^LS
                if i<=j
                    M[i,j] = rhoA(i,j)
                else
                    # Elements below diagonals are replaced by the elements above the diagonal.
                    M[i,j] = M[j,i]' 
                end
            end
        end
        return M
    end;
    
    w = eigvals(rhoA_Matrix()) # Eigenvalues of the reduced density matrix.
    #=
    The following loop calculates S = - sum \lamba_i * log(\lambda_i).
    =#
    
    # Array to store the log of the eigenvalues.
    DL = zeros(ComplexF64,2^LS)
    for i=1:length(w)
        if abs(w[i]) < 1.e-8 # Avoid zeros.
            continue
        else
            DL[i] = log(w[i])
        end
    end
    return real(-sum(w.*DL)) # S = -tr(rho *log(rho)).
end;

Bin2Dec(BinaryNumber) = parse(Int, string(BinaryNumber); base=2);
Dec2Bin(DecimalNumber) = string(DecimalNumber, base=2);

List = [i for i=0:2^L-1]; # List with numbers from 0 to 2^L-1.

#=
The following function converts all numbers in decimals in the above list 
 from 0 to 2^L -1 to binary.
=#

function List_Bin(Lst)
    
    l = []
    
    for i in Lst
        
        i_Bin = Dec2Bin(i)
        
        #=
        While converting numbers from decimal to binary, for example, 1
        is mapped to 1, to make sure that
        every numbers have N qubits in them, the following loop adds leading 
        zeros to make the
        length of the binary string equal to N. Now, 1 is mapped to 000.....1
        (string of length N).
        =#
        
        while length(i_Bin) < L
            i_Bin = "0"*i_Bin
        end
            
        # Puts the binary number in the list l after its length is L.
        push!(l,i_Bin)
    end
    return l
end;

#=
    The following function takes a binary string as input and rolls the qubits by one and
    returns the rolled string.
=#

Roll_String(Binary_String) = last(Binary_String)*Binary_String[1:L-1];

#=
    The following function takes a wavefunction as input and performs one roll
    on the qubits and returns the resultant wavefunction.
=#

function Psi_Roll(Initial_Psi)
    
    #=
        The following list contains all possible 2^N qubits after one roll 
        is performed on them.
        For example, the first position 0001 is changed to 1000.
    =#
    
    # Rolls every string in the list List by one qubit.
    Rl = [ Roll_String(i) for i in List_Bin(List)]
    
    #=
        The following list contains the previous list but in decimal numbers. For example,
        for N =4, the first position 1 is changed to 8.
    =#
    
    Rl_d = [Bin2Dec(i) for i in Rl]
    
    #=
        The following loop rearranges the coefficients of Psi after rolling. 
        For example, for N = 4, if the first coefficient 0001 is mapped to the
        eighth coefficient 1000 after one rotation of the qubits. 
        The coefficient of the rolled Psi in the i ^ th position is in the
        Rl_d[i] ^ th positon of the initial Psi.
    =#
    
    Psi_Rolled = []
    
    for i=1:2^L
        
        # Rearranging the coefficients according to the list l_d.
        push!(Psi_Rolled,Initial_Psi[Rl_d[i]+1])
        
        #= The addition of 1 to the index is necessary because Julia counts from 1,
           rather than 0. But we need to represent all the numbers from 1 to 16 using 
           four binary digits. So we started with the List = [0 to 15], then after
           rolling we just add 1 to each index to make it work for Julia.
        =#
    end
    return Psi_Rolled
end

#=
The following function performs specified number of rolls Num on the qubits.
=#

function N_Rolled(Num, Initial_Psi)
    
    if Num == 0 
        return Initial_Psi
    else
        
        s = Psi_Roll(Initial_Psi)
        for i=1:Num-1
            s = Psi_Roll(s)
        end
        return s
    end
end


function Average_Entropy(Initial_Psi)
    
    list_of_entropies = []
    #=
    The loop calculates all the entropies and returns a list containing them.
    =#
    for i=1:L
        S = Entropy(N_Rolled(i,Initial_Psi))
        push!(list_of_entropies,S)
    end
    return sum(list_of_entropies)/length(list_of_entropies)
end;

py"""
f = open('plot_data'+'.txt', 'w')
def Write_file(Noise, Energy, Entropy):
    f = open('plot_data'+'.txt', 'a')
    f.write(str(Noise) +'\t'+ str(Energy)+ '\t' + str(Entropy) +'\n')
"""
# delta_index runs from 0 to 255.
delta_index = parse(Int64,ARGS[1])

Delta = LinRange(0.0,0.35,256+1)
Delta_start = Delta[delta_index+1]
Delta_end = Delta[delta_index+2]
Num = 5

for i=0:Num
    delta = Delta_start+(i/Num)*(Delta_end-Delta_start)
    Op = Grover_operator(delta)
    EIGU = py"eigu"(Op)
    X = string(delta)
    Y = real(1im*log.(EIGU[1]))
    V = EIGU[2]
    
    for j=1:2^L
        py"Write_file"(delta, real(Y[j]), Average_Entropy(V[1:2^L,j:j]))
    end
end

#Delta = LinRange(0.0,0.35,17)

#=
for i = 0:length(Delta)-2
    println(Delta[i+1],Delta[i+2])
end
=#
