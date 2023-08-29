L = 10;

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


# Good seeds = 10,14, 1945, 1337, 141421, 1414, 173205075, 1642, 1942.
SEED = 4000
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
    return -GROVER_DELTA
end;

#=
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
=#

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
                M[i+1,j+1] = M[j+1,i+1]'
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

# delta_index runs from 0 to 128.
delta_index = parse(Int64,ARGS[1])

Delta = LinRange(0.0,0.3,64+1)
delta_start = Delta[delta_index+1]
delta_end = Delta[delta_index+2]
Num = 4

#=
Arrays to hold delta, energy and entropy before they are written into the file.              
=#
deltas = []
Ys = []
Entropies = []
              
for i=0:Num
    delta = delta_start+(i/Num)*(delta_end-delta_start)
    Op = Grover_delta(delta)
    EIGU = py"eigu"(collect(Op))
    X = string(delta)
    Y = real(1im*log.(EIGU[1]))
    V = EIGU[2]
    
    for j=1:2^L
        py"Write_file"(delta, real(Y[j]), average_entanglement_entropy(V[1:2^L,j:j]))
    end
end

#=
using Plots
using DelimitedFiles
using ColorSchemes
#using CSV
using LaTeXStrings
#using PyPlot
file = raw"plot_data.txt"
M = readdlm(file)
delta = M[:,1]; # index
energy = M[:,2]; # eigenvalue
entropy = M[:,3]; # entropy
S_Page = 0.5*L*log(2)-0.5
quasienergy = pi-atan(2/sqrt(2^L-1));
gr()
plot_font = "Computer Modern"
default(fontfamily=plot_font)
MyTitle = "L = "*string(L)*", Page Value = "*string(round(0.5*L*log(2)-0.5;digits = 2))*" ";
p = plot(delta,energy,
    seriestype = :scatter,
    markerstrokecolor = "grey30",
    markerstrokewidth=0.0,
    markersize=2,
    thickness_scaling = 1.4,
    xlims=(0,0.4), 
    ylims=(-3.14,3.14),
    title = MyTitle,
    label = "",
    #legend = :bottomleft,
    dpi=600,
    zcolor = entropy,
    grid = false,
    #colorbar_title = "Average entanglement entropy",
    font="CMU Serif",
    color = :jet1,
    #:linear_bmy_10_95_c78_n256,#:diverging_rainbow_bgymr_45_85_c67_n256,#:linear_bmy_10_95_c78_n256,#:rainbow1,
    right_margin = 5Plots.mm,
    left_margin = Plots.mm,
    titlefontsize=16,
    guidefontsize=16,
    tickfontsize=16,
    legendfontsize=16,
    framestyle = :box
    )
plot!(size=(900,700))
#plot!(yticks = ([(-pi) : (-pi/2): (-pi/4): 0: (pi/4) : (pi/2) : pi;], ["-\\pi", "-\\pi/2", "-\\pi/4","0","\\pi/4","\\pi/2","\\pi"]))
hline!([[-quasienergy]],lc=:magenta,linestyle= :dashdotdot,legend=false)
hline!([ [0]],lc=:magenta,linestyle= :dashdotdot,legend=false)
hline!([ [quasienergy]],lc=:magenta,linestyle= :dashdotdot,legend=false)
xlabel!("Noise \$\\delta\$")
ylabel!("Energy \$\\phi_{F}\$")
savefig(string(L)*"_delta_energy_"*string(SEED)*".png")
=#
