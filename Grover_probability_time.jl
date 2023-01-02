L = 6;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
file = raw"6_Grover_gates_data.txt" # Change for every L.
M = readdlm(file)
Gates_data_1 = M[:,1];
Gates_data_2 = M[:,2];
Gates_data_3 = M[:,3];

Number_of_Gates = 2*(2*L^2-6*L+5)+2*L+4*L-4;
SEED = 1000
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

#length(NOISE)

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)=exp(-1im*(theta/2)*[1 1;1 1]);
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*[1 1;1 1]);
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

function x_bar(n)
    k_n = (2*pi*n)/(2^L-2)
    s = zeros(2^L-2,1)
    for j = 1:2^L-2
        sigma_z_basis = zeros(2^L-2,1);
        sigma_z_basis[j] = 1
        s += exp(1im*j*k_n) * sigma_z_basis
    end
    return s/sqrt(2^L-1)
end;
#= 
The following function returns the basis transformation matrix U such that
U\x_bar_k> = \k> for k = 1 2 ... 2^L-2. The two special states are neglected. 
=#
function Basis_Change_Matrix()
    local U = zeros((2^L)-2,(2^L)-2)
    for k = 1:2^L-2
        sigma_z_basis = zeros(2^L-2,1);
        sigma_z_basis[k] = 1
        U += sigma_z_basis * (x_bar(k))'
    end
    return U
end;

function Grover_operator(DELTA)
    U_list = [];
    U_noise_list = [];
    U_x_delta = sparse(Identity(2^L));
    #ux_list = []
    NOISE_list = []

    Gates_data_new_1 = []
    Gates_data_new_2 = []
    Gates_data_new_3 = []
    
    # U_x
    for i = (2*L^2-4*L+5)+1 : 2*(2*L^2-6*L+5)+2*L+4*L-4
        if Gates_data_1[i] == "H"
            
            
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)
            h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_x_delta *= h_matrix
        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
            push!(U_noise_list,h_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= x_matrix
        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            push!(U_noise_list,x_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(ux_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            U_x_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
            push!(U_noise_list,rx_matrix) # Noise.
        
            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiselss.
            
        end
    end
    
    U_0_delta = sparse(Identity(2^L));
    
    #u0_list = []
    # U_0
    for i = 1 : 2*L^2-4*L+5
        if Gates_data_1[i] == "H"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            h_matrix = Matrix_Gate(Hadamard(DELTA*epsilon), Gates_data_3[i])
            U_0_delta *= h_matrix
        
            push!(Gates_data_new_1,"H")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i])
        
            push!(U_noise_list,h_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Hadamard(0.0), Gates_data_3[i])) # Noiseless.
            
        elseif Gates_data_1[i] == "X"

        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            x_matrix = Matrix_Gate(CX(DELTA*epsilon),Gates_data_3[i])
            U_0_delta *= x_matrix
        
            push!(Gates_data_new_1,"X")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            push!(U_noise_list,x_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(CX(0.0),Gates_data_3[i])) # Noiseless.
            
        else
            #push!(u0_list,"CRX")
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            rx_matrix = CU(Rx(Gates_data_1[i]+DELTA*epsilon), Gates_data_2[i], Gates_data_3[i])
            U_0_delta *= rx_matrix
        
            push!(Gates_data_new_1,Gates_data_1[i])
            push!(Gates_data_new_2,Gates_data_2[i])
            push!(Gates_data_new_3,Gates_data_3[i])
        
            push!(U_noise_list,rx_matrix) # Noise.
        
            push!(U_list,CU(Rx(Gates_data_1[i]), Gates_data_2[i], Gates_data_3[i])) # Noiseless.
            
        end
    end
        
    GROVER_DELTA = U_x_delta*U_0_delta
    
    return collect(GROVER_DELTA)
    #return h_eff_D
end;

U_0 = -Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = 1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;

U = Grover_operator(0.0);

py"""
f = open('probability_data'+'.txt', 'w')
def Write_file(p1, p2, i):
    f = open('probability_data'+'.txt', 'a')
    f.write(str(p1) +'\t'+ str(p2)+ '\t' + str(i) +'\n')
"""

function Pxbar(full_wavefunction)
    #= full wavefunction = \sum_{j=0 to 2^{L}-1} \alpha_{j} |j>.
    x_bar_wavefunction =  \sum_{j=0 to 2^{L}-1} \alpha_{j} |j> - \alpha_{0}|0>.
    =#
    x_bar_wavefunction = full_wavefunction[2:2^L]
    p_xbar = sum(x_bar_wavefunction)
    return abs(p_xbar)^2/(2^L-1)
end

Psi_0(L) = (1/sqrt(2^L))*ones(ComplexF64,2^L);

p_0l = []
p_x_barl = []
psi = Psi_0(L);
p_0 = psi[1]*conj.(psi[1])
p_xbar = Pxbar(psi)
py"Write_file"(real(p_0),real(p_xbar),0)
push!(p_0l,p_0)
push!(p_x_barl,p_xbar)
#println(p_xbar)
#println(p_0)
for i=1:200
    psi = U*psi
    p_0 = abs(psi[1])^2
    p_xbar = Pxbar(psi)
    py"Write_file"(real(p_0),real(p_xbar),i)
    push!(p_0l,p_0)
    push!(p_x_barl,p_xbar)
    #println(p_xbar)
    #println(p_0)
end;
using Plots
plot(p_0l,label="p0")
plot!(p_x_barl,label="p_x_bar")
