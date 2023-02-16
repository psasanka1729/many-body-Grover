L = 10;

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

SEED = 100000+parse(Int64,ARGS[1])
Random.seed!(SEED)
NOISE = 2*rand(Float64,Number_of_Gates).-1;

I2 = [1 0; 0 1];
Z = [1 0;0 -1];
H = (1/sqrt(2))*[1 1;1 -1]
Rx(theta)= exp(-1im*(theta/2)*([1 0;0 1]-[0 1;1 0]));
Hadamard(noise) = exp(-1im*(pi/2+noise)*(I2-H)) #Ry(pi/2+noise)*Pauli_Z;
CX(noise) = exp(-1im*((pi/2+noise))*([1 0;0 1]-[0 1;1 0])); # This is X gate.
Z_gate(noise) = exp(-1im*(pi/2+noise)*(I2-Z)) #Hadamard(noise)*CX(noise)*Hadamard(noise); # noise
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

#=
U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0
# V(x_bar basis) = (Z basis).
V = py"eigu"(G_exact)[2];=#

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
U (x bar basis) = (Z basis)
=#
#=
function Basis_Change_Matrix()
    local U = zeros(2^L,2^L)
    for k = 1:2^L
        sigma_z_basis = zeros(2^L,1);
        sigma_z_basis[k] = 1
        U += sigma_z_basis * (x_bar(k))'
    end
    return U
end;
=#

#=
The following function converts a matrix from a given basis to Z basis.
The matrix VM has the old bases as its column vectors.
=#
function Basis_Change_Matrix(VM)
    local U = zeros(2^L,2^L)
    for k = 1:2^L
        sigma_z_basis = zeros(2^L,1);
        sigma_z_basis[k] = 1
        U += sigma_z_basis * (VM[:,k])'
    end
    return U #  Matrix_Z = U*Matrix_old_basis.
end;

#DELTA = 0.01
function Eigenvectors(DELTA)
    
    U_list = [];
    U_noise_list = [];
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
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix
        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            push!(U_noise_list,z_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
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
    for i = 1 : U_0_gate_number
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
            
        elseif Gates_data_1[i] == "Z"
        
            epsilon = NOISE[i]
            push!(NOISE_list,epsilon)        
            z_matrix = Matrix_Gate(Z_gate(DELTA*epsilon),Gates_data_3[i])
            U_x_delta *= z_matrix
        
            push!(Gates_data_new_1,"Z")
            push!(Gates_data_new_2,0.0)
            push!(Gates_data_new_3,Gates_data_3[i]) 
        
            push!(U_noise_list,z_matrix) # Noise.
        
            push!(U_list,Matrix_Gate(Z_gate(0.0),Gates_data_3[i])) # Noiseless.
            
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
    
    function kth_term(k)

            f_k = Identity(2^L);
    
            for i = k:length(U_list)-1
                f_k = f_k*collect(U_list[length(U_list)-i+k])
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
    
    
        return f_k*H_k*(f_k')
    end;         
    
    #= The following loop sums over all epsilon to get H_eff. =#
    h_eff = zeros(2^L,2^L);
    for i = 1:length(U_list)
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
    h_eff_eigenvectors = eigvecs(h_eff) # Diagonalizing h_eff.

    return h_eff_eigenvectors
end;

H_eff_Eigvecs = Eigenvectors(0.0);

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

        # The sum goes from 1 to dim(H) i.e length of an eigenvector.
        for i = 1:2^L
            p = abs(Eigenvectors_Matrix[:,n][i])^2 + 1.e-9 # To avoid singularity in log.
            q = abs(Eigenvectors_Matrix[:,n+1][i])^2 + 1.e-9           

            KLd_sum += p*log(p/q)
        end
        #println(KLd_sum)
        push!(KL,KLd_sum)  
    end
    return KL
end;

H_eff_D = Eigenvectors(0.0) # Any value of \delta works.
py"""
f = open('KL_data'+'.txt', 'w')
def Write_file(Index, KL_div):
    f = open('KL_data'+'.txt', 'a')
    f.write(str(Index) +'\t'+ str(KL_div)+'\n')
"""
KLd_calculated = KLd(H_eff_D)
for k = 1:2^L-1
    py"Write_file"(k,KLd_calculated[k])
end
