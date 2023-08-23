L = 12;
using JLD
using PyCall
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
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

SEED = 10000+parse(Int64,ARGS[1])
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
           
    return sparse(PI_0_matrix + PI_1_matrix)     
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

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;
#V = py"eigu"(G_exact)[2];

function x_bar(n)
    s = zeros(2^L)
    k_n = (2*pi*n)/(2^L-1)
    for j=1:2^L-1
        ket_j    = zeros(2^L);
        ket_j[j+1] = 1 
        s = s+ exp(1im*(j-1)*k_n)*ket_j
    end
    return s/sqrt(2^L-1)
end;

function sigma_z_to_x_bar_basis_change_matrix(L)
    V     = spzeros(2^L,2^L)
    ket_0 = spzeros(2^L)
    ket_0[1] = 1
    ket_1 = spzeros(2^L)
    ket_1[2] = 1
    #ket_xbar = x_bar(0)
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x    = (1/sqrt(N))   * ones(N)
    ket_xbar = sqrt(N/(N-1)) * ket_x - 1/sqrt(N-1)*ket_0 # Normalization checked.
    eigenstate_1 = ket_0 #(ket_0-1im*ket_xbar)/sqrt(2)
    eigenstate_2 = ket_xbar#(ket_0+1im*ket_xbar)/sqrt(2)
    V = V+ ket_0*(eigenstate_1')
    V = V+ ket_1*(eigenstate_2')
    
    # The bulk.
    for n=2:2^L-1
        # ket_n has n+1 th position as 1 in computational basis.
        ket_n    = spzeros(2^L);
        ket_n[n+1] = 1 
        
        V = V+ket_n*(x_bar(n-1)')
    end
    return V
end;

basis_change_matrix = sigma_z_to_x_bar_basis_change_matrix(L);

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
    #h_eff_matrix = 1im*((Grover_delta(h)*(-G_exact)')-Identity(2^L))/h
    h_eff_matrix = 1im*((Grover_delta(h)-Grover_delta(-h))/(2*h))*(-G_exact)'
    #= h_eff_xbar = V * h_eff_z * V^{\dagger}.=#
    #h_eff_matrix_xbar_basis = (basis_change_matrix)*h_eff_matrix *(basis_change_matrix') # Matrix in |0> and |xbar> basis.
    return h_eff_matrix_xbar_basis
end;

H_EFF_MATRIX = h_eff_from_derivative(1.e-8);

#=
ket_0 = spzeros(2^L)
ket_0[1] = 1
# Defining the state |x_bar> in sigma_z basis.
N = 2^L
ket_x    = (1/sqrt(N))   * ones(N)
ket_xbar = sqrt(N/(N-1)) * ket_x - 1/sqrt(N-1)*ket_0 # Normalization checked.

eigenstate_1 = ket_0#(ket_0-1im*ket_xbar)/sqrt(2)
eigenstate_2 = ket_xbar#(ket_0+1im*ket_xbar)/sqrt(2)
h_00 = eigenstate_1'*H_EFF_MATRIX*eigenstate_1
h_0x = eigenstate_1'*H_EFF_MATRIX*eigenstate_2
h_x0 = eigenstate_2'*H_EFF_MATRIX*eigenstate_1
h_xx = eigenstate_2'*H_EFF_MATRIX*eigenstate_2
=#

save("h_eff_matrix.jld","h_eff",H_EFF_MATRIX)

#=
h_eff_block_matrix = [h_00 h_0x; h_x0 h_xx]

h_eff_elements_file       = open("h_eff_elements.txt", "w")
write(h_eff_elements_file , string(h_00))
write(h_eff_elements_file , "\n")
write(h_eff_elements_file , string(h_0x))
write(h_eff_elements_file , "\n")
write(h_eff_elements_file , string(h_x0))
write(h_eff_elements_file , "\n")
write(h_eff_elements_file , string(h_xx))
write(h_eff_elements_file , "\n")
close(h_eff_elements_file)

h_spec_eigenvalues_file  = open("h_spec_eigenvalues.txt", "w")
h_spec_eigenvalues = eigvals(h_eff_block_matrix)
write(h_spec_eigenvalues_file , string(real(h_spec_eigenvalues[1])))
write(h_spec_eigenvalues_file, "\t")
write(h_spec_eigenvalues_file , string(real(h_spec_eigenvalues[2])))
=#
