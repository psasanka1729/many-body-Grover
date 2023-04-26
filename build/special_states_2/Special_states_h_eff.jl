L = 12;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
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

SEED = 25599+parse(Int64,ARGS[1])
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

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0;

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


#=
Derivative of G(\delta) is calculated using forward difference.
=#
function h_eff_from_derivative(h)
    h_eff_matrix = 1im*((Grover_delta(h)*(-G_exact)')-Identity(2^L))/h
    return h_eff_matrix
end;

# h_eff matrix is created once to use in the following calculation.
h = 1.e-8
matrix_of_h_effective = h_eff_from_derivative(h);


function block_h_eff_matrix(h_eff_matrix)

    #= 
    Construction of block 2x2 H_eff matrix.
    =#
    
    # Defining the state |0> in sigma_z basis.
    ket_0    = zeros(2^L)
    ket_0[1] = 1
    
    # Defining the state |x_bar> in sigma_z basis.
    N = 2^L
    ket_x    = (1/sqrt(N))*ones(N)
    ket_xbar = sqrt(N/(N-1))*ket_x-1/sqrt(N-1)*ket_0 # Normalization checked.
    
    # Matrix elements of h_eff in |0> and |xbar> basis.
    h_0_0       = ket_0'    * h_eff_matrix * ket_0
    h_0_xbar    = ket_0'    * h_eff_matrix * ket_xbar
    h_xbar_0    = ket_xbar' * h_eff_matrix * ket_0
    h_xbar_xbar = ket_xbar' * h_eff_matrix * ket_xbar
    

    # h_eff block matrix.
    h_eff_block = [ h_0_0 h_0_xbar; h_xbar_0 h_xbar_xbar]
    

    # G_0 block matrix.
    N = 2^L
    G_0_block = [2/N-1 -2*sqrt(N-1)/N;2*sqrt(N-1)/N 2/N-1]
    
    return h_eff_block*G_0_block # h_eff * G_0.
end;

# Calculate the 2x2 matrix h_eff*G_0 in the basis |0> and |x_bar>.
H_eff_G_0 = block_h_eff_matrix(matrix_of_h_effective);

# Calculate the matrix B in sigma_y basis.
function B_matrix_y_basis()
    
    N = 2^L
    theta = asin(2*sqrt(N-1)/N)
    
    # Eigenstates of tau_y in the |0> and |xbar> basis as row vectors.
    y_s_p = (1/sqrt(2))*[1  1im]'  # corresponding to +1 eigenvalue.
    y_s_n = (1/sqrt(2))*[1 -1im]'  # corresponding to -1 eigenvalue.
    
    # Value of the integration for each possible eigenvalues.
    I_11 = exp(-im*theta)                          # -1 -1.
    I_12 = (1im/(sin(theta)))*log(exp(-1im*theta)) # -1  1.
    I_21 = I_12                                    #  1 -1.
    I_22 = exp(im*theta)                           #  1  1.
    
    return [[(I_11*y_s_n'*H_eff_G_0*y_s_n) (I_12*y_s_n'*H_eff_G_0*y_s_p)];
            [(I_21*y_s_p'*H_eff_G_0*y_s_n) (I_22*y_s_p'*H_eff_G_0*y_s_p)]]
end;

function sigma_y_to_sigma_z_basis_change(Matrix)
    
    sigma_y_n = (1/sqrt(2))*[1 -1im]'   # corresponding to -1 eigenvalue.
    sigma_y_p = (1/sqrt(2))*[1  1im]'   # corresponding to +1 eigenvalue.
    
    sigma_z_n = [0 1]'                   # corresponding to -1 eigenvalue.
    sigma_z_p = [1 0]'                   # corresponding to +1 eigenvalue.
    
    V = spzeros(2,2)
    
    V = V + sigma_z_n * sigma_y_n'
    V = V + sigma_z_p * sigma_y_p'
    
    return V*Matrix*V'
end;

#=
Basis transformation matrix from sigma_y to sigma_z.
=#
#sigma_y_to_sigma_z(Matrix) = ((1/sqrt(2))*[[1,1] [-1im, 1im]])*Matrix*inv((1/sqrt(2))*[[1,1] [-1im,1im]]);

# Changing the B matrix from sigma_y basis to sigma_z basis.
B_y_basis = B_matrix_y_basis()
B_matrix_z_basis = sigma_y_to_sigma_z_basis_change(B_y_basis);

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

py"""
f = open('Pauli_coefficients_data'+'.txt', 'w')
def Write_file_Pauli(b_0, b_1, b_2, b_3):
    f = open('Pauli_coefficients_data'+'.txt', 'a')
    f.write(str(b_0) +'\t'+ str(b_1)+ '\t' + str(b_2) +'\t' + str(b_3) +'\n')
"""

PC = Pauli_coefficients(B_matrix_z_basis)
py"Write_file_Pauli"(PC[1],PC[2],PC[3],PC[4])

py"""
f = open('special_states_eigenvalues_data'+'.txt', 'w')
def Write_file(eigenvalue_1, eigenvalue_2):
    f = open('special_states_eigenvalues_data'+'.txt', 'a')
    f.write(str(eigenvalue_1) +'\t'+ str(eigenvalue_2)+'\n')
"""

# Diagonalize the special state matrix.
Special_eigenvalues = eigvals(B_matrix_z_basis)
# Write the two eigenvalue to the data file.
py"Write_file"(Special_eigenvalues[1],Special_eigenvalues[2])

#=
function exp_h_spec(delta)
    theta = asin(2*sqrt(2^L-1)/2^L)
    H_0_spec = [-theta 0;0 theta]
    H_spec   = B_y_basis
    return exp(-1im*(H_0_spec + delta*H_spec))
end;
=#

#=
Exact_list = []
Effec_list = []
delta_list = []
Num = 10;
for i = 1:Num
    delta = 0.15*(i/Num)

    eigu_h_eff = py"eigu"(-exp_h_spec(delta))
    h_eff_energies = real(1im*log.(eigu_h_eff[1])); # Eigenvalue.
    
    EIGU = py"eigu"(collect(Grover_delta(delta)))
    G_delta_energies = real(1im*log.(EIGU[1])); # Eigenvalue.
    special_states_exact = [G_delta_energies[1];G_delta_energies[2^L]]
    
    for j = 1:2
        #py"Write_file2"(delta,Exact[j],Effec[j])
        push!(delta_list,delta)
        push!(Exact_list,special_states_exact[j])
        push!(Effec_list,h_eff_energies[j])
        #println(delta);
    end
end;
=#

#=
using Plots
using DelimitedFiles
using ColorSchemes
using LaTeXStrings

Markersize = 4
delta = delta_list
exact = Exact_list # exact energy.
effec = Effec_list # effective energy.

S_Page = 0.5*L*log(2)-0.5

plot_font = "Computer Modern"
default(fontfamily=plot_font)
gr()
S_Page = 0.5*L*log(2)-0.5
MyTitle = "L = 4 ";
p = plot(delta,exact,
    seriestype = :scatter,
    markercolor = "firebrick1 ",#"red2",
    markerstrokewidth=0.0,
    markersize=Markersize,
    thickness_scaling = 1.4,
    xlims=(0,0.35), 
    ylims=(-3.14,3.14),
    #title = MyTitle,
    label = "Exact",
    legend = :bottomleft,
    dpi=500,
    #zcolor = entropy,
    grid = false,
    #colorbar_title = "Average entanglement entropy",
    font="CMU Serif",
    color = :jet1,
    right_margin = 5Plots.mm,
    left_margin = Plots.mm,
    titlefontsize=10,
    guidefontsize=13,
    tickfontsize=13,
    legendfontsize=15,
    framestyle = :box
    )

p = plot!(delta,effec,
    seriestype = :scatter,
    markercolor = "blue2",
    markershape=:pentagon,#:diamond,
    markerstrokewidth=0.0,
    markersize=3,
    thickness_scaling = 1.4,
    xlims=(0,0.15), 
    ylims=(-3.14,3.14),
    #title = MyTitle,
    label = "Effective",
    legend = :bottomleft,
    dpi=500,
    #zcolor = entropy,
    grid = false,
    #colorbar_title = "Average entanglement entropy",
    font="CMU Serif",
    right_margin = 5Plots.mm,
    left_margin = Plots.mm,
    titlefontsize=10,
    guidefontsize=18,
    tickfontsize=18,
    legendfontsize=20,
    framestyle = :box
    )
#=
Adjust the length of the axis tick.
=#
function ticks_length!(;tl=0.02)
    p = Plots.current()
    xticks, yticks = Plots.xticks(p)[1][1], Plots.yticks(p)[1][1]
    xl, yl = Plots.xlims(p), Plots.ylims(p)
    x1, y1 = zero(yticks) .+ xl[1], zero(xticks) .+ yl[1]
    sz = p.attr[:size]
    r = sz[1]/sz[2]
    dx, dy = tl*(xl[2] - xl[1]), tl*r*(yl[2] - yl[1])
    plot!([xticks xticks]', [y1 y1 .+ dy]', c=:black, labels=false,linewidth = 1.5)
    plot!([x1 x1 .+ dx]', [yticks yticks]', c=:black, labels=false,linewidth = 1.5, xlims=xl, ylims=yl)
    return Plots.current()
end
ticks_length!(tl=0.015)
plot!(size=(900,700))

xlabel!("Noise, "*L"\delta")
ylabel!("Energy of the bulk")
#savefig("exact_effec_"*string(L)*"_"*string(SEED)*".png")
=#
