L = 9;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
#using PyCall
#
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

SEED = 3000+parse(Int64,ARGS[1])
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

function Grover_operator(DELTA)

    U_x_delta = Identity(2^L);

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
py"""
f = open('probability_data'+'.txt', 'w')
def Write_file(p1, p2, i):
    f = open('probability_data'+'.txt', 'a')
    f.write(str(p1) +'\t'+ str(p2)+ '\t' + str(i) +'\n')
"""
=#

probability_time_file       = open("probability_time.txt", "w")

function Pxbar(full_wavefunction)
    #= full wavefunction = \sum_{j=0 to 2^{L}-1} \alpha_{j} |j>.
    x_bar_wavefunction =  \sum_{j=0 to 2^{L}-1} \alpha_{j} |j> - \alpha_{0}|0>.
    =#
    x_bar_wavefunction = full_wavefunction[2:2^L]
    p_xbar = sum(x_bar_wavefunction)
    return abs(p_xbar)^2/(2^L-1)
end

#delta_index = 1+parse(Int64,ARGS[1])
#DELTAS = [0.0,0.01,0.02,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.21,0.24,0.3,0.4]
Delta = 0.06#DELTAS[delta_index]

G_delta = Grover_operator(Delta);

ket_0    = zeros(2^L)
ket_0[1] = 1
N = 2^L
ket_x    = (1/sqrt(N))   * ones(N)
ket_x_bar = sqrt(N/(N-1)) * ket_x - 1/sqrt(N-1)*ket_0 # Normalization checked.

Psi_0(L) = sparse((1/sqrt(2^L))*ones(ComplexF64,2^L));
p_0l = []
p_x_barl = []
ket_psi = Psi_0(L);
p_0 = ket_psi[1]*conj.(ket_psi[1])
p_xbar = Pxbar(ket_psi)

#py"Write_file"(real(p_0),real(p_xbar),0)
write(probability_time_file, string(real(p_0)))
write(probability_time_file, "\t")
write(probability_time_file, string(real(p_xbar)))
write(probability_time_file, "\t")
write(probability_time_file, string(0))
write(probability_time_file, "\n")

push!(p_0l,p_0)
push!(p_x_barl,p_xbar)
for i=1:500
    global ket_psi = G_delta*ket_psi
    global p_0 = abs(ket_psi[1])^2
    global p_xbar = abs(ket_x_bar'*ket_psi)^2
    #global psi = U*psi
    #global p_0 = abs(psi[1])^2
    #global p_xbar = Pxbar(psi)
    #py"Write_file"(real(p_0),real(p_xbar),i)
    push!(p_0l,p_0)
    push!(p_x_barl,p_xbar)
    write(probability_time_file, string(real(p_0)))
    write(probability_time_file, "\t")
    write(probability_time_file, string(real(p_xbar)))
    write(probability_time_file, "\t")
    write(probability_time_file, string(i))
    write(probability_time_file, "\n")
end;

#using Plots
#plot(p_0l,label="p0")
#plot!(p_x_barl,label="p_x_bar")
using LsqFit
model(t, p) = p[1] .+ p[2] * cos.(p[3] .* t .+ p[4])
# Define the first order data set.
xdata = [i for i = 50:70];
ydata = p_0l[50:70]
# Define an initial guess for the parameters
p0 = [  0.1,   0.1,   0.14, 10.0]
# Call the curve_fit function
fit = curve_fit(model, xdata, ydata, p0)
# Extract the best-fit parameters
A_1 = fit.param[1]
B_1 = fit.param[2]
omega_1 = fit.param[3]
phi_1 = fit.param[4]
model(t, p) = p[1] .+ p[2] * cos.(p[3] .* t .+ p[4])
# Define the second order data set
xdata = [i for i = 50:210];
ydata = p_0l[50:210]
# Define an initial guess for the parameters
p0 = [  A_1,   B_1,   omega_1, phi_1]
# Call the curve_fit function
fit = curve_fit(model, xdata, ydata, p0)
# Extract the best-fit parameters
A_2 = fit.param[1]
B_2 = fit.param[2]
omega_2 = fit.param[3]
phi_2 = fit.param[4]
#scatter(xdata,ydata)
#plot!(xdata,A_2 .+ B_2 .* cos.(omega_2 .* xdata .+ phi_2))

fit_data_file       = open("fit_data.txt", "w")

write(fit_data_file, string(A_2))
write(fit_data_file, "\t")
write(fit_data_file, string(B_2))
write(fit_data_file, "\t")
write(fit_data_file, string(omega_2))
write(fit_data_file, "\t")
write(fit_data_file, string(phi_2))
write(fit_data_file, "\t")
write(fit_data_file, string(p_0l[200]-(A_2+B_2*cos(omega_2*200+phi_2))))
close(fit_data_file)
