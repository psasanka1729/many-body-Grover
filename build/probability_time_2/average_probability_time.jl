L = 14;

using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using PyCall
file = raw"14_new_Grover_gates_data.txt" # Change for every L.
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
Delta = 0.00
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
    
    return GROVER_DELTA
end;

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

U_0 = Identity(2^L)#[-1 0 0 0; 0 1 0 0; 0 0 1 0;0 0 0 1];
U_0[1,1] = -1
A = ones(2^L,2^L);
U_x = (2/2^L)*A-Identity(2^L); # 2\s><s|-I
G_exact = U_x*U_0
U = G_exact
#U = Grover_operator(Delta);

Psi_0(L) = sparse((1/sqrt(2^L))*ones(ComplexF64,2^L));
p_0l = []
p_x_barl = []
psi = Psi_0(L);
p_0 = psi[1]*conj.(psi[1])
p_xbar = Pxbar(psi)
py"Write_file"(real(p_0),real(p_xbar),0)
push!(p_0l,p_0)
push!(p_x_barl,p_xbar)
for i=1:200
    global psi = U*psi
    p_0 = abs(psi[1])^2
    p_xbar = Pxbar(psi)
    py"Write_file"(real(p_0),real(p_xbar),i)
    push!(p_0l,p_0)
    push!(p_x_barl,p_xbar)
end;

using LsqFit

model(t, p) = p[1] .+ p[2] * cos.(p[3] .* t .+ p[4])

# Define the first order data set.
xdata = [i for i = 50:70];
ydata = p_0l[50:70]

# Define an initial guess for the parameters
p0 = [  0.31,   0.31,   0.12, -9.67]

# Call the curve_fit function
fit = curve_fit(model, xdata, ydata, p0)

# Extract the best-fit parameters
A_1 = fit.param[1]
B_1 = fit.param[2]
omega_1 = fit.param[3]
phi_1 = fit.param[4]

model(t, p) = p[1] .+ p[2] * cos.(p[3] .* t .+ p[4])

# Define the second order data set
xdata = [i for i = 70:150];
ydata = p_0l[70:150]

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

py"""
f = open('fitted_data'+'.txt', 'w')
def Write_file_fit(A, B, omega, phi, error):
    f = open('fitted_data'+'.txt', 'a')
    f.write(str(A) +'\t'+ str(B)+ '\t' + str(omega)+'\t' + str(phi) + '\t' +str(error) + '\n')
"""
py"Write_file_fit"(A_2,B_2,omega_2,phi_2,p_0l[100]-(A_2+B_2*cos(omega_2*100+phi_2)))
