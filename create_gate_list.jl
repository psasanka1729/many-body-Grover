using Random
using LinearAlgebra
using SparseArrays
using PyCall
using Statistics

L = 3;

py"""
f = open('Grover_gates_data'+'.txt', 'w')
def Write_file1(X, Y, Z):
    f = open('Grover_gates_data'+'.txt', 'a')
    f.write(str(X) + '\t' + str(Y)+ '\t' + str(Z) +'\n')
"""

Identity(dimension) = 1* Matrix(I, dimension, dimension);

function Write_Gates_to_File(L)
    Gate_count = 0
    
    U0_XHL_Gates = []
    # Left side of MCX of U_0.
    for i = 1:L
        push!(U0_XHL_Gates,["X",i])
    end    
    # H on the L^th qubit.
    push!(U0_XHL_Gates,["H",L])

    # Right side of MCX of U_0.
    U0_XHR_Gates = []
    # H on the L^th qubit.
    push!(U0_XHR_Gates,["H",L])
    for i = 1:L
        push!(U0_XHR_Gates,["X",i])
    end
    
    MCX = sparse(Identity(2^L));
    
    # Multiplying the gates to get the left side of MCX of U_0.
    XHL_Matrix = sparse(Identity(2^L))
    for i in U0_XHL_Gates
        if i[1] == "H"   
            
            py"Write_file1"("H",0.0,i[2])
            Gate_count +=1
            
        elseif i[1] == "X"

            py"Write_file1"("X",0.0,i[2])
            Gate_count +=1
 
        end
    end
    
    #= Constructing the multicontrolled Toffoli gate. =# 
    # C_1.
    for i = 1:L-2
        for j = 1:i

            py"Write_file1"(pi/2^j, L-i, L-i+j)
            Gate_count +=1
            
        end
    end

    # C_2.
    for i = 2:L
        
        py"Write_file1"(pi/2^(i-2), 1, i)
        Gate_count +=1

    end

    # C3 = - C1.
    for i = L-2:-1:1
        for j = i:-1:1

            py"Write_file1"(-pi/2^j, L-i, L-i+j)
            Gate_count +=1

            
        end
    end

    # C_4.
    for i = 1:L-3
        for j = 1:i

            py"Write_file1"(pi/2^j, L-i-1, L-i-1+j)
            Gate_count +=1
   
        end    
    end

    # C_5.
    for i = 2:L-1

        py"Write_file1"(-pi/2^(i-2), 1, i)
        Gate_count +=1
  
    end

    # C6 = - C4.
    for i = L-3:-1:1
        for j = i:-1:1

            py"Write_file1"(-pi/2^j, L-i-1, L-i-1+j)
            Gate_count +=1
                
        end    
    end

    # Multiplying the gates to get the right side of MCX of U_0.
    XHR_Matrix = sparse(Identity(2^L))
    for j in U0_XHR_Gates
        if j[1] == "H"

            py"Write_file1"("H", 0.0,j[2])
            Gate_count +=1

                
        elseif j[1] == "X"
            
            py"Write_file1"("X",0.0,j[2])
            Gate_count +=1
  
        end
    end

    #U0_matrix = sparse(XHL_Matrix*MCX*XHR_Matrix)    

    
    #= Ux matrix. =#
    # Left side of MCX of U_x.
    Ux_XHL_Gates = []
    # H gate L-1 time
    for i = 1:L-1
        push!(Ux_XHL_Gates,["H",i])
    end    
    # X gate L-1 time
    for i = 1:L-1
        push!(Ux_XHL_Gates,["X",i])
    end  
    # Z gate on L.
    push!(Ux_XHL_Gates,["Z",L])
    
    Ux_XHR_Gates = []
    # Right side of MCX of U_x.
    push!(Ux_XHR_Gates,["Z",L])
    # X gate L-1 time
    for i = 1:L-1
        push!(Ux_XHR_Gates,["X",i])
    end    
    # H gate L-1 time
    for i = 1:L-1
        push!(Ux_XHR_Gates,["H",i])
    end
    
    # Multiplying the matrices to get the left side of MCX of U_x.
    MCX = sparse(Identity(2^L));
    XHL_Matrix = sparse(Identity(2^L))
    for i in Ux_XHL_Gates
        
        if i[1] == "H"

            py"Write_file1"("H", 0.0,i[2])
            Gate_count +=1
            
        elseif i[1] == "X"

            py"Write_file1"("X", 0.0,i[2])
            Gate_count +=1
        elseif i[1] == "Z"
            py"Write_file1"("Z", 0.0,i[2])
            Gate_count +=1
        end
    end
    
    #= MCX gate in linear depth.=#
    #= Contructing the multicontrolled Toffoli gate. =#
    # C_1.
    for i = 1:L-2
        for j = 1:i
            
            py"Write_file1"(pi/2^j, L-i, L-i+j)
            Gate_count +=1
                
        end
    end
    # C_2.
    for i = 2:L
        
        py"Write_file1"(pi/2^(i-2), 1, i)
        Gate_count +=1
  
    end
    # C3 = - C1.
    for i = L-2:-1:1
        for j = i:-1:1

            py"Write_file1"(-pi/2^j, L-i, L-i+j)
            Gate_count +=1
   
        end
    end
    # C_4.
    for i = 1:L-3
        for j = 1:i

            py"Write_file1"(pi/2^j, L-i-1, L-i-1+j)
            Gate_count +=1
   
        end    
    end
    # C_5.
    for i = 2:L-1

        py"Write_file1"(-pi/2^(i-2), 1, i)
        Gate_count +=1
    
    end
    # C6 = - C4.
    for i = L-3:-1:1
        for j = i:-1:1

            py"Write_file1"(-pi/2^j, L-i-1, L-i-1+j)
            Gate_count +=1
   
        end    
    end

    # Right side of MCX of U_x.
    XHR_Matrix = sparse(Identity(2^L))
    for j in Ux_XHR_Gates
        if j[1] == "H"          
            
            py"Write_file1"("H", 0.0,j[2])
            Gate_count +=1
   
        elseif j[1] == "X"         
            
            py"Write_file1"("X", 0.0,j[2])
            Gate_count +=1
        elseif j[1] == "Z"
            py"Write_file1"("Z", 0.0,j[2])
            Gate_count +=1            
        end
    end
    return Gate_count
end; 
Write_Gates_to_File(L)

Number_of_Gates(L) = 4*L^2-6*L+10;
print("Number of gates ",Number_of_Gates(L) )
