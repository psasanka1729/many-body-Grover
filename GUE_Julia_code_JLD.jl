using JLD
using Random
using LinearAlgebra

function generate_gaussian_unitary_matrix(n::Int)
    # Generate a random complex matrix
    A = randn(ComplexF64, n, n) + im * randn(ComplexF64, n, n)
    
    # Perform QR decomposition to get a unitary matrix
    Q, R = qr(A)
    
    return Q
end;

# Specify the size of the matrix (e.g., n=3 for a 3x3 matrix)
n = 2^10

# Generate a Gaussian Unitary Matrix
gu_matrix = generate_gaussian_unitary_matrix(n)

save("GUEmatrix.jld","GUE",gu_matrix)
