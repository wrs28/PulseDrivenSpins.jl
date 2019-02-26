module PulseDrivenSpins

export Spins
export Ix, Iy, Iz
export dipolar_xx, dipolar_yy, dipolar_zz
export dipolar_xy, dipolar_yz, dipolar_zx

using LinearAlgebra
using SparseArrays
using StaticArrays

const σ₀ = sparse(1.0im*I,2,2)
const σ₁ = sparse([1,2],[2,1],ComplexF64[1,1],2,2)
const σ₂ = sparse([1,2],[2,1],1.0im*[-1,1],2,2)
const σ₃ = sparse([1,2],[1,2],ComplexF64[1,-1],2,2)

struct Spins{M}
    N::Int
    r::Vector{SVector{3,Float64}}

    Spins(N) = new{N}(N)
    Spins(N,r) = new{N}(N,[SVector(r[1][i],r[2][i],r[3][i]) for i ∈ 1:N])
    Spins(N,x,y,z) = new{N}(N,[SVector(x[i],y[i],z[i]) for i ∈ 1:N])
end

Ix(spins) = Ixyz(spins,Ix)
Iy(spins) = Ixyz(spins,Iy)
Iz(spins) = Ixyz(spins,Iz)

Ix(spins,i) = Ixyz(spins,i,σ₁)
Iy(spins,i) = Ixyz(spins,i,σ₂)
Iz(spins,i) = Ixyz(spins,i,σ₃)

function Ixyz(spins::Spins{N},I1::T) where {N,T<:Function}
    s = spzeros(2^N,2^N)
    for i ∈ 1:N
        s += I1(spins,i)
    end
    return s
end
function Ixyz(spins::Spins{N},i,σ) where N
    @assert 1 ≤ i ≤ N "spin site must satisfy 1 ≤ i ≤ N. here N=$N, i=$i"
    iz = 1
    for j ∈ 1:N
        iz = j==i ? kron(σ,iz) : kron(σ₀,iz)
    end
    return iz
end

dipolar_xx(spins) = dipolar_ij(spins,Ix,Ix)
dipolar_yy(spins) = dipolar_ij(spins,Iy,Iy)
dipolar_zz(spins) = dipolar_ij(spins,Iz,Iz)

dipolar_xy(spins) = dipolar_ij(spins,Ix,Iy)
dipolar_yz(spins) = dipolar_ij(spins,Iy,Iz)
dipolar_zx(spins) = dipolar_ij(spins,Iz,Ix)

function dipolar_ij(spins::Spins{N},I1,I2) where N
    s = spzeros(2^N,2^N)
    for i ∈ 1:N, j ∈ 1:N
        i==j ? nothing : s += dipolar_ij(spins,i,j,I1,I2)
    end
    return s
end
dipolar_ij(spins,i,j,I1,I2) = dipolar_strength(spins,i,j)*I1(spins,i)*I2(spins,j)

dipolar_strength(spins,i,j) = 1/(norm(spins.r[i]-spins.r[j]))^3




end # module
