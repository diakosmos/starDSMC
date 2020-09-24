function ix(i::Int,j::Int,k::Int)
    @assert 0 < i <= 3
    @assert 0 < j <= 4
    @assert 0 < k <= 5
    i + 3*(j-1) + 3*4*(k-1)
end
function ijk(n::Int)
    @assert 0 < n <= 3*4*5
    i  = (n -1) % 3 + 1
    jk = (n -1) ÷ 3 + 1
    j  = (jk-1) % 4 + 1
    k  = (jk-1) ÷ 4 + 1
    (i,j,k)
end
