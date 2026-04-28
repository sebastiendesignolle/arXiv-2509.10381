using Combinatorics
using DynamicPolynomials
using LinearAlgebra
using JuMP
using SCS
using Oscar

# this is not stricto sensu the definition of pinched-marginalizable
# we are typically missing [1, 2, 1, 3, 1] (added later by hand)
# but this is only a proof of principle
function is_marginalizable(s, k::T) where {T <: Integer}
    appear_twice = Tuple{T, T}[]
    for i in 1:k
        tmp = [j for j in eachindex(s) if s[j] == i]
        if length(tmp) > 2
            return false
        elseif length(tmp) == 2
            if tmp[2] == tmp[1] + 1
                return false
            end
            push!(appear_twice, Tuple(tmp))
        end
    end
    # sort by the first index
    sort!(appear_twice)
    # return whether the second index are now in reverse order
    return issorted(getindex.(appear_twice, (2,)); rev = true)
end

# returns the coefficient for normalization,
# then the one of the first marginal,
# and a flag for the positivity of the corresponding variable
function coefficients(s, n::T, k::T) where {T <: Integer}
    coeff_normalization = Rational{T}(n^k)
    coeff_marginal = Rational{T}(n^(k-1))
    flag_positivity = false
    ind = findfirst(s .== 1)
    if ind === nothing
        ind = 0
    end
    for i in 1:k
        tmp = [j for j in eachindex(s) if s[j] == i]
        if any(tmp[i] < ind < tmp[i+1] for i in 1:length(tmp)-1)
            coeff_marginal /= n
            flag_positivity = true
        end
        if length(tmp) > 0
            coeff_normalization /= n
            if i > 1
                coeff_marginal /= n
            end
        end
    end
    return coeff_normalization, coeff_marginal, flag_positivity
end

function get_monoms(k::T) where {T <: Integer}
    sets_marginalizable = [T[]]
    for len in 1:2k-1
        for ci in CartesianIndices(Tuple(fill(k, len)))
            if is_marginalizable(ci.I, k)
                push!(sets_marginalizable, collect(ci.I))
            end
        end
    end
    if k ≥ 3
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1]]))[1].seeds)
    end
    if k ≥ 4
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 3, 1, 4, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 4, 1]]))[1].seeds)
    end
    if k ≥ 5
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4, 1, 5]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4, 1, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4, 5]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 1, 4, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 3, 1, 4, 1, 5]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 3, 1, 4, 1, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 4, 1, 5]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 4, 1, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 3, 1, 4, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 3, 4, 1, 5, 1]]))[1].seeds)
        append!(sets_marginalizable, orbits(gset(symmetric_group(Int(k)), [[1, 2, 1, 3, 4, 5, 1]]))[1].seeds)
    end
    terms = orbits(gset(symmetric_group(Int(k)), sets_marginalizable))
    return sets_marginalizable, terms
end

function universal_parent_povm(n::TI, k::TI, sets_marginalizable::Vector, terms; verbose = false, solver = SCS.Optimizer, maxdeg = 2k, full = true) where {TI <: Integer}
    model = GenericModel{Float64}(solver)
    if solver == SCS.Optimizer
        set_attribute(model, "eps_abs", 1e-8)
        set_attribute(model, "eps_rel", 1e-8)
    end
    @variable(model, c[1:length(terms)])
    if full
        # full set of monoms
        monoms = filter(set -> length(set) ≤ maxdeg, sets_marginalizable)
    else
        # subset of monoms for speed
        monoms = filter(set -> length(set) ≤ maxdeg && (isempty(set) || vcat(set, reverse(set[1:end-1])) in sets_marginalizable), sets_marginalizable)
    end
    # println(monoms)
    nb = length(monoms)
    println("There are ", nb, " monomials")
    M = Matrix{Vector{TI}}(undef, nb, nb)
    for i in 1:nb, j in 1:nb
        if isempty(monoms[i]) || isempty(monoms[j])
            M[i, j] = vcat(monoms[i], reverse(monoms[j]))
        elseif monoms[i][end] == monoms[j][end]
            M[i, j] = vcat(monoms[i], reverse(monoms[j][1:end-1]))
        else
            M[i, j] = vcat(monoms[i], reverse(monoms[j]))
        end
    end
    println("There are ", length(unique(M)), " constraints")
    obj = zero(GenericAffExpr{Float64, GenericVariableRef{Float64}})
    con = zero(GenericAffExpr{Float64, GenericVariableRef{Float64}})
    flags_positivity = falses(length(terms))
    @variable(model, X[1:nb, 1:nb] in PSDCone())
    for u in unique(M)
        expr = zero(GenericAffExpr{Float64, GenericVariableRef{Float64}})
        for i in 1:nb, j in 1:nb
            if M[i, j] == u
                add_to_expression!(expr, X[i, j])
            end
        end
        if u in sets_marginalizable
            i = findfirst(orb -> u in orb, terms)
            coeff_normalization, coeff_marginal, flag_positivity = coefficients(u, n, k)
            add_to_expression!(obj, coeff_marginal * c[i])
            add_to_expression!(con, coeff_normalization * c[i])
            flags_positivity[i] |= flag_positivity
            @constraint(model, expr == c[i])
        else
            @constraint(model, expr == 0)
        end
    end
    # println(obj)
    # println(con)
    @objective(model, Max, obj)
    @constraint(model, con == 1)
    for ind in 1:length(terms)
        if flags_positivity[ind]
            @constraint(model, c[ind] ≥ 0)
        end
    end
    println("SDP ready!")
    verbose && set_silent(model)
    optimize!(model)
    if !JuMP.is_solved_and_feasible(model)
        println("Something went wrong: $(JuMP.raw_status(model))")
    end
    println("ηᵍ ≥ ", round(objective_value(model); digits = 4))
    res = inv(objective_value(model)) - 1
    println("SR ≤ ", round(res; digits = 4))
    println("degree: ", maximum(length.(M[abs.(value.(X)) .> 1e-6]))) # checks the degree
    return res
    return value.(c), value.(X) # inspect SDP variables
end

# obtain values from Table Ib
function check(d, k; kwargs...)
    s, t = get_monoms(Int8(k))
    return universal_parent_povm(d, k, s, t; kwargs...)
end
