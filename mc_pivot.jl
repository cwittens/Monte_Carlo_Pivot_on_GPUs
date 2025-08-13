using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Pkg.add("KernelAbstractions")
# Pkg.add("CUDA")
# Pkg.add("LinearAlgebra")
# Pkg.add("Plots")
# Pkg.add("Random")
# Pkg.add("Statistics")
# Pkg.add("BenchmarkTools")
# Pkg.add("LaTeXStrings")
# Pkg.add("Adapt")
# Pkg.add("AMDGPU")
# Pkg.add("Measurements")
# Pkg.add("LsqFit")

using LinearAlgebra
using Plots
using Random
using Statistics
using BenchmarkTools
using LaTeXStrings
using KernelAbstractions
using CUDA
using Adapt
using BenchmarkTools
using AMDGPU
using Measurements
using LsqFit

########################################################
# select the backend you want to use
########################################################

# backend = CUDABackend() # for NVIDIA GPUs
# backend = ROCBackend() # for AMD GPUs
backend = CPU() # for CPU only

# for small chain length, the computational overhead of the GPU can be too high
# with dynamic=true, the backend will switch to CPU for small N when running the simulation
global dynamic = true # use CPU backend for small N

# set a random seed for reproducibility
Random.seed!(42)

# set folder where all plots should be saved to
plots_folder = joinpath(@__DIR__, "plots")
mkpath(plots_folder) # create the plots folder if it does not exist

########################################################
# define all necessary functions and kernels
########################################################
begin

    # set default plotting options for einheitliche und schöne Plots
    default(
        grid=true,
        box=:on,
        size=(700, 500),
        dpi=100,
        titlefont=font(16),
        linewidth=3, gridlinewidth=2,
        markersize=4, markerstrokewidth=2,
        xtickfontsize=14, ytickfontsize=14,
        xguidefontsize=16, yguidefontsize=16,
        ztickfontsize=14, zguidefontsize=16,
        legendfontsize=14
    )


    # Function to create an initial chain of N points in 2D or 3D
    function create_initial_chain(N, dim3::Bool, backend)
        bond_diameter = 1.0f0 # bond diameter in Float32

        # chain now needs to be a 2D array 
        # and not a vector of vectors to be able
        # to put on to the GPU
        chain = zeros(Float32, N, 2 + dim3) # 2D array with N rows and 2 or 3 columns

        # Create straight chain along x-axis 
        for i in 2:N
            chain[i, 1] = (i - 1) * bond_diameter
        end

        return adapt(backend, chain)
    end


    # before we looped over i ∈ [1, N-2] and j ∈ [i+2, N]
    # now we create a list of pairs (i, j)
    # to only loop over k ∈ [1, (N-2)*(N-1)/2] 
    function build_index_map(N, backend)
        pairs = Tuple{Int,Int}[]
        for i in 1:(N-2)
            for j in (i+2):N
                push!(pairs, (i, j))
            end
        end
        return adapt(backend, pairs)
    end

    # is valid check now as a kernel function
    @kernel function kernel_is_valid_chain2D(@Const(chain), @Const(index_map), bond_diameter_2, valid_flag)
        k = @index(Global)
        i, j = index_map[k]
        # TODO: improve by not keep checking if already one has failed.
        @inbounds a = chain[j, 1] - chain[i, 1]
        @inbounds b = chain[j, 2] - chain[i, 2]
        if a^2 + b^2 < bond_diameter_2
            valid_flag[1] = false
        end
    end

    # is valid check now as a kernel function
    @kernel function kernel_is_valid_chain3D(@Const(chain), @Const(index_map), bond_diameter_2, valid_flag)
        k = @index(Global)
        i, j = index_map[k]
        @inbounds a = chain[j, 1] - chain[i, 1]
        @inbounds b = chain[j, 2] - chain[i, 2]
        @inbounds c = chain[j, 3] - chain[i, 3]
        if a^2 + b^2 + c^2 < bond_diameter_2
            valid_flag[1] = false
        end
    end


    # kernel to rotate the chain in 2D
    @kernel function kernel_rotate_2D(chain, I, @Const(range_to_rotate), cθ, sθ)
        k = @index(Global)
        @inbounds j = range_to_rotate[k]
        @inbounds xrel = chain[j, 1] - chain[I, 1]
        @inbounds yrel = chain[j, 2] - chain[I, 2]
        x_new_rel = xrel * cθ - yrel * sθ
        y_new_rel = xrel * sθ + yrel * cθ
        @inbounds chain[j, 1] = chain[I, 1] + x_new_rel
        @inbounds chain[j, 2] = chain[I, 2] + y_new_rel
    end

    # kernel to rotate the chain in 3D around the x-axis
    @kernel function kernel_rotate_3D_X(chain, I, @Const(range_to_rotate), cθ, sθ)
        k = @index(Global)
        @inbounds j = range_to_rotate[k]
        @inbounds yrel = chain[j, 2] - chain[I, 2]
        @inbounds zrel = chain[j, 3] - chain[I, 3]
        y_new_rel = yrel * cθ - zrel * sθ
        z_new_rel = yrel * sθ + zrel * cθ
        @inbounds chain[j, 2] = chain[I, 2] + y_new_rel
        @inbounds chain[j, 3] = chain[I, 3] + z_new_rel
    end


    # kernel to rotate the chain in 3D around the y-axis
    @kernel function kernel_rotate_3D_Y(chain, I, @Const(range_to_rotate), cθ, sθ)
        k = @index(Global)
        @inbounds j = range_to_rotate[k]
        @inbounds xrel = chain[j, 1] - chain[I, 1]
        @inbounds zrel = chain[j, 3] - chain[I, 3]
        x_new_rel = xrel * cθ + zrel * sθ
        z_new_rel = -xrel * sθ + zrel * cθ
        @inbounds chain[j, 1] = chain[I, 1] + x_new_rel
        @inbounds chain[j, 3] = chain[I, 3] + z_new_rel
    end

    # kernel to rotate the chain in 3D around the z-axis
    @kernel function kernel_rotate_3D_Z(chain, I, @Const(range_to_rotate), cθ, sθ)
        k = @index(Global)
        @inbounds j = range_to_rotate[k]
        @inbounds xrel = chain[j, 1] - chain[I, 1]
        @inbounds yrel = chain[j, 2] - chain[I, 2]
        x_new_rel = xrel * cθ - yrel * sθ
        y_new_rel = xrel * sθ + yrel * cθ
        @inbounds chain[j, 1] = chain[I, 1] + x_new_rel
        @inbounds chain[j, 2] = chain[I, 2] + y_new_rel
    end



    function end_to_end_length2(chain) # works in 2D and 3D
        # move to CPU (with Array) and then calculate the end-to-end length
        return sum((Array(chain[1, :]) - Array(chain[end, :])) .^ 2)
    end



    function make_mc_step2D!(chain, index_map, backend)

        N = size(chain, 1)

        # choose a random point in the chain
        I = rand(1:N)

        # choose if to rotate to the right or left (not sure if this is necessary)
        # either rotate all points before I or after I
        range_to_rotate = 1:0
        while length(range_to_rotate) == 0 # to ensure we have a non-empty range
            range_to_rotate = rand(Bool) ? (1:(I-1)) : ((I+1):N)
        end

        # choose a random angle
        # use cospi, sinpi for better numerical stability 
        θ = 2 * rand(Float32) # same as:  θ = 2π * rand()
        cθ = cospi(θ)         #           cθ = cos(θ)
        sθ = sinpi(θ)         #           sθ = sin(θ)

        kernel_rotate = kernel_rotate_2D(backend)
        kernel_rotate(chain, I, range_to_rotate, cθ, sθ, ndrange=length(range_to_rotate))


        kernel_check = kernel_is_valid_chain2D(backend)

        bond_diameter_2 = 1.0^2 # square of the bond diameter
        valid_flag = adapt(backend, [true]) # mutable array to store the validity flag
        kernel_check(chain, index_map, bond_diameter_2, valid_flag, ndrange=length(index_map))

        return Array(valid_flag)[1] # return the validity flag as a boolean
    end


    function make_mc_step3D!(chain3D, index_map, backend)
        N = size(chain3D, 1)

        # choose a random point in the chain
        I = rand(1:N)

        # choose if to rotate to the right or left (not sure if this is necessary)
        # either rotate all points before I or after I
        range_to_rotate = 1:0
        while length(range_to_rotate) == 0 # to ensure we have a non-empty range
            range_to_rotate = rand(Bool) ? (1:(I-1)) : ((I+1):N)
        end



        # choose axis to rotate around (x, y, z)
        axis = rand(1:3)

        # choose a random angle π/2 or 3π/2
        θ = rand([0.0f0, 0.5f0, 1.5f0]) # in units of π
        cθ = cospi(θ)
        sθ = sinpi(θ)

        # generate rotation matrix
        if axis == 1 # x-axis
            kernel_rotate = kernel_rotate_3D_X(backend)
        elseif axis == 2 # y-axis
            kernel_rotate = kernel_rotate_3D_Y(backend)
        else # z-axis
            kernel_rotate = kernel_rotate_3D_Z(backend)
        end

        kernel_rotate(chain3D, I, range_to_rotate, cθ, sθ, ndrange=length(range_to_rotate))

        kernel_check = kernel_is_valid_chain3D(backend)

        bond_diameter_2 = 1.0^2 # square of the bond diameter
        valid_flag = adapt(backend, [true]) # mutable array to store the validity flag
        kernel_check(chain3D, index_map, bond_diameter_2, valid_flag, ndrange=length(index_map))


        return Array(valid_flag)[1] # return the validity flag as a boolean
    end

    function make_mc_step!(chain2d_or3D, index_map, backend)
        n = size(chain2d_or3D, 2)
        if n == 2
            return make_mc_step2D!(chain2d_or3D, index_map, backend)
        elseif n == 3
            return make_mc_step3D!(chain2d_or3D, index_map, backend)
        else
            throw(ArgumentError("Invalid chain dimension"))
        end
    end


    function simulation(backend, N_MC, N; dim3_bool::Bool, random_walk::Bool=false, n_samples=100)
        if dynamic && N <= 100
            backend = CPU() # change backend to CPU for small N
        end

        println("\n Running simulation with N: $N, N_MC: $N_MC, dim3_bool: $dim3_bool, random_walk: $random_walk and backend: $backend \n")

        # initialize

        chain = create_initial_chain(N, dim3_bool, backend)
        try_chain = similar(chain) # cache it already to reuse later

        index_map = build_index_map(N, backend)

        step = 0
        # warm up 
        while step < 10_000 # to have a warm up, do 1000 successful steps
            copyto!(try_chain, chain) # try_chain = chain
            is_valid = make_mc_step!(try_chain, index_map, backend)

            # if random_walk is true, we always accept the new chain
            if is_valid || random_walk # is a valid chain or we are in random walk mode
                copyto!(chain, try_chain) # chain = try_chain
                step += 1
            end
        end

        chain_lengths = Float64[]
        counter_total = 0
        step = 0
        while step < N_MC # do N_MC successful steps
            counter_total += 1
            copyto!(try_chain, chain) # try_chain = chain
            is_valid = make_mc_step!(try_chain, index_map, backend)

            chain_updated = is_valid || random_walk

            # if random_walk is true, we always accept the new chain
            if chain_updated # is a valid chain or we are in random walk mode
                copyto!(chain, try_chain) # chain = try_chain
                step += 1
            end

            # avoid resemble if step go rejected (and step % n_samples is still 0)
            if chain_updated && step % n_samples == 0
                push!(chain_lengths, end_to_end_length2(chain))
            end
        end

        return (chain_lengths, N_MC / counter_total) # return the chain lengths and the acceptance ratio
    end


    # for fitting the scaling law

    fit_scaling1(N, params) = @. N^(2 * params[1])
    fit_scaling2(N, params) = @. params[2] * N^(2 * params[1])

    function both_fits(N_values, chain_lengths, chain_lengths_std)
        weights = 1 ./ chain_lengths_std .^ 2

        fit1 = curve_fit(fit_scaling1, N_values, chain_lengths, weights, [1.])
        fit2 = curve_fit(fit_scaling2, N_values, chain_lengths, weights, [1., 1.])
        coef(fit1)
        coeficients1 = Measurements.correlated_values(coef(fit1), estimate_covar(fit1))
        ν1 = coeficients1[1]

        coef(fit2)
        coeficients2 = Measurements.correlated_values(coef(fit2), estimate_covar(fit2))
        A2 = coeficients2[1]
        ν2 = coeficients2[2]

        return (coeficients1, coeficients2)
    end
end


###################################################
# calculate the autocorrelation
###################################################

N_MC_acc_steps = 50_000
N = 200 # chain length
dim3_bool = false
random_walk = false

chain = create_initial_chain(N, dim3_bool, backend)
try_chain = similar(chain) # cache it already to reuse later
index_map = build_index_map(N, backend)



# do a MC Simulation and save the chain length after every successful step
chain_lengths = Float64[]
step = 0
total_steps = 0
warmup_steps = 10_000
while step < N_MC_acc_steps + warmup_steps
    copyto!(try_chain, chain) # try_chain = chain
    is_valid = make_mc_step!(try_chain, index_map, backend)
    step += is_valid
    total_steps += 1

    # if random_walk is true, we always accept the new chain
    if is_valid || random_walk # is a valid chain or we are in random walk mode
        copyto!(chain, try_chain) # chain = try_chain
        if step > warmup_steps # to have a warm up
            push!(chain_lengths, end_to_end_length2(chain))
        end
    end

end

println("Accepted Steps: $step, Total Steps: $total_steps")

# Autocorrelation calculation
N = length(chain_lengths)
mean_val = sum(Float64, chain_lengths) / N           # <x>
mean_squared = mean_val^2                            # <x>²  
squared_mean = sum(Float64, chain_lengths .^ 2) / N  # <x²>

τ_max = 50
ρ = zeros(Float64, τ_max + 1) # normalized autocorrelation ρ(τ)
# Calculate normalized autocorrelation ρ(τ) for τ = 0,1,2,...,N
for τ in 0:τ_max
    # Calculate cross-product: <x(t+τ) * x(t)>
    cross_product_sum = 0.0
    valid_pairs = N - τ
    for t in 1:valid_pairs
        cross_product_sum += chain_lengths[t+τ] * chain_lengths[t]
    end
    cross_product_mean = cross_product_sum / (N - τ)

    ρ_τ = (cross_product_mean - mean_squared) / (squared_mean - mean_squared)
    ρ[τ+1] = ρ_τ
end

iact = 0.5 + sum(ρ[2:end])
p_autocorr = plot(0:τ_max, ρ, title="Autocorrelation Plot with τ = $(round(iact, sigdigits=3))", xlabel="t", ylabel=L"\rho(t)", label=L"\rho(t)", legend=:topright, marker=:circle, line=:solid,)

savefig(p_autocorr, joinpath(plots_folder, "N_$(N_MC_acc_steps)_autocorrelation_plot.pdf"))



###################################################
# production runs of the mc pivot algorithm
###################################################

# define chain lengths
N_values = [10, 15, 20, 30, 40, 60, 80, 100, 150, 200]#, 300, 400, 500, 700, 1000]
N_MC = 50_000

# define xticks for the plots
xticks = (N_values[1:2:end], string.(N_values[1:2:end]))




# 1. 2D Self-Avoiding Walk
chain_lengths_2d_saw = zeros(Float64, length(N_values))
chain_lengths_std_2d_saw = zero(chain_lengths_2d_saw)

for (i, N) in enumerate(N_values)
    (chain_length, counter_accepted) = simulation(backend, N_MC, N; dim3_bool=false, random_walk=false)
    println("2D SAW - N: $N, Acceptance rate: $(counter_accepted)")
    chain_lengths_2d_saw[i] = mean(chain_length)
    chain_lengths_std_2d_saw[i] = std(chain_length) / sqrt(length(chain_length))
end

p1 = plot(N_values, chain_lengths_2d_saw,
    xscale=:log10, yscale=:log10,
    yerr=chain_lengths_std_2d_saw,
    xlabel="Chain Length", ylabel=L"\langle R^2_{ee} \rangle",
    title="2D Self-Avoiding Walk", size=(800, 600),
    label="2D SAW Data",
    xticks=xticks,
)

c1, c2 = both_fits(N_values, chain_lengths_2d_saw, chain_lengths_std_2d_saw)
ref_line1 = N_values .^ (2 * c1[1].val)
ref_line2 = c2[2].val .* N_values .^ (2 * c2[1].val)
# plot!(p1, N_values, ref_line1, label=latexstring("\$N^{2 \\cdot $(round(c1[1].val, digits=3)) \\pm $(round(c1[1].err, digits=4))}\$"), linestyle=:dash)
plot!(p1, N_values, ref_line2, label=latexstring("\$$(round(c2[2].val, digits=2)) N^{2 \\cdot $(round(c2[1].val, digits=3)) \\pm $(round(c2[1].err, digits=4))}\$"), linestyle=:dash)



# 2. 2D Random Walk
chain_lengths_2d_rw = zeros(Float64, length(N_values))
chain_lengths_std_2d_rw = zero(chain_lengths_2d_rw)

for (i, N) in enumerate(N_values)
    (chain_length, counter_accepted) = simulation(backend, N_MC, N; n_samples=10, dim3_bool=false, random_walk=true)
    println("2D RW - N: $N, Acceptance rate: $(counter_accepted)")
    chain_lengths_2d_rw[i] = mean(chain_length)
    chain_lengths_std_2d_rw[i] = std(chain_length) / sqrt(length(chain_length))
end

p2 = plot(N_values, chain_lengths_2d_rw, xscale=:log10, yscale=:log10,
    yerr=chain_lengths_std_2d_rw,
    xlabel="Chain Length", ylabel=L"\langle R^2_{ee} \rangle",
    title="2D Random Walk", size=(800, 600),
    label="2D RW Data",
    xticks=xticks,
)

c1, c2 = both_fits(N_values, chain_lengths_2d_rw, chain_lengths_std_2d_rw)
ref_line1 = N_values .^ (2 * c1[1].val)
ref_line2 = c2[2].val .* N_values .^ (2 * c2[1].val)
# plot!(p2, N_values, ref_line1, label=latexstring("\$N^{2 \\cdot $(round(c1[1].val, digits=3)) \\pm $(round(c1[1].err, digits=4))}\$"), linestyle=:dash)
plot!(p2, N_values, ref_line2, label=latexstring("\$$(round(c2[2].val, digits=2)) N^{2 \\cdot $(round(c2[1].val, digits=3)) \\pm $(round(c2[1].err, digits=3))}\$"), linestyle=:dash)



# 3. 3D Self-Avoiding Walk
chain_lengths_3d_saw = zeros(Float64, length(N_values))
chain_lengths_std_3d_saw = zero(chain_lengths_3d_saw)

for (i, N) in enumerate(N_values)
    (chain_length, counter_accepted) = simulation(backend, N_MC, N; dim3_bool=true, random_walk=false)
    println("3D SAW - N: $N, Acceptance rate: $(counter_accepted)")
    chain_lengths_3d_saw[i] = mean(chain_length)
    chain_lengths_std_3d_saw[i] = std(chain_length) / sqrt(length(chain_length))
end

p3 = plot(N_values, chain_lengths_3d_saw, xscale=:log10, yscale=:log10,
    yerr=chain_lengths_std_3d_saw,
    xlabel="Chain Length", ylabel=L"\langle R^2_{ee} \rangle",
    title="3D Self-Avoiding Walk", size=(800, 600),
    label="3D SAW Data",
    xticks=xticks,)

c1, c2 = both_fits(N_values, chain_lengths_3d_saw, chain_lengths_std_3d_saw)
ref_line1 = N_values .^ (2 * c1[1].val)
ref_line2 = c2[2].val .* N_values .^ (2 * c2[1].val)
# plot!(p3, N_values, ref_line1, label=latexstring("\$N^{2 \\cdot $(round(c1[1].val, digits=3)) \\pm $(round(c1[1].err, digits=4))}\$"), linestyle=:dash)
plot!(p3, N_values, ref_line2, label=latexstring("\$$(round(c2[2].val, digits=2)) N^{2 \\cdot $(round(c2[1].val, digits=3)) \\pm $(round(c2[1].err, digits=3))}\$"), linestyle=:dash)



# 4. 3D Random Walk
chain_lengths_3d_rw = zeros(Float64, length(N_values))
chain_lengths_std_3d_rw = zero(chain_lengths_3d_rw)

for (i, N) in enumerate(N_values)
    (chain_length, counter_accepted) = simulation(backend, N_MC, N; dim3_bool=true, random_walk=true)
    println("3D RW - N: $N, Acceptance rate: $(counter_accepted)")
    chain_lengths_3d_rw[i] = mean(chain_length)
    chain_lengths_std_3d_rw[i] = std(chain_length) / sqrt(length(chain_length))
end

p4 = plot(N_values, chain_lengths_3d_rw, xscale=:log10, yscale=:log10,
    yerr=chain_lengths_std_3d_rw,
    xlabel="Chain Length", ylabel=L"\langle R^2_{ee} \rangle",
    title="3D Random Walk", size=(800, 600),
    label="3D RW Data",
    xticks=xticks,)

c1, c2 = both_fits(N_values, chain_lengths_3d_rw, chain_lengths_std_3d_rw)
ref_line1 = N_values .^ (2 * c1[1].val)
ref_line2 = c2[2].val .* N_values .^ (2 * c2[1].val)
# plot!(p4, N_values, ref_line1, label=latexstring("\$N^{2 \\cdot $(round(c1[1].val, digits=3)) \\pm $(round(c1[1].err, digits=4))}\$"), linestyle=:dash)
plot!(p4, N_values, ref_line2, label=latexstring("\$$(round(c2[2].val, digits=2)) N^{2 \\cdot $(round(c2[1].val, digits=3)) \\pm $(round(c2[1].err, digits=3))}\$"), linestyle=:dash)



# Save plots
ending = ".pdf"
savefig(p1, joinpath(plots_folder, "N_$(N_MC)_2d_saw_plot$ending"))
savefig(p2, joinpath(plots_folder, "N_$(N_MC)_2d_rw_plot$ending"))
savefig(p3, joinpath(plots_folder, "N_$(N_MC)_3d_saw_plot$ending"))
savefig(p4, joinpath(plots_folder, "N_$(N_MC)_3d_rw_plot$ending"))
