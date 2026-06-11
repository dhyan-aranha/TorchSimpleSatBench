
#include <cuda_runtime.h>
#include <torch/extension.h> 

__global__ void update_ref_kernel(const int* subs, int* out_roots, int num_nodes) {
    int batch_id = blockIdx.x;
    int curr_idx = threadIdx.x;
    int start_node = curr_idx;

    int next_idx = subs[(batch_id * num_nodes) + curr_idx];

    while (curr_idx != next_idx) {
        curr_idx = next_idx;
        next_idx = subs[(batch_id * num_nodes) + curr_idx];
    }

    int out_flat_index = (batch_id * num_nodes) + start_node;
    out_roots[out_flat_index] = curr_idx;
}

__global__ void unify_kernel(
    const int* left_roots,
    const int* right_roots,
    int* subs,
    const bool* is_var_mask,
    const int* nodes,       
    const int* children,
    bool* success_mask,
    int* next_frontier_left,
    int* next_frontier_right,
    int* next_frontier_size,
    int num_nodes,
    int max_arity
){
    int batch_id = blockIdx.x;
    int pair_idx = threadIdx.x; 

    int left_node = left_roots[(batch_id * num_nodes) + pair_idx];
    int right_node = right_roots[(batch_id * num_nodes) + pair_idx];

    if (left_node == right_node || left_node == -1 || right_node == -1){
        return;
    }

    bool l_is_var = is_var_mask[left_node];
    bool r_is_var = is_var_mask[right_node];

    if (l_is_var){
        int flat_var_idx = (batch_id * num_nodes) + left_node;
        int returned_value = atomicCAS(&subs[flat_var_idx], left_node, right_node);

        if (returned_value != left_node){
            int write_idx = atomicAdd(next_frontier_size, 1);
            next_frontier_left[write_idx] = returned_value;
            next_frontier_right[write_idx] = right_node;
        }
    }
    else if (r_is_var){
        int flat_var_idx = (batch_id * num_nodes) + right_node;
        int returned_value = atomicCAS(&subs[flat_var_idx], right_node, left_node);

        if (returned_value != right_node){
            int write_idx = atomicAdd(next_frontier_size, 1);
            next_frontier_left[write_idx] = returned_value;
            next_frontier_right[write_idx] = left_node;
        }
    }
    else {
        int l_sym = nodes[left_node];
        int r_sym = nodes[right_node];

        if (l_sym != r_sym){
            success_mask[batch_id] = false;
        } else {
            
            for (int i = 0; i < max_arity; i++){ 
                int l_child = children[(left_node * max_arity) + i];
                int r_child = children[(right_node * max_arity) + i];

                if (l_child != -1 && r_child != -1){
                    int write_idx = atomicAdd(next_frontier_size, 1);
                    next_frontier_left[write_idx] = l_child;
                    next_frontier_right[write_idx] = r_child;
                }
            }
        }
    }
}

torch::Tensor launch_update_ref(torch::Tensor subs, int num_nodes){
    int num_batches = subs.size(0);
    torch::Tensor out_roots = torch::empty_like(subs);
    dim3 blocks(num_batches);
    dim3 threads(num_nodes);

    update_ref_kernel<<<blocks, threads>>>(
        subs.data_ptr<int>(),
        out_roots.data_ptr<int>(),
        num_nodes
    );
    return out_roots;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> launch_unify(
    torch::Tensor left_roots, 
    torch::Tensor right_roots, 
    torch::Tensor batch_indices, 
    torch::Tensor subs, 
    torch::Tensor is_var_mask,
    torch::Tensor nodes,
    torch::Tensor children,
    int num_nodes,
    int max_arity) 
{
    int num_pairs = left_roots.size(0);
    auto success_mask = torch::ones({num_pairs}, torch::dtype(torch::kBool).device(left_roots.device()));

    int stack_capacity = num_pairs * max_arity;
    if (stack_capacity < num_pairs) stack_capacity = num_pairs; 
    
    auto next_frontier_left = torch::empty({stack_capacity}, left_roots.options());
    auto next_frontier_right = torch::empty({stack_capacity}, left_roots.options());
    auto next_batch_indices = torch::empty({stack_capacity}, left_roots.options());
    auto next_frontier_size = torch::zeros({1}, torch::dtype(torch::kInt32).device(left_roots.device()));

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads; 

    unify_kernel<<<blocks, threads>>>(
        left_roots.data_ptr<int>(),
        right_roots.data_ptr<int>(),
        subs.data_ptr<int>(),
        is_var_mask.data_ptr<bool>(),
        nodes.data_ptr<int>(),
        children.data_ptr<int>(),
        success_mask.data_ptr<bool>(),
        next_frontier_left.data_ptr<int>(),
        next_frontier_right.data_ptr<int>(),
        next_batch_indices.data_ptr<int>(),
        next_frontier_size.data_ptr<int>(),
        num_nodes,
        max_arity
    );

    int actual_size = next_frontier_size.item<int>();

    return std::make_tuple(
        success_mask, 
        next_frontier_left.slice(0, 0, actual_size),
        next_frontier_right.slice(0, 0, actual_size),
        next_batch_indices.slice(0, 0, actual_size)
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_update_ref", &launch_update_ref, "Parallel pointer chaser kernel wrapper");
    m.def("launch_unify", &launch_unify, "Parallel lock-free unification kernel wrapper");
}