#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint> // Para uint32_t

#define CUDA_CHECK(call)                                 \
do {                                                    \
    cudaError_t err = (call);                           \
    if (err != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err,                \
                cudaGetErrorString(err), #call);       \
        exit(1);                                       \
    }                                                   \
} while (0)


// CUDA kernel para reinterpretar float32 a uint32 y hacer bit-flip
__global__ void bit_flip_kernel(const float* input_ptr, uint32_t* output_ptr, int num_elements, int random_bit_to_flip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Reinterpretar float32 a uint32_t.
        // Esto es una reinterpretación de bits, no un casting numérico.
        uint32_t float_bits;
        memcpy(&float_bits, &input_ptr[idx], sizeof(float));

        // Realizar el bit-flip del bit elegido aleatoriamente
        float_bits ^= (1U << random_bit_to_flip); // 1U asegura que es un unsigned int literal

        // Guardar el resultado en el tensor de salida (uint32_t)
        output_ptr[idx] = float_bits;
    }
}

// Función de envoltura para llamar al kernel desde Python
torch::Tensor bit_flip_float32_to_uint32_gpu(torch::Tensor input_tensor, int random_bit_to_flip) {
    TORCH_CHECK(input_tensor.is_cuda(), "input_tensor must be a CUDA tensor");
    TORCH_CHECK(input_tensor.dtype() == torch::kFloat32, "input_tensor must be of dtype float32");
    TORCH_CHECK(input_tensor.numel() > 0, "input_tensor must not be empty");
    TORCH_CHECK(random_bit_to_flip >= 0 && random_bit_to_flip < 32, "random_bit_to_flip must be between 0 and 31");

    int num_elements = input_tensor.numel();
    auto output_tensor = torch::empty_like(input_tensor, input_tensor.options().dtype(torch::kUInt32));

    // Obtener punteros a los datos de los tensores
    const float* input_ptr = input_tensor.data_ptr<float>();
    uint32_t* output_ptr = output_tensor.data_ptr<uint32_t>();

    // Configurar la ejecución del kernel
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Llamar al kernel CUDA
    bit_flip_kernel<<<num_blocks, threads_per_block>>>(input_ptr, output_ptr, num_elements, random_bit_to_flip);
    CUDA_CHECK(cudaGetLastError()); // Comprobar errores de ejecución

    return output_tensor;
}

// Función de envoltura para reinterpretar uint32 a float32
__global__ void uint32_to_float32_kernel(const uint32_t* input_ptr, float* output_ptr, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float float_val;
        memcpy(&float_val, &input_ptr[idx], sizeof(uint32_t)); // uint32_t tiene el mismo tamaño que float
        output_ptr[idx] = float_val;
    }
}

torch::Tensor uint32_to_float32_gpu(torch::Tensor input_tensor) {
    TORCH_CHECK(input_tensor.is_cuda(), "input_tensor must be a CUDA tensor");
    TORCH_CHECK(input_tensor.dtype() == torch::kUInt32, "input_tensor must be of dtype uint32");
    TORCH_CHECK(input_tensor.numel() > 0, "input_tensor must not be empty");

    int num_elements = input_tensor.numel();
    auto output_tensor = torch::empty_like(input_tensor, input_tensor.options().dtype(torch::kFloat32));

    const uint32_t* input_ptr = input_tensor.data_ptr<uint32_t>();
    float* output_ptr = output_tensor.data_ptr<float>();

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    uint32_to_float32_kernel<<<num_blocks, threads_per_block>>>(input_ptr, output_ptr, num_elements);
    CUDA_CHECK(cudaGetLastError());

    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bit_flip_float32_to_uint32", &bit_flip_float32_to_uint32_gpu, "Reinterprets float32 to uint32 and performs a bit-flip (CUDA)");
    m.def("uint32_to_float32", &uint32_to_float32_gpu, "Reinterprets uint32 to float32 (CUDA)");
}