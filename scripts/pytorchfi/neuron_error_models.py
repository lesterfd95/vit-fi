"""pytorchfi.error_models provides different error models out-of-the-box for use."""

import logging
import random

import torch
import numpy as np

from pytorchfi import core
from pytorchfi.util import random_value
from torch.utils.cpp_extension import load

clip_stats = {"nan_inf": 0, "too_big": 0}
correct_nan_train = False
correct_nan_test = False 
correct_big_train = False
correct_big_test = False
umbral_por_tipo = {'Linear': 90.8779998779297, 'Dropout': 90.8779998779297, 'LayerNorm': 19.010177421569825, 'Softmax': 1.00, 'GELU': 20.302083492279053}

# Load Extension

# Compilar el kernel CUDA
# Asegúrate de que el archivo bit_manipulation_cuda.cu esté en el mismo directorio
# o proporciona la ruta completa.
print("Cargando kernel CUDA...")
bit_manip_extension = load(
    name="bit_manip_extension",
    sources=["bit_manipulation_cuda.cu"],
    verbose=True
)
print("Kernel CUDA cargado.")


# Helper Functions
def set_correction(nan_train: bool, nan_test: bool, big_train: bool, big_test: bool):
    global correct_nan_train 
    global correct_nan_test 
    global correct_big_train
    global correct_big_test

    correct_nan_train = nan_train
    correct_nan_test = nan_test
    correct_big_train = big_train
    correct_big_test = big_test

    return {"correct_nan_train": correct_nan_train, "correct_nan_test": correct_nan_test, "correct_big_train": correct_big_train, "correct_big_test": correct_big_test}



def random_batch_element(pfi: core.FaultInjection):
    return random.randint(0, pfi.batch_size - 1)


def random_neuron_location(pfi: core.FaultInjection, layer: int = -1):
    if layer == -1:
        layer = random.randint(0, pfi.get_total_layers() - 1)

    dim = pfi.get_layer_dim(layer)
    shape = pfi.get_layer_shape(layer)

    dim1_shape = shape[1]
    dim1_rand = random.randint(0, dim1_shape - 1)
    if dim > 2:
        dim2_shape = shape[2]
        dim2_rand = random.randint(0, dim2_shape - 1)
    else:
        dim2_rand = None
    if dim > 3:
        dim3_shape = shape[3]
        dim3_rand = random.randint(0, dim3_shape - 1)
    else:
        dim3_rand = None

    return (layer, dim1_rand, dim2_rand, dim3_rand)


# Neuron Perturbation Models

# single random neuron error in single batch element
def random_neuron_inj(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    b = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi.declare_neuron_fault_injection(
        batch=[b], layer_num=[layer], dim1=[C], dim2=[H], dim3=[W], value=[err_val]
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True,
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for _ in range(6))

    if not rand_loc:
        (layer, C, H, W) = random_neuron_location(pfi)
    if not rand_val:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi.batch_size):
        if rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi)
        if rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi: core.FaultInjection, min_val: int = -1, max_val: int = 1):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi)
    for i in range(pfi.get_total_layers()):
        (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        batch.append(b)
        layer_num.append(layer)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi: core.FaultInjection,
    min_val: int = -1,
    max_val: int = 1,
    rand_loc: bool = True,
    rand_val: bool = True,
):
    batch, layer_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi.get_total_layers()):
        if not rand_loc:
            (layer, C, H, W) = random_neuron_location(pfi, layer=i)
        if not rand_val:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi.batch_size):
            if rand_loc:
                (layer, C, H, W) = random_neuron_location(pfi, layer=i)
            if rand_val:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            layer_num.append(layer)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi.declare_neuron_fault_injection(
        batch=batch,
        layer_num=layer_num,
        dim1=c_rand,
        dim2=h_rand,
        dim3=w_rand,
        value=value,
    )


class single_bit_flip_func(core.FaultInjection):
    def __init__(self, model, batch_size, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)
    
        #logging.basicConfig(
        #    level=logging.INFO,
        #    format="%(asctime)s %(levelname)s %(message)s"
        #)

        self.inyecciones = 0

        self.bits = kwargs.get("bits", 8)
        self.layer_ranges = []

    def set_conv_max(self, data):
        self.layer_ranges = data

    def reset_conv_max(self, data):
        self.layer_ranges = []

    def get_conv_max(self, layer):
        return self.layer_ranges[layer]

    @staticmethod
    def _twos_comp(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val

    def _twos_comp_shifted(self, val, nbits):
        return (1 << nbits) + val if val < 0 else self._twos_comp(val, nbits)
    
    def _flip_bit_np(self, value, bit_pos):

        #print(f"valor original {value}")

        if bit_pos < 0 or bit_pos > 31:
            raise ValueError("bit_pos debe estar entre 0 y 31.")
        if value.dtype != torch.float32:
            raise ValueError("El tensor debe ser torch.float32.")
    
        # Convertir a numpy (si está en GPU, se copia a CPU)
        numpy_array = value.cpu().numpy() if value.is_cuda else value.numpy()

        # Convertir el float32 a uint32 (sin copiar datos, misma memoria)
        uint32_val = np.array([numpy_array.item()], dtype=np.float32).view(np.uint32)[0]

        #print(f"valor entero {uint32_val}")
        #print(f"valor binario {bin(uint32_val)}")
        #print(f"bit a cambiar {bit_pos}")

        # Aplicar XOR para invertir el bit deseado (operación bitwise)
        uint32_val ^= (1 << bit_pos)

        #print(f"nuevo valor binario {bin(uint32_val)}")
        #print(f"nuevo valor entero {uint32_val}")

        # Convertir de vuelta a float32
        new_float = np.array([uint32_val], dtype=np.uint32).view(np.float32)[0]

        #print(f"nuevo valor flotante {new_float}")

        # Devolver como tensor (en el mismo dispositivo original)
        return torch.tensor(new_float, dtype=torch.float32, device=value.device)
      
    def _flip_bit_cuda(self, value, bit_pos):
        # Asegúrate de que CUDA esté disponible
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. Este código requiere una GPU.")

        device = torch.device("cuda")
        
        # Reinterpretar float32 como uint32 y hacer el bit-flip, todo en GPU
        flipped_uint32_tensor = bit_manip_extension.bit_flip_float32_to_uint32(value, bit_pos)
        #print(f"\nTensor uint32 después de reinterpretar y bit-flip (en GPU):\n{flipped_uint32_tensor}")
        #print(f"Tipo de dato: {flipped_uint32_tensor.dtype}, Dispositivo: {flipped_uint32_tensor.device}")

        # Reinterpretar el uint32 flippeado de nuevo a float32
        final_float_tensor = bit_manip_extension.uint32_to_float32(flipped_uint32_tensor)
        #print(f"\nTensor float32 final después del bit-flip y reinterpretación inversa (en GPU):\n{final_float_tensor}")
        #print(f"Tipo de dato: {final_float_tensor.dtype}, Dispositivo: {final_float_tensor.device}")

        # Devolver como tensor (en el mismo dispositivo original)
        return final_float_tensor

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info(f"Original Value: {orig_value}")
        print(f"Original Value: {orig_value}")
        
       
        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        if(abs(orig_value) > abs(max_value)):
            print(f"Original value {orig_value} grater than maximum value {max_value}")
        if(abs(quantum) > abs(2.0 ** (total_bits - 1))):
            print(f"Quantified value {quantum} grater data type max value {(2.0 ** (total_bits - 1))}")
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info(f"Quantum: {quantum}")
        print(f"Quantum: {quantum}")
        logging.info(f"Twos Couple: {twos_comple}")
        print(f"Twos Couple: {twos_comple}")

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info(f"Bits: {bits}")
        print(f"Bits: {bits}")

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        if len(bits) != total_bits:
            raise AssertionError
        logging.info(f"Sign extend bits {bits}")
        print(f"Sign extend bits {bits}")

        # flip a bit
        # use MSB -> LSB indexing
        if bit_pos >= total_bits:
            raise AssertionError

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info(f"New bits: {bits_str_new}")
        print((f"New bits: {bits_str_new}"))

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")
            print(f"Error: Not all the bits are digits (0/1). New bits: {bits_str_new}")
            print(f"Original Value: {orig_value}")
            print(f"Quantum: {quantum}")
            print(f"Twos Couple: {twos_comple}")
            print(f"Bits: {bits}")
            print(f"Sign extend bits {bits}")
            print((f"New bits: {bits_str_new}"))

        # convert to quantum
        if not bits_str_new.isdigit():
            raise AssertionError
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info(f"Out: {out}")
        print((f"Out: {out}"))

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info(f"New Value: {new_value}")
        print(f"New Value: {new_value}")

        return torch.tensor(new_value, dtype=save_type)

    def single_bit_flip_signed_across_batch(self, module, input_val, output):
        global clip_stats
        global correct_nan_train 
        global correct_nan_test 
        global correct_big_train
        global correct_big_test
        corrupt_conv_set = self.corrupt_layer
        range_max = self.get_conv_max(self.current_layer)
        logging.info(f"Current layer: {self.current_layer}")
        logging.info(f"Range_max: {range_max}")

        tipo_capa = type(module).__name__  # 'Linear', 'GELU', 'Softmax', etc.
        threshold = umbral_por_tipo.get(tipo_capa)
     
        # Solo clonamos la salida si es Softmax para evitar errores inplace
        if isinstance(module, torch.nn.Softmax):
            # operar sobre clon de la salida para evitar modificaciones in-place de salidas de softmax
            # ya que en esos casos se altera el grafo de gradientes y da error
            #print("Es softmax")
            output_clone = output.clone()
            target_tensor = output_clone
        else:
        # Para otras capas, modificamos output directamente
            #print("NO es softmax")
            target_tensor = output
        
        tipo_capa = type(module).__name__  # Ej: "Linear", "Dropout", etc.

        
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                prev_value = target_tensor[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")

                new_value = self._flip_bit_cuda(prev_value.detach(), rand_bit)

                #print(f'prev_value:{prev_value}')
                #print(f"value position: element [{self.corrupt_batch[i]}] position [{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]")
                #print(f'new_value:{new_value}')

                if correct_nan_train:
                    if torch.isnan(new_value) or torch.isinf(new_value): # si el flip genera nan o inf, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["nan_inf"] += 1

                if correct_big_train:
                    if torch.abs(new_value) > 1e4: # si el flip genera valor muy grande, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["too_big"] += 1

                if correct_nan_test:
                    if torch.isnan(new_value) or torch.isinf(new_value): # si el flip genera nan o inf, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["nan_inf"] += 1
                        
                if correct_big_test:
                    if torch.abs(new_value) > threshold: # si el flip genera valor muy grande, hacer 0
                        new_value = torch.zeros_like(new_value)
                        #new_value = torch.full_like(new_value, fill_value=threshold)
                        clip_stats["too_big"] += 1
                
                target_tensor[self.corrupt_batch[i]][self.corrupt_dim[0][i]][ 
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = new_value 

                #output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                #    self.corrupt_dim[1][i]
                #][self.corrupt_dim[2][i]] = new_value             

                #print(f'output:{output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}')
        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = target_tensor[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value = self._flip_bit_cuda(prev_value.detach(), rand_bit)

                #print(f'prev_value:{prev_value}')
                #print(f"value position: element [{self.corrupt_batch[i]}] position [{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]")
                #print(f'new_value:{new_value}')

                if correct_nan_train:
                    if torch.isnan(new_value) or torch.isinf(new_value): # si el flip genera nan o inf, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["nan_inf"] += 1

                if correct_big_train:
                    if torch.abs(new_value) > 1e4: # si el flip genera valor muy grande, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["too_big"] += 1

                if correct_nan_test:
                    if torch.isnan(new_value) or torch.isinf(new_value): # si el flip genera nan o inf, hacer 0
                        new_value = torch.zeros_like(new_value)
                        clip_stats["nan_inf"] += 1
                        
                if correct_big_test:
                    if torch.abs(new_value) > threshold: # si el flip genera valor muy grande, hacer 0
                        new_value = torch.zeros_like(new_value)
                        #new_value = torch.full_like(new_value, fill_value=threshold)
                        clip_stats["too_big"] += 1
                
                target_tensor[self.corrupt_batch[i]][self.corrupt_dim[0][i]][ 
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = new_value 

                target_tensor[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][ 
                    self.corrupt_dim[2]
                ] = new_value

                #output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                #    self.corrupt_dim[2]
                #] = new_value

                #print(f'output:{output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}')

        
        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

        # en los casos softmax se devuelve un valor que sustituye a la salida de la capa como entrada de la proxima 
        # la salida original no modificada sigue existiendo sin influenciar el f.prop y disponible para el grafo de b.prop
        if isinstance(module, torch.nn.Softmax):
            return output_clone 


def random_neuron_single_bit_inj_batched(
    pfi: core.FaultInjection, layer_ranges, batch_random=True
):
    """
    Args:
        pfi: The core.FaultInjection in which the neuron fault injection should be instantiated.
        layer_ranges:
        batch_random (default True): True if each batch should have a random location, if false, then each
                                     batch will use the same randomly generated location.
    """
    pfi.set_conv_max(layer_ranges)

    locations = (
        [random_neuron_location(pfi) for _ in range(pfi.batch_size)]
        if batch_random
        else [random_neuron_location(pfi)] * pfi.batch_size
    )
    # Convert list of tuples [(1, 3), (2, 4)] to list of list [[1, 2], [3, 4]]
    random_layers, random_c, random_h, random_w = map(list, zip(*locations))

    return pfi.declare_neuron_fault_injection(
        batch=range(pfi.batch_size),
        layer_num=random_layers,
        dim1=random_c,
        dim2=random_h,
        dim3=random_w,
        function=pfi.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi: core.FaultInjection, layer_ranges):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (layer, C, H, W) = random_neuron_location(pfi)

    return pfi.declare_neuron_fault_injection(
        batch=[batch],
        layer_num=[layer],
        dim1=[C],
        dim2=[H],
        dim3=[W],
        function=pfi.single_bit_flip_signed_across_batch,
    )


def random_neuron_multiple_bit_inj(pfi: core.FaultInjection, layer_ranges, n_inj):
     
    pfi.set_conv_max(layer_ranges)

    locations = [random_neuron_location(pfi) for _ in range(n_inj)]

    # Convert list of tuples [(1, 3), (2, 4)] to list of list [[1, 2], [3, 4]]
    random_layers, random_c, random_h, random_w = map(list, zip(*locations))

    random_batch = [random_batch_element(pfi) for _ in range(n_inj)]

    return pfi.declare_neuron_fault_injection(
        batch=random_batch,
        layer_num=random_layers,
        dim1=random_c,
        dim2=random_h,
        dim3=random_w,
        function=pfi.single_bit_flip_signed_across_batch,
    )