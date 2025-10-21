# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import csv
import time
import random
import copy

from vit import ViT
from pytorchfi.neuron_error_models import single_bit_flip_func
from pytorchfi.neuron_error_models import random_neuron_multiple_bit_inj
from pytorchfi.neuron_error_models import clip_stats
from pytorchfi.neuron_error_models import set_correction

from randomaug import RandAugment

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--train', action='store_true', help='realizar entrenamiento')

    parser.add_argument('--name', type=str, help='nombre a usar en archivos de checkpoints y logs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--dp', action='store_true', help='use data parallel')
    parser.add_argument('--bs', default='512')
    parser.add_argument('--size', default="32")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
    parser.add_argument('--ber_t', type=float, default=1e-9, help='Tasa de error de bit (bit error rate) durante entrenamiento')
    parser.add_argument('--shape_t', type=int, nargs='+', default=[3,224,224], help='Dimensiones de entrada, ej: -s 3 224 224')
    parser.add_argument('--layers_t', type=str, nargs='+', default=['Linear'], help='Lista de capas a usar, ej: -l Linear LayerNorm GELU Dropout Softmax')

    parser.add_argument('--batchsize', '-b', type=int, default=500, help='Tamaño del batch')
    parser.add_argument('--shape', '-s',type=int, nargs='+', default=[3,224,224], help='Dimensiones de entrada, ej: -s 3 224 224')
    parser.add_argument('--ber', '-B', type=float, default=1e-9, help='Tasa de error de bit (bit error rate)')
    parser.add_argument('--layers', '-l',type=str, nargs='+', default=['Linear'], help='Lista de capas a usar, ej: -l Linear LayerNorm GELU Dropout Softmax')
    parser.add_argument('--seed', '-S', type=int, default=1, help='Semilla para todas las funciones aleatorias')
    parser.add_argument('--model', type=str, help='nombre del modelo a usar en las pruebas')
    
    args = parser.parse_args()
    return args

# obtener tamaño en bits del dato usado
def get_bits_from_dtype(tensor):
    dtype_bits_map = {
        torch.float32: 32,
        torch.float64: 64,
        torch.float16: 16,
        torch.int8: 8,
        torch.int16: 16,
        torch.int32: 32,
        torch.int64: 64,
        torch.uint8: 8,
    }
    return dtype_bits_map.get(tensor.dtype, None)

def main_train():
        
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    size = imsize

    # Set up normalization based on the dataset cifar10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    print("Cargando datasets")
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    # Set up class names based on the dataset cifar10  
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load model
    print('==> Building model..')
    net = ViT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead), #dimension interna en las cabezas (revisar que el valor de dim_head en la clase ViT es esto dividido entre heads)
        #depth = 6, 
        depth = 12, #numero de encoders
        #heads = 8,
        heads = 12, #numero de cabezas en cada mecanismo de atencion de cada encoder
        #mlp_dim = 512,
        mlp_dim = 768, #dimension interna en los MLP
        dropout = 0.1,
        emb_dropout = 0.1
    )

    net.to(device)

    # Obtener el numero de bits del tipo de dato con que se opera
    type_bits = get_bits_from_dtype(next(net.parameters()))
    print("Cantidad de bits en tipo de datos: "+str(type_bits))

    # Calcular cantidad de bits totales en capas inyectables
    
    # aqui se usa un valor variable basado en la cantidad de valores de salida de las capas elejidas
    # con esto la cantidad de bit flips para un mismo BER varia cuando varian los tipos/cantidad de capa inyectadas
    # sirve para ver como reaccionan distintas capas para un % determinado de error
    #total_bits = total_values*type_bits

    # aqui se usa un valor fijo basado en la cantidad de valores de salida de los 5 tipos de capa (toda la red)
    # con esto la cantidad de bit flips para un mismo BER es fija aunque cambien los tipos de capa inyectados
    # ayuda a facilitar la comparacion 
    total_bits = (bs/200) * 5535544400 * type_bits  
                            
    # Configurar inyeccion
    print("Configurando inyeccion")
    n_inyecciones = round(ber_t * total_bits)
    if n_inyecciones == 0:
        n_inyecciones = 1
        print("Numero de inyecciones muy bajo, fijado a 1")
    print(f"{n_inyecciones} bit flips individuales en cada batch (capa y posicion aleatorias)")
    pfi_model = single_bit_flip_func(net, 
                                bs,
                                input_shape=shape_t,
                                layer_types=layers_t,
                                use_cuda=True,
                                bits=type_bits
                                )

    #print(pfi_model.print_pytorchfi_layer_summary())
    #print(pfi_model.get_total_layers())

    layer_max_list = [0 for _ in range(pfi_model.get_total_layers())]
    if (ber_t != 0):
        corrupted_model = random_neuron_multiple_bit_inj(pfi_model, layer_ranges=layer_max_list, n_inj=n_inyecciones)#repetir esto en cada batch para generar un nuevo modelo con nuevas posiciones aleatorios
    else:
        corrupted_model = net
        print(f"BER_training = {ber_t}. Using model without injection hooks")
    clean_model = copy.deepcopy(net)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_path = './checkpoint/{}.t7'.format(filename)
        checkpoint = torch.load(checkpoint_path)
        corrupted_model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    # For Multi-GPU
    if dp:
        print("using data parallel")
        corrupted_model = torch.nn.DataParallel(corrupted_model) # make parallel
        cudnn.benchmark = True

    # Loss is CE
    criterion = nn.CrossEntropyLoss()
    if opt == "adam":
        optimizer = optim.Adam(corrupted_model.parameters(), lr=learning_rate)
    elif opt == "sgd":
        optimizer = optim.SGD(corrupted_model.parameters(), lr=learning_rate)  
        
    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    corrupted_model.to(device)

    ##### Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    def train(epoch, corrupted_model):
        print('\nEpoch: %d' % epoch)
        corrupted_model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch}", unit="batch")):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = corrupted_model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tqdm.write(f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
            
            if np.isnan(train_loss/(batch_idx+1)):
                print(outputs)
                print(loss)
                print(train_loss)

            # cambiar posiciones de bitflip
            if (ber_t != 0):
                corrupted_model = random_neuron_multiple_bit_inj(pfi_model, layer_ranges=layer_max_list, n_inj=n_inyecciones) 

        return train_loss/(batch_idx+1)


    ##### Validation
    def test(epoch, best_acc):
        clean_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc=f"Validation Epoch {epoch}", unit="batch")):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = clean_model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tqdm.write(f'Val Loss: {test_loss/(batch_idx+1):.3f} | Val Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        
        print(f"Clip stats: {clip_stats}")

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                "net": corrupted_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}.t7'.format(filename))
            best_acc = acc
        
        os.makedirs("log", exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
        print(content)
        log_file = f'log/log_{filename}.txt'
        with open(log_file, 'a') as appender:
            appender.write(content + "\n")
        return test_loss, acc, best_acc

    list_loss = []
    list_acc = []

    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        trainloss = train(epoch, corrupted_model)
        clean_model.load_state_dict(corrupted_model.state_dict()) # copiar parámetros entrenados SIN hooks
        clean_model.to(device)
        val_loss, acc, best_acc = test(epoch, best_acc)
        
        scheduler.step(epoch-1) # step cosine scheduling
        
        list_loss.append(val_loss)
        list_acc.append(acc)
        
        # Write out csv..
        csv_file = f'log/log_{filename}.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss) 
            writer.writerow(list_acc) 
        print(list_loss)


def main_test():

    class explorer:
        def __init__(self, model, device, layers, input_shape = [3,224,224]):
            self.model = model
            self.device = device
            self.layers = layers
            self.input_shape = input_shape
            self.layer_max_values = {layer_type.__name__: [] for layer_type in layers}
            self.type_counters = {layer_type.__name__: 0 for layer_type in layers}
            self.layer_order = []
            self.layer_max_list = []
            self.simple_dict_list = []
            self.total_values = 0
            self.handles = []
            self.golden_accuracy = 0
                
        # hook para ver valores maximos en salidas de capas
        def max_value_hook(self, name, simple_type, index):
            def hook(module, input, output):
                max_val = output.abs().max().item()
                # Si aún no existe valor máximo para esta capa, lo inicializamos con -inf
                if len(self.layer_max_values[simple_type]) <= index:
                    self.layer_max_values[simple_type].append(max_val)
                else:
                    # Actualizamos máximo si el nuevo valor es mayor
                    if max_val > self.layer_max_values[simple_type][index]:
                        self.layer_max_values[simple_type][index] = max_val
            return hook
        
        # añadir hook de valores maximos
        def add_max_value_hook(self):
            for name, module in self.model.named_modules():
                for layer_type in self.layers:
                    if isinstance(module, layer_type):
                        simple_type = layer_type.__name__
                        index = self.type_counters[simple_type]
                        self.type_counters[simple_type] += 1

                        self.layer_order.append((simple_type, index))
                        h = module.register_forward_hook(self.max_value_hook(name, simple_type, index))
                        self.handles.append(h)

        # hook para sacar numero de elementos del modelo
        def count_elements_hook(self, module, input, output):
            if isinstance(output, torch.Tensor):
                self.total_values += output.numel()

        # añadir hook de numero de elementos
        def add_count_elements_hook(self):           
            for name, module in self.model.named_modules():
                for layer_type in self.layers:
                    if isinstance(module, layer_type):
                        h = module.register_forward_hook(self.count_elements_hook)
                        self.handles.append(h)
        
        # dummy forward pass
        def dummy_forward_pass(self):
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(*self.input_shape).to(self.device)  
                self.model(dummy_input)

        # full forward pass
        def full_forward_pass(self, testloader):
            self.model.to(self.device)
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(testloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)    
                    _, predicted = torch.max(outputs, 1)  
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            self.golden_accuracy = 100 * correct / total     

        # quitar hooks
        def remove_hooks(self):
            for h in self.handles:
                h.remove()
        
        # hacer inferencia de prueba con todos los hooks y obtener resultados 
        def explore(self,testloader):
            self.add_count_elements_hook() 
            self.dummy_forward_pass()
            self.remove_hooks()
            self.add_max_value_hook()
            self.full_forward_pass(testloader)
            self.remove_hooks()
            self.dict_max_list = [(typ, self.layer_max_values[typ][idx]) for typ, idx in self.layer_order]
            self.layer_max_list = [val for _, val in self.dict_max_list]

        # hacer inferencia en un batch para obtener la cantidad de elementos  
        def count_elements(self):
            self.add_count_elements_hook() 
            self.dummy_forward_pass()
            self.remove_hooks()

    # Transformaciones para CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Media y std de CIFAR-10
    ])

    # Cargar conjunto de prueba
    print("Cargando dataset")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # Recrear modelo 
    print("Creando modelo")
    model = ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 10,
        dim = 768, #dimension interna en las cabezas (revisar que el valor de dim_head en la clase ViT es esto dividido entre heads)
        depth = 12, #numero de encoders
        heads = 12, #numero de cabezas en cada mecanismo de atencion de cada encoder
        mlp_dim = 768, #dimension interna en los MLP
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # Cargar pesos desde archivo
    print("Cargando pesos desde archivo")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    model.eval()

    # Representar arquitectura
    #print(model)
    #summary(model, input_size=(3, 224, 224), batch_size=500)
    #  Contar tipos de capa
    #from collections import Counter
    #layer_types = [type(module).__name__ for module in model.modules()]
    #counter = Counter(layer_types)
    #for layer_type, count in counter.items():
    #    print(f"{layer_type}: {count}")

    # explorar red para sacar maximos, numero de elementos y precision
    print("Explorando la red")
    exp = explorer(model, device, layers, [batch_size,*shape])     
    exp.count_elements()
    total_values = exp.total_values
    print("Total de valores en salida de capas: " + str(total_values))
    exp.full_forward_pass(testloader)
    golden_accuracy = exp.golden_accuracy
    #golden_accuracy = 86.48
    print(f"golden_accuracy: {golden_accuracy}")

    # Obtener el numero de bits del tipo de dato con que se opera
    type_bits = get_bits_from_dtype(next(model.parameters()))
    print("Cantidad de bits en tipo de datos: "+str(type_bits))

    # Calcular cantidad de bits totales en capas inyectables
    
    # aqui se usa un valor variable basado en la cantidad de valores de salida de las capas elejidas
    # con esto la cantidad de bit flips para un mismo BER varia cuando varian los tipos/cantidad de capa inyectadas
    # sirve para ver como reaccionan distintas capas para un % determinado de error
    #total_bits = total_values*type_bits
   
    # aqui se usa un valor fijo basado en la cantidad de valores de salida de los 5 tipos de capa (toda la red)
    # con esto la cantidad de bit flips para un mismo BER es fija aunque cambien los tipos de capa inyectados
    # ayuda a facilitar la comparacion 
    total_bits = (batch_size/200)*5535544400*type_bits  

    # Configurar inyeccion
    print("Configurando inyeccion")
    n_inyecciones = round(ber * total_bits)
    if n_inyecciones == 0:
        n_inyecciones = 1
        print("Numero de inyecciones muy bajo, fijado a 1")
    print(f"{n_inyecciones} bit flips individuales en cada batch (capa y posicion aleatorias)")
    pfi_model = single_bit_flip_func(model, 
                                batch_size,
                                input_shape=shape,
                                layer_types=layers,
                                use_cuda=True,
                                bits=type_bits
                                )

    #print(pfi_model.print_pytorchfi_layer_summary())
    #print(pfi_model.get_total_layers())

    layer_max_list = [0 for _ in range(pfi_model.get_total_layers())]
    corrupted_model = random_neuron_multiple_bit_inj(pfi_model, layer_ranges=layer_max_list, n_inj=n_inyecciones)

    corrupted_model.to(device)
    corrupted_model.eval()

    correct = 0
    total = 0

    print("Evaluando modelo con inyeccion")
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = corrupted_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Precisión modelo corrupto: {accuracy:.2f}%')

    dif = golden_accuracy - accuracy
    print(f'Diferencia en precisión: {dif:.2f}%')

    print(f"Clip stats: {clip_stats}")
    print("---------------------------------------------------\n")



if __name__ == '__main__':
    
    args = parse_args()
    
    layer_map = {
        "Linear": nn.Linear,
        "GELU": nn.GELU,
        "LayerNorm": nn.LayerNorm,
        "Dropout": nn.Dropout,
        "Softmax": nn.Softmax,
    }
    
    do_train = args.train
    filename = args.name
    
    if do_train:
        # Configuración entrenamiento
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = args.lr
        opt = args.opt
        aug = args.noaug
        use_amp = not args.noamp
        dp = args.dp
        bs = int(args.bs)
        imsize = int(args.size)
        n_epochs = args.n_epochs
        patch = args.patch
        dimhead = args.dimhead
        layers_t = [layer_map[name] for name in args.layers_t if name in layer_map]
        shape_t = args.shape_t
        ber_t = args.ber_t
        resume = args.resume

        print(f"TRAINING ARGUMENTS")
        print(f"device: {device}")
        print(f"learning_rate: {learning_rate}")
        print(f"optimizer: {opt}")
        print(f"disable use randomaug: {aug}")
        print(f"using mixed precision training: {use_amp}")
        print(f"use data parallel: {dp}")
        print(f"training batch size: {bs}")
        print(f"header internal dimension: {dimhead}")
        print(f"layers: {layers_t}")
        print(f"Shape: {shape_t}")
        print(f"training BER: {ber_t}")
        print(f"RESUME?: {resume}")
    
    else: 
        print(f"NO TRAINING")
        
    # Configuración prueba
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = args.model
    model_path = './checkpoint/{}.t7'.format(model) 
    batch_size = args.batchsize
    layers = [layer_map[name] for name in args.layers if name in layer_map]
    shape = args.shape
    ber = args.ber
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"TESTING ARGUMENTS")
    print(f"device: {device}")
    print(f"model_path: {model_path}")
    print(f"batch size: {batch_size}")
    print(f"layers: {layers}")
    print(f"Shape: {shape}")
    print(f"BER: {ber}")
    print(f"seed: {seed}")

    if do_train:
        train_correction = set_correction(True, False, True, False)
        print(train_correction)
        main_train()
    test_correction = set_correction(False, True, False ,True)
    #test_correction = set_correction(False, False, False , False)
    print(test_correction)
    main_test()