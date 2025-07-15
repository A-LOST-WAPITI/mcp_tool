import os
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pandas as pd
import numpy as np
from ast import literal_eval
from functools import partial
import matgl
import warnings
import torch
torch.set_default_device("cpu")
warnings.filterwarnings("ignore")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.lattice import norm_lattice
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer

from crystalformer.extension.experimental import make_cond_logp, make_multi_cond_logp, make_mcmc_step
from crystalformer.extension.matgl_utils import make_forward_fn


import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

group = parser.add_argument_group('base transformer parameters')
group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the first atom, 0 means we simply use a table for first aw_logit')
group.add_argument('--transformer_layers', type=int, default=16, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=16, help='The number of heads')
group.add_argument('--key_size', type=int, default=64, help='The key size')
group.add_argument('--model_size', type=int, default=64, help='The model size')
group.add_argument('--embed_size', type=int, default=32, help='The enbedding size')
group.add_argument('--dropout_rate', type=float, default=0.5, help='The dropout rate')
group.add_argument('--restore_path', default='./', help='The path to restore the base model')

group.add_argument('--cond_restore_path',
                   default='/data/zdcao/website/matgl/pretrained_models/MEGNet-MP-2018.6.1-Eform',
                   help='The path to restore the conditional model')

group = parser.add_argument_group('conditional generation parameters')
group.add_argument('--spacegroup', type=int, help='The space group')
group.add_argument('--remove_radioactive', action='store_true', help='remove radioactive elements and noble gas')
group.add_argument('--input_path', default='./', help='The path to load the input data')
group.add_argument('--mode', type=str, default="single", help='single or multi')
group.add_argument('--target', type=str, default="-4, 3", help='target value for formation energy and bandgap')
group.add_argument('--alpha', type=str, default="10, 3", help='guidance strength')
group.add_argument('--output_path', default='./', help='The path to output the generated data')

group = parser.add_argument_group('MCMC parameters')
group.add_argument('--mc_steps', type=int, default=1000, help='The number of MCMC steps')
group.add_argument('--mc_width', type=float, default=0.1, help='The width of MCMC proposal')
group.add_argument('--init_temp', type=float, default=10.0, help='The initial temperature')
group.add_argument('--end_temp', type=float, default=1.0, help='The final temperature')
group.add_argument('--decay_step', type=int, default=10, help='The number of decay steps')


def make_cond_forward_fn(batch_forward_fn):
    def forward_fn(G, L, XYZ, A, W, target):
        x = (G, L, XYZ, A, W)
        return  np.abs(batch_forward_fn(x) - target).squeeze()
    return forward_fn


args = parser.parse_args()
key = jax.random.PRNGKey(42)

target = [float(x) for x in args.target.split(',')]
alpha = [float(x) for x in args.alpha.split(',')]

print("target:", target)
print("alpha:", alpha)

if args.remove_radioactive:
    from crystalformer.src.elements import radioactive_elements_dict, noble_gas_dict
    # remove radioactive elements and noble gas
    atom_mask = [1] + [1 if i not in radioactive_elements_dict.values() and i not in noble_gas_dict.values() else 0 for i in range(1, args.atom_types)]
    atom_mask = jnp.array(atom_mask)
    atom_mask = atom_mask.at[0].set(0) # set the probability of padding atom to 0
    # # TODO: set probability of Yb element to 0 due to the MP dataset
    # atom_mask = atom_mask.at[70].set(0)
    # print('set logit of Yb element to 0')
    print('set logit of padding atom to 0')
    atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
    print('sampling structure formed by non-radioactive elements and non-noble gas')
    print(atom_mask)
        
else:
    atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
    atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
    print(atom_mask)

# print(f'there is total {jnp.sum(atom_mask[0])-1} elements')
print(atom_mask.shape)   

################### Load BASE Model #############################
base_params, base_transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                    args.h0_size, 
                                    args.transformer_layers, args.num_heads, 
                                    args.key_size, args.model_size, args.embed_size, 
                                    args.atom_types, args.wyck_types,
                                    args.dropout_rate)
print ("# of transformer params", ravel_pytree(base_params)[0].size) 

_, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, base_transformer)

print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    base_params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

################### Load Conditional Model #############################
if args.mode == "single":
    print("\n========== Load single conditional model ==========")
    # Load the pre-trained MEGNet formation energy model.
    model = matgl.load_model(args.cond_restore_path)
    if "BandGap" in args.cond_restore_path:
        model = partial(model.predict_structure, state_attr=torch.tensor([0]))
    else:
        model = model.predict_structure

    _, batch_forward_fn = make_forward_fn(model)
    forward_fn = make_cond_forward_fn(batch_forward_fn)
    cond_logp_fn = make_cond_logp(logp_fn, forward_fn, 
                                    target=jnp.array(target[0]),
                                    alpha=alpha[0])

elif args.mode == "multi":
    print("\n========== Load multiple conditional models ==========")
    restore_path = args.cond_restore_path.split(',')
    batch_forward_fns = []

    for path in restore_path:
        model = matgl.load_model(path)
        # if path contains "BandGap"
        if "BandGap" in path:
            model = partial(model.predict_structure, state_attr=torch.tensor([0]))
        else:
            model = model.predict_structure
        
        _, batch_forward_fn = make_forward_fn(model)
        forward_fn = make_cond_forward_fn(batch_forward_fn)
        batch_forward_fns.append(forward_fn)

    cond_logp_fn = make_multi_cond_logp(logp_fn, batch_forward_fns,
                                        targets=jnp.array(target),
                                        alphas=jnp.array(alpha))
    
elif args.mode == "force":
    print("\n========== Load force conditional model ==========")
    restore_path = args.cond_restore_path.split(',')
    models = []
    batch_forward_fns = []

    print("===========load property prediction models===========")
    for path in restore_path:
        model = matgl.load_model(path)
        # if path contains "BandGap"
        if "BandGap" in path:
            model = partial(model.predict_structure, state_attr=torch.tensor([0]))
        else:
            model = model.predict_structure
        
        _, batch_forward_fn = make_forward_fn(model)
        forward_fn = make_cond_forward_fn(batch_forward_fn)
        batch_forward_fns.append(forward_fn)

    print("===========load force prediction models===========")
    from crystalformer.extension.matgl_utils import make_force_forward_fn

    model = matgl.load_model("/data/zdcao/website/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES")
    _, batch_forward_fn =  make_force_forward_fn(model)
    forward_fn = make_cond_forward_fn(batch_forward_fn)
    batch_forward_fns.append(forward_fn)

    target.append(0.0)
    alpha.append(1.0)
    print("target:", target)
    print("alpha:", alpha)

    cond_logp_fn = make_multi_cond_logp(logp_fn, batch_forward_fns,
                                        targets=jnp.array(target),
                                        alphas=jnp.array(alpha))

else:
    raise NotImplementedError

print("\n========== Load sampled data ==========")
csv_file = f"{args.input_path}/output_{args.spacegroup}.csv"
origin_data = pd.read_csv(csv_file)
L, XYZ, A, W = origin_data['L'], origin_data['X'], origin_data['A'], origin_data['W']
L = L.apply(lambda x: literal_eval(x))
XYZ = XYZ.apply(lambda x: literal_eval(x))
A = A.apply(lambda x: literal_eval(x))
W = W.apply(lambda x: literal_eval(x))

# convert array of list to numpy ndarray
G = jnp.array([args.spacegroup]*len(L))
L = jnp.array(L.tolist())
XYZ = jnp.array(XYZ.tolist())
A = jnp.array(A.tolist())
W = jnp.array(W.tolist())

L = norm_lattice(G, W, L)

print(G.shape, L.shape, XYZ.shape, A.shape, W.shape)

################### MCMC ############################
print("\n========== Start MCMC ==========")
mcmc = make_mcmc_step(base_params, n_max=args.n_max, atom_types=args.atom_types, atom_mask=atom_mask)
x = (G, L, XYZ, A, W)

print("====== before mcmc =====")
print ("XYZ:\n", XYZ)  # fractional coordinate 
print ("A:\n", A)  # element type
print ("W:\n", W)  # Wyckoff positions
print ("L:\n", L)  # lattice

from time import time
start = time()
temp = args.init_temp
for i in range(args.decay_step):
    alpha = i/(args.decay_step-1)
    temp = 1/(alpha/args.end_temp + (1-alpha)/args.init_temp)
    # temp = init_temp - (init_temp - end_temp) * i / (decay_step-1)
    key, subkey = jax.random.split(key)
    x, acc = mcmc(cond_logp_fn, x_init=x, key=subkey,
                  mc_steps=args.mc_steps//args.decay_step,
                  mc_width=args.mc_width, temp=temp)
    print("i, temp, acc", i, temp, acc)
print("MCMC Time elapsed: ", time()-start)

G, L, XYZ, A, W = x

key, subkey = jax.random.split(key)
logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(base_params, subkey, G, L, XYZ, A, W, False)
logp = logp_w + logp_xyz + logp_a + logp_l
key, subkey = jax.random.split(key)
logp_new = jax.jit(cond_logp_fn, static_argnums=7)(base_params, subkey, G, L, XYZ, A, W, False)

print("====== after mcmc =====")
M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) 
num_atoms = jnp.sum(M, axis=1)

#scale length according to atom number since we did reverse of that when loading data
length, angle = jnp.split(L, 2, axis=-1)
length = length*num_atoms[:, None]**(1/3)
angle = angle * (180.0 / jnp.pi) # to deg
L = jnp.concatenate([length, angle], axis=-1)

print ("XYZ:\n", XYZ)  # fractional coordinate 
print ("A:\n", A)  # element type
print ("W:\n", W)  # Wyckoff positions
print ("L:\n", L)  # lattice

data = pd.DataFrame()
data['L'] = np.array(L).tolist()
data['X'] = np.array(XYZ).tolist()
data['A'] = np.array(A).tolist()
data['W'] = np.array(W).tolist()
data['M'] = np.array(M).tolist()
data['logp'] = np.array(logp).tolist()
data['logp_new'] = np.array(logp_new).tolist()

filename = f'{args.output_path}/cond_output_{args.spacegroup}.csv'
header = False if os.path.exists(filename) else True
data.to_csv(filename, mode='a', index=False, header=header)

print ("Wrote samples to %s"%filename)
