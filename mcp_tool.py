import os
# don’t pre-allocate all GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'
# cap JAX’s total footprint to 50% of GPU RAM
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
# throttle threading
os.environ['OMP_NUM_THREADS']                 = '2'
os.environ['DP_INTRA_OP_PARALLELISM_THREADS'] = '2'
os.environ['DP_INTER_OP_PARALLELISM_THREADS'] = '2'

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pandas as pd
import numpy as np
from ast import literal_eval
from typing import Sequence, Tuple

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.lattice import norm_lattice
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer

from crystalformer.extension.experimental import make_cond_logp, make_multi_cond_logp, make_mcmc_step
from crystalformer.extension.matgl_utils import make_forward_fn, revert

from pymatgen.core import Structure
from deepmd.infer.deep_property import DeepProperty
from tempfile import TemporaryDirectory
from subprocess import Popen
from os import symlink
from os.path import realpath, exists
import argparse
import torch

def make_cond_forward_fn(batch_forward_fn, target_type: str = 'equal'):
    if target_type == 'equal':
        diff = lambda preds, target: np.abs(preds - target)
    elif target_type == 'greater':
        diff = lambda preds, target: np.clip(target - preds, 0, None)
    elif target_type == 'less':
        diff = lambda preds, target: np.clip(preds - target, 0, None)
    elif target_type == 'minimize':
        diff = lambda preds, eps: (preds ** 2) + 1.0 / (preds + eps)  # avoid division by zero
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    def forward_fn(G, L, XYZ, A, W, target):
        # pull everything into pure NumPy float32 and ensure 1-D shape
        target_np = np.atleast_1d(np.asarray(target, dtype=np.float32))
        preds_np = np.atleast_1d(
            np.asarray(batch_forward_fn((G, L, XYZ, A, W)), dtype=np.float32)
        )
        diff_np = diff(preds_np, target_np)
        return diff_np.reshape(-1).astype(np.float32)
    return forward_fn

# 定义用于存储设置的类并给定默认值
class Config(object):
    def __init__(self, conf_dict):
        # default configuration
        config = {
            # Physics parameters
            'n_max': 21,
            'atom_types': 119,
            'wyck_types': 28,

            # Base transformer parameters
            'Nf': 5,
            'Kx': 16,
            'Kl': 4,
            'h0_size': 256,
            'transformer_layers': 16,
            'num_heads': 16,
            'key_size': 64,
            'model_size': 64,
            'embed_size': 32,
            'dropout_rate': 0.5,
            'restore_path': './model/mp20/epoch_003800.pkl',

            # Conditional generation parameters
            'cond_model_path': None,
            'spacegroup': None,  # None indicates it's an optional argument
            'remove_radioactive': True,
            'mode': 'single',
            'target': '-4',
            'alpha': '10',
            'output_path': './',

            # MCMC parameters
            'mc_steps': 1000,
            'mc_width': 0.1,
            'init_temp': 10.0,
            'end_temp': 1.0,
            'decay_step': 10,

            # MISC
            'seed': 42
        }

        # Update default config with provided config
        config.update(conf_dict)

        # Set attributes from config
        for (key, value) in config.items():
            setattr(self, key, value)

def simple_property_forward(model, struc_list: Sequence[Structure]) -> np.ndarray:
    def _pad_batch_structures(
            strucs: Sequence[Structure]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pads a list/tuple of pymatgen Structures to the largest atom count.
        Returns
        -------
        coords   : (B, N_max, 3) float32
        lattice  : (B, 3, 3)     float32
        atom_Z   : (B, N_max)    int32      (padding == -1)
        """
        if len(strucs) == 0:
            raise ValueError("Got an empty batch.")
        B        = len(strucs)
        natoms   = np.array([len(s) for s in strucs])
        N_max    = max(len(s) for s in strucs)
        coords   = np.zeros((B, N_max, 3), dtype=np.float32)        # pad 0
        lattice  = np.zeros((B, 3, 3),     dtype=np.float32)        # one per struct
        atom_Z   = np.full((B, N_max), 0, dtype=np.int32)           # pad 0
        for i, s in enumerate(strucs):
            n = natoms[i]
            coords[i, :n] = s.cart_coords
            atom_Z[i, :n] = s.atomic_numbers
            lattice[i]    = s.lattice.matrix

        atom_Z -= 1  # convert to 0-based indexing
        
        result_size_raito = natoms / N_max
        return coords, lattice, atom_Z, result_size_raito
    
    cart_coords, lattice, atom_Z, result_size_raito = _pad_batch_structures(struc_list)

    # forward pass
    n_struc = len(struc_list)
    with torch.no_grad():
        torch.cuda.empty_cache()
        output = model.eval(
            cart_coords,
            lattice,
            atom_Z,
            mixed_type=True
        )[0].reshape(n_struc, -1)[:, 0] / result_size_raito

    # sync
    torch.cuda.synchronize()

    return output.astype(np.float32)

def sound_forward(model_forward_fn_list, struc_list: Sequence[Structure]):
    density = [
        float(struc.density) * 1e3 # g/cm^3 to kg/m^3
        for struc in struc_list
    ]

    log_G, log_K = [forward_fn(struc_list) for forward_fn in model_forward_fn_list]
    G_Pa = np.exp(log_G) * 1e9  # GPa to Pa
    K_Pa = np.exp(log_K) * 1e9  # GPa to Pa

    # Longitudinal velocity
    v_L = np.sqrt((K_Pa + (4/3) * G_Pa) / density)

    # Shear velocity
    v_S = np.sqrt(G_Pa / density)
    # Average velocity
    v_m = (1/3 * (1/v_L**3 + 2/v_S**3)) ** (-1/3)
    
    return v_m

def load_property_model(model_type: str):
    base_dir = '/opt/agents/'
    ava_model_dict = {
        'simple': [
            'bandgap',
            'shear_modulus',
            'bulk_modulus',
            'ambient_pressure',
            'high_pressure'
        ],
        'hybrid': [
            'sound'
        ]
    }

    if model_type in ava_model_dict['simple']:
        if model_type.endswith("pressure"):
            model_dir = f'{base_dir}/superconductor/models/{model_type}/'
        else:
            model_dir = f'{base_dir}/thermal_properties/models/{model_type}/'

        model_file = f'{model_dir}/{os.listdir(model_dir)[0]}'
        if model_type == 'high_pressure':
            model = DeepProperty(model_file, auto_batch_size=True, head='tc')
        else:
            model = DeepProperty(model_file, auto_batch_size=True)

        return lambda struc_list: simple_property_forward(model, struc_list)
    elif model_type in ava_model_dict['hybrid']:
        if model_type == 'sound':
            model_forward_fn_list = [
                load_property_model('shear_modulus'),
                load_property_model('bulk_modulus')
            ]

            return lambda struc_list: sound_forward(model_forward_fn_list, struc_list)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported in hybrid models. Available types: {ava_model_dict['hybrid']}")
    else:
        raise ValueError(f"Model type '{model_type}' is not supported. Available types: {ava_model_dict['simple'] + ava_model_dict['hybrid']}")
            
def init_cond_model(config):
    rand_key = jax.random.PRNGKey(config.seed)
    # 将字符串转换为numpy数组
    target_vec = jnp.array(config.target)
    alpha_vec = jnp.array(config.alpha)

    if config.remove_radioactive:
        from crystalformer.src.elements import radioactive_elements_dict, noble_gas_dict
        # remove radioactive elements and noble gas
        atom_mask = [1] + [1 if i not in radioactive_elements_dict.values(
        ) and i not in noble_gas_dict.values() else 0 for i in range(1, config.atom_types)]
        atom_mask = jnp.array(atom_mask)
        # set the probability of padding atom to 0
        atom_mask = atom_mask.at[0].set(0)
        # # TODO: set probability of Yb element to 0 due to the MP dataset
        # atom_mask = atom_mask.at[70].set(0)
        # print('set logit of Yb element to 0')
        print('set logit of padding atom to 0')
        atom_mask = jnp.stack([atom_mask] * config.n_max, axis=0)
        print('sampling structure formed by non-radioactive elements and non-noble gas')
        # print(atom_mask)
    else:
        # we will do nothing to a_logit in sampling
        atom_mask = jnp.zeros((config.atom_types), dtype=int)
        atom_mask = jnp.stack([atom_mask] * config.n_max, axis=0)
        # print(atom_mask)

    ################### Load BASE Model #############################
    base_params, base_transformer = make_transformer(
        rand_key,
        config.Nf, config.Kx, config.Kl, config.n_max,
        config.h0_size,
        config.transformer_layers, config.num_heads,
        config.key_size, config.model_size, config.embed_size,
        config.atom_types, config.wyck_types,
        config.dropout_rate
    )
    print("# of transformer params", ravel_pytree(base_params)[0].size)

    _, logp_fn = make_loss_fn(
        config.n_max, config.atom_types,
        config.wyck_types, config.Kx, config.Kl, base_transformer
    )

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(config.restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        base_params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    ################### Load Conditional Model #############################
    if config.mode == "single":
        print("\n========== Load single conditional model ==========")
        # Load the pre-trained MEGNet formation energy model.
        cond_model_forward_fn = load_property_model(config.cond_model_type[0])

        batch_forward_fn = make_forward_fn(cond_model_forward_fn)
        forward_fn = make_cond_forward_fn(batch_forward_fn, config.target_type[0])
        cond_logp_fn = make_cond_logp(
            logp_fn, forward_fn, 
            target=target_vec[0],
            alpha=alpha_vec[0]
        )
    elif config.mode == "multi":
        print("\n========== Load multiple conditional models ==========")
        assert len(config.cond_model_type) == len(target_vec) == len(alpha_vec), \
            "The number of models, targets, and alphas must match."
        batch_forward_fns = []

        for model_type, target_type in zip(config.cond_model_type, config.target_type):
            cond_model_forward_fn = load_property_model(model_type)
            
            batch_forward_fn = make_forward_fn(cond_model_forward_fn)
            forward_fn = make_cond_forward_fn(batch_forward_fn, target_type)
            batch_forward_fns.append(forward_fn)

        cond_logp_fn = make_multi_cond_logp(
            logp_fn, batch_forward_fns,
            targets=target_vec,
            alphas=alpha_vec
        )
    else:
        raise NotImplementedError
    
    return rand_key, cond_logp_fn, logp_fn, base_params, atom_mask

def generate_sample(spacegroup, num_samples: int, temperature: float):
    with TemporaryDirectory() as tmpdir:
        # 准备权重文件到临时目录
        symlink(realpath('model/mp20/epoch_003800.pkl'), f'{tmpdir}/epoch_003800.pkl')

        sub_process = Popen(
            [
                'uv', 'run', 'python', './main.py',
                '--optimizer', 'none',
                '--restore_path', f'{tmpdir}/',
                '--spacegroup', *[str(item) for item in spacegroup],
                '--num_samples', str(num_samples),
                '--temperature', str(temperature)
            ],
        )

        sub_process.wait()
        midfix = "-".join([str(spg) for spg in spacegroup])
        if os.path.exists(f'{tmpdir}/output_{midfix}.csv'):
            origin_data = pd.read_csv(f'{tmpdir}/output_{midfix}.csv')
            L, XYZ, A, W, G = origin_data['L'], origin_data['X'], origin_data['A'], origin_data['W'], origin_data['G']
            L = L.apply(lambda x: literal_eval(x))
            XYZ = XYZ.apply(lambda x: literal_eval(x))
            A = A.apply(lambda x: literal_eval(x))
            W = W.apply(lambda x: literal_eval(x))

            # convert array of list to numpy ndarray
            G = jnp.array(G.tolist())
            L = jnp.array(L.tolist())
            XYZ = jnp.array(XYZ.tolist())
            A = jnp.array(A.tolist())
            W = jnp.array(W.tolist())

            L = norm_lattice(G, W, L)
        else:
            raise FileNotFoundError(f"Output file for spacegroup {spacegroup} not found in temporary directory.")

    return G, L, XYZ, A, W

def get_args():
    parser = argparse.ArgumentParser(description='Script for running conditional generation with MCMC')

    # Paths and model info
    parser.add_argument('--cond_model_type', type=str, nargs='+', required=True,
                        help='Types of conditional models to use.')

    # Generation mode and targets
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                        help='Generation mode: single or multi')
    parser.add_argument('--target', type=float, nargs='+', required=True,
                        help='Target values (e.g., formation energy, bandgap)')
    parser.add_argument('--target_type', type=str, nargs='+',
                        help='Type of target: equal, greater, less, minimize')
    parser.add_argument('--alpha', type=float, nargs='+', required=True,
                        help='Guidance strength for generation')

    # Physics & constraints
    parser.add_argument('--spacegroup', type=int, required=True, nargs='+',
                        help='Minimum spacegroup number to sample from')
    parser.add_argument('--random_spacegroup_num', type=int, default=0,
                        help='Number of random spacegroups to sample from')
    parser.add_argument('--init_sample_temperature', type=float, default=1.0,
                        help='Initial sampling temperature')
    parser.add_argument('--init_sample_num', type=int, default=100,
                        help='Number of samples to initialize generation')

    # MCMC parameters
    parser.add_argument('--mc_steps', type=int, default=100,
                        help='Number of MCMC steps')
    parser.add_argument('--mc_width', type=float, default=0.1,
                        help='Width of the MCMC proposal')
    parser.add_argument('--init_temp', type=float, default=10.0,
                        help='Initial MCMC temperature')
    parser.add_argument('--end_temp', type=float, default=1.0,
                        help='Final MCMC temperature')
    parser.add_argument('--decay_step', type=int, default=10,
                        help='MCMC temperature decay steps')

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    config_dict = get_args()
    config = Config(config_dict)

    print("\n======= Generate Samples =======")
    # Wether smaple from random spacegroup
    if config.random_spacegroup_num > 0:
        spacegroup = np.random.choice(
            range(config.spacegroup[0], 231),
            size=config.random_spacegroup_num,
        )
        print(f"Randomly sample from spacegroups: {spacegroup}")
    else:
        spacegroup = config.spacegroup
    G, L, XYZ, A, W = generate_sample(
        spacegroup=spacegroup,
        num_samples=config.init_sample_num,  # Number of samples to generate
        temperature=config.init_sample_temperature  # Temperature for sampling
    )

    ################### MCMC ############################
    key, cond_logp_fn, logp_fn, base_params, atom_mask = init_cond_model(config)

    print("\n========== Start MCMC ==========")
    mcmc = make_mcmc_step(base_params, n_max=config.n_max, atom_types=config.atom_types, atom_mask=atom_mask)
    x = (G, L, XYZ, A, W)

    print("====== before mcmc =====")
    print ("XYZ:\n", XYZ)  # fractional coordinate
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("L:\n", L)  # lattice

    from time import time
    start = time()
    temp = config.init_temp
    for i in range(config.decay_step):
        alpha = i/(config.decay_step-1)
        temp = 1/(alpha/config.end_temp + (1-alpha)/config.init_temp)
        # temp = init_temp - (init_temp - end_temp) * i / (decay_step-1)
        key, subkey = jax.random.split(key)
        x, acc = mcmc(cond_logp_fn, x_init=x, key=subkey,
                    mc_steps=config.mc_steps // config.decay_step,
                    mc_width=config.mc_width, temp=temp)
        print("i, temp, acc", i, temp, acc)
    print("MCMC Time elapsed: ", time() - start)

    G, L, XYZ, A, W = x

    key, subkey = jax.random.split(key)
    logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(base_params, subkey, G, L, XYZ, A, W, False)
    logp = logp_w + logp_xyz + logp_a + logp_l
    key, subkey = jax.random.split(key)
    logp_new = jax.jit(cond_logp_fn, static_argnums=7)(base_params, subkey, G, L, XYZ, A, W, False)
    sorted_idx_vec = jnp.argsort(logp_new, axis=0, descending=True)

    print("====== after mcmc =====")
    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) 
    num_atoms = jnp.sum(M, axis=1)

    struc_list = list(map(revert, G, L, XYZ, A, W))

    if not exists("target"):
        os.makedirs("target")
    for (idx, sorted_idx) in enumerate(sorted_idx_vec):
        struc = struc_list[sorted_idx]
        struc.sort()
        struc.to("target/POSCAR_{}".format(idx + 1))