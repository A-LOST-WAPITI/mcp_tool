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
from os.path import realpath
import argparse

def make_cond_forward_fn(batch_forward_fn):
    def forward_fn(G, L, XYZ, A, W, target):
        # pull everything into pure NumPy float32 and ensure 1-D shape
        target_np = np.atleast_1d(np.asarray(target, dtype=np.float32))
        preds_np  = np.atleast_1d(
            np.asarray(batch_forward_fn((G, L, XYZ, A, W)), dtype=np.float32)
        )
        diff_np   = np.abs(preds_np - target_np)
        # 寻找小于目标值
        # diff_np = np.clip(preds_np - target_np, 0, None)
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

def load_property_model(model_path: str, model_head: str = None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    else:
        try:
            property_model = DeepProperty(model_path) if model_head is None else DeepProperty(model_path, model_head=model_head)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
    return property_model

def property_model_forward(model, struc: Structure):
    """
    Forward function for the property model.
    config:
        model: The property model to be used for prediction.
        struc: A pymatgen Structure object.
    Returns:
        The predicted property value.
    """
    if not isinstance(struc, Structure):
        raise TypeError("Input must be a pymatgen Structure object.")
    
    quantity = model.eval(
        struc.cart_coords[np.newaxis, ...],
        struc.lattice.matrix.reshape(1, 3, 3),
        [elem.Z for elem in struc.species]
    )[0].reshape(-1)

    return quantity
    
def init_cond_model(config):
    rand_key = jax.random.PRNGKey(config.seed)
    # 将字符串转换为numpy数组
    target_vec = jnp.array(literal_eval(config.target))
    alpha_vec = jnp.array(literal_eval(config.alpha))

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
        model = load_property_model(config.cond_model_path, config.model_head)
        model_forward = lambda x: property_model_forward(model, x)

        _, batch_forward_fn = make_forward_fn(model_forward)
        forward_fn = make_cond_forward_fn(batch_forward_fn)
        cond_logp_fn = make_cond_logp(
            logp_fn, forward_fn, 
            target=target_vec[0],
            alpha=alpha_vec[0]
        )
    elif config.mode == "multi":
        print("\n========== Load multiple conditional models ==========")
        model_path_list = config.cond_model_path.split(',')
        model_head_list = config.model_head.split(',') if config.model_head else [None] * len(model_path_list)
        assert len(model_path_list) == len(target_vec) == len(alpha_vec) == len(model_head_list), \
            "The number of models, targets, and alphas must match."
        batch_forward_fns = []

        for model_path, model_head in zip(model_path_list, model_head_list):
            model = load_property_model(model_path, model_head)
            model_forward = lambda x: property_model_forward(model, x)
            
            _, batch_forward_fn = make_forward_fn(model_forward)
            forward_fn = make_cond_forward_fn(batch_forward_fn)
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
    parser.add_argument('--cond_model_path', type=str, required=True,
                        help='Comma-separated paths to conditional models')
    parser.add_argument('--model_head', type=str, default=None,
                        help='Head of the model for specific task.')

    # Generation mode and targets
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                        help='Generation mode: single or multi')
    parser.add_argument('--target', type=str, default='-4, 3',
                        help='Target values (e.g., formation energy, bandgap)')
    parser.add_argument('--alpha', type=str, default='10, 3',
                        help='Guidance strength for generation')

    # Physics & constraints
    parser.add_argument('--spacegroup', type=int, required=True, nargs='+',
                        help='Space group number(s) for generation')
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
    G, L, XYZ, A, W = generate_sample(
        spacegroup=config.spacegroup,
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
    print("MCMC Time elapsed: ", time()-start)

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

    struc_list = map(revert, G, L, XYZ, A, W)

    for (idx, sorted_idx) in enumerate(sorted_idx_vec):
        struc = struc_list[sorted_idx]
        struc.sort()
        struc.to("target/POSCAR_{}".format(idx + 1))