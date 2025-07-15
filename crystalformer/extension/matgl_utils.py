import jax
import jax.numpy as jnp
import numpy as np
import joblib
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
# from matgl.ext.ase import PESCalculator

from crystalformer.src.wyckoff import mult_table, wmax_table, symops


mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)
symops = np.array(symops)


def make_forward_fn(model, dummy_value=5):

    def forward_fn(x):
        try: 
            struct = revert(*x)
            quantity = model(struct)
            # if quantity is nan, return a dummy value
            quantity = quantity if not np.isnan(quantity) else np.array(dummy_value)
        except:
            quantity = np.array(dummy_value)  #TODO: check if this is a good idea
        
        return quantity

    def batch_forward_fn(x):
        output = map(forward_fn, zip(*x))
        output = np.array(list(output))

        return output

    def parallel_batch_forward(x, batch_size=50):
        x = jax.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)

        xs = [[_x[i:i+batch_size] for _x in x] for i in range(0, G.shape[0], batch_size)]
        
        output = joblib.Parallel(
            n_jobs=-1,
        )(joblib.delayed(batch_forward_fn)(_x) for _x in xs)

        # unpack the output
        output = np.array(output)
        # reshape it back to the original shape
        output = np.reshape(output, G.shape)
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()
        
        return output

    # TODO: 没有使用parallel版本可能带来的性能问题
    return forward_fn, batch_forward_fn, 


# def make_force_forward_fn(pot):

    ase_adaptor = AseAtomsAdaptor()
    calc = PESCalculator(pot)

    def forward_fn(x):
        try: 
            struct = revert(*x)
            atoms = ase_adaptor.get_atoms(struct)
            atoms.calc = calc
            # if quantity is nan, return a dummy value
            forces = atoms.get_forces()
        except:
            forces = np.ones((1, 3))*np.inf # avoid nan
        forces = np.linalg.norm(forces, axis=-1)
        forces = np.clip(forces, 1e-2, 1e2)  # avoid too large or too small forces
        forces = np.mean(forces)
        
        return np.log(forces)

    def batch_forward_fn(x):
        output = map(forward_fn, zip(*x))
        output = np.array(list(output))

        return output

    def parallel_batch_forward(x, batch_size=50):
        x = jax.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)

        xs = [[_x[i:i+batch_size] for _x in x] for i in range(0, G.shape[0], batch_size)]
        
        output = joblib.Parallel(
            n_jobs=-1,
        )(joblib.delayed(batch_forward_fn)(_x) for _x in xs)

        # unpack the output
        output = np.array(output)
        # reshape it back to the original shape
        output = np.reshape(output, G.shape)
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()
        
        return output

    return forward_fn, parallel_batch_forward


def symmetrize_atoms(g, w, x):

    # apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs


def revert(G, L, X, A, W):
    """
    NOTE Elements must within 1-95 due to pyxtal restriction
    Given normalized GLXAW, returns pymatgen.Structure object

    Args:
        G: space group number
        L: lattice parameters
        A: element number list
        W: wyckoff letter list
        X: fractional coordinates list
    
    Returns:
        struct: pymatgen.Structure object
    """

    # un-normalize data
    M = mult_table[G-1, W]
    num_atoms = np.sum(M)
    length, angle = np.split(L, 2)
    length = length*num_atoms**(1/3)
    angle = angle / (np.pi / 180) # to rad
    L = np.concatenate([length, angle])

    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    A_list = np.repeat(A, [len(xs) for xs in xs_list])
    X_list = np.concatenate(xs_list)
    struct = Structure(lattice, A_list, X_list)
    return struct


if __name__  == "__main__":
    import matgl
    # from crystalformer.src.utils import GLXYZAW_from_file
    from crystalformer.cli.classifier import GLXYZAW_from_sample
    from functools import partial
    from time import time
    import torch
    import warnings
    torch.set_default_device("cpu")
    warnings.filterwarnings("ignore")

    def make_callback_forward(forward):
        def callback_forward(x):
            G, _, _, _, _ = x
            result_shape = jax.ShapeDtypeStruct(G.shape, jnp.float32)
            return jax.experimental.io_callback(forward, result_shape, x)
        return callback_forward

    # formation energy model
    model = matgl.load_model("/data/zdcao/website/matgl/pretrained_models/MEGNet-MP-2018.6.1-Eform")
    model = model.predict_structure

    # band gap model
    model = matgl.load_model("/data/zdcao/website/matgl/pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi")
    model = partial(model.predict_structure, state_attr=torch.tensor([0]))

    forward_fn, parallel_batch_forward = make_forward_fn(model)

    # data = GLXYZAW_from_file('./data/mini.csv', n_max=21, wyck_types=28, atom_types=119)
    data = GLXYZAW_from_sample(225, "/home/zdcao/website/distance/crystal_gpt/experimental/base/output_225.csv")

    ##################### single forward #####################
    s_time = time()
    output = forward_fn(jax.tree.map(lambda x: x[0], data))
    print("Single forward Time taken: ", time() - s_time)

    ##################### parallel batch forward #####################
    s_time = time()
    output = parallel_batch_forward(data)
    print("batch forward Time taken: ", time() - s_time)
    # print(output)

    ##################### callback parallel batch forward #####################
    s_time = time()
    callback_forward = make_callback_forward(parallel_batch_forward)
    output = callback_forward(data)
    print(output.shape)
    print("callback batch forward Time taken: ", time() - s_time)

    ##################### force forward #####################
    print("================= Force forward =====================")
    model = matgl.load_model("/data/zdcao/website/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES")
    forward_fn, parallel_batch_forward =  make_force_forward_fn(model)

    ##################### single forward #####################
    s_time = time()
    output = forward_fn(jax.tree.map(lambda x: x[0], data))
    print("Single forward Time taken: ", time() - s_time)

    ##################### parallel batch forward #####################
    s_time = time()
    output = parallel_batch_forward(data, batch_size=50)
    print("batch forward Time taken: ", time() - s_time)
    # print(output)

    ##################### callback parallel batch forward #####################
    s_time = time()
    callback_forward = make_callback_forward(parallel_batch_forward)
    output = callback_forward(data)
    print(output.shape)
    print("callback batch forward Time taken: ", time() - s_time)
