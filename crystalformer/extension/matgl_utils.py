import jax
import numpy as np
import pathos.multiprocessing as mp
from pymatgen.core import Structure, Lattice

from crystalformer.src.wyckoff import mult_table, wmax_table, symops


mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)
symops = np.array(symops)


def make_forward_fn(forward_fn):
    def parallel_batch_forward(x):
        # x = jax.tree.map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)

        struc_list = [
            revert(*info)
            for info in zip(*x)
        ]
        
        # TODO: 并行运行
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     output = pool.map(forward_fn, struc_list)
        output = list(map(forward_fn, struc_list))

        # unpack the output
        output = np.array(output)
        # reshape it back to the original shape
        output = np.reshape(output, G.shape)
        # output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()
        
        return output

    return parallel_batch_forward

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
    import os
    from deepmd.infer.deep_property import DeepProperty
    from os import listdir
    from pymatgen.core import Structure
    import numpy as np
    model = DeepProperty('/Users/roy/Work/mcp_tool/cond_model/0415_h20_dpa3a_shareft_nosel_128_64_32_scp1_e1a_csilu3_rc6_arc_4_expsw_l16_128GPU_240by3_matbench_dielectric_fold1/model.ckpt-200000.pt')

    struc_list = []
    for file in listdir('target'):
        if file.startswith("POSCAR"):
            struc_list.append(Structure.from_file(f'target/{file}'))

    parallel_batch_forward = make_forward_fn(model)
    results = parallel_batch_forward(struc_list)
    print(results)