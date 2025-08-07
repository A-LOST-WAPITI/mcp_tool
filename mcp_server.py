from typing import List
from pathlib import Path
import logging
import shutil
import subprocess
from dp.agent.server import CalculationMCPServer
from typing_extensions import TypedDict
import numpy as np

import shutil

import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    '''Parse command line arguments for MCP server.'''
    parser = argparse.ArgumentParser(description='DPA Calculator MCP Server')
    parser.add_argument('--port', type=int, default=50001,
                        help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()


# Initialize MCP server
mcp = CalculationMCPServer(
    'CrystalFormerServer',
    host=args.host,
    port=args.port
)


# ================ Tool to generate structures with conditional properties via CrystalFormer ===================
class GenerateCryFormerStructureResult(TypedDict):
    poscar_paths: Path
    message: str


@mcp.tool()
def generate_crystalformer_structures(
    cond_model_type: List[str],
    target_values: List[float],
    target_type: List[str],
    alpha: List[float],
    space_group_min: int,
    random_spacegroup_num: int,
    init_sample_num: int,
    mc_steps: int
) -> GenerateCryFormerStructureResult:
    '''
    Generate structures using CrystalFormer with specified conditional properties.
    Args:
        cond_model_type (List[str]): List of conditional model types (e.g., 'bandgap', 'shear_modulus', 'bulk_modulus', 'ambient_pressure', 'high_pressure', 'sound').
        target_values (List[float]): Target values for the properties.
        target_type (List[str]): Type of target values ('equal', 'greater', 'less', 'minimize').
        alpha (List[float]): Alpha values for different values.
        space_group_min (int): Minimum space group number.
        random_spacegroup_num (int): Number of random space groups to consider.
        init_sample_num (int): Initial number of samples for each space group.
        mc_steps (int): Number of Monte Carlo steps.
    '''
    try:
        assert len(cond_model_type) == len(target_values) == len(target_type) == len(alpha), \
            'Length of cond_model_type, target_values, target_type, and alpha must be the same.'
        
        ava_cond_model = [
            'bandgap',
            'shear_modulus',
            'bulk_modulus',
            'ambient_pressure',
            'high_pressure',
            'sound'
        ]
        assert np.all([model_type in ava_cond_model for model_type in cond_model_type]), \
            'Model type must be one of the following: ' + ', '.join(ava_cond_model)
        
        ava_value_type = ['equal', 'greater', 'less', 'minimize']
        assert np.all([t in ava_value_type for t in target_type]), \
            'Target type must be one of the following: ' + ', '.join(ava_value_type)

        # activate uv
        workdir = Path('/opt/agents/crystalformer')
        output_path = workdir / 'outputs'

        if output_path.exists():
            shutil.rmtree(output_path)

        mode = 'multi' if len(cond_model_type) > 1 else 'single'

        cmd = [
            'uv', 'run', 'python',
            'crystalformer_mcp.py',
            '--mode', mode,
            '--cond_model_type', *cond_model_type,
            '--target', *target_values,
            '--target_type', *target_type,
            '--alpha', *alpha,
            '--spacegroup', space_group_min,
            '--init_sample_num', init_sample_num,
            '--random_spacegroup_num', random_spacegroup_num,
            '--mc_steps', mc_steps,
            '--output_path', str(output_path)
        ]
        subprocess.run(cmd, cwd=workdir, check=True)
        
        return {
            'poscar_paths': output_path,
            'message': 'CrystalFormer structure generation successfully!'
        }
    except Exception:
        return {
            'poscar_paths': None,
            'message': 'CrystalFormer Execution failed!'
        }


# ====== Run Server ======
if __name__ == '__main__':
    logging.info(f'Starting CrystalFormerServer on port {args.port}...')
    mcp.run(transport='sse')
