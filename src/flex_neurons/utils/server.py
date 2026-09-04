#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
- server related functions
"""

import platform


def is_on_server():
    """pretty self-explanatory"""
    import os

    if os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"):
        return True
    return False


if __name__ == "__main__":
    print(is_on_server())
