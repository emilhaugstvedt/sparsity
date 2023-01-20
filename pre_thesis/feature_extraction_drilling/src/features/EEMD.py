from PyEMD import EEMD

def get_EEMD(data, n_imfs=None):
    eemd = EEMD()
    eemd.eemd(data)
    imfs = eemd.get_imfs_and_residue()
    if n_imfs is None:
        return imfs
    return imfs[:n_imfs]