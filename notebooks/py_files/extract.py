"""Functions for extracting raw data."""

import re
import numpy as np

def extract_data(raw_path, hea_path, raw_dtype='int16'):
    """Extract raw and meta data from .dat and .hea files.

    Parameters
    ----------
    raw_path : str
        Path to .dat file.
    hea_path : str
        Path to .hea file.
    raw_dtype : {'int16', 'uint16'}
        Raw datatype. Matlab scripts uses int16 but the
        publications states uint16. int16 looks correct.

    Returns
    -------
    sigs : 2d array
        Extract signals.
    meta : dict
        Meta-data.
    """

    meta = extract_metadata(hea_path)

    sigs = np.fromfile(raw_path, dtype=raw_dtype)
    sigs = sigs.reshape(meta['n_samples'], -1).T

    return sigs, meta


def extract_metadata(hea_path):
    """Extract metadata from .hea files.

    Parameters
    ----------
    hea_path : str
        Path to .hea file.

    Returns
    -------
    meta_dict : dict
        Metadata contents.
    """

    with open(hea_path, 'r') as f:
        meta = f.read().split('\n')

    meta = [m for m in meta if m != '']

    meta_dict = {}

    header_vars = ['id', 'n_sigs', 'fs', 'n_samples']
    meta_vars = ['signal_name', 'bit_res', 'bit_gain',
                 'baseline', 'units', 'first_bit']

    for d, h in zip(meta[0].split(' '), header_vars):

        if h == 'id':
            meta_dict[h] = d
        elif h in ['n_sigs', 'n_samples']:
            meta_dict[h] = int(d)
        else:
            meta_dict[h] = float(d)

    for ind, m in enumerate(meta[1:]):

        m = m.split(' ')
        m = [i for i in m if i != '']

        ind_str = str(ind).zfill(2)

        signal_name = m[-1]
        bit_res = int(m[1])
        bit_gain = float(re.search('[^(]*', m[2])[0])
        baseline = float(re.search('\(.*\)', m[2])[0][1:-1])
        units = re.search('\/.*', m[2])[0][1:]
        first_bit = int(m[6])

        dat = [signal_name, bit_res, bit_gain, baseline, units, first_bit]

        meta_dict['sig'+ind_str] = {
            k: v for k, v in zip(meta_vars, dat)
        }

    return meta_dict