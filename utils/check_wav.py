#!/usr/bin/env python3

import argparse
import struct
import numpy as np
from scipy.signal import argrelextrema

parser = argparse.ArgumentParser(description='Check output wav file versus reference file')

parser.add_argument("--input", dest="infile", default=None, help="File containing the input values to be checked")
parser.add_argument("--skiped", dest="skiped", default=0, type=int, help="Time skipped (in ms) at the beginning of the file to be checked")
parser.add_argument("--tolerance", dest="tolerance", default=200, type=float, help="Maximum tolerated difference between values")
parser.add_argument("--ref", dest="reffile", default=None, help="File containing the reference values used to check the input file")

args = parser.parse_args()



def _get_field(wav, offset, lent):
    wav.seek(0)
    wav.seek(offset, 0)
    return int.from_bytes(wav.read(lent), byteorder='little')

def _get_spec(fin):
    fin_header_size = _get_field(fin, 16, 4)
    sample_rate = _get_field(fin, 24, 4)
    fin_bit_rate = _get_field(fin, 34, 2)
    num_channels = _get_field(fin, 22, 2)
    if fin_header_size == 16:
        fin.seek(44)
    elif fin_header_size == 18:
        fin.seek(46)
    else:
        print('Unknown wav format\n')
        fin.seek(0)
        fin_bit_rate = 32
    return fin_bit_rate, sample_rate, num_channels


def _read_signed_integer(fin, num_bytes):
    # Read the specified number of bytes from the file
    data = fin.read(num_bytes)
    # Check the sign bit in the last byte
    if not data:
        return None
    if data[-1] & 0x80:
        # If negative, pad with 1s in the remaining bytes
        pad_value = (2 ** (8 * (4 - num_bytes)) - 1).to_bytes(4 - num_bytes, 'big', signed=False)
        data += pad_value
    else:
        # If positive, pad with 0s in the remaining bytes
        pad_value = b'\x00' * (4 - num_bytes)
        data += pad_value
    # Unpack the 4-byte chunk as a signed 32-bit integer
    value, = struct.unpack("i", data)
    return value


def get_data(infile):
    data_list = []
    with open(infile, "rb") as fin:
        fin_bit_rate, sample_rate, num_channels = _get_spec(fin)
        while True:
            value = _read_signed_integer(fin, int(fin_bit_rate/8))
            if value is None:  # Break the loop if there is no more data to read
                break
            data_list.append(value)
    return np.array(data_list).reshape(-1, num_channels).T, [fin_bit_rate, sample_rate, num_channels]


def mse_idx_correl(i, ref_signal, checking_section):
    most_correspondig_ref_section = ref_signal[i:i + len(checking_section)]
    return np.mean((most_correspondig_ref_section - checking_section)**2)

def find_corresponding_section(ref_signal, checking_section):
    correlation = np.correlate(ref_signal, checking_section, mode='valid')
    p = list(argrelextrema(correlation, np.greater)[0])
    index_max_correl = p[np.argmin([mse_idx_correl(i, ref_signal, checking_section) for i in p])]
    
    most_correspondig_ref_section = ref_signal[index_max_correl:index_max_correl + len(checking_section)]
    return most_correspondig_ref_section


# get data from both file
in_signal, fin_specs = get_data(args.infile)
ref_signal, ref_specs = get_data(args.reffile)
if fin_specs != ref_specs:
    print("Files have differents properties")
    exit(-1)


for i in range(fin_specs[2]):
     # select part of file used to check
    checking_section = in_signal[i, int(fin_specs[1] * args.skiped / 1000) :]
    most_correspondig_ref_section = find_corresponding_section(ref_signal[i, :], checking_section)
    if np.max(np.abs(most_correspondig_ref_section - checking_section)) > args.tolerance:
        print("Files don't match")
        exit(-1)