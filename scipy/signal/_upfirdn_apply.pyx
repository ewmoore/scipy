# -*- coding: utf-8 -*-

# Code adapted from "upfirdn" python library with permission:
#
# Copyright (c) 2009, Motorola, Inc
#
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of Motorola nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cimport cython
cimport numpy as np


@cython.cdivision(True)
def _output_len(Py_ssize_t in_len,
                Py_ssize_t up,
                Py_ssize_t down,
                Py_ssize_t len_h):
    """The output length that results from a given input"""
    cdef Py_ssize_t np
    if len_h != 0:  # figure out what the padding needs to be
        # the modulo below is the C-equivalent of a Python modulo,
        # i.e. -1 % 10 = 9
        in_len = in_len + (len_h + ((-len_h % up) + up) % up) // up - 1
    np = in_len * up
    cdef Py_ssize_t need = np // down
    if np % down > 0:
        need += 1
    return need



@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
def _apply(double [:] x, Py_ssize_t len_x, double [:] out,
           double [:] h_trans_flip, Py_ssize_t h_per_phase,
           Py_ssize_t up, Py_ssize_t down):
    cdef Py_ssize_t padded_len = len_x + h_per_phase - 1
    cdef Py_ssize_t x_idx = 0
    cdef Py_ssize_t y_idx = 0
    cdef Py_ssize_t h_idx = 0
    cdef Py_ssize_t t = 0
    cdef Py_ssize_t x_ptr_idx = 0
    cdef np.float64_t acc

    while x_idx < len_x:
        acc = 0.
        h_idx = t * h_per_phase
        x_ptr_idx = x_idx - h_per_phase + 1
        if x_ptr_idx < 0:
            h_idx -= x_ptr_idx
            x_ptr_idx = 0
        for x_ptr_idx in range(x_ptr_idx, x_idx + 1):
            acc += x[x_ptr_idx] * h_trans_flip[h_idx]
            h_idx += 1
        # store and increment
        out[y_idx] = acc
        y_idx += 1
        t += down
        x_idx += t / up  # integer div
        # which phase of the filter to use
        t = t % up

    # Use a second loop to flush out the last bits
    while x_idx < padded_len:
        acc = 0.
        h_idx = t * h_per_phase
        x_ptr_idx = x_idx - h_per_phase + 1
        while x_ptr_idx <= x_idx:
            if x_ptr_idx < len_x and x_ptr_idx > 0:
                acc += x[x_ptr_idx] * h_trans_flip[h_idx]
            x_ptr_idx += 1
            h_idx += 1
        # store and increment
        out[y_idx] = acc
        y_idx += 1
        t += down
        x_idx += t / up  # integer div
        # which phase of the filter to use
        t = t % up
