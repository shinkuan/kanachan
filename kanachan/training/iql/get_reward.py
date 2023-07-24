from __future__ import print_function
from ctypes import *
import pathlib
from typing import Optional
import torch

so =  str(pathlib.Path(__file__).parent.resolve()) + "/get_reward.so"
lib = cdll.LoadLibrary(so)

# define class GoSlice to map to:
# C type struct { void *data; GoInt len; GoInt cap; }
class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_void_p)), ("len", c_longlong), ("cap", c_longlong)]

# sparse = GoSlice((c_void_p * 29)(4, 6, 9, 11, 15, 24, 208, 286, 291, 308, 309, 328, 332, 345, 350, 353, 362, 366, 377, 381, 389, 394, 395, 409, 413, 442, 445, 449, 515), 29, 29) 
# numeric = GoSlice((c_void_p * 6)(0, 0, 24000, 29000, 23000, 24000), 6, 6) 
# progression = GoSlice((c_void_p * 71)(0, 157, 345, 581, 131, 281, 425, 569, 131, 297, 421, 593, 149, 277, 373, 583, 145, 265, 375, 523, 53, 173, 345, 565, 61, 241, 347, 519, 41, 253, 409, 952, 493, 19, 217, 391, 515, 125, 167, 335, 499, 15, 1318, 465, 145, 209, 319, 455, 135, 281, 417, 511, 147, 269, 423, 519, 113, 773, 293, 447, 535, 101, 237, 357, 587, 141, 207, 435, 527, 91, 289), 71, 71) 
# option = GoSlice((c_void_p * 15)(0, 12, 16, 28, 32, 40, 48, 64, 68, 100, 104, 105, 106, 107, 108), 15, 15) 
# action = GoSlice((c_void_p * 1)(12), 1, 1) 

lib.getReward.argtypes = [GoSlice, GoSlice, GoSlice, GoSlice, GoSlice]
lib.getReward.restype = c_longlong

# print("getReward = %d" % lib.getReward(sparse, numeric, progression, option, action))

def get_reward(
    sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
    candidate: torch.Tensor, index: torch.Tensor, game_rank: Optional[int],
    game_score: Optional[int]) -> float:

    l_sparse = sparse.tolist()
    _sparse = GoSlice((c_void_p * len(l_sparse))(), len(l_sparse), len(l_sparse))
    _sparse._fields_[0][:] = l_sparse

    l_numeric = numeric.tolist()
    _numeric = GoSlice((c_void_p * len(l_numeric))(), len(l_numeric), len(l_numeric))
    _numeric._fields_[0][:] = l_numeric

    l_progression = progression.tolist()
    _progression = GoSlice((c_void_p * len(l_progression))(), len(l_progression), len(l_progression))
    _progression._fields_[0][:] = l_progression

    l_candidate = candidate.tolist()
    _candidate = GoSlice((c_void_p * len(l_candidate))(), len(l_candidate), len(l_candidate))
    _candidate._fields_[0][:] = l_candidate

    l_index = index.tolist()
    _index = GoSlice((c_void_p * len(l_index))(), len(l_index), len(l_index))
    _index._fields_[0][:] = l_index

    reward = lib.getReward(_sparse, _numeric, _progression, _candidate, _index)

    if game_rank is not None:
        if game_rank == 1:
            reward += 10
        elif game_rank == 2:
            reward += 4
        elif game_rank == 3:
            reward -=4
        elif game_rank == 4:
            reward -=10

    if game_score is not None:
        if game_score > 25000:
            reward += 4

    return reward



