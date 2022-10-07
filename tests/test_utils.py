import sys
sys.path.append('.')

from utils import get_all_h_params_combo

#test to check if all hyperparameter combos are indeed getting created
def test_get_h_param_combo():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    
    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list

    h_para_comb = get_all_h_params_combo(params)

    assert len(h_para_comb) == len(gamma_list)*len(c_list)