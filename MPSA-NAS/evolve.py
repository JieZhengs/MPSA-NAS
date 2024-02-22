# !/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import numpy as np


def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)


def adapt_inertia_weight(weight_set, c1_set, c2_set, err_set, previous_err_set, best_num_parameters, best_flops,
                         gbest_err):
    Tp = float(__read_ini_file('SEARCH', 'Tp'))
    Tf = float(__read_ini_file('SEARCH', 'Tf'))
    wp = list(map(float, __read_ini_file('SEARCH', 'wp').split(',')))
    wf = list(map(float, __read_ini_file('SEARCH', 'wf').split(',')))
    wa = list(map(float, __read_ini_file('SEARCH', 'wa').split(',')))
    print('动态自适应更改pso权重中....')
    for i in range(0, len(weight_set)):
        c1_set[i] *= pow(Tp / best_num_parameters[i], wp[int(bool(best_num_parameters[i] > Tp))]) * pow(
            Tf / best_flops[i],
            wf[int(bool(best_flops[i] > Tf))])
        c2_set[i] *= pow((1 - gbest_err) / (1 - err_set[i]), wa[int(bool((1 - err_set[i]) > (1 - gbest_err)))])
        weight_set[i] *= pow((1 - gbest_err) / (1 - err_set[i]), wa[int(bool((1 - err_set[i]) > (1 - gbest_err)))])


    return weight_set, c1_set, c2_set


def aconpso(adapt_iteration, weight_set, c1_set, c2_set, particle, gbest, pbest, velocity, params, previous_err_set,
            err_set, best_num_parameters, gbest_err, best_flops, curr_gen):
    """
    pso for architecture evolution
    fixed-length PSO, use standard formula, but add a strided layer number constraint
    """
    particle_length = params['particle_length']
    max_output_channel = params['max_output_channel']
    cur_len = len(particle)

    if curr_gen >= adapt_iteration:
        weight_set, c1_set, c2_set = adapt_inertia_weight(weight_set, c1_set, c2_set, err_set, previous_err_set,
                                                          best_num_parameters, best_flops, gbest_err)
    r1 = np.random.random(cur_len)
    r2 = np.random.random(cur_len)
    new_velocity = np.asarray(velocity) * np.asarray(weight_set) + np.asarray(c1_set) * r1 * (
            np.asarray(pbest) - np.asarray(particle)) + np.asarray(c2_set) * r2 * (
                           np.asarray(gbest) - np.asarray(particle))

    new_particle = list(particle + new_velocity)
    new_particle = [round(par, 2) for par in new_particle]  
    new_velocity = list(new_velocity)
    subparticle_length = particle_length // 3
    subParticles = [new_particle[0:subparticle_length], new_particle[subparticle_length:2 * subparticle_length],
                    new_particle[2 * subparticle_length:]]

    for j, subParticle in enumerate(subParticles):
        valid_particle = [dimen for dimen in subParticle if 0 <= dimen <= 12.99]
        if len(valid_particle) == 0:
            new_particle[j * subparticle_length] = 0.00

    updated_particle1 = []
    for k, par in enumerate(new_particle):
        if (0.00 <= par <= 12.99):
            updated_particle1.append(par)
        elif par > 12.99:
            updated_particle1.append(12.99)
        else:
            updated_particle1.append(0.00)

    updated_particle = []
    for k, par in enumerate(updated_particle1):
        if int(round(par - int(par), 2) * 100) + 1 > max_output_channel:
            updated_particle.append(round(int(par) + float(max_output_channel - 1) / 100, 2))
        else:
            updated_particle.append(par)

    return updated_particle, new_velocity, weight_set, c1_set, c2_set
