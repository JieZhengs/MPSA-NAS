# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from utils_ghostv2 import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import aconpso
import copy, os, time
import configparser


def create_directory():
    dirs = ['./log', './populations', './scripts', './trained_models']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def fitness_evaluate(population, curr_gen):
    filenames = []
    population_downscale = []
    for i, particle in enumerate(population):
        particle_downscale = [int(dimen) + round((dimen - int(dimen) + 0.01) / 2 - 0.01, 2) if round(dimen - int(dimen),
                                                                                                     2) >= 0.02 else round(
            dimen // 1 + 0.00, 2) for dimen in particle]
        filename = decode(particle_downscale, curr_gen, i)
        filenames.append(filename)
        population_downscale.append(particle_downscale)

    err_set, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False,
                                                     population=population_downscale)
    return err_set, num_parameters, flops


def evolve(adapt_iteration, weight_set, c1_set, c2_set, population, gbest_individual, pbest_individuals, velocity_set,
           params,
           previous_err_set, err_set, best_num_parameters, gbest_err, best_flops,
           curr_gen):
    offspring = []
    new_velocity_set = []
    new_weight_set = []
    new_c1_set = []
    new_c2_set = []
    for i, particle in enumerate(population):
        new_particle, new_velocity, new_weight, new_c1, new_c2 = aconpso(adapt_iteration, weight_set[i], c1_set[i],
                                                                         c2_set[i], particle,
                                                                         gbest_individual,
                                                                         pbest_individuals[i],
                                                                         velocity_set[i], params,
                                                                         previous_err_set,
                                                                         err_set, best_num_parameters, gbest_err,
                                                                         best_flops,
                                                                         curr_gen)
        offspring.append(new_particle)
        new_velocity_set.append(new_velocity)
        new_weight_set.append(new_weight)
        new_c1_set.append(new_c1)
        new_c2_set.append(new_c2)
    return offspring, new_velocity_set, new_weight_set, new_c1_set, new_c2_set


def update_best_particle(population, err_set, num_parameters, flops, gbest, pbest):
    fitnessSet = [
        (1 - err_set[i]) * pow(num_parameters[i] / Tp, wp[int(bool(num_parameters[i] > Tp))]) * pow(flops[i] / Tf, wf[
            int(bool(flops[i] > Tf))]) for i in range(len(population))]
    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_errSet = copy.deepcopy(err_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness = getGbest(
            [pbest_individuals, pbest_errSet, pbest_params, pbest_flops])
    else:
        gbest_individual, gbest_err, gbest_params, gbest_flops = gbest
        pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest

        pbest_fitnessSet = [
            (1 - pbest_errSet[i]) * pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
                pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))]) for i in range(len(pbest_individuals))]

        gbest_fitness = (1 - gbest_err) * pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(
            gbest_flops / Tf, wf[int(bool(gbest_flops > Tf))])

        for i, fitness in enumerate(fitnessSet):
            if fitness > pbest_fitnessSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_errSet[i] = copy.deepcopy(err_set[i])
                pbest_params[i] = copy.deepcopy(num_parameters[i])
                pbest_flops[i] = copy.deepcopy(flops[i])
            if fitness > gbest_fitness:
                gbest_fitness = copy.deepcopy(fitness)
                gbest_individual = copy.deepcopy(population[i])
                gbest_err = copy.deepcopy(err_set[i])
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])

    return [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params,
                                                                      pbest_flops]


def getGbest(pbest):
    pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest
    gbest_err = 1.0
    gbest_params = 10e6
    gbest_flops = 10e9
    gbest = None

    gbest_fitness = (1 - gbest_err) * pow(gbest_params / Tp, wp[int(bool(gbest_params > Tp))]) * pow(gbest_flops / Tf,
                                                                                                     wf[int(bool(
                                                                                                         gbest_flops > Tf))])

    pbest_fitnessSet = [(1 - pbest_errSet[i]) * pow(pbest_params[i] / Tp, wp[int(bool(pbest_params[i] > Tp))]) * pow(
        pbest_flops[i] / Tf, wf[int(bool(pbest_flops[i] > Tf))]) for i in range(len(pbest_individuals))]

    for i, indi in enumerate(pbest_individuals):
        if pbest_fitnessSet[i] > gbest_fitness:
            gbest = copy.deepcopy(indi)
            gbest_err = copy.deepcopy(pbest_errSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
            gbest_fitness = copy.deepcopy(pbest_fitnessSet[i])
    return gbest, gbest_err, gbest_params, gbest_flops, gbest_fitness


def fitness_test(final_individual):
    final_individual = copy.deepcopy(final_individual)
    filename = Utils.generate_pytorch_file(final_individual, -1, -1)
    err_set, num_parameters, flops = fitnessEvaluate([filename], -1, True, [final_individual], [batch_size],
                                                     [weight_decay])
    return err_set[0], num_parameters[0], flops[0]


def calculate_fitness(err_set, num_parameters, flops):
    err_set_mean = np.mean(err_set)
    num_parameters_mean = np.mean(num_parameters)
    flops_mean = np.mean(flops)
    weighted_avg = (1 - err_set_mean) * 0.7 - num_parameters_mean / 1000000 * 0.15 - flops_mean / 1000000 * 0.15
    return weighted_avg


def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness（第一代初始化种群的的适应度评估）' % (gen_no))
    weight_avg = float('inf')  
    best_population = None
    best_err_set = None
    best_num_parameters = None
    best_flops = None
    for i in range(0, start_init_gen):
        population = initialize_population(params)
        print('第 %d 个population：' % i + ':' + str(population))
        err_set, num_parameters, flops = fitness_evaluate(population, -(i + 1))
        print('第 %d 个错误率：' % i + ':' + str(np.mean(err_set)))
        current_weight_avg = calculate_fitness(err_set, num_parameters,
                                               flops)
        if current_weight_avg < weight_avg:
            weight_avg = current_weight_avg
            best_err_set = err_set
            best_num_parameters = num_parameters
            best_flops = flops
            best_population = population

    print('最佳polulation：' + str(best_population))
    print('初始化第一代运行的错误率' + str(np.mean(best_err_set)))
    Log.info('EVOLVE[%d-gen]-当前代粒子群更新初始化的最佳加权适应度[ %d ],最佳初始化准确率为[ %.4f ]' % (gen_no, weight_avg, 1 - np.mean(best_err_set)))
    previous_err_set = best_err_set
    Log.info('EVOLVE[%d-gen]-Finish the evaluation（第一代初始化种群的适应度评估完成）' % (gen_no))
    [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params,
                                                               pbest_flops] = update_best_particle(best_population,
                                                                                                   best_err_set,
                                                                                                   best_num_parameters,
                                                                                                   best_flops,
                                                                                                   gbest=None,
                                                                                                   pbest=None)
    print('第一代的gbest_err' + str(gbest_err))
    print('第一代的pbest_errSet' + str(np.mean(pbest_errSet)))
    Log.info('EVOLVE[%d-gen]-Finish the updating（初始化第一代的个体最佳解、全局最佳解完成）' % (gen_no))

    Utils.save_population_and_err('population', best_population, best_err_set, best_num_parameters, best_flops, gen_no)
    Utils.save_population_and_err('pbest', pbest_individuals, pbest_errSet, pbest_params, pbest_flops, gen_no)
    Utils.save_population_and_err('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], gen_no)

    gen_no += 1
    velocity_set = []
    weight_set = []
    c1_set = []
    c2_set = []
    for ii in range(len(best_population)):
        velocity_set.append([0.01] * len(best_population[ii]))
        weight_set.append([weight] * len(best_population[ii]))
        c1_set.append([c1] * len(best_population[ii]))
        c2_set.append([c2] * len(best_population[ii]))

    print('运行前pso权重：' + str(weight_set))
    print('运行前c1权重：' + str(c1_set))
    print('运行前c2权重：' + str(c2_set))

    for curr_gen in range(gen_no, params['num_iteration']):
        params['gen_no'] = curr_gen
        Log.info('EVOLVE[%d-gen]-Begin pso evolution（当前代粒子群更新）' % (curr_gen))

        print('前一代的运行的错误率' + str(np.mean(previous_err_set)))
        print('后一代及以后的运行的错误率' + str(np.mean(best_err_set)))
        print('进化前 %d 代pso权重：' % (curr_gen) + str(weight_set))
        print('进化前 %d 代c1：' % (curr_gen) + str(c1_set))
        print('进化前 %d 代c2：' % (curr_gen) + str(c2_set))

        best_population, velocity_set, weight_set, c1_set, c2_set = evolve(adapt_iteration, weight_set, c1_set, c2_set,
                                                                           best_population,
                                                                           gbest_individual,
                                                                           pbest_individuals,
                                                                           velocity_set,
                                                                           params,
                                                                           previous_err_set,
                                                                           best_err_set, best_num_parameters, gbest_err,
                                                                           best_flops,
                                                                           curr_gen)
        print('进化后 %d 代pso权重：' % (curr_gen) + str(weight_set))
        print('进化后 %d 代c1：' % (curr_gen) + str(c1_set))
        print('进化后 %d 代c2：' % (curr_gen) + str(c2_set))

        Log.info('EVOLVE[%d-gen]-Finish pso evolution（当前代粒子群更新完成）' % (curr_gen))
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness（当前代粒子适应度评估）' % (curr_gen))
        previous_err_set = best_err_set
        best_err_set, best_num_parameters, best_flops = fitness_evaluate(best_population, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation（当前代粒子适应度评估完成）' % (curr_gen))

        [gbest_individual, gbest_err, gbest_params, gbest_flops], [pbest_individuals, pbest_errSet, pbest_params,
                                                                   pbest_flops] = update_best_particle(best_population,
                                                                                                       best_err_set,
                                                                                                       best_num_parameters,
                                                                                                       best_flops,
                                                                                                       gbest=[
                                                                                                           gbest_individual,
                                                                                                           gbest_err,
                                                                                                           gbest_params,
                                                                                                           gbest_flops],
                                                                                                       pbest=[
                                                                                                           pbest_individuals,
                                                                                                           pbest_errSet,
                                                                                                           pbest_params,
                                                                                                           pbest_flops])
        print('第%d代的gbest_err' % (curr_gen) + str(gbest_err))
        print('第%d代的pbest_errSet' % (curr_gen) + str(np.mean(pbest_errSet)))
        Log.info('EVOLVE[%d-gen]-Finish the updating（当前代个体最佳解、全局最佳解更新完成）' % (curr_gen))

        Utils.save_population_and_err('population', best_population, best_err_set, best_num_parameters, best_flops,
                                      curr_gen)
        Utils.save_population_and_err('pbest', pbest_individuals, pbest_errSet, pbest_params, pbest_flops, curr_gen)
        Utils.save_population_and_err('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end - start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    print('运行后pso权重：' + str(weight_set))
    print('运行后c1权重：' + str(c1_set))
    print('运行后c2权重：' + str(c2_set))

    search_time = str("%02dh:%02dm:%02ds" % (h, m, s))
    equipped_gpu_ids, _ = GPUTools._get_equipped_gpu_ids_and_used_gpu_info()
    num_GPUs = len(equipped_gpu_ids)

    proxy_err = copy.deepcopy(gbest_err)

    gbest_err, num_parameters, flops = fitness_test(gbest_individual)
    Log.info('Error=[%.5f], #parameters=[%d], FLOPs=[%d]' % (gbest_err, gbest_params, gbest_flops))
    Utils.save_population_and_err('final_gbest', [gbest_individual], [gbest_err], [num_parameters], [flops], -1,
                                  proxy_err, search_time + ', GPUs:%d' % num_GPUs)


def __read_ini_file(section, key):
    config = configparser.ConfigParser()
    config.read('global.ini')
    return config.get(section, key)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    create_directory()
    params = Utils.get_init_params()
    start_init_gen = int(__read_ini_file('PSO', 'start_init_gen'))
    weight = float(__read_ini_file('PSO', 'weight'))
    c1 = float(__read_ini_file('PSO', 'c1'))
    c2 = float(__read_ini_file('PSO', 'c2'))
    adapt_iteration = int(__read_ini_file('PSO', 'adapt_iteration'))
    batch_size = int(__read_ini_file('SEARCH', 'batch_size'))
    weight_decay = float(__read_ini_file('SEARCH', 'weight_decay'))
    Tp = float(__read_ini_file('SEARCH', 'Tp'))
    Tf = float(__read_ini_file('SEARCH', 'Tf'))
    wp = list(map(float, __read_ini_file('SEARCH', 'wp').split(',')))
    wf = list(map(float, __read_ini_file('SEARCH', 'wf').split(',')))

    evolveCNN(params)