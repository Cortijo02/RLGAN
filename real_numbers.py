import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import gymnasium as gym
import numpy as np
import pygame
import gymnasium.utils.play
from MLP import MLP

# Define operadores de números reales

rang = (-1, 1) # al no hacerlo con clases, debemos definir el rango como variable global

def crossover_real_numbers (ind1, ind2, pcross): # devuelve el cruce (emparejamiento) de dos individuos
    ind1_copy, ind2_copy = [*ind1], [*ind2]
    for i in range(len(ind1)):
        if random.random() > pcross:
            beta = random.uniform(1e-6, 1-1e-6)
            ind1_copy[i] = beta * ind1[i] + (1 - beta) * ind2[i]
            ind2_copy[i] = beta * ind2[i] + (1 - beta) * ind1[i]

    return ind1_copy, ind2_copy


def mutate_swap_real_numbers (ind, pmut): # devuelve individuo mutado; la mutación consistirá en intercambiar elementos
    ind_copy = [*ind]
    for i in range(len(ind)):
        if random.random() < pmut:
            ind_copy[i] = random.uniform(rang[0], rang[1])
    return ind_copy

# fitness para himmelblau: valor mínimo de la función
global architecture 

def get_architecture():
    return architecture

def set_architecture(arch):
    global architecture
    architecture = arch

def fitness(ch):
    env = gym.make("LunarLander-v3", render_mode=None)
    reward_list = []
    
    for _ in range(20):  
        observation, _ = env.reset()
        racum = 0  
        # Se crea el modelo una sola vez por episodio:
        model = MLP(get_architecture())
        model.from_chromosome(ch)
        
        while True:
            action = policy(model, observation)
            # Observation: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
            observation, reward, terminated, truncated, _ = env.step(action)
            racum += reward  # Recompensa del entorno
            
            # Penalización progresiva del ángulo y velocidad angular
            racum -= abs(observation[4]) * 50  
            racum -= abs(observation[5]) * 50  
            
            # Recompensa/Penalización según la acción:
            if action == 2:  # Motor principal (vertical)
                if observation[3] < -0.1:  # Si la velocidad vertical (vy) es negativa (caída)
                    racum += 50  
                else:
                    racum += 10
            elif action in [1, 3]:  
                if abs(observation[4]) > 0.1:  # Si el ángulo es significativo
                    racum += 30  
                else:
                    racum += 5   
            elif action == 0:  # No hacer nada
                racum -= 50
                
            # Bonificación por aterrizaje exitoso (ambas piernas en contacto)
            if observation[6] == 1 and observation[7] == 1:
                racum += 20

            if terminated or truncated:
                reward_list.append(racum)
                break

    return sum(reward_list) / len(reward_list)



def policy (model, observation):
    s = model.forward(observation)
    # action = np.argmax(s)
    probabilities = np.exp(s) / np.sum(np.exp(s))  # Softmax
    action = np.random.choice(len(s), p=probabilities)
    return action

def select (pop, T): # devuelve un individuo seleccionado por torneo, devuelve una copia para evitar efectos laterales
    # pop se supone ya ordenada por fitness
    selected = [random.randint(0, len(pop)-1) for _ in range(T)]
    return [*pop[min(selected)]]

def sort_pop (pop, fit): # devuelve una tupla: la población ordenada por fitness, y la lista de fitness.
    fitness_list = list(map(fit, pop))
    sorted_pop_fitness = sorted(zip(pop, fitness_list), key=lambda x: x[1], reverse=True)
    return [x[1] for x in sorted_pop_fitness], [x[0] for x in sorted_pop_fitness]

def evolve_himmelblau (pop, fit, pmut, pcross=0.7, ngen=100, T=2, trace=0):
    initial_pop = [*pop]
    historical_best = []
    best_fitness = sys.maxsize * -1
    for i in range(ngen):
        sorted_fitnesses, sorted_pop = sort_pop(initial_pop, fit)
        current_best = sorted_pop[0]
        selected_pop = [select(sorted_pop, T) for _ in range(len(initial_pop))]

        crossed_pop = []
        for j in range(0, len(selected_pop)-1, 2):
            crossed_pop.extend(crossover_real_numbers(selected_pop[j], selected_pop[j+1], pcross))
        if len(selected_pop) % 2 != 0:
            crossed_pop.append(selected_pop[-1])
        
        mutated_pop = [mutate_swap_real_numbers(ind, pmut) for ind in crossed_pop]
        
        if  sorted_fitnesses[0] > best_fitness:
            historical_best = current_best
            best_fitness = sorted_fitnesses[0]

        initial_pop = mutated_pop

        if trace and i % trace == 0:
            print(f"[{i}] Best: {sorted_fitnesses[0]}")

    initial_pop.insert(0, historical_best)
    return initial_pop