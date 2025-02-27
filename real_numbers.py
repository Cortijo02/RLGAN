import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import gymnasium as gym
import numpy as np
import pygame
import gymnasium.utils.play
from MLP import MLP
from tqdm import tqdm
import concurrent.futures

from loky import get_reusable_executor


EXECUTOR = get_reusable_executor()

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

def simulated_binary_crossover(ind1, ind2, pcross, eta=2):
    ind1_copy, ind2_copy = [*ind1], [*ind2]
    for i in range(len(ind1)):
        if random.random() < pcross:
            u = random.random()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            ind1_copy[i] = 0.5 * ((1 + beta) * ind1[i] + (1 - beta) * ind2[i])
            ind2_copy[i] = 0.5 * ((1 - beta) * ind1[i] + (1 + beta) * ind2[i])
    return ind1_copy, ind2_copy

def blend_crossover(ind1, ind2, pcross, alpha=0.5):
    ind1_copy, ind2_copy = [*ind1], [*ind2]
    for i in range(len(ind1)):
        if random.random() < pcross:
            gamma = (1 + 2 * alpha) * random.random() - alpha
            ind1_copy[i] = (1 - gamma) * ind1[i] + gamma * ind2[i]
            ind2_copy[i] = gamma * ind1[i] + (1 - gamma) * ind2[i]
    return ind1_copy, ind2_copy

def polynomial_mutation(ind, pmut, eta=2):
    ind_copy = [*ind]
    for i in range(len(ind)):
        if random.random() < pmut:
            u = random.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            ind_copy[i] += delta
    return ind_copy


def gaussian_mutation(ind, pmut, sigma=0.2): # Probar 0.05
    ind_copy = [*ind]
    for i in range(len(ind)):
        if random.random() < pmut:
            ind_copy[i] += random.gauss(0, sigma)
    return ind_copy

def random_mutation(ind, pmut):
    options = [polynomial_mutation, gaussian_mutation]

    return random.choice(options)(ind, pmut)


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

def fitness (ch):
    env = gym.make("LunarLander-v3", render_mode=None)

    rewards_list = []
    # En las diapos pone *3*
    for _ in range(3):
        observation, _ = env.reset()
        racum = 0
        while True:
            model = MLP(get_architecture())
            model.from_chromosome(ch)
            action = policy(model, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            # reward = custom_reward(observation, action, terminated)
            racum += reward

            if terminated or truncated:
                rewards_list.append(racum)
                break
    
    return sum(rewards_list) / len(rewards_list)

def show(ind):
    env = gym.make("LunarLander-v3", render_mode="human")

    observation, _ = env.reset()
    iters = 0
    while True:
        model = MLP(get_architecture())
        model.from_chromosome(ind)
        action = policy(model, observation)
        observation, _, terminated, truncated, _ = env.step(action)

        if any([truncated, terminated]):
            observation, _ = env.reset()
            break

    env.close()

"""
def policy (model, observation):
    s = model.forward(observation)
    action = np.argmax(s)
    return action
"""

def policy(model, observation, epsilon=0.01):
    """
    ε-greedy policy: selects the optimal action with probability (1 - epsilon)
    and a random action with probability epsilon.
    
    Args:
    - model: the model with a forward method to predict action values.
    - observation: the current input (observed state).
    - epsilon: exploration probability (between 0 and 1).

    Returns:
    - action: the selected action.
    """
    # Copiada de otro grupo
    s = model.forward(observation) 
    if np.random.rand() < epsilon:  
        action = np.random.randint(len(s))
    else: 
        action = np.argmax(s)
    return action


def select (pop, T): # devuelve un individuo seleccionado por torneo, devuelve una copia para evitar efectos laterales
    # pop se supone ya ordenada por fitness
    selected = [random.randint(0, len(pop)-1) for _ in range(T)]
    return [*pop[min(selected)]]

def sort_pop (pop, fit): # devuelve una tupla: la población ordenada por fitness, y la lista de fitness.
    fitness_list = EXECUTOR.map(fit, pop)
    sorted_pop_fitness = sorted(zip(pop, fitness_list), key=lambda x: x[1], reverse=True)
    return [x[1] for x in sorted_pop_fitness], [x[0] for x in sorted_pop_fitness]

def evolve_himmelblau (pop, fit, pmut, pcross=0.7, ngen=100, T=2, trace=0):
    initial_pop = [*pop]
    historical_best = []
    best_fitness = sys.maxsize * -1
    pbar = tqdm(range(ngen), desc="Processing")
    for i in pbar:
        sorted_fitnesses, sorted_pop = sort_pop(initial_pop, fit)
        current_best = sorted_pop[0]
        selected_pop = [select(sorted_pop, T) for _ in range(len(initial_pop))]

        crossed_pop = []
        for j in range(0, len(selected_pop)-1, 2):
            crossed_pop.extend(simulated_binary_crossover(selected_pop[j], selected_pop[j+1], pcross))
        if len(selected_pop) % 2 != 0:
            crossed_pop.append(selected_pop[-1])
        
        mutated_pop = [random_mutation(ind, pmut) for ind in crossed_pop]
        
        if  sorted_fitnesses[0] > best_fitness:
            show(current_best)
            historical_best = current_best
            best_fitness = sorted_fitnesses[0]
            np.save("current_best_chromosome.npy", historical_best)
            np.save("current_best_architecture.npy", architecture)
            # print(f"[{i:>4}] New Best: {best_fitness:>5.2f}")

        initial_pop = mutated_pop
        # if trace and i % trace == 0:
            # print(f"[{i:>4}] Best:     {sorted_fitnesses[0]:>5.2f}")
        
        pbar.set_postfix(current_best=sorted_fitnesses[0], best_fitness=best_fitness)


    initial_pop.insert(0, historical_best)
    return initial_pop

def custom_reward(state, action, done):
    """
    Custom reward function for the Lunar Lander environment.
    
    Parameters:
        state (list): The state vector from the environment.
        action (int): The action taken by the agent.
        done (bool): Whether the episode has ended.
    
    Returns:
        float: The computed reward.
    """
    # Extract state variables
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, left_leg, right_leg = state

    # Reward Components
    reward = 0.0

    # **1. Distance to Landing Pad (Encourage moving towards the center)**
    reward -= 10 * abs(pos_x)  # Penalize being far from the landing zone

    # **2. Stable Descent (Encourage slow and steady landings)**
    reward -= 100 * abs(vel_x)  # Penalize horizontal velocity
    reward -= 100 * abs(vel_y)  # Penalize vertical velocity

    # **3. Proper Orientation (Encourage upright landing)**
    reward -= 50 * abs(angle)  # Penalize tilting too much

    # **4. Leg Contact (Encourage a safe landing)**
    if left_leg or right_leg:
        reward += 100  # Bonus for making contact safely

    # **6. Successful Landing Bonus**
    if done:
        if abs(pos_x) < 0.1 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1 and abs(angle) < 0.1:
            reward += 1000  # Big reward for a safe landing
        else:
            reward -= 500  # Big penalty for a crash

    return reward
