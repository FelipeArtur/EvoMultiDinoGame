import random
import numpy as np
import pickle
from chrome_trex import MultiDinoGame, ACTION_UP, ACTION_FORWARD, ACTION_DOWN
import json

# Parâmetros do Algoritmo Genético
POPULATION_SIZE = 100  # Número de dinos na população
GENERATIONS = 100  # Número de gerações
MUTATION_RATE = 0.4  # Probabilidade de mutação
CROSSOVER_RATE = 0.5  # Probabilidade de crossover
FPS = 0 # Sem limite de FPS

# Inicializa o jogo com múltiplos dinos
game = MultiDinoGame(POPULATION_SIZE, fps=FPS)
state_size = len(game.get_state()[0])
gene_length = state_size  # O comprimento dos genes é o mesmo do estado


# Funções para Algoritmos Genéticos
def create_individual():
    """Cria um indivíduo com genes aleatórios (pesos para decisões)."""
    return np.random.uniform(-1, 1, gene_length)


def create_population():
    """Cria uma população inicial."""
    return [create_individual() for _ in range(POPULATION_SIZE)]


def get_action(weights, state):
    """Determina a ação com base no estado e nos pesos (gene)."""
    decision = np.dot(state, weights)
    if decision > 0.5:
        return ACTION_UP
    elif decision < -0.5:
        return ACTION_DOWN
    else:
        return ACTION_FORWARD


def evaluate_population(population):
    """Avalia a população jogando uma rodada com todos os indivíduos."""
    fitness_scores = np.zeros(POPULATION_SIZE)

    game.reset()
    done = False

    while not done:
        states = game.get_state()
        actions = [
            get_action(population[dino_idx], states[dino_idx])
            for dino_idx in range(POPULATION_SIZE)
        ]
        game.step(actions)

        scores = game.get_scores()
        for idx, score in enumerate(scores):
            fitness_scores[idx] = max(fitness_scores[idx], score)

        if game.game_over:
            done = True

    return fitness_scores


def crossover(parent1, parent2):
    """Realiza o cruzamento entre dois indivíduos."""
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(0, gene_length - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2


def mutate(individual):
    """Aplica mutação em um indivíduo."""
    for i in range(gene_length):
        if random.random() < MUTATION_RATE:
            individual[i] += np.random.normal(0, 0.1)  # Pequena perturbação gaussiana
    return individual


def select_parents(fitness_scores):
    """Seleciona dois pais com base na aptidão (fitness) usando torneio."""
    tournament_size = 3
    tournament = random.sample(range(len(fitness_scores)), tournament_size)
    best_individual_idx = max(tournament, key=lambda idx: fitness_scores[idx])
    return best_individual_idx


# Início da evolução
population = create_population()

for generation in range(GENERATIONS):
    print(f"Geração {generation + 1}/{GENERATIONS}")

    # Avaliação da população
    fitness_scores = evaluate_population(population)

    # Seleção dos melhores indivíduos
    new_population = []
    for _ in range(POPULATION_SIZE // 2):  # Para gerar dois filhos por vez
        parent1_idx = select_parents(fitness_scores)
        parent2_idx = select_parents(fitness_scores)

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        # Cruzamento e mutação
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.extend([child1, child2])

    # Atualiza a população
    population = new_population

# Avalia a população final para encontrar o melhor indivíduo
final_fitness_scores = evaluate_population(population)
best_individual_idx = np.argmax(final_fitness_scores)
best_individual = population[best_individual_idx]

# Salva os pesos do melhor indivíduo e os pontos de cada indivíduo em um arquivo
results = {
    "best_individual_weights": best_individual.tolist(),
    "fitness_scores": final_fitness_scores.tolist()
}

# Organiza o arquivo JSON
with open('best_individual_weights_and_scores.json', 'w') as f:
    json.dump(results, f, indent=4, sort_keys=True)

with open('best_individual_weights_and_scores.json', 'w') as f:
    json.dump(results, f)

# Fecha o jogo após as gerações
game.close()
