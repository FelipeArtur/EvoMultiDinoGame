# EvoMultiDinoGame

### Descrição

O código implementa o processo de treinamento do jogo do dinossauro usando Algoritmo Genético (AG). O objetivo do treinamento é otimizar a performance de vários dinossauros controlados por IA ao longo de múltiplas gerações, onde cada geração representa uma população de dinossauros que joga o jogo.


### Processo de Treinamento

#### Criação da População Inicial

Uma população de dinossauros (definida pelo parâmetro `POPULATION_SIZE`) é criada. Cada dinossauro é representado por um conjunto de pesos (genes), que são vetores aleatórios. Esses pesos determinam as ações que o dinossauro deve tomar em diferentes situações do jogo.
```python
def create_individual():
    """Cria um indivíduo com genes aleatórios (pesos para decisões)."""
    return np.random.uniform(-1, 1, gene_length)


def create_population():
    """Cria uma população inicial."""
    return [create_individual() for _ in range(POPULATION_SIZE)]
```

#### Avaliação

Cada dinossauro na população é testado jogando o jogo. A função `get_action()` usa o produto interno entre o estado atual do jogo (um vetor de características do ambiente) e os pesos (genes) do dinossauro para determinar a ação (pular, abaixar ou continuar correndo). A pontuação (fitness) de cada dinossauro é monitorada durante o jogo, e a pontuação máxima alcançada é registrada para determinar o desempenho daquele indivíduo.
```python
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
```

#### Seleção dos Melhores

Após cada rodada do jogo, os dinossauros são avaliados com base em suas pontuações. O código usa seleção por torneio, onde um subconjunto aleatório de indivíduos é escolhido, e o melhor entre eles é selecionado como um dos pais para a próxima geração.
``` python
def select_parents(fitness_scores):
    """Seleciona dois pais com base na aptidão (fitness) usando torneio."""
    tournament_size = 3
    tournament = random.sample(range(len(fitness_scores)), tournament_size)
    best_individual_idx = max(tournament, key=lambda idx: fitness_scores[idx])
    return best_individual_idx
```

#### Cruzamento (Crossover)

Dois dinossauros selecionados (pais) são combinados para gerar dois filhos. Um ponto de cruzamento é escolhido aleatoriamente, e parte dos genes de um pai é combinada com parte dos genes do outro. Esse processo simula a troca de informações entre indivíduos para gerar diversidade.
``` python
def crossover(parent1, parent2):
    """Realiza o cruzamento entre dois indivíduos."""
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(0, gene_length - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2
```

#### Mutação

Após o cruzamento, os genes dos filhos podem sofrer mutações aleatórias, introduzindo pequenas mudanças nos valores dos pesos. Isso é feito com uma pequena probabilidade (definida por `MUTATION_RATE`), adicionando variabilidade ao processo de evolução e permitindo que novas soluções sejam exploradas.
``` python
def mutate(individual):
    """Aplica mutação em um indivíduo."""
    for i in range(gene_length):
        if random.random() < MUTATION_RATE:
            individual[i] += np.random.normal(0, 0.1)  # Pequena perturbação gaussiana
    return individual
```

#### Nova Geração

A nova população é formada com base nos filhos gerados pelos pais da geração anterior. O processo é repetido por determinadas gerações (definido por `GENERATIONS`), com a expectativa de que, com o tempo, os dinossauros evoluam para desempenhos melhores, adaptando-se ao ambiente do jogo.
``` python
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
```

#### Encerramento

Após o número especificado de gerações, o treinamento é finalizado, e o jogo é encerrado.

### Vantagens do Algoritmo Genético

#### Sem Gradientes Necessários

Ao contrário de redes neurais com backpropagation, os Algoritmos Genéticos não precisam de derivadas ou gradientes. Isso os torna uma boa escolha para problemas onde a função de custo não é contínua ou diferenciável, como é o caso de alguns jogos.

#### Exploração de Múltiplas Soluções

Algoritmos Genéticos exploram o espaço de soluções de maneira ampla, já que envolvem uma população inteira de candidatos, ao invés de se focar em melhorar um único indivíduo como ocorre no Aprendizado por Reforço tradicional.

#### Adaptação e Diversidade

O uso de mutação e crossover permite que novas soluções sejam descobertas e evita o problema de convergência prematura para mínimos locais. A diversidade gerada pela mutação possibilita a exploração de diferentes partes do espaço de soluções.

#### Simples de Implementar

Algoritmos Genéticos são conceitualmente mais simples e fáceis de implementar em comparação com métodos de aprendizado de máquina baseados em gradientes, o que os torna uma abordagem prática para problemas complexos, onde modelar diretamente a função de fitness ou a dinâmica do problema é difícil.


#### Comparação com Outras Abordagens de IA

### Aprendizado por Reforço (RL)

No RL tradicional (como Q-Learning ou DQN), um agente aprende a partir de feedbacks de recompensa e usa gradientes para otimizar sua política de ações. Embora isso possa ser mais eficiente em termos de otimização, é mais complexo de configurar, requer uma função de recompensa explícita e pode sofrer com problemas de convergência local.

### Redes Neurais com Backpropagation

Redes neurais treinadas com backpropagation exigem uma função de custo diferenciável. Se o ambiente do problema não for contínuo ou suave, pode ser difícil aplicar gradientes. Além disso, redes neurais exigem dados rotulados ou feedback contínuo de recompensas para treinar, o que pode não ser tão fácil de obter em alguns cenários.

Portanto, o uso de Algoritmos Genéticos neste código traz simplicidade e eficácia, explorando amplamente as soluções possíveis e permitindo o uso em cenários onde os métodos tradicionais podem falhar ou ser difíceis de implementar.

# Apresentação 

<video width="900" height="400" controls>
  <source src="https://youtu.be/3FTPhwXqAtc" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Instalando Dependências
```
poetry install
```
## Rodando Código
```
poetry run src/app/main.py
```