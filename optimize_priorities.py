#!/usr/bin/env python3
import random
import numpy as np
import time
import sys
import os

# Try different import paths to handle different directory structures
try:
    # First try to import from source.processed
    from source.processed.modefb_data import modefb_data
    print("Imported data from source.processed.modefb_data")
except ImportError:
    try:
        # Try importing directly from processed if we're in the source directory
        from processed.modefb_data import modefb_data
        print("Imported data from processed.modefb_data")
    except ImportError:
        try:
            # If we're in the parent directory and source is in path
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from source.processed.modefb_data import modefb_data
            print("Imported data using absolute path")
        except ImportError:
            print("ERROR: Could not import modefb_data. Please make sure you're running the script from the correct directory.")
            print("The data file should be at source/processed/modefb_data.py")
            sys.exit(1)

# Constants
POPULATION_SIZE = 100
GENERATIONS = 500
EARLY_STOP = 100  # Stop when no improvement for this long
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
NUM_MODES = 77  # Modes from 0 to 76
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 5  # Number of best individuals to keep unchanged

def calculate_fitness(priority_assignment, data):
    """
    Calculate the fitness of a priority assignment.
    Fitness is the number of lines that have a value of 1.
    
    Args:
        priority_assignment: A list where index is the mode and value is the priority
        data: List of lists of (mode, value) tuples
    
    Returns:
        The number of lines with value 1
    """
    count_ones = 0
    
    for line in data:
        # Find the mode with the highest priority (lowest priority number)
        best_priority = float('inf')
        best_value = None
        
        for mode, value in line:
            if mode < len(priority_assignment):  # Ensure mode is within bounds
                priority = priority_assignment[mode]
                if priority < best_priority:
                    best_priority = priority
                    best_value = value
        
        if best_value == 1:
            count_ones += 1
    
    return count_ones

def create_random_individual():
    """Create a random priority assignment (a permutation of 0-76)"""
    # Create a list of priorities (0 to 76)
    priorities = list(range(NUM_MODES))
    # Shuffle the priorities
    random.shuffle(priorities)
    return priorities

def create_initial_population(size):
    """Create an initial population of random individuals"""
    return [create_random_individual() for _ in range(size)]

def tournament_selection(population, fitnesses, tournament_size):
    """Select an individual using tournament selection"""
    # Randomly select tournament_size individuals
    tournament_indices = random.sample(range(len(population)), tournament_size)
    # Select the best individual from the tournament
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def ordered_crossover(parent1, parent2):
    """
    Perform ordered crossover between two parents.
    This preserves the relative order of elements from each parent while ensuring
    each element appears exactly once in the offspring.
    """
    size = len(parent1)
    # Choose two random crossover points
    start, end = sorted(random.sample(range(size), 2))
    
    # Create offspring
    offspring = [-1] * size
    
    # Copy the segment between start and end from parent1
    offspring[start:end+1] = parent1[start:end+1]
    
    # Fill the remaining positions with elements from parent2 in the order they appear
    # while skipping elements that are already in the offspring
    remaining_elements = [x for x in parent2 if x not in offspring]
    
    # Fill positions before start
    for i in range(start):
        offspring[i] = remaining_elements.pop(0)
    
    # Fill positions after end
    for i in range(end+1, size):
        offspring[i] = remaining_elements.pop(0)
    
    return offspring

def swap_mutation(individual, mutation_rate):
    """
    Perform swap mutation on an individual.
    Each position has a mutation_rate chance of being swapped with another random position.
    """
    mutated = individual.copy()
    size = len(individual)
    
    for i in range(size):
        if random.random() < mutation_rate:
            # Select another position to swap with
            j = random.randint(0, size - 1)
            # Swap the values
            mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def genetic_algorithm(data):
    """
    Implement a genetic algorithm to find an optimal priority assignment
    
    Args:
        data: List of lists of (mode, value) tuples
    
    Returns:
        The best priority assignment found
    """
    # Create initial population
    population = create_initial_population(POPULATION_SIZE)
    
    best_fitness = -1
    best_individual = None
    best_generation = 0
    
    # Track progress
    progress = []
    
    start_time = time.time()
    
    # Main loop
    for generation in range(GENERATIONS):
        # Calculate fitness for each individual
        fitnesses = [calculate_fitness(ind, data) for ind in population]
        
        # Find the best individual
        current_best_index = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_index]
        current_best_individual = population[current_best_index]
        
        # Update the best individual overall
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
            best_generation = generation
            
            # Print progress
            elapsed_time = time.time() - start_time
            print(f"Generation {generation}: Best fitness = {best_fitness}, Time = {elapsed_time:.2f}s")
        
        # Save the progress
        progress.append((generation, current_best_fitness))
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        sorted_indices = np.argsort(fitnesses)[::-1]  # Sort in descending order
        for i in range(ELITISM_COUNT):
            new_population.append(population[sorted_indices[i]])
        
        # Create the rest of the new population
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            
            # Crossover
            if random.random() < CROSSOVER_RATE:
                offspring = ordered_crossover(parent1, parent2)
            else:
                offspring = parent1.copy()
            
            # Mutation
            offspring = swap_mutation(offspring, MUTATION_RATE)
            
            # Add to new population
            new_population.append(offspring)
        
        # Replace the old population
        population = new_population
        
        # Optimization: if no improvement for EARLY_STOP generations, break
        if generation - best_generation > EARLY_STOP:
            print(f"No improvement for 100 generations. Stopping at generation {generation}.")
            break
    
    return best_individual, best_fitness, progress

def validate_assignment(assignment):
    """
    Validate that the assignment is correct:
    - Each mode gets exactly one priority
    - All priorities from 0 to 76 are used exactly once
    
    Args:
        assignment: A list where index is the mode and value is the priority
    
    Returns:
        True if the assignment is valid, False otherwise
    """
    # Check length
    if len(assignment) != NUM_MODES:
        print(f"Invalid assignment length: {len(assignment)} != {NUM_MODES}")
        return False
    
    # Check that all priorities from 0 to 76 are used exactly once
    priorities_used = set(assignment)
    if len(priorities_used) != NUM_MODES:
        print(f"Not all priorities are used: {len(priorities_used)} != {NUM_MODES}")
        return False
    
    for i in range(NUM_MODES):
        if i not in priorities_used:
            print(f"Priority {i} is not used")
            return False
    
    return True

def invert_assignment(assignment):
    """
    Invert the assignment to get the mapping from priorities to modes.
    
    Args:
        assignment: A list where index is the mode and value is the priority
    
    Returns:
        A list where index is the priority and value is the mode
    """
    inverted = [-1] * len(assignment)
    for mode, priority in enumerate(assignment):
        inverted[priority] = mode
    return inverted

def main():
    print("Loading data...")
    data = modefb_data
    
    # Verify data format
    if not data or not isinstance(data, list):
        print("ERROR: Data format is incorrect. Expected a list of lists of tuples.")
        sys.exit(1)
        
    print(f"Loaded {len(data)} lines of data")
    
    print("Starting genetic algorithm...")
    best_assignment, best_fitness, progress = genetic_algorithm(data)
    
    print("\nBest assignment found:")
    print(f"Fitness: {best_fitness} out of {len(data)} lines")
    
    if validate_assignment(best_assignment):
        print("Assignment is valid!")
    else:
        print("Warning: Assignment is not valid!")
    
    # Print the assignment in a readable format
    print("\nPriority -> Mode mapping:")
    for priority in range(NUM_MODES):
        mode = invert_assignment(best_assignment)[priority]
        print(f"Priority {priority} -> Mode {mode}")
    
    # Save the results to a file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"optimization_result_{timestamp}.txt"
    
    with open(result_file, "w") as f:
        f.write(f"Optimization Results - {timestamp}\n")
        f.write(f"Fitness: {best_fitness} out of {len(data)} lines\n\n")
        
        f.write("Priority -> Mode mapping:\n")
        for priority in range(NUM_MODES):
            mode = invert_assignment(best_assignment)[priority]
            f.write(f"Priority {priority} -> Mode {mode}\n")
    
    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    main()

