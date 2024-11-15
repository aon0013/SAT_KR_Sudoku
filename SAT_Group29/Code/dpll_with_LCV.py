import numpy as np

def parse_dimacs(file_path):
    """Load clauses from a DIMACS file."""
    clauses_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(('p', 'c')):
                continue
            clause = [int(x) for x in line.strip().split() if x != '0']
            clauses_list.append(clause)
    return clauses_list

def load_puzzles(file_path):
    """Load Sudoku puzzles from a text file."""
    with open(file_path, 'r') as file:
        puzzles = [line.strip() for line in file]
    return puzzles

def sudoku_to_cnf(puzzle_string):
    """Convert a Sudoku puzzle into CNF clauses."""
    cnf_clauses = []
    for row in range(9):
        for col in range(9):
            value = puzzle_string[row * 9 + col]
            if value != '.':
                var = 100 * (row + 1) + 10 * (col + 1) + int(value)
                cnf_clauses.append([var])
    return cnf_clauses

def has_empty_clause(cnf_clauses):
    """Check if there are any empty clauses in the list."""
    return any(len(clause) == 0 for clause in cnf_clauses)

def dp_solver(cnf_clauses, assignments, stats, use_pure_literals=True):
    """Davis-Putnam solver with correct backtrack implementation."""

    # Unsatisfiability due to empty clauses
    if has_empty_clause(cnf_clauses):
        stats['conflicts'] += 1
        return False, []

    # Clauses satisfied?
    if not cnf_clauses:
        return True, assignments

    # Check for tautology and remove tautological clauses
    cnf_clauses = [c for c in cnf_clauses if not any(-lit in c for lit in c)]

    # Unit propagation
    unit_literals = [c[0] for c in cnf_clauses if len(c) == 1]
    while unit_literals:
        unit = unit_literals.pop(0)
        stats['unit_propagations'] += 1
        result = apply_literal(cnf_clauses, assignments[:], unit, stats)
        if result is None:
            # Conflict detected during unit propagation
            stats['conflicts'] += 1
            return False, []
        cnf_clauses, assignments = result  # Update clauses and assignments
        if has_empty_clause(cnf_clauses):  # Check again after propagation
            stats['conflicts'] += 1
            return False, []
        unit_literals = [c[0] for c in cnf_clauses if len(c) == 1]

    # Pure literal assignment
    if use_pure_literals:
        literals = set(lit for clause in cnf_clauses for lit in clause)
        pure_literals = [l for l in literals if -l not in literals]
        for pure in pure_literals:
            stats['pure_literals'] += 1
            result = apply_literal(cnf_clauses, assignments[:], pure, stats)
            if result is None:
                # Conflict detected during pure literal assignment
                stats['conflicts'] += 1
                return False, []
            cnf_clauses, assignments = result  # Update clauses and assignments

    # Branching
    if not cnf_clauses:
        return True, assignments  # All clauses are satisfied

    stats['decisions'] += 1

    # Use LCV to select the literal
    literal = select_lcv_literal(cnf_clauses)

    # Assigning literal to true
    result = apply_literal(cnf_clauses, assignments[:], literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        sat, final_assignments = dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)
        if sat:
            return True, final_assignments

    # Increment backtracks here since we're backtracking from a failed assignment
    stats['backtracks'] += 1 #! TODO

    # Assigning literal to false
    result = apply_literal(cnf_clauses, assignments[:], -literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        sat, final_assignments = dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)
        if sat:
            return True, final_assignments

    # Increment backtracks again since both assignments failed
    stats['backtracks'] += 1

    # Conflict detected in both assignments
    return False, []

def select_lcv_literal(cnf_clauses):
    """Select the literal using Least Constraining Value (LCV) heuristic."""
    literal_impact = {}

    # Calculate the impact of each literal on the remaining clauses
    for clause in cnf_clauses:
        for lit in clause:
            if lit not in literal_impact:
                literal_impact[lit] = 0
            literal_impact[lit] += len([c for c in cnf_clauses if -lit in c])

    # Select the literal with the minimum impact
    return min(literal_impact, key=lambda lit: literal_impact[lit])


def apply_literal(cnf_clauses, assignments, literal, stats):
    """Apply a literal by updating clauses and assignments."""
    updated_clauses = []
    new_assignments = assignments[:]
    new_assignments.append(literal)

    for clause in cnf_clauses:
        if literal in clause:
            continue  # Clause is satisfied
        elif -literal in clause:
            new_clause = [lit for lit in clause if lit != -literal]
            if not new_clause:
                # Empty clause generated, conflict
                stats['conflicts'] += 1
                return None  # Indicate conflict
            updated_clauses.append(new_clause)
        else:
            updated_clauses.append(clause)

    return updated_clauses, new_assignments

def solve_puzzles(sudoku_rules, sudoku_puzzles, output_prefix):
    """Solve multiple Sudoku puzzles and save solutions in DIMACS format."""
    rules = parse_dimacs(sudoku_rules)
    puzzles = load_puzzles(sudoku_puzzles)
    backtracker = []

    for idx, puzzle in enumerate(puzzles):
        stats = {
            'backtracks': 0,
            'decisions': 0,
            'unit_propagations': 0,
            'conflicts': 0,
            'pure_literals': 0
        }
        print(f"Solving puzzle {idx + 1}")
        puzzle_clauses = sudoku_to_cnf(puzzle)
        clauses = rules + puzzle_clauses
        assignments = []

        solvable, solution = dp_solver(clauses, assignments, stats, use_pure_literals=True)

        # Output to DIMACS
        output_filename = f"{output_prefix}_puzzle_{idx + 1}_solution.txt"
        if solvable:
            print(f"Puzzle {idx + 1} is solvable.")
            save_solution(output_filename, solution)
            print(f"Solution written to {output_filename}")
            grid = format_grid(solution)
            print("Solution Grid:")
            print_grid(grid)
        else:
            print(f"Puzzle {idx + 1} is unsolvable.")
            with open(output_filename, 'w') as f:
                f.write("UNSAT\n")
        print(f"Stats for puzzle {idx + 1}: {stats}\n")
        backtracker.append(stats['backtracks'])

    print("Backtracks per puzzle:", backtracker)
    print(f"Mean: {np.mean(backtracker)}, SD: {np.std(backtracker)}")

def save_solution(filename, solution):
    """Save solution in DIMACS format."""
    with open(filename, 'w') as file:
        file.write("c Sudoku solution\n")
        file.write("p cnf 729 0\n")
        for lit in solution:
            if lit > 0:
                file.write(f"{lit} 0\n")

def format_grid(solution):
    """Format solution as a 9x9 grid for Sudoku display."""
    grid = [['.' for _ in range(9)] for _ in range(9)]
    for lit in solution:
        if lit > 0:
            num = lit % 10
            col = (lit // 10) % 10 - 1
            row = (lit // 100) - 1
            grid[row][col] = str(num)
    return grid

def print_grid(grid):
    """Display Sudoku solution in a readable format."""
    for i, row in enumerate(grid):
        if i % 3 == 0 and i != 0:
            print("- - - + - - - + - - -")
        row_display = ''
        for j, val in enumerate(row):
            if j % 3 == 0 and j != 0:
                row_display += '| '
            row_display += f"{val} "
        print(row_display.strip())
    print("\n")

# Example usage:
solve_puzzles(
    "sudoku-rules-9x9.txt",
    "1000 sudokus.txt",
    "C:/Users/floor/Documents/GitHub/SAT_KR_Sudoku/SAT_Group29/Code/solutions_1000_LCV/solution"
)

""" 
for 50 extreme:
Backtracks per puzzle: [62, 10, 2, 2, 1, 9, 7, 1, 10, 5, 2, 1, 228, 12, 27, 20, 2, 2, 69, 74, 7, 2, 11, 5, 5, 0, 1, 0, 1, 2, 0, 6, 6, 8, 10, 15, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 3, 0, 0, 8]
Mean: 12.8, SD: 34.725206982824446

50 extreme, not the first backtrack:
Backtracks per puzzle: [29, 4, 0, 0, 0, 3, 2, 0, 3, 1, 0, 0, 112, 4, 12, 8, 0, 0, 32, 34, 2, 0, 4, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3]
Mean: 5.5, SD: 16.983815825661793

1000 puzzles, not the first backtrack:
Backtracks per puzzle: [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 9, 0, 0, 0, 5, 6, 0, 0, 0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 12, 0, 7, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 5, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 6, 8, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 1, 0, 5, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 4, 2, 0, 4, 0, 0, 0, 0, 0, 1, 0, 1, 0, 4, 0, 0, 0, 1, 0, 4, 0, 1, 0, 0, 6, 0, 9, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 0, 0, 0, 55, 0, 2, 1, 0, 0, 2, 0, 0, 3, 0, 3, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 3, 3, 21, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 8, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 7, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 12, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 9, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 2, 0, 3, 0, 0, 0, 3, 2, 3, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 5, 0, 0, 3, 0, 1, 8, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 7, 2, 2, 1, 23, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 3, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 11, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 4, 0, 3, 0, 5, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 5, 1, 0, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 2, 3, 0, 0, 1, 0, 1, 15, 3, 9, 5, 1, 2, 0, 2, 0, 1, 1, 0, 5, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 1, 0, 7, 2, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 4, 0, 0, 9, 8, 5, 4, 5, 0, 0, 0, 0, 0, 3, 1, 1, 4, 12, 3, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0]     
Mean: 0.8921859545004945, SD: 2.7122552707752683

1000 puzzles:
Backtracks per puzzle: [1, 0, 7, 0, 0, 0, 0, 0, 0, 2, 9, 0, 0, 1, 6, 7, 0, 0, 0, 0, 0, 1, 0, 0, 1, 12, 1, 8, 0, 0, 1, 0, 0, 0, 0, 2, 1, 3, 0, 6, 19, 0, 0, 1, 12, 13, 0, 0, 1, 11, 0, 0, 5, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 3, 0, 3, 0, 0, 27, 0, 15, 3, 0, 0, 0, 0, 7, 0, 0, 1, 0, 5, 0, 0, 0, 0, 1, 2, 1, 5, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 5, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 11, 1, 0, 0, 9, 0, 0, 0, 0, 1, 1, 1, 11, 0, 0, 9, 0, 1, 1, 0, 3, 0, 1, 0, 5, 0, 2, 0, 0, 0, 0, 0, 9, 3, 0, 0, 10, 0, 0, 0, 0, 0, 4, 0, 1, 5, 1, 6, 0, 0, 1, 1, 0, 1, 5, 0, 3, 3, 0, 1, 9, 0, 0, 0, 4, 0, 1, 0, 11, 3, 0, 0, 3, 1, 0, 4, 0, 2, 0, 0, 0, 5, 0, 0, 0, 7, 1, 0, 3, 0, 1, 0, 0, 3, 12, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 5, 1, 1, 7, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 3, 0, 0, 5, 1, 4, 0, 1, 1, 0, 0, 0, 16, 21, 1, 5, 1, 0, 1, 0, 0, 1, 4, 1, 0, 0, 0, 1, 1, 0, 7, 1, 0, 3, 6, 0, 9, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 4, 1, 1, 1, 1, 0, 0, 3, 4, 7, 0, 1, 0, 0, 0, 0, 0, 1, 0, 14, 0, 0, 0, 5, 1, 1, 4, 0, 12, 0, 1, 1, 1, 3, 3, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 27, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1, 3, 8, 0, 2, 1, 3, 9, 6, 1, 9, 0, 0, 0, 0, 0, 4, 0, 3, 2, 9, 0, 1, 0, 3, 0, 9, 0, 4, 2, 0, 14, 0, 19, 3, 0, 2, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 0, 9, 0, 0, 0, 0, 0, 3, 1, 6, 6, 0, 3, 1, 1, 1, 0, 3, 112, 0, 5, 3, 1, 0, 6, 1, 0, 8, 0, 7, 1, 2, 5, 0, 1, 0, 4, 1, 0, 0, 2, 5, 1, 0, 7, 1, 0, 7, 9, 44, 3, 2, 1, 1, 0, 3, 0, 11, 1, 0, 1, 27, 1, 1, 0, 0, 0, 0, 0, 19, 4, 7, 3, 0, 0, 0, 1, 0, 0, 0, 1, 11, 0, 6, 3, 5, 0, 0, 1, 0, 0, 0, 6, 0, 11, 1, 0, 0, 0, 0, 5, 1, 0, 0, 3, 0, 0, 1, 4, 1, 8, 0, 0, 1, 0, 4, 1, 1, 1, 3, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 4, 1, 0, 1, 0, 5, 0, 4, 1, 20, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 6, 1, 0, 0, 4, 7, 0, 2, 3, 5, 0, 2, 0, 1, 2, 1, 1, 1, 1, 8, 1, 15, 2, 0, 0, 1, 0, 6, 1, 0, 0, 13, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 6, 1, 5, 8, 1, 1, 2, 0, 0, 0, 0, 0, 0, 13, 3, 4, 3, 0, 3, 2, 0, 4, 0, 0, 26, 1, 1, 2, 10, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 20, 0, 0, 0, 0, 26, 0, 2, 0, 1, 0, 12, 2, 3, 11, 1, 6, 1, 9, 2, 0, 0, 7, 5, 8, 3, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 7, 4, 0, 5, 1, 1, 0, 1, 3, 0, 3, 9, 9, 1, 0, 1, 0, 0, 1, 1, 0, 10, 1, 2, 3, 1, 3, 7, 5, 1, 1, 1, 0, 2, 0, 0, 0, 3, 0, 0, 1, 3, 7, 6, 0, 0, 0, 1, 6, 3, 1, 0, 1, 0, 5, 12, 0, 1, 8, 0, 3, 18, 1, 0, 3, 1, 1, 0, 7, 0, 0, 1, 0, 6, 0, 7, 0, 1, 1, 3, 1, 1, 0, 5, 3, 2, 16, 6, 6, 3, 50, 0, 0, 0, 0, 4, 0, 1, 2, 1, 1, 1, 6, 3, 0, 5, 0, 7, 0, 0, 0, 3, 0, 2, 0, 9, 0, 1, 0, 0, 1, 24, 5, 1, 1, 0, 0, 0, 1, 6, 0, 1, 4, 1, 1, 10, 1, 7, 0, 12, 0, 0, 1, 1, 5, 4, 1, 1, 1, 3, 3, 0, 2, 12, 3, 2, 0, 1, 3, 20, 0, 0, 0, 3, 1, 0, 0, 9, 0, 1, 1, 0, 0, 6, 6, 7, 0, 0, 4, 2, 4, 32, 7, 20, 12, 3, 5, 0, 6, 1, 4, 4, 0, 12, 0, 0, 0, 5, 0, 2, 1, 6, 1, 1, 0, 0, 1, 1, 1, 7, 5, 0, 0, 2, 6, 0, 3, 1, 16, 6, 0, 0, 1, 0, 0, 0, 24, 0, 1, 0, 7, 0, 0, 0, 5, 0, 3, 0, 2, 2, 1, 5, 1, 10, 1, 0, 19, 20, 11, 10, 13, 0, 1, 1, 1, 1, 8, 3, 3, 9, 26, 7, 5, 5, 0, 1, 3, 1, 0, 0, 0, 0, 1, 1, 3, 4, 0, 0, 0, 2, 2, 0, 0, 1, 5, 0, 1, 7, 2]
Mean: 2.5410484668644906, SD: 5.814563258225178
"""
