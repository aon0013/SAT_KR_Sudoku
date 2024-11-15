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
    """Davis-Putnam solver with MOM heuristic for literal selection."""

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

    # MOM heuristic for literal selection
    # Step 1: Find the smallest clause size (minimum k)
    min_size = min(len(clause) for clause in cnf_clauses)
    smallest_clauses = [clause for clause in cnf_clauses if len(clause) == min_size]

    # Step 2: Count occurrences of literals in these clauses
    literal_counts = {}
    for clause in smallest_clauses:
        for literal in clause:
            literal_counts[literal] = literal_counts.get(literal, 0) + 1

    # Step 3: Calculate MOM scores
    scores = {}
    for literal in literal_counts:
        pos_count = literal_counts.get(literal, 0)
        neg_count = literal_counts.get(-literal, 0)
        scores[literal] = (pos_count + neg_count) * (2 ** min_size) + pos_count * neg_count

    # Step 4: Select the literal with the highest score
    literal = max(scores, key=scores.get)

    # Assigning literal to true
    result = apply_literal(cnf_clauses, assignments[:], literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        sat, final_assignments = dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)
        if sat:
            return True, final_assignments
    
    # Increment backtracks here since we're backtracking from a failed assignment
    #stats['backtracks'] += 1 # !TODO 

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
    "C:/Users/floor/Documents/GitHub/SAT_KR_Sudoku/SAT_Group29/Code/solutions_1000_MOM/solution"
)

"""
for 50 extreme:
Backtracks per puzzle: [195, 76, 0, 14, 589, 1, 23, 237, 612, 0, 188, 356, 575, 731, 303, 10, 1, 19, 641, 3126, 22, 378, 305, 344, 67, 0, 497, 0, 2, 418, 1, 119, 1, 564, 228, 19, 0, 3, 1, 0, 8, 0, 0, 0, 0, 1, 0, 0, 25, 4]
Mean: 214.08, SD: 469.0206323819881


50 extreme, not the first backtrack:
Backtracks per puzzle: [97, 36, 0, 6, 293, 0, 11, 117, 305, 0, 93, 177, 287, 364, 150, 4, 0, 9, 319, 1562, 9, 188, 152, 171, 32, 0, 247, 0, 0, 208, 0, 58, 0, 281, 113, 9, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 12, 1]
Mean: 106.3, SD: 234.3451514326678


1000 with both backtracks:
Backtracks per puzzle: [1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 119, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 21, 0, 0, 0, 0, 1, 0, 2, 0, 0, 7, 0, 1, 0, 0, 91, 0, 0, 2, 0, 11, 0, 0, 45, 0, 0, 0, 5, 0, 1, 24, 0, 0, 8, 4, 43, 0, 1, 0, 0, 8, 0, 12, 0, 0, 7, 1, 13, 1, 0, 0, 0, 0, 19, 3, 8, 10, 0, 1, 0, 1, 1, 0, 0, 3, 1, 2, 1, 0, 0, 24, 2, 1, 1, 0, 0, 0, 4, 1, 0, 0, 0, 5, 89, 0, 1, 0, 25, 0, 17, 0, 0, 0, 2, 5, 3, 0, 21, 1, 5, 1, 0, 4, 1, 8, 1, 14, 1, 5, 0, 0, 6, 0, 4, 31, 1, 0, 0, 2, 0, 8, 0, 14, 28, 0, 0, 6, 2, 9, 3, 0, 1, 6, 0, 0, 0, 0, 0, 59, 0, 5, 8, 3, 4, 0, 0, 7, 19, 0, 14, 1, 1, 8, 1, 0, 0, 65, 0, 0, 0, 3, 0, 0, 0, 5, 16, 1, 1, 5, 2, 0, 1, 3, 3, 9, 0, 0, 1, 0, 0, 2, 0, 15, 3, 1, 0, 18, 1, 1, 1, 7, 0, 0, 0, 5, 27, 0, 0, 2, 0, 0, 0, 0, 6, 3, 0, 0, 0, 0, 0, 0, 26, 0, 0, 2, 0, 0, 3, 2, 14, 23, 1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 12, 14, 0, 6, 1, 0, 1, 0, 7, 16, 78, 1, 0, 1, 1, 2, 0, 0, 53, 5, 0, 0, 1, 1, 36, 0, 1, 0, 0, 0, 6, 0, 1, 7, 45, 0, 3, 0, 5, 0, 26, 29, 1, 7, 0, 1, 1, 1, 42, 8, 0, 0, 0, 4, 0, 10, 8, 2, 0, 0, 3, 1, 1, 17, 39, 0, 38, 8, 1, 1, 11, 4, 13, 0, 23, 0, 0, 1, 1, 1, 4, 1, 0, 1, 0, 0, 1, 0, 13, 2, 0, 0, 1, 5, 0, 0, 3, 1, 2, 0, 11, 0, 0, 5, 0, 0, 4, 0, 1, 0, 6, 1, 14, 1, 1, 0, 1, 0, 5, 0, 1, 0, 1, 0, 0, 0, 0, 3, 1, 28, 0, 21, 8, 1, 34, 1, 0, 1, 2, 0, 5, 12, 0, 4, 1, 1, 1, 16, 0, 1, 1, 0, 0, 5, 1, 12, 1, 4, 1, 6, 24, 2, 1, 1, 0, 0, 15, 0, 8, 0, 0, 2, 1, 0, 21, 3, 8, 0, 0, 1, 0, 0, 3, 1, 39, 19, 6, 5, 4, 0, 0, 0, 18, 17, 0, 1, 5, 2, 1, 8, 1, 1, 2, 0, 2, 0, 2, 0, 3, 0, 0, 3, 2, 1, 1, 3, 4, 0, 1, 1, 2, 0, 10, 4, 5, 3, 4, 1, 0, 3, 0, 0, 0, 7, 0, 0, 10, 1, 20, 0, 7, 19, 0, 0, 83, 0, 5, 0, 0, 1, 0, 0, 1, 0, 0, 20, 19, 0, 10, 0, 0, 15, 0, 2, 0, 0, 0, 6, 0, 0, 0, 0, 2, 3, 2, 4, 1, 0, 51, 0, 5, 9, 1, 1, 0, 6, 1, 0, 0, 25, 1, 1, 15, 0, 2, 0, 1, 17, 16, 1, 0, 0, 0, 0, 13, 0, 0, 0, 1, 16, 2, 0, 4, 4, 9, 1, 0, 8, 0, 0, 1, 77, 0, 0, 19, 0, 2, 1, 0, 0, 1, 28, 0, 5, 0, 2, 17, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2, 18, 9, 0, 18, 0, 0, 27, 0, 2, 1, 0, 1, 0, 1, 0, 7, 0, 11, 0, 1, 4, 3, 15, 0, 17, 17, 0, 1, 2, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 2, 0, 61, 1, 1, 1, 0, 13, 70, 0, 46, 0, 24, 28, 7, 1, 11, 24, 0, 0, 16, 0, 8, 1, 1, 12, 1, 1, 9, 1, 24, 10, 0, 1, 1, 7, 0, 0, 20, 0, 1, 103, 1, 0, 0, 5, 2, 2, 9, 1, 0, 37, 15, 0, 1, 0, 14, 1, 1, 11, 2, 114, 0, 5, 0, 0, 9, 2, 2, 1, 5, 0, 0, 4, 0, 18, 0, 2, 3, 1, 2, 0, 5, 23, 1, 2, 1, 1, 2, 7, 1, 6, 1, 32, 15, 0, 17, 0, 4, 10, 2, 3, 2, 2, 0, 12, 3, 5, 1, 0, 1, 2, 15, 11, 3, 0, 1, 25, 16, 13, 25, 1, 1, 0, 1, 3, 1, 29, 0, 52, 0, 36, 1, 1, 7, 11, 1, 11, 13, 13, 12, 1, 0, 2, 1, 43, 0, 0, 6, 0, 2, 38, 1, 0, 0, 0, 9, 35, 16, 15, 3, 5, 17, 5, 44, 3, 1, 85, 0, 0, 0, 0, 0, 30, 1, 25, 1, 1, 16, 44, 0, 0, 7, 0, 13, 25, 0, 1, 17, 0, 3, 0, 0, 0, 1, 1, 4, 37, 17, 10, 2, 0, 5, 0, 11, 1, 11, 0, 2, 0, 1, 2, 51, 1, 1, 2, 65, 5, 0, 11, 47, 2, 1, 3, 0, 1, 1, 1, 1, 13, 4, 1, 2, 0, 1, 31, 1, 5, 0, 0, 0, 49, 0, 5, 1, 0, 5, 1, 1, 0, 100, 11, 1, 0, 3, 6, 3, 54, 0, 4, 60, 27, 1, 2, 75, 21, 13, 3, 31, 11, 0, 0, 0, 0, 10, 1, 25, 5, 44, 1, 0, 15, 3, 6, 0, 3, 17, 2, 5, 30, 1, 3, 0, 2, 2, 5, 0, 0, 19, 2, 0, 0, 9, 63, 0, 6, 10, 2, 5, 0, 10, 1, 1, 2, 1, 3, 2, 0, 0, 8, 7, 25, 0, 2, 23, 5, 13, 6, 0, 22, 7, 0, 11, 2, 0, 23, 37, 28, 10, 0, 11, 0, 9, 2, 1, 0, 59, 1, 7, 4, 1, 7, 1, 1, 1, 0, 24, 31, 0, 0, 0, 3, 0, 7, 0, 10]
Mean: 6.425321463897132, SD: 13.826861848452316

1000, not the first backtrack:
Backtracks per puzzle: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 44, 0, 0, 0, 0, 4, 0, 0, 22, 0, 0, 0, 2, 0, 0, 10, 0, 0, 2, 1, 20, 0, 0, 0, 0, 3, 0, 4, 0, 0, 2, 0, 6, 0, 0, 0, 0, 0, 9, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 43, 0, 0, 0, 12, 0, 8, 0, 0, 0, 0, 2, 0, 0, 10, 0, 1, 0, 0, 1, 0, 3, 0, 6, 0, 2, 0, 0, 2, 0, 1, 15, 0, 0, 0, 0, 0, 2, 0, 6, 13, 0, 0, 2, 0, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 29, 0, 2, 2, 1, 1, 0, 0, 3, 8, 0, 6, 0, 0, 3, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 7, 0, 0, 2, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 2, 12, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 1, 0, 6, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 2, 0, 0, 0, 0, 3, 6, 37, 0, 0, 0, 0, 0, 0, 0, 26, 1, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 2, 0, 0, 3, 21, 0, 1, 0, 1, 0, 12, 14, 0, 3, 0, 0, 0, 0, 20, 3, 0, 0, 0, 1, 0, 4, 2, 0, 0, 0, 0, 0, 0, 8, 19, 0, 18, 3, 0, 0, 4, 1, 6, 0, 10, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 13, 0, 9, 3, 0, 16, 0, 0, 0, 0, 0, 2, 5, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2, 0, 5, 0, 0, 0, 2, 10, 0, 0, 0, 0, 0, 7, 0, 3, 0, 0, 0, 0, 0, 9, 1, 1, 0, 0, 0, 0, 0, 1, 0, 18, 9, 2, 1, 1, 0, 0, 0, 8, 8, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 2, 1, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 9, 0, 2, 9, 0, 0, 40, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 25, 0, 1, 3, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 7, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 1, 1, 4, 0, 0, 2, 0, 0, 0, 37, 0, 0, 9, 0, 0, 0, 0, 0, 0, 13, 0, 2, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0, 8, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 1, 1, 7, 0, 7, 8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 6, 33, 0, 22, 0, 11, 13, 3, 0, 4, 11, 0, 0, 7, 0, 3, 0, 0, 5, 0, 0, 3, 0, 10, 3, 0, 0, 0, 3, 0, 0, 8, 0, 0, 50, 0, 0, 0, 2, 0, 0, 4, 0, 0, 17, 6, 0, 0, 0, 6, 0, 0, 5, 0, 56, 0, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, 1, 0, 7, 0, 0, 1, 0, 0, 0, 2, 11, 0, 0, 0, 0, 0, 3, 0, 2, 0, 15, 7, 0, 7, 0, 1, 4, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 7, 5, 1, 0, 0, 12, 7, 6, 11, 0, 0, 0, 0, 0, 0, 14, 0, 25, 0, 17, 0, 0, 2, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 21, 0, 0, 2, 0, 0, 17, 0, 0, 0, 0, 4, 17, 7, 6, 0, 2, 8, 2, 20, 1, 0, 40, 0, 0, 0, 0, 0, 14, 0, 11, 0, 0, 7, 21, 0, 0, 2, 0, 6, 12, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 18, 8, 4, 0, 0, 2, 0, 5, 0, 5, 0, 0, 0, 0, 0, 24, 0, 0, 0, 32, 2, 0, 5, 22, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 14, 0, 2, 0, 0, 0, 24, 0, 2, 0, 0, 2, 0, 0, 0, 49, 5, 0, 0, 1, 2, 1, 26, 0, 1, 28, 12, 0, 0, 37, 10, 6, 1, 15, 5, 0, 0, 0, 0, 4, 0, 11, 0, 21, 0, 0, 7, 0, 1, 0, 0, 7, 0, 2, 14, 0, 1, 0, 0, 0, 2, 0, 0, 9, 0, 0, 0, 4, 31, 0, 1, 3, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 11, 0, 0, 10, 2, 6, 1, 0, 10, 2, 0, 5, 0, 0, 11, 18, 12, 4, 0, 4, 0, 3, 0, 0, 0, 28, 0, 3, 1, 0, 2, 0, 0, 0, 0, 11, 15, 0, 0, 0, 0, 0, 3, 0, 3]
Mean: 2.67457962413452, SD: 6.681975748045493
"""



