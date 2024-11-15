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

def calculate_jw_weights(cnf_clauses, unassigned_vars):
    """Calculate Jeroslow-Wang weights for each unassigned variable."""
    weights = {var: 0 for var in unassigned_vars}
    
    for clause in cnf_clauses:
        clause_length = len(clause)
        for literal in clause:
            var = abs(literal)
            if var in weights:
                weights[var] += 2 ** -clause_length  # Jeroslow-Wang weighting scheme
                
    return weights






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

    # Branching with Jeroslow-Wang heuristic
    if not cnf_clauses:
        return True, assignments  # All clauses are satisfied

    stats['decisions'] += 1

    # Calculate Jeroslow-Wang weights for unassigned variables
    literals = set(abs(lit) for clause in cnf_clauses for lit in clause)
    assigned_vars = set(abs(lit) for lit in assignments)
    unassigned_vars = literals - assigned_vars
    if not unassigned_vars:
        stats['conflicts'] += 1
        return False, []  # No unassigned variables left but clauses remain

    # Select the variable with the highest Jeroslow-Wang weight
    jw_weights = calculate_jw_weights(cnf_clauses, unassigned_vars)
    literal = max(jw_weights, key=jw_weights.get)

    # Assigning literal to true
    result = apply_literal(cnf_clauses, assignments[:], literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        sat, final_assignments = dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)
        if sat:
            return True, final_assignments

    # Increment backtracks here since we're backtracking from a failed assignment
    stats['backtracks'] += 1

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
    "50 easy sudokus.txt",
    "C:/Users/floor/Documents/GitHub/SAT_KR_Sudoku/SAT_Group29/Code/solutions_easy_JW/solution.txt"
)

"""
for 50 extreme:
Backtracks per puzzle: [3202, 4043, 86601, 103380, 5034, 54930, 75021, 50052, 18182, 54016, 6130, 2885, 31323, 2436, 36835, 579, 580, 512, 35431, 157638, 45012, 5540, 250, 49228, 9110, 0, 19569, 0, 2496, 889, 4, 21277, 5839, 5818, 6473, 11530, 0, 44, 179, 9, 35, 0, 0, 0, 0, 9, 88, 0, 3704, 20]
Mean: 18318.66, SD: 31628.360678106605
"""

""""
for 1000:
Backtracks per puzzle: [5, 0, 195, 0, 0, 34, 0, 0, 0, 93, 126, 0, 17, 15, 11, 57, 0, 0, 0, 0, 0, 4, 162, 0, 5, 6, 10, 4200, 0, 0, 8, 0, 6, 0, 164, 2636, 4, 51, 0, 1, 4799, 0, 0, 11, 108, 22, 0, 0, 5, 106, 0, 0, 3, 0, 103, 370, 0, 0, 481, 616, 147, 0, 15, 0, 0, 12, 0, 233, 0, 0, 3642, 110, 83, 185, 0, 0, 0, 0, 76, 19, 145, 262, 0, 288, 0, 5, 0, 0, 8, 24, 82, 8, 192, 0, 0, 9, 15, 55, 14, 0, 0, 0, 42, 9, 0, 0, 0, 1195, 2255, 0, 14, 0, 257, 0, 3, 9, 11, 0, 2, 5, 6, 623, 384, 9, 285, 31, 0, 146, 96, 365, 24, 0, 4, 127, 0, 0, 21, 0, 35, 159, 2, 869, 23, 6, 0, 7, 0, 11, 246, 0, 0, 153, 123, 205, 3, 0, 101, 3, 4, 0, 0, 0, 0, 58, 0, 9, 5, 0, 256, 0, 0, 4, 123, 0, 132, 2, 4, 9, 12, 0, 17, 1689, 0, 0, 0, 62, 0, 28, 0, 1, 259, 5, 17, 7, 7, 0, 5, 151, 331, 313, 0, 0, 109, 0, 4, 45, 19, 364, 6, 4, 0, 10, 10, 10, 80, 570, 15, 0, 0, 17, 14, 0, 0, 19, 0, 16, 11, 0, 8, 40, 13, 65, 0, 90, 505, 0, 238, 0, 610, 15, 0, 0, 30, 9, 191, 1055, 22, 0, 0, 113, 9, 574, 0, 21, 7, 58, 0, 0, 100, 157, 10, 464, 255, 0, 42, 0, 62, 112, 1360, 8, 0, 3, 12, 1, 9, 0, 321, 40, 0, 17, 385, 165, 262, 2, 4, 0, 37, 374, 203, 0, 5, 7, 134, 0, 1, 0, 2, 13, 184, 1, 9, 266, 0, 9, 3, 0, 200, 11, 0, 9, 28, 91, 0, 6, 337, 149, 0, 0, 26, 198, 196, 3, 88, 0, 2367, 657, 10, 14, 5, 2, 6, 0, 1891, 0, 655, 185, 3, 215, 547, 14, 0, 32, 0, 0, 0, 0, 78, 1077, 0, 0, 133, 3, 0, 0, 14, 21, 5, 0, 1, 0, 5, 22, 0, 155, 109, 0, 23, 431, 17, 12, 7, 2, 4, 210, 4, 0, 77, 0, 0, 0, 10, 0, 0, 0, 11, 16, 20, 14, 0, 89, 247, 11, 531, 146, 41, 123, 149, 0, 7, 360, 0, 71, 10, 3, 93, 54, 0, 0, 26, 536, 0, 78, 2, 18, 5, 6, 33, 3, 150, 12, 3, 172, 1, 563, 12, 7, 114, 0, 10, 6, 12, 0, 6, 6, 76, 2, 0, 17, 0, 0, 0, 908, 220, 1, 523, 102, 77, 4, 7, 0, 277, 1103, 0, 16, 2, 222, 66, 2, 3, 45, 10, 0, 56, 457, 62, 25, 326, 201, 0, 4, 15, 10, 540, 261, 5, 0, 27, 140, 93, 0, 4, 16, 544, 8, 120, 28, 1, 3, 23, 0, 83, 4, 0, 38, 3036, 239, 255, 0, 7, 40, 0, 0, 287, 160, 108, 649, 3, 1, 0, 215, 1340, 0, 0, 317, 96, 0, 3, 17, 16, 17, 0, 3, 0, 0, 0, 3, 0, 76, 68, 1, 6, 12, 85, 62, 9, 5, 0, 5, 0, 9, 7, 6, 15, 164, 446, 18, 3, 2, 153, 0, 75, 6, 12, 1, 0, 543, 50, 9, 0, 0, 0, 0, 681, 0, 133, 12, 6, 41, 2, 0, 150, 121, 47, 7, 6, 537, 0, 141, 3, 1024, 33, 0, 4, 1, 230, 40, 35, 3, 47, 12, 30, 327, 0, 328, 16, 95, 0, 41, 17, 17, 0, 13, 178, 1, 114, 78, 457, 1, 8, 25, 3, 240, 2, 0, 259, 111, 33, 56, 5, 15, 0, 12, 0, 218, 0, 1, 0, 13, 1, 0, 118, 13, 8, 11, 8, 0, 71, 11, 2, 140, 145, 12, 0, 0, 0, 14, 4, 496, 164, 9, 9, 0, 131, 1640, 0, 1121, 0, 12, 408, 3, 10, 111, 378, 44, 0, 280, 17, 1, 9, 7, 8, 19, 12, 55, 3, 100, 634, 0, 361, 6, 2, 0, 0, 182, 5, 53, 1751, 595, 5, 1053, 1, 19, 114, 851, 645, 170, 148, 413, 0, 10, 2, 112, 90, 220, 5, 62, 40, 0, 114, 3, 0, 53, 8, 19, 3, 1, 0, 0, 14, 0, 100, 137, 254, 9, 65, 283, 13, 53, 0, 46, 556, 211, 3, 82, 194, 8, 6, 1, 57, 257, 62, 5, 0, 16, 7, 56, 612, 44, 168, 8, 143, 11, 4, 348, 0, 230, 35, 67, 422, 3, 87, 10, 5, 72, 148, 1713, 5, 8, 0, 111, 414, 4, 1, 0, 25, 0, 277, 106, 5, 66, 64, 1, 11, 191, 3, 2, 102, 37, 315, 362, 5, 0, 0, 11, 0, 9, 517, 11, 0, 16, 119, 19, 218, 95, 3, 176, 5, 4, 627, 559, 10, 184, 1308, 0, 0, 0, 0, 2, 341, 7, 210, 68, 52, 150, 327, 12, 0, 112, 405, 11, 0, 0, 40, 6, 0, 2997, 0, 15, 0, 84, 16, 59, 142, 194, 439, 19, 68, 0, 0, 60, 17, 10, 5, 21, 63, 137, 6, 280, 70, 37, 7, 2, 7, 7, 168, 23, 132, 70, 23, 8, 49, 10, 79, 5, 1, 470, 9, 12, 10, 59, 8, 196, 5, 17, 0, 16, 0, 0, 197, 104, 119, 9, 63, 14, 3, 182, 408, 72, 0, 6, 17, 7, 7, 447, 534, 368, 76, 137, 3, 1, 1, 0, 219, 2, 3, 275, 0, 1, 0, 25, 12, 144, 6, 595, 4, 2, 4, 2, 216, 4, 85, 2, 8, 353, 204, 6, 17, 0, 69, 9, 36, 216, 0, 202, 439, 0, 1, 15, 210, 0, 12, 68, 93, 93, 6, 158, 139, 13, 67, 5, 97, 272, 20, 459, 21, 298, 27, 6, 108, 136, 362, 1, 42, 79, 18, 185, 64, 0, 1211, 4, 1214, 576, 3, 327, 9, 199, 4, 156, 6, 237, 0, 1030, 26, 19, 280, 13, 24, 83, 6, 37, 127, 297, 0, 15, 84, 9, 6, 0, 2, 24, 352]
Mean: 124.96241345202769, SD: 361.5915152609576
"""

"""
for 50 easy:
Backtracks per puzzle: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Mean: 0.0, SD: 0.0
"""
