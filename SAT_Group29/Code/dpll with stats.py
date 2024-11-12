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
    """Davis-Putnam solver."""
    stats['backtracks'] += 1

    # Unsatisfiability due to empty clauses
    if has_empty_clause(cnf_clauses): #checks if any clause is empty
        stats['conflicts'] += 1
        return False, []

    # Clauses satisfied?
    if not cnf_clauses:  # All clauses are satisfied # If `cnf_clauses` is empty, all clauses are satisfied
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
                return False, []
            cnf_clauses, assignments = result  # Update clauses and assignments

    # Branching
    if not cnf_clauses:
        return True, assignments  # All clauses are satisfied

    stats['decisions'] += 1

    # Variable selection heuristic: choose the first unassigned literal
    literals = set(abs(lit) for clause in cnf_clauses for lit in clause)
    assigned_vars = set(abs(lit) for lit in assignments)
    unassigned_vars = literals - assigned_vars
    if not unassigned_vars:
        return False, []  # No unassigned variables left but clauses remain
    literal = next(iter(unassigned_vars))

    # Assigning literal to true
    result = apply_literal(cnf_clauses, assignments[:], literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        sat, final_assignments = dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)
        if sat:
            return True, final_assignments

    # Assigning literal to false
    result = apply_literal(cnf_clauses, assignments[:], -literal, stats)
    if result is not None:
        cnf_clauses_new, assignments_new = result
        return dp_solver(cnf_clauses_new, assignments_new, stats, use_pure_literals)

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
            'backtracks': -1,
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

    print(backtracker)

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
    "50 extreme sudokus.txt",
    "solutions_extreme/solution"
)
