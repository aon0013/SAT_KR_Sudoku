def parse_dimacs(file_path):
    """Load clauses from a DIMACS file."""
    clauses_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(('p', 'c')):
                continue
            clause = [int(x) for x in line.split()[:-1]]
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

def dp_solver(cnf_clauses, assignments):
    """Simplified Davis-Putnam solver with enhanced error handling."""
    #unsatisfiability due to empty clauses
    if has_empty_clause(cnf_clauses):
        return False, []
    #clauses satisfied?
    if not cnf_clauses:  # If `cnf_clauses` is empty, all clauses are satisfied
        return True, assignments

    #check for tautology
    cnf_clauses = [c for c in cnf_clauses if not any(-lit in c for lit in c)]

    #unit propagation
    unit_literals = [c[0] for c in cnf_clauses if len(c) == 1]
    while unit_literals:
        unit = unit_literals.pop(0)
        cnf_clauses, new_assignments = apply_literal(cnf_clauses, assignments[:], unit)
        if has_empty_clause(cnf_clauses):  #check again after propagation
            return False, []
        unit_literals = [c[0] for c in cnf_clauses if len(c) == 1]
        assignments = new_assignments  #update assignments

    #check pure literal assignment
    literals = set(lit for clause in cnf_clauses for lit in clause)
    pure_literals = [l for l in literals if -l not in literals]
    for pure in pure_literals:
        cnf_clauses, new_assignments = apply_literal(cnf_clauses, assignments[:], pure)
        assignments = new_assignments  #update assignment

    #branching
    if not cnf_clauses:
        return True, assignments  #clauses satisfied

    literal = abs(cnf_clauses[0][0])  #select first literal from first clause

    #assigning literal to true
    sat, final_assignments = dp_solver(*apply_literal(cnf_clauses, assignments[:], literal))
    if sat:
        return True, final_assignments

    return dp_solver(*apply_literal(cnf_clauses, assignments[:], -literal))

def apply_literal(cnf_clauses, assignments, literal):
    """Apply a literal by updating clauses and assignments."""
    updated_clauses = []
    new_assignments = assignments[:]  #copy to avoid in-place modification
    new_assignments.append(literal)

    for clause in cnf_clauses:
        if literal in clause:
            continue  #clause satisfied, skip
        new_clause = [lit for lit in clause if lit != -literal]
        if not new_clause and len(clause) > 1:
            #if we create an empty clause and it was not a unit clause, return the original list to indicate conflict
            return cnf_clauses, assignments
        updated_clauses.append(new_clause)

    return updated_clauses, new_assignments

def solve_puzzles(sudoku_rules, sudoku_puzzles, output_prefix):
    """Solve multiple Sudoku puzzles and save solutions_easy in DIMACS format."""
    rules = parse_dimacs(sudoku_rules)
    puzzles = load_puzzles(sudoku_puzzles)

    for idx, puzzle in enumerate(puzzles):
        print(f"Solving puzzle {idx + 1}")
        puzzle_clauses = sudoku_to_cnf(puzzle)
        clauses = rules + puzzle_clauses
        assignments = []

        solvable, solution = dp_solver(clauses, assignments)

        #output to DIMACS
        output_filename = f"{output_prefix}_puzzle_{idx + 1}_solution.txt"
        if solvable:
            print(f"Puzzle {idx + 1} is solvable.")
            save_solution(output_filename, solution)
            print(f"Solution written to {output_filename}")
        else:
            print(f"Puzzle {idx + 1} is unsolvable.")
            with open(output_filename, 'w') as f:
                f.write("UNSAT\n")

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
    for row in grid:
        print(" ".join(row))
    print("\n")

solve_puzzles(
    "/Users/noah/PycharmProjects/SAT_Group29/sudoku-rules-9x9.txt",
    "/Users/noah/PycharmProjects/SAT_Group29/top91.sdk.txt",
    "/Users/noah/PycharmProjects/SAT_Group29/solutions_easy/solution"
)