from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)

def solve_hungarian_method(cost_matrix):
    """Solve the assignment problem using the Hungarian Method."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    assignments = [(r + 1, c + 1, cost_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
    return assignments, total_cost

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        size = int(request.form.get("size"))
        matrix = []
        
        # Collecting matrix values from the form
        for i in range(size):
            row = list(map(int, request.form.getlist(f"row{i}")))
            matrix.append(row)
        
        cost_matrix = np.array(matrix)
        assignments, total_cost = solve_hungarian_method(cost_matrix)
        
        return render_template("result.html", assignments=assignments, total_cost=total_cost)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
