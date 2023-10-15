import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpInteger

# Read player data from the "DKSalaries.csv" file
salary_data = pd.read_csv('DKSalaries.csv', skiprows=7)

# Read player projections from the "projections.csv" file
projections_data = pd.read_csv('projections.csv')

# Create a new 'Name' column by concatenating 'first_name' and 'last_name' in 'projections_data'
projections_data['Name'] = projections_data['first_name'] + ' ' + projections_data['last_name']

# Format the 'Name' column in the salary_data DataFrame
salary_data['Name'] = salary_data['Name'].str.strip()

# Before merging, separate DST players from the rest
dst_players = projections_data[projections_data['position'] == 'DST']
remaining_players = projections_data[projections_data['position'] != 'DST']

# Merge the remaining players based on 'Name'
merged_data = pd.merge(salary_data, remaining_players, on='Name')

# Now you can add DST players without considering the last name
for i in range(len(dst_players)):
    merged_data = merged_data.append(dst_players.iloc[i], ignore_index=True)

# Define the salary cap and roster constraints
salary_cap = 50000
max_players_per_position = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}

# Create the PuLP optimization problem
prob = LpProblem("DraftKings_Optimization", LpMaximize)

# Create binary decision variables for each player (1 if selected, 0 if not)
player_vars = LpVariable.dicts("Player", range(len(merged_data)), 0, 1, LpInteger)

# Objective function: maximize projected points
prob += lpSum(player_vars[i] * merged_data.at[i, "ppg_projection"] for i in range(len(merged_data)))

# Constraints
# 1. Stay within the salary cap
prob += lpSum(player_vars[i] * merged_data.at[i, "Salary"] for i in range(len(merged_data))) <= salary_cap

# 2. Roster constraints
for position, max_count in max_players_per_position.items():
    prob += lpSum(player_vars[i] for i in range(len(merged_data)) if merged_data.at[i, "Position"] == position) == max_count

# Solve the optimization problem
prob.solve()

# Extract the selected player indices
selected_players = [i for i in range(len(merged_data)) if player_vars[i].varValue == 1]

# Print the selected lineup
print("Selected Lineup:")
for i in selected_players:
    print(f"Name: {merged_data.at[i, 'Name']}, Position: {merged_data.at[i, 'Position']}, Salary: {merged_data.at[i, 'Salary']}, Projection: {merged_data.at[i, 'ppg_projection']}")

# Print the total projected points
total_points = prob.objective.value()
print("Total Projected Points:", total_points)
