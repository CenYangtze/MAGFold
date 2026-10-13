lifelongagentbench_db_few_shots = [
"""LifelongAgentBench / DB task
Instruction: Which employees are in the Sales department? Return employee names ordered by employee id.
Table: employees
Columns:
- employee_id INT
- name TEXT
- department TEXT

Query[SELECT employee_id, name, department FROM employees LIMIT 5]
Finish[SELECT name FROM employees WHERE department = 'Sales' ORDER BY employee_id]""",
"""LifelongAgentBench / DB task
Instruction: Increase the stock to 50 for item_id 12.
Table: inventory
Columns:
- item_id INT
- item_name TEXT
- stock INT

Query[UPDATE inventory SET stock = 50 WHERE item_id = 12]
Finish[done]""",
]


lifelongagentbench_os_few_shots = [
"""LifelongAgentBench / OS task
Instruction: Create a file named report.txt containing the word ready.

Bash[printf 'ready\\n' > report.txt]
Finish[done]""",
]


lifelongagentbench_kg_few_shots = [
"""LifelongAgentBench / Knowledge Graph task
Question: which films were directed by the given director?
Entities:
- Example Director: m.example_director

Action: get_relations(m.example_director)
Action: get_neighbors(m.example_director, film.director.film)
Finish[m.example_film_1, m.example_film_2]""",
]


lifelongagentbench_solver_system_prompt = """
Solve LifelongAgentBench tasks by interacting with the provided environment. The task description tells you which sub-environment you are in and lists the valid action syntax.

General rules:
1. Use only the actions supported by the current task description.
2. Keep actions concise and machine-readable. Do not include markdown fences unless the action format explicitly asks for a shell script.
3. DB tasks: use Query[<SQL>] to inspect or mutate the table, then Finish[<SQL>] for final SQL. For mutation tasks, Query[UPDATE/INSERT/DELETE ...] followed by Finish[done] is also valid.
4. OS tasks: use Bash[<script>] for shell operations, then Finish[done].
5. Knowledge Graph tasks: use Action: api_name(arguments) for KG API calls, or Finish[<entity_id>, ...] when submitting final Freebase IDs.
6. Do NOT output <think> or </think> tags.
"""


def get_lifelongagentbench_few_shots(task_config: dict, few_shots_num: int) -> list[str]:
    lab_env = task_config.get("lab_env", "db_bench")
    if lab_env == "os_interaction":
        return lifelongagentbench_os_few_shots[:few_shots_num]
    if lab_env == "knowledge_graph":
        return lifelongagentbench_kg_few_shots[:few_shots_num]
    return lifelongagentbench_db_few_shots[:few_shots_num]
