from ProductSystem import ProductSystem
from TransitionSystem import TransitionSystem
from BuchiAutomaton import BuchiAutomaton

class ModelChecker:
    def __init__(self, product):
        self.product = product
        self.visited = set()     # States seen in First DFS
        self.on_stack = set()    # States currently being explored (to find cycles)
        self.flagged = set()     # States seen in Second DFS

    def check(self):
        # Start searching from every possible initial pair
        for start_state in self.product.get_initial_states():
            if start_state not in self.visited:
                if self.first_dfs(start_state):
                    return "BUG FOUND: Property Violated!"
        return "SUCCESS: Property Holds."

    def first_dfs(self, state):
        self.visited.add(state)
        self.on_stack.add(state)  # track current path
        for next_state in self.product.get_successors(state):
            if next_state not in self.visited:
                if self.first_dfs(next_state):
                    return True
        if self.product.is_accepting(state):
            if self.second_dfs(state):
                return True
        self.on_stack.remove(state)
        return False

    def second_dfs(self, state):
        self.flagged.add(state)
        for next_state in self.product.get_successors(state):
            if next_state in self.on_stack:  # check cycle on current path only
                return True
            if next_state not in self.flagged:
                if self.second_dfs(next_state):
                    return True
        return False


ts = TransitionSystem()
ts.add_state("s0", labels={"red"}, is_initial=True)
ts.add_state("s1", labels={"green"})
ts.add_state("s2", labels={"yellow"})

ts.add_transition("s0", "s1")
ts.add_transition("s1", "s2")
ts.add_transition("s2", "s0")
# ts.add_transition("s1", "s1")  # bug
ts.add_transition("s2", "s1")  # bug

ba = BuchiAutomaton()
ba.states = {"q0"}
ba.initial_states = {"q0"}
ba.add_transition("q0", "q0", lambda labels: "red" not in labels)
ba.accepting_states = {"q0"}

product = ProductSystem(ts, ba)
checker = ModelChecker(product)
result = checker.check()
print(result)
