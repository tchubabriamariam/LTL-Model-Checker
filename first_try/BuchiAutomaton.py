class BuchiAutomaton:
    def __init__(self):
        self.states = set()
        self.initial_states = set()
        self.accepting_states = set()
        self.transitions = [] # List of (from_state, to_state, condition_function)

    def add_transition(self, f, t, condition):
        self.transitions.append((f, t, condition))

# This Automaton accepts if the state is NOT red
ba = BuchiAutomaton()
ba.states = {"q0"}
ba.initial_states = {"q0"}
ba.accepting_states = {"q0"} # It "stays" in a violation state

# Condition: "Is 'red' NOT in the labels of the current state?"
ba.add_transition("q0", "q0", lambda labels: "red" not in labels) 