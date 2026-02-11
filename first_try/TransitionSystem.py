class TransitionSystem: # TS as in the book
    def __init__(self):
        self.states = set()
        self.initial_states = set()
        self.transitions = {}  # Map state -> list of next states
        self.labels = {}       # Map state -> set of atomic propositions

    def add_state(self, name, labels=None, is_initial=False):
        self.states.add(name)
        if is_initial:
            self.initial_states.add(name)
        self.labels[name] = set(labels) if labels else set()
        if name not in self.transitions:
            self.transitions[name] = []

    def add_transition(self, from_state, to_state):
        if from_state in self.states and to_state in self.states:
            self.transitions[from_state].append(to_state)

# Test model for traffic light system
# s0 is Red, s1 is Green, s2 is Yellow
ts = TransitionSystem()
ts.add_state("s0", labels={"red"}, is_initial=True)
ts.add_state("s1", labels={"green"})
ts.add_state("s2", labels={"yellow"})

ts.add_transition("s0", "s1") # Red to Green
ts.add_transition("s1", "s2") # Green to Yellow
ts.add_transition("s2", "s0") # Yellow back to Red


ts.add_transition("s1", "s1") # Green can stay Green this is a bug 