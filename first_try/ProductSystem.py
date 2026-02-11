class ProductSystem:
    def __init__(self, ts, ba):
        self.ts = ts
        self.ba = ba

    def get_initial_states(self):
        initials = []
        for s in self.ts.initial_states:
            for q in self.ba.initial_states:
                initials.append((s, q))  
        return initials


    def get_successors(self, pair):
        s, q = pair
        successors = []
        # Look where the TS can go next
        for next_s in self.ts.transitions[s]:
            next_labels = self.ts.labels[next_s]
            # Look where the BA can go based on those new labels
            for q_from, q_to, condition in self.ba.transitions:
                if q_from == q and condition(next_labels):
                    successors.append((next_s, q_to))
        return successors

    def is_accepting(self, pair):
        s, q = pair
        return q in self.ba.accepting_states