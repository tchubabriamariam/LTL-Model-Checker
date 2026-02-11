from dataclasses import dataclass, field
from typing import Optional

"""
this is a simple LTL model checker for finite-state models, using the lasso method.
here i use symbols which are from internet which helps visualize the results better.
"""
@dataclass
class KripkeStructure:
    """
    M = (states, init, trans, label)
      states : list of state names
      init   : list of initial state names
      trans  : dict  state -> list of successor states
      label  : dict  state -> set of atomic propositions true in that state
    """
    states: list
    init:   list
    trans:  dict   # {state: list(str)}
    label:  dict   # {state: set(str)}


@dataclass
class LTLFormula:
    """A node in the LTL syntax tree."""
    op:    str                        # ATOM, !, &&, ||, ->, X, F, G, U
    name:  Optional[str] = None       # for ATOM nodes, example 'p', 'q'
    left:  Optional['LTLFormula'] = None
    right: Optional['LTLFormula'] = None

    def __repr__(self):
        if self.op == 'ATOM': return self.name
        if self.op == '!':    return f'!{self.left}'
        if self.op in ('X','F','G'): return f'{self.op}({self.left})'
        return f'({self.left} {self.op} {self.right})'


# ─── LTL PARSER ───────────────────────────────────────────────────────────────
# Grammar (right-associative, precedence low->high):
#   formula = formula -> formula
#           | formula || formula
#           | formula && formula
#           | formula U formula
#           | ! formula | X formula | F formula | G formula
#           | atom | ( formula )

def tokenize(src: str) -> list:
    tokens = []
    i = 0
    while i < len(src):
        if src[i].isspace():
            i += 1
        elif src[i:i+2] == '->':
            tokens.append('->')
            i += 2
        elif src[i:i+2] == '&&':
            tokens.append('&&')
            i += 2
        elif src[i:i+2] == '||':
            tokens.append('||')
            i += 2
        elif src[i] in '!()':
            tokens.append(src[i])
            i += 1
        elif src[i] in 'XFGUxfgu':
            tokens.append(src[i].upper())
            i += 1
        elif src[i].isalpha() or src[i] == '_':
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] == '_'):
                j += 1
            tokens.append(('ATOM', src[i:j]))
            i = j
        else:
            raise SyntaxError(f"Unknown character '{src[i]}' at position {i}")
    tokens.append('EOF')
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def eat(self, expected=None):
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise SyntaxError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self):
        f = self.parse_impl()
        if self.peek() != 'EOF':
            raise SyntaxError(f"Unexpected token '{self.peek()}'")
        return f

    def parse_impl(self):
        left = self.parse_or()
        if self.peek() == '->':
            self.eat('->')
            right = self.parse_impl()   # right-associative
            return LTLFormula('->', left=left, right=right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == '||':
            self.eat('||')
            right = self.parse_and()
            left = LTLFormula('||', left=left, right=right)
        return left

    def parse_and(self):
        left = self.parse_until()
        while self.peek() == '&&':
            self.eat('&&')
            right = self.parse_until()
            left = LTLFormula('&&', left=left, right=right)
        return left

    def parse_until(self):
        left = self.parse_unary()
        while self.peek() == 'U':
            self.eat('U')
            right = self.parse_unary()
            left = LTLFormula('U', left=left, right=right)
        return left

    def parse_unary(self):
        tok = self.peek()
        if tok == '!':
            self.eat('!')
            return LTLFormula('!', left=self.parse_unary())
        if tok in ('X', 'F', 'G'):
            self.eat(tok)
            return LTLFormula(tok, left=self.parse_unary())
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()
        if tok == '(':
            self.eat('(')
            f = self.parse_impl()
            self.eat(')')
            return f
        if isinstance(tok, tuple) and tok[0] == 'ATOM':
            self.eat()
            return LTLFormula('ATOM', name=tok[1])
        raise SyntaxError(f"Unexpected token '{tok}'")


def parse_ltl(src: str) -> LTLFormula:
    return Parser(tokenize(src)).parse()


# Find all lasso paths: (prefix_states, loop_start_index)
# The infinite path is: prefix[0], prefix[1], ..., prefix[n-1],
#                        prefix[loop_start], prefix[loop_start+1], ...  (repeating)

def enumerate_lassos(model: KripkeStructure) -> list:
    """
    DFS from each initial state. When we revisit a state, we've found a lasso.
    Returns list of (seq, loop_start) pairs.
    """
    lassos = []

    def dfs(state, path):
        # Check if we've looped back
        if state in path:
            loop_start = path.index(state)
            lassos.append((list(path), loop_start))
            return
        # Limit depth to avoid combinatorial explosion on large models
        if len(path) > len(model.states) * 2:
            return
        succs = model.trans.get(state, [])
        if not succs:
            # Deadend we treat this as self loop
            lassos.append((path + [state], len(path)))
            return
        for succ in succs:
            dfs(succ, path + [state])

    for s in model.init:
        dfs(s, [])

    return lassos


def path_at(seq: list, loop_start: int, i: int) -> str:
    """Return the state at position i on the infinite lasso path."""
    if i < len(seq):
        return seq[i]
    loop_len = len(seq) - loop_start
    offset = (i - loop_start) % loop_len
    return seq[loop_start + offset]

def eval_ltl(phi: LTLFormula, seq: list, loop_start: int,
             model: KripkeStructure, pos: int) -> bool:
    """
    Evaluate phi on the infinite lasso path starting at position pos.

    The lasso: seq[0..n-1] then loops from loop_start forever.

    For G and F, it suffices to check one full loop period past the
    lasso start, because after that the path repeats.
    """
    # How far we need to look to cover one full loop
    check_horizon = len(seq) + (len(seq) - loop_start) + 1

    state = path_at(seq, loop_start, pos)
    aps   = model.label.get(state, set())

    if phi.op == 'ATOM':
        return phi.name in aps

    if phi.op == '!':
        return not eval_ltl(phi.left, seq, loop_start, model, pos)

    if phi.op == '&&':
        return (eval_ltl(phi.left,  seq, loop_start, model, pos) and
                eval_ltl(phi.right, seq, loop_start, model, pos))

    if phi.op == '||':
        return (eval_ltl(phi.left,  seq, loop_start, model, pos) or
                eval_ltl(phi.right, seq, loop_start, model, pos))

    if phi.op == '->':
        return (not eval_ltl(phi.left, seq, loop_start, model, pos) or
                eval_ltl(phi.right, seq, loop_start, model, pos))

    if phi.op == 'X':
        # shifting path by one step
        return eval_ltl(phi.left, seq, loop_start, model, pos + 1)

    if phi.op == 'F':
        # Finally means exists i >= pos such that phi holds at i
        for i in range(pos, pos + check_horizon):
            if eval_ltl(phi.left, seq, loop_start, model, i):
                return True
        return False

    if phi.op == 'G':
        # Globally means for all i >= pos, phi holds at i
        for i in range(pos, pos + check_horizon):
            if not eval_ltl(phi.left, seq, loop_start, model, i):
                return False
        return True

    if phi.op == 'U':
        # Until means exists j >= pos s.t. right holds at j,
        #        and left holds at all i in [pos, j)
        for j in range(pos, pos + check_horizon):
            if eval_ltl(phi.right, seq, loop_start, model, j):
                # Check left holds from pos to j-1
                if all(eval_ltl(phi.left, seq, loop_start, model, i)
                       for i in range(pos, j)):
                    return True
        return False

    raise ValueError(f"Unknown operator: {phi.op}")

def model_check(model: KripkeStructure, phi: LTLFormula,
                verbose: bool = True) -> bool:
    """
    Check M |= phi.
    Returns True if all lasso paths from all initial states satisfy phi.
    """
    lassos = enumerate_lassos(model)

    if not lassos:
        print("  No paths found.")
        return False

    all_hold = True
    for seq, loop_start in lassos:
        result = eval_ltl(phi, seq, loop_start, model, pos=0)
        if verbose:
            loop_state = seq[loop_start]
            path_str = ' -> '.join(seq) + f' -> [{loop_state}...] (loop)'
            status = '✓' if result else '✗'
            print(f"  {status} Path: {path_str}")
        if not result:
            all_hold = False

    return all_hold


# this is examples part
def run_example(title, model, formula_str, expected=None):
    phi = parse_ltl(formula_str)
    print(f"\n{'='*60}")
    print(f"Example: {title}")
    print(f"Formula: {formula_str}  (parsed: {phi})")
    print(f"States:  {model.states}")
    print(f"Init:    {model.init}")
    print(f"Trans:   {dict(model.trans)}")
    print(f"Labels:  {dict(model.label)}")
    print(f"Paths:")
    result = model_check(model, phi)
    verdict = "TRUE  (M |= phi)" if result else "FALSE (M ⊭ phi)"
    print(f"Result:  {verdict}")
    if expected is not None:
        ok = "OK" if result == expected else "MISMATCH"
        print(f"         [{ok}: expected {expected}]")


if __name__ == '__main__':
    # Example 1
    # Response property  G(p -> F q)
    # A simple 3-state cycle where p always leads to q
    # Should be true
    m1 = KripkeStructure(
        states = ['s0', 's1', 's2'],
        init   = ['s0'],
        trans  = {'s0': ['s1'], 's1': ['s2'], 's2': ['s0']},
        label  = {'s0': {'p'}, 's1': {'p', 'q'}, 's2': {'q'}}
    )
    run_example("Response: G(p -> F q)", m1, "G(p -> F q)", expected=True)

    # Example 2
    # Safety  G !error
    # A model that can reach an error state
    # Should be false
    m2 = KripkeStructure(
        states = ['ok', 'err'],
        init   = ['ok'],
        trans  = {'ok': ['ok', 'err'], 'err': ['err']},
        label  = {'ok': set(), 'err': {'error'}}
    )
    run_example("Safety: G !error (error reachable)", m2, "G !error", expected=False)

    # Example 3
    # Liveness  G F done
    # Two states alternating; 'done' holds in one of them
    # Should be true
    m3 = KripkeStructure(
        states = ['s0', 's1'],
        init   = ['s0'],
        trans  = {'s0': ['s1'], 's1': ['s0']},
        label  = {'s0': set(), 's1': {'done'}}
    )
    run_example("Liveness: G F done", m3, "G F done", expected=True)

    # Example 4
    # Until  p U q
    # p holds in a and b, q holds in c. Path: a->b->c (then loops)
    # Should be true
    m4 = KripkeStructure(
        states = ['a', 'b', 'c'],
        init   = ['a'],
        trans  = {'a': ['b'], 'b': ['c'], 'c': ['c']},
        label  = {'a': {'p'}, 'b': {'p'}, 'c': {'q'}}
    )
    run_example("Until: p U q", m4, "p U q", expected=True)

    # Example 5
    # Next  X q
    # From initial state s0, is q true in the very next step?
    # s0->{p}, s1->{q}. Should be true
    m5 = KripkeStructure(
        states = ['s0', 's1'],
        init   = ['s0'],
        trans  = {'s0': ['s1'], 's1': ['s0']},
        label  = {'s0': {'p'}, 's1': {'q'}}
    )
    run_example("Next: X q", m5, "X q", expected=True)

    # Example 6
    # Mutual exclusion  G !(crit1 && crit2)
    # Two processes model ensures they're never both critical.
    # Should be true
    m6 = KripkeStructure(
        states = ['idle', 'one', 'two'],
        init   = ['idle'],
        trans  = {'idle': ['one', 'two'], 'one': ['idle'], 'two': ['idle']},
        label  = {'idle': set(), 'one': {'crit1'}, 'two': {'crit2'}}
    )
    run_example("Mutual exclusion: G !(crit1 && crit2)", m6,
                "G !(crit1 && crit2)", expected=True)

    print(f"\n{'='*60}")
    print("Done.")
