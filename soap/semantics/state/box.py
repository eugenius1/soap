from soap.expression.variable import Variable
from soap.label import Identifier
from soap.lattice.map import map
from soap.semantics.error import cast, ErrorSemantics, IntegerInterval
from soap.semantics.state.base import BaseState
from soap.semantics.state.identifier import IdentifierBaseState


class BoxState(BaseState, map(None, (IntegerInterval, ErrorSemantics))):
    __slots__ = ()

    def _cast_key(self, key):
        if isinstance(key, str):
            return Variable(key)
        if isinstance(key, Variable):
            return key
        raise TypeError(
            'Do not know how to convert {!r} into a variable'.format(key))

    def _cast_value(self, value=None, top=False, bottom=False):
        if top or bottom:
            return IntegerInterval(top=top, bottom=bottom)
        return cast(value)

    def is_fixpoint(self, other):
        """Checks if `self` is equal to `other` in the value ranges.

        For potential non-terminating loops, states are not the bottom element
        in the evaluation of loop statements even if a fixpoint is reached.
        This computation would result in a fixpoint of value ranges but
        the resulting error terms are strictly greater. Consequently for
        non-terminating loops the fixpoint for the error terms are always
        [-inf, inf] = ⊤. To gain any useful information about the program we
        wish to disregard the error terms and warn about non-termination.
        """
        if self.is_top() and other.is_top():
            return True
        if self.is_bottom() and other.is_bottom():
            return True
        non_bottom_keys = lambda d: set(
            [k for k, v in d.items() if not v.is_bottom()])
        if non_bottom_keys(self) != non_bottom_keys(other):
            return False
        for k, v in self.items():
            u = other[k]
            if type(v) is not type(u):
                return False
            if isinstance(v, ErrorSemantics):
                if not v.is_top() and not u.is_top():
                    u, v = u.v, v.v
            if u != v:
                return False
        return True

    def widen(self, other):
        """Simple widening operator, jumps to infinity if interval widens.

        self.widen(other) => self ∇ other
        """
        if self.is_top() or other.is_bottom():
            return self
        if self.is_bottom() or other.is_top():
            return other
        mapping = dict(self)
        for k, v in other.items():
            if k not in mapping:
                mapping[k] = v
            else:
                mapping[k] = mapping[k].widen(v)
        return self.__class__(mapping)


class IdentifierBoxState(IdentifierBaseState, BoxState):
    __slots__ = ()

    def _cast_value(self, value=None, top=False, bottom=False):
        if not isinstance(value, Identifier):
            return super()._cast_value(value, top=top, bottom=bottom)
        return value

    def _annotated_transition(self, annotation):
        mapping = dict(self)
        for k, v in self.items():
            if not k.annotation.is_bottom():
                continue
            annotated_key = Identifier(k.variable, annotation=annotation)
            v |= mapping.get(annotated_key, self._cast_value(bottom=True))
            mapping[annotated_key] = v
        return self.__class__(mapping)

    def visit_AssignFlow(self, flow):
        state = super().visit_AssignFlow(flow)
        return state._annotated_transition(flow.annotation)

    def visit_IfFlow(self, flow):
        true_annotation = flow.annotation.attributed_true()
        false_annotation = flow.annotation.attributed_false()
        exit_annotation = flow.annotation.attributed_exit()

        # for each split, annotate respective changes to values
        true_state, false_state = self._split_states(flow)
        true_state = true_state._annotated_transition(true_annotation)
        false_state = false_state._annotated_transition(false_annotation)
        true_state = true_state.transition(flow.true_flow)
        false_state = false_state.transition(flow.false_flow)

        # join transitioned states together, annotate joined values
        state = true_state | false_state
        return state._annotated_transition(exit_annotation)

    def visit_WhileFlow(self, flow):
        # use super()'s method to compute fixpoint for us.
        fixpoint = self._solve_fixpoint(flow)

        # check if we've given up of computing an unrolled fixpoint, and
        # if there is still potential non-terminating condition after the
        # loop kernel; if non-termination is possible, last_entry (without
        # annotations for program statements, which is not relevant to our
        # analysis), should not be tested bottom.
        last_entry_predicate = lambda k: k.label.is_bottom()
        last_entry = fixpoint['last_entry'].filter(last_entry_predicate)
        last_entry._warn_non_termination(flow)

        entry_annotation = flow.annotation.attributed_entry()
        exit_annotation = flow.annotation.attributed_exit()

        # we need to get the annotated joined value mapping from all entries to
        # the loop, so to produce a complete mapping due to the effects of loop
        # statements for loop exit.
        entry = fixpoint['entry']._annotated_transition(entry_annotation)
        # we should not include any current values into this as these values
        # are not relevant after the loop, all relevant loop execution effects
        # are recorded with annotations.
        entry = entry.filter(lambda k: not k.label.is_bottom())

        # annotate loop exit values.
        exit = fixpoint['exit']._annotated_transition(exit_annotation)

        # stitch all things together, without introducing extraneous lubs.
        return entry | exit
