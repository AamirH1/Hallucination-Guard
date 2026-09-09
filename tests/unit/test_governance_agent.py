from app.agents.governance import GovernanceAgent, GovernanceDecision


def test_approve_when_grounding_high():
    agent = GovernanceAgent(grounding_threshold=0.85, retrieval_threshold=0.7, max_regeneration_attempts=2)
    outcome = agent.decide(0.9, 0.9, 0, False, [])
    assert outcome.decision == GovernanceDecision.APPROVE


def test_refuse_on_low_retrieval_confidence():
    agent = GovernanceAgent(grounding_threshold=0.85, retrieval_threshold=0.7, max_regeneration_attempts=2)
    outcome = agent.decide(0.95, 0.3, 0, False, [])
    assert outcome.decision == GovernanceDecision.REFUSE


def test_refuse_on_insufficient_evidence_flag():
    agent = GovernanceAgent()
    outcome = agent.decide(0.0, 0.9, 0, True, [])
    assert outcome.decision == GovernanceDecision.REFUSE
    assert outcome.reason == "insufficient_evidence"


def test_revise_when_grounding_below_threshold_and_attempts_remain():
    agent = GovernanceAgent(grounding_threshold=0.85, retrieval_threshold=0.7, max_regeneration_attempts=2)
    outcome = agent.decide(0.5, 0.9, 0, False, ["claim a"])
    assert outcome.decision == GovernanceDecision.REVISE
    assert outcome.unsupported_claims == ["claim a"]


def test_forces_refuse_after_max_attempts_never_loops_forever():
    agent = GovernanceAgent(grounding_threshold=0.85, retrieval_threshold=0.7, max_regeneration_attempts=2)
    outcome = agent.decide(0.5, 0.9, 2, False, ["claim a"])
    assert outcome.decision == GovernanceDecision.REFUSE
    assert outcome.reason == "max_regeneration_attempts_exceeded"
