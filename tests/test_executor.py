from data_analyst_agent.sandbox.executor import KernelSession


def test_state_persists_across_runs():
    with KernelSession() as session:
        assert session.run("x = 21").ok
        result = session.run("print(x * 2)")
        assert result.ok
        assert "42" in result.output


def test_error_is_captured():
    with KernelSession() as session:
        result = session.run("undefined_name")
        assert not result.ok
        assert "NameError" in result.error
