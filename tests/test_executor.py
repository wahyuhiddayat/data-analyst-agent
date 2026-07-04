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


def test_generated_files_do_not_pollute_host_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with KernelSession() as session:
        assert session.run("open('leaked.txt', 'w').write('x')").ok
    assert not (tmp_path / "leaked.txt").exists()
